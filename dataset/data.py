import os
import datetime
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, Dataset # <<<--- Dataset をインポート
from torchvision import datasets, transforms
from .data_transform import GaussianBlur, get_moco_base_augmentation

data_folder = '/' # for greene,  '../dataset' for local

# ▼▼▼ Wrapper Dataset Class ▼▼▼
class NoisySubsetWrapper(Dataset):
    """Wraps a Subset and returns data with potentially modified (noisy) labels."""
    def __init__(self, subset, noisy_targets_list):
        self.subset = subset
        # Ensure targets are stored as a standard Python list or numpy array for easy indexing
        self.noisy_targets = list(noisy_targets_list) if not isinstance(noisy_targets_list, np.ndarray) else noisy_targets_list

        # Verify lengths match
        if len(self.subset) != len(self.noisy_targets):
             raise ValueError(f"Subset size ({len(self.subset)}) and noisy targets length ({len(self.noisy_targets)}) mismatch!")
        
        # Make targets accessible if needed elsewhere (copying logic from create_imbalanced_dataset)
        self.targets = self.noisy_targets

    def __getitem__(self, index):
        # Get original data using the subset's internal index mapping
        data, _ = self.subset[index] # Retrieve data, ignore original label from subset
        # Get the noisy label using the direct index into our stored list
        noisy_label = self.noisy_targets[index]
        # Ensure the label is returned as a type PyTorch expects (e.g., int or tensor)
        return data, int(noisy_label)

    def __len__(self):
        return len(self.subset)
# ▲▲▲ End Wrapper Class ▲▲▲


# --- create_imbalanced_dataset function remains the same ---
def create_imbalanced_dataset(dataset, imbalance_ratio, num_classes=10):
    """Creates an exponentially imbalanced dataset from the given dataset."""
    targets = np.array(dataset.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    max_count = class_counts.max()
    target_counts = [int(max_count * (imbalance_ratio ** (i / (num_classes - 1.0)))) for i in range(num_classes)]

    print(f"Creating exponentially imbalanced dataset. Imbalance ratio: {imbalance_ratio}")
    print(f"Target counts per class: {target_counts}")
    indices_to_keep = []
    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        target_count = target_counts[cls]
        if target_count > len(cls_indices):
             print(f"Warning: Target count {target_count} for class {cls} exceeds available samples {len(cls_indices)}. Using all available samples.")
             target_count = len(cls_indices)
        if target_count > 0:
            chosen_indices = np.random.choice(cls_indices, size=target_count, replace=False)
            indices_to_keep.extend(chosen_indices)
    print(f"Total samples after imbalance: {len(indices_to_keep)}")
    imbalanced_subset = Subset(dataset, indices_to_keep)
    # Add targets attribute to the Subset for potential later use (like calculating noise ratio)
    imbalanced_subset.targets = targets[indices_to_keep]
    return imbalanced_subset
# --- End of create_imbalanced_dataset ---


def get_dataloader(args):

    if args.dset in ["cifar10", "cifar100"]:
        # ... (normalize, transforms definition remain the same) ...
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        if hasattr(args, 'aug') and args.aug == 'pc':
            transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), normalize])
        elif hasattr(args, 'aug') and args.aug == 'rs':
             transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)), transforms.ToTensor(), normalize])
        else:
             transform_train = transforms.Compose([transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        num_classes = 10 if args.dset == 'cifar10' else 100
        dataset_class = datasets.CIFAR10 if args.dset == 'cifar10' else datasets.CIFAR100

        # 1. Load original full dataset
        train_dataset_full = dataset_class('../dataset', train=True, download=True, transform=transform_train)

        # 2. Apply imbalance if specified
        if hasattr(args, 'imbalance_ratio') and args.imbalance_ratio < 1.0:
            train_dataset_maybe_imbalanced = create_imbalanced_dataset(
                train_dataset_full,
                imbalance_ratio=args.imbalance_ratio,
                num_classes=num_classes
            )
            print(f"Using imbalanced dataset (ratio={args.imbalance_ratio}).")
        else:
            train_dataset_maybe_imbalanced = train_dataset_full
            print("Using balanced dataset.")

        # --- ▼▼▼ Label Noise Application (Modified Section) ▼▼▼ ---
        train_dataset_final = train_dataset_maybe_imbalanced # Start with the (potentially imbalanced) dataset

        if hasattr(args, 'noise_ratio') and args.noise_ratio > 0.0:
            print(f"Preparing to add symmetric label noise with ratio: {args.noise_ratio}")

            # Get the base labels (either from original or the Subset's added attribute)
            if not hasattr(train_dataset_maybe_imbalanced, 'targets'):
                 # This might happen if the base dataset object doesn't have 'targets'
                 # or if create_imbalanced_dataset failed to add it. Needs robust handling.
                 # For torchvision datasets, targets should exist.
                 try:
                     # Attempt to get targets from the underlying dataset if it's a Subset
                     if isinstance(train_dataset_maybe_imbalanced, Subset):
                         original_labels = np.array([train_dataset_maybe_imbalanced.dataset.targets[i] for i in train_dataset_maybe_imbalanced.indices])
                     else:
                          original_labels = np.array(train_dataset_maybe_imbalanced.targets)
                 except AttributeError:
                      raise ValueError("Could not retrieve 'targets' from the dataset object.")
            else:
                 original_labels = np.array(train_dataset_maybe_imbalanced.targets)


            noisy_labels_array = original_labels.copy()
            num_samples_to_noise = int(args.noise_ratio * len(original_labels))
            num_samples_to_noise = min(num_samples_to_noise, len(original_labels)) # Ensure not exceeding dataset size

            if num_samples_to_noise > 0:
                noise_indices = np.random.choice(len(original_labels), size=num_samples_to_noise, replace=False)

                noise_count = 0
                for i in noise_indices:
                    original_label = original_labels[i]
                    new_label = np.random.randint(0, num_classes)
                    while new_label == original_label:
                        new_label = np.random.randint(0, num_classes)
                    noisy_labels_array[i] = new_label
                    noise_count += 1

                # **Use the Wrapper class**
                train_dataset_final = NoisySubsetWrapper(train_dataset_maybe_imbalanced, noisy_labels_array)
                print(f"Label noise applied to {noise_count} samples. Using NoisySubsetWrapper.")
            else:
                print("No noise added (num_noise is zero or less).")
        # --- ▲▲▲ End of Label Noise Application ---

        # Create DataLoaders using the final dataset object
        train_loader = torch.utils.data.DataLoader(
            train_dataset_final, # <<< Use the final (potentially wrapped) dataset
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_class('../dataset', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
        )

    # ... (rest of the dataset handling: stl10, fmnist, etc. remain unchanged) ...
    elif args.dset == 'stl10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2434, 0.2615])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        test_tranform = get_moco_base_augmentation(min_scale=args.min_scale, normalize=normalize, size=96) if hasattr(args, 'test_ood') and args.test_ood else transform
        train_dataset = datasets.STL10('data', split='train', download=True, transform=transform)
        # TODO: Add imbalance/noise logic for STL10 if needed
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(datasets.STL10('data', split='test', download=True, transform=test_tranform), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    # ... (other datasets) ...

    elif args.dset in ['mnist', 'fmnist', 'kmnist']:
        if args.dset == 'mnist':
            DARASER_CLASS = datasets.MNIST
            normalize_mean = (0.1307,)
            normalize_std = (0.3081,)
        elif args.dset == 'fmnist':
            DARASER_CLASS = datasets.FashionMNIST
            normalize_mean = (0.2860,)
            normalize_std = (0.3530,)
        elif args.dset == 'kmnist':
            DARASER_CLASS = datasets.KMNIST
            normalize_mean = (0.1904,)
            normalize_std = (0.3475,)
        transform =transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
        train_dataset_full = DARASER_CLASS(root='../dataset', train=True, download=True, transform=transform)
        valset = DARASER_CLASS(root='../dataset', train=False, download=True, transform=transform)
        num_classes = 10
        if hasattr(args, 'imbalance_ratio') and args.imbalance_ratio < 1.0:
            train_dataset_maybe_imbalanced = create_imbalanced_dataset(
                train_dataset_full,
                imbalance_ratio=args.imbalance_ratio,
                num_classes=num_classes
            )
            print(f"Using imbalanced dataset (ratio={args.imbalance_ratio}).")
        else:
            train_dataset_maybe_imbalanced = train_dataset_full
            print("Using balanced dataset.")
        
        train_dataset_final = train_dataset_maybe_imbalanced

        if hasattr(args, 'noise_ratio') and args.noise_ratio > 0.0:
            print(f"Preparing to add symmetric label noise with ratio: {args.noise_ratio}")

            if not hasattr(train_dataset_maybe_imbalanced, 'targets'):
                raise ValueError("Dataset onject does not have 'targets' attribute for adding noise.")
            
            original_labels = np.array(train_dataset_maybe_imbalanced.targets)
            noisy_labels_array = original_labels.copy()

            num_samples_to_noise = int(args.noise_ratio * len(original_labels))
            num_samples_to_noise = min(num_samples_to_noise, len(original_labels))

            if num_samples_to_noise > 0:
                noise_indices = np.random.choice(len(original_labels), size=num_samples_to_noise, replace=False)

                noise_count = 0
                for i in noise_indices:
                    original_label = original_labels[i]
                    new_label = np.random.randint(0, num_classes)
                    while new_label == original_label:
                        new_label = np.random.randint(0, num_classes)
                    noisy_labels_array[i] = new_label
                    noise_count += 1

                train_dataset_final = NoisySubsetWrapper(train_dataset_maybe_imbalanced, noisy_labels_array)
                print(f"Label noise applied to {noise_count} samples. Using NoisySubsetWrapper.")
            else:
                print("No noise added (num_noise is zero or less).")
        
        train_loader = DataLoader(
            train_dataset_final, 
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
        )
        test_loader = DataLoader(
            valset, 
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dset}")


    return train_loader, test_loader