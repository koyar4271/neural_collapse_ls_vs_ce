import os
import datetime
import numpy as np # <<< Ensure numpy is imported
import torch
import torchvision
from torch.utils.data import DataLoader, Subset # <<< Ensure Subset is imported
from torchvision import datasets, transforms
from .data_transform import GaussianBlur, get_moco_base_augmentation

data_folder = '/' # for greene,  '../dataset' for local

def create_imbalanced_dataset(dataset, imbalance_ratio, num_classes=10):
    """Creates an exponentially imbalanced dataset from the given dataset."""
    targets = np.array(dataset.targets)
    classes, class_counts = np.unique(targets, return_counts=True)

    # Original maximum class sample count (e.g., 5000 for CIFAR-10)
    max_count = class_counts.max()

    # Calculate target sample counts for each class exponentially
    # For class index i (0, 1, ..., K-1), sample count is N_max * (ratio^(i / (K-1)))
    target_counts = [int(max_count * (imbalance_ratio ** (i / (num_classes - 1.0)))) for i in range(num_classes)]

    print(f"Creating exponentially imbalanced dataset. Imbalance ratio: {imbalance_ratio}")
    print(f"Target counts per class: {target_counts}")

    indices_to_keep = []

    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        target_count = target_counts[cls]

        # Limit target count to the number of available samples if necessary
        if target_count > len(cls_indices):
             print(f"Warning: Target count {target_count} for class {cls} exceeds available samples {len(cls_indices)}. Using all available samples.")
             target_count = len(cls_indices)

        # Randomly select indices up to the target count (without replacement)
        # Also handle the case where target_count might be 0
        if target_count > 0:
            chosen_indices = np.random.choice(cls_indices, size=target_count, replace=False)
            indices_to_keep.extend(chosen_indices)

    print(f"Total samples after imbalance: {len(indices_to_keep)}")

    # Extract the subset from the original dataset using the selected indices
    imbalanced_subset = Subset(dataset, indices_to_keep)
    # Add the 'targets' attribute to the Subset object for later access (e.g., for adding noise)
    imbalanced_subset.targets = targets[indices_to_keep]

    return imbalanced_subset


def get_dataloader(args):

    if args.dset in ["cifar10", "cifar100"]:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        # Set data augmentations based on args.aug
        if hasattr(args, 'aug') and args.aug == 'pc':  # padded crop
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize])
        elif hasattr(args, 'aug') and args.aug == 'rs':  # resized crop
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
                transforms.ToTensor(),
                normalize])
        else: # (args.aug is None) or (args.aug == 'null'):
             transform_train = transforms.Compose([transforms.ToTensor(), normalize])

        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        num_classes = 10 if args.dset == 'cifar10' else 100
        dataset_class = datasets.CIFAR10 if args.dset == 'cifar10' else datasets.CIFAR100

        # 1. Load the original *full* training dataset
        train_dataset_full = dataset_class('../dataset', train=True, download=True, transform=transform_train)

        # 2. Apply dataset imbalance based on the imbalance ratio
        if hasattr(args, 'imbalance_ratio') and args.imbalance_ratio < 1.0: # Check if the argument exists and is less than 1.0
            train_dataset = create_imbalanced_dataset(
                train_dataset_full,
                imbalance_ratio=args.imbalance_ratio,
                num_classes=num_classes
            )
            print(f"Using imbalanced dataset (ratio={args.imbalance_ratio}).")
        else:
            train_dataset = train_dataset_full # Use the original dataset if ratio is 1.0 or arg doesn't exist
            print("Using balanced dataset.")

        # 3. Apply label noise (after potential imbalancing)
        if hasattr(args, 'noise_ratio') and args.noise_ratio > 0.0: # Check if the argument exists and is greater than 0.0
            print(f"Adding symmetric label noise with ratio: {args.noise_ratio}")

            # Ensure the dataset object has the 'targets' attribute (should exist after create_imbalanced_dataset)
            if not hasattr(train_dataset, 'targets'):
                 raise ValueError("Dataset object must have 'targets' attribute for adding noise.")

            original_labels = np.array(train_dataset.targets)
            noisy_labels = original_labels.copy()

            num_noise = int(args.noise_ratio * len(original_labels))
            # Clip the number of noisy samples to the actual dataset size (important after imbalancing)
            num_noise = min(num_noise, len(original_labels))

            if num_noise > 0:
                noise_indices = np.random.choice(len(original_labels), size=num_noise, replace=False)

                for i in noise_indices:
                    original_label = original_labels[i]
                    new_label = np.random.randint(0, num_classes)
                    while new_label == original_label: # Ensure the new label is different
                        new_label = np.random.randint(0, num_classes)
                    noisy_labels[i] = new_label

                # Overwrite the targets in the dataset object
                train_dataset.targets = noisy_labels.tolist()
                print("Label noise has been added.")
            else:
                print("No noise added (num_noise is zero or less).")

        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, # Use the potentially modified dataset
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_class('../dataset', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
        )

    elif args.dset == 'stl10': # Changed 'if' to 'elif' and corrected indentation
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2434, 0.2615])
        transform = transforms.Compose([
            # transforms.RandomCrop(96, padding=4), # for stl10
            transforms.ToTensor(),
            normalize
        ])
        test_tranform = get_moco_base_augmentation(min_scale=args.min_scale, normalize=normalize, size=96) if hasattr(args, 'test_ood') and args.test_ood else transform

        # Load STL10 training dataset (imbalance/noise not implemented here)
        train_dataset = datasets.STL10('data', split='train', download=True, transform=transform)
        # TODO: Add imbalance/noise logic for STL10 if needed

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True
            )
        test_loader = torch.utils.data.DataLoader(
            datasets.STL10('data', split='test', download=True, transform=test_tranform),
            batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
            )

    elif args.dset == 'fmnist':
        fashion_mnist = torchvision.datasets.FashionMNIST(download=True, train=True, root="data").train_data.float()
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((fashion_mnist.mean() / 255,), (fashion_mnist.std() / 255,))
                                        ])
        # Load FMNIST training dataset (imbalance/noise not implemented here)
        train_dataset = datasets.FashionMNIST("data", download=True, train=True, transform=transform)
        # TODO: Add imbalance/noise logic for FMNIST if needed

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST("data", download=True, train=False, transform=transform),
            batch_size=args.batch_size, shuffle=False)

    elif args.dset == 'mnist':
        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
        # Load MNIST training dataset (imbalance/noise not implemented here)
        train_dataset = datasets.MNIST(root='../dataset', train=True, download=True, transform=transform)
        # TODO: Add imbalance/noise logic for MNIST if needed

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        valset = datasets.MNIST(root='../dataset', train=False, download=True, transform=transform)
        test_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    elif args.dset == 'tinyi': # image_size:64 x 64
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = transforms.Compose([transforms.ToTensor(),
                                        normalize,
                                        ])
        test_tranform = get_moco_base_augmentation(min_scale=args.min_scale, normalize=normalize, size=64) if hasattr(args, 'test_ood') and args.test_ood else transform

        # Load TinyImageNet training dataset (imbalance/noise needs custom implementation for ImageFolder)
        # Note: ImageFolder has a 'targets' attribute, but careful implementation is needed.
        train_dataset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'train'), transform)
        # TODO: Add imbalance/noise logic for TinyImageNet (Note: 200 classes)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'val'), test_tranform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    else:
        raise ValueError(f"Unknown dataset: {args.dset}")

    return train_loader, test_loader