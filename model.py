import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GroupNorm32(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, **kargs):
        super().__init__(num_groups, num_channels, **kargs)


class ResNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, small_kernel=True, backbone='resnet18', args=None):
        super(ResNet, self).__init__()

        self.loss_type = args.loss if args is not None and hasatter(args, 'loss') else 'ce'

        # Load the pretrained ResNet model
        if args.norm == 'bn':
            resnet_model = models.__dict__[backbone](pretrained=pretrained)
        else:
            resnet_model = models.__dict__[backbone](pretrained=pretrained, norm_layer=GroupNorm32)

        if small_kernel:
            conv1_out_ch = resnet_model.conv1.out_channels
            if args.dset in ['fmnist', 'mnist', 'kmnist']:
                resnet_model.conv1 = nn.Conv2d(1, conv1_out_ch, kernel_size=3, stride=1, padding=1, bias=False)  # Small dataset filter size used by He et al. (2015)
            else:
                resnet_model.conv1 = nn.Conv2d(3, conv1_out_ch, kernel_size=3, stride=1, padding=1, bias=False)  # Small dataset filter size used by He et al. (2015)
        resnet_model.maxpool = nn.Identity()

        # Isolate the feature extraction layers
        self.features = nn.Sequential(*list(resnet_model.children())[:-1])

        # Isolate the classifier layer
        self.classifier = nn.Linear(resnet_model.fc.in_features, num_classes)
        self.feat_dim = resnet_model.fc.in_features

        if args.ETF_fc:
            weight = torch.sqrt(torch.tensor(num_classes / (num_classes - 1))) * (
                    torch.eye(num_classes) - (1 / num_classes) * torch.ones((num_classes, num_classes)))
            weight /= torch.sqrt((1 / num_classes * torch.norm(weight, 'fro') ** 2))

            self.classifier.weight = nn.Parameter(torch.mm(weight, torch.eye(num_classes, resnet_model.fc.in_features)))
            self.classifier.weight.requires_grad_(False) # Freeze the weights of the final layer 
            """ 
            example K = 3, d = 5
            W = [[ 0.8165, 0.4082, 0.4082, 0.0000, 0.0000],
                [-0.4082, 0.8165, -0.4082, 0.0000, 0.0000],
                [-0.4082, -0.4082, 0.8165, 0.0000, 0.0000]]
            """


    def forward(self, x, ret_feat=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        if self.load_type == 'dr':
            x = F.normalize(x, p=2, dim =1)

        out = self.classifier(x)

        if ret_feat:
            return out, x
        else:
            return out


class MLP(nn.Module):
    def __init__(self, hidden, depth=6, fc_bias=True, num_classes=10):
        # Depth means how many layers before final linear layer

        super(MLP, self).__init__()
        layers = [nn.Linear(3072, hidden), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]
        for i in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]

        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden, num_classes, bias=fc_bias)
        print(fc_bias)

    def forward(self, x, ret_feat=False):
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        features = F.normalize(x)
        x = self.classifier(x)
        if ret_feat:
            return x, features
        else:
            return x
        
class MNIST_MLP(nn.Module):
    def __init__(self, hidden, depth=3, fc_bias=True, num_classes=10, args=None):
        # Depth means how many layers before final linear layer

        super(MNIST_MLP, self).__init__()
        
        input_dim = 784
        self.feat_dim = hidden

        layers = []
        layers += [nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
        for i in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]

        layers += [nn.Linear(hidden, hidden)]

        layers += [nn.BatchNorm1d(hidden)]

        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden, num_classes, bias=fc_bias)
        print(fc_bias)

        if args is not None and args.ETF_fc:
            log_print = print if args is None else print
            log_print(f"Setting Fixed ETF Classifier for MNIST_MLP (feat_dim={self.feat_dim}, num_classes={num_classes})")
            
            weight = torch.sqrt(torch.tensor(num_classes / (num_classes - 1))) * (
                    torch.eye(num_classes) - (1 / num_classes) * torch.ones((num_classes, num_classes)))
            
            if self.feat_dim < num_classes:
                 log_print("Warning: Feature dimension (hidden) is less than num_classes. ETF initialization may be complex.")

            if self.feat_dim >= num_classes:
                W_etf = torch.zeros(num_classes, self.feat_dim)
                W_etf[:, :num_classes] = weight
            else:
                W_etf = weight[:, :self.feat_dim]

            self.classifier.weight = nn.Parameter(W_etf)
            self.classifier.weight.requires_grad_(False)
            
            if self.classifier.bias is not None:
                self.classifier.bias.data.fill_(0)


    def forward(self, x, ret_feat=False):
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        features = F.normalize(x)
        out = self.classifier(x)

        if ret_feat:
            return out, x 
        else:
            return out
