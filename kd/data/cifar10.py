from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_cifar10_dataloaders(batch_size=64, root='kd/data/data', fraction=1/20):
    # Define standard CIFAR-10 normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # mean
                             (0.2023, 0.1994, 0.2010))  # std
    ])

    # Load full datasets
    train_dataset_full = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset_full = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    # Calculate subset size
    train_size = int(len(train_dataset_full) * fraction)
    test_size = int(len(test_dataset_full) * fraction)

    # Randomly select indices for the subset
    train_indices = np.random.choice(len(train_dataset_full), train_size, replace=False)
    test_indices = np.random.choice(len(test_dataset_full), test_size, replace=False)

    # Create subset datasets
    train_dataset = Subset(train_dataset_full, train_indices)
    test_dataset = Subset(test_dataset_full, test_indices)

    # Wrap in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader