import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import os

class MFCCBinaryDataset(Dataset):
    """
    Custom Dataset class for loading MFCC data from binary CIFAR-like format.

    File structure:
    - Each sample: 1 label byte + 3072 feature bytes (32×32×3 normalized MFCC)
    - Total: 492 samples × 3073 bytes = 1,511,916 bytes

    Labels: 0 = positive, 1 = negative
    """

    def __init__(self, bin_file_path='kd/data/data', transform=None):
        """
        Args:
            bin_file_path (str): Path to the binary MFCC file
            transform (callable, optional): Optional transform to be applied on images
        """
        # Validate file exists
        if not os.path.exists(bin_file_path):
            raise FileNotFoundError(f"Binary file not found: {bin_file_path}")

        # Load binary data
        self.data_raw = np.fromfile(bin_file_path, dtype=np.uint8)

        # Validate file size (must be divisible by 3073)
        bytes_per_sample = 3073
        if len(self.data_raw) % bytes_per_sample != 0:
            raise ValueError(
                f"Invalid file size. Expected multiple of {bytes_per_sample}, "
                f"got {len(self.data_raw)}"
            )

        # Reshape to (num_samples, 3073)
        self.data = self.data_raw.reshape(-1, bytes_per_sample)

        # Extract labels (first column) and features (remaining 3072 columns)
        self.labels = torch.from_numpy(self.data[:, 0]).long()
        self.features = torch.from_numpy(self.data[:, 1:]).float()  # Shape: (N, 3072)

        # Reshape features to (32, 32, 3) format
        # Note: features are already in [0, 255] range (uint8 before conversion)
        self.features = self.features.reshape(-1, 32, 32, 3)  # Shape: (N, 32, 32, 3)

        # Convert from [0, 255] to [0, 1] for standard PyTorch compatibility
        self.features = self.features / 255.0

        # Permute from (H, W, C) to (C, H, W) for PyTorch convention
        self.features = self.features.permute(0, 3, 1, 2)  # Shape: (N, 3, 32, 32)

        self.transform = transform
        self.num_samples = len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Feature tensor of shape (3, 32, 32), values in [0, 1]
            label (int): Class label (0 or 1)
        """
        image = self.features[idx]  # Already in [0, 1] range
        label = self.labels[idx].item()

        # Apply transform if provided (e.g., normalization, augmentation)
        if self.transform:
            # Transform expects PIL Image or np.ndarray, so convert
            image = transforms.ToPILImage()(image)
            image = self.transform(image)

        return image, label


def load_wuw_dataloaders(
    bin_file_path='kd/data/data/mfcc_dataset.bin',  # ← Now has default value
    batch_size=64,
    test_fraction=0.2,
    normalize=True,
    seed=42
):
    """
    Load MFCC binary dataset and return train/test DataLoaders.

    Args:
        bin_file_path (str): Path to the MFCC binary file
        batch_size (int): Batch size for DataLoaders. Default: 64
        test_fraction (float): Fraction of data to use for testing. Default: 0.2
        normalize (bool): Whether to apply normalization. Default: True
        seed (int): Random seed for reproducibility. Default: 42

    Returns:
        tuple: (train_loader, test_loader, dataset_info)
            - train_loader (DataLoader): Training data loader
            - test_loader (DataLoader): Test data loader
            - dataset_info (dict): Metadata about the dataset

    Example:
        >>> train_loader, test_loader, info = load_mfcc_dataloaders(
        ...     'mfcc_dataset.bin',
        ...     batch_size=32,
        ...     test_fraction=0.2
        ... )
        >>> for images, labels in train_loader:
        ...     print(images.shape, labels.shape)  # torch.Size([32, 3, 32, 32]), torch.Size([32])
    """

    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define transforms
    if normalize:
        # Normalization for MFCC data
        # Using neutral values since MFCC is already normalized [0, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # Load full dataset
    full_dataset = MFCCBinaryDataset(bin_file_path, transform=transform)

    # Calculate train/test split
    total_size = len(full_dataset)
    test_size = int(total_size * test_fraction)
    train_size = total_size - test_size

    # Randomly split indices
    all_indices = np.arange(total_size)
    np.random.shuffle(all_indices)

    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:]

    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Dataset info
    dataset_info = {
        'total_samples': total_size,
        'train_samples': train_size,
        'test_samples': test_size,
        'batch_size': batch_size,
        'image_shape': (3, 32, 32),
        'num_classes': 2,  # 0=positive, 1=negative
        'class_names': ['positive', 'negative'],
        'normalize': normalize,
        'label_distribution': {
            'positive': int((full_dataset.labels == 0).sum()),
            'negative': int((full_dataset.labels == 1).sum())
        }
    }

    return train_loader, test_loader, dataset_info