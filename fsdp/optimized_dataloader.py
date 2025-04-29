import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import numpy as np

class FSPDDataset(Dataset):
    """Custom dataset for FSDP training with performance optimizations for T4 GPUs"""
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize the dataset
        
        Args:
            data_dir (str): Directory containing the data
            split (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.split = split
        
        # Load data - in a real scenario you would load your actual data
        try:
            self.features = np.load(os.path.join(self.data_dir, 'features.npy'), mmap_mode='r')
            self.labels = np.load(os.path.join(self.data_dir, 'labels.npy'), mmap_mode='r')
            print(f"Loaded {len(self.features)} samples from {self.data_dir}")
        except FileNotFoundError:
            # If files don't exist, create dummy data for demonstration
            print(f"No data found in {self.data_dir}, creating dummy data")
            self.features = np.random.randn(10000 if split == 'train' else 2000, 1024).astype(np.float32)
            self.labels = np.random.randint(0, 10, size=len(self.features))
            
            # Create directories if they don't exist
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Save dummy data
            np.save(os.path.join(self.data_dir, 'features.npy'), self.features)
            np.save(os.path.join(self.data_dir, 'labels.npy'), self.labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset with optimized memory access patterns"""
        feature = self.features[idx].copy()  # Copy to avoid issues with mmap_mode
        label = self.labels[idx]
        
        # Convert to torch tensors with correct dtype for mixed precision
        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        # Apply transforms if any
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label

def create_optimized_data_loaders(data_dir, batch_size, world_size, rank, num_workers=8):
    """
    Create optimized data loaders for FSDP training on T4 GPUs
    
    Args:
        data_dir (str): Directory containing the data
        batch_size (int): Batch size per GPU
        world_size (int): Total number of processes
        rank (int): Rank of the current process
        num_workers (int): Number of data loading workers
        
    Returns:
        tuple: (train_loader, val_loader, train_sampler)
    """
    # Initialize datasets
    train_dataset = FSPDDataset(data_dir, split='train')
    val_dataset = FSPDDataset(data_dir, split='val')
    
    # Create distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True  # Drop last incomplete batch for consistent sizes
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders with optimized settings for T4 GPUs
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,      # Optimized for n1-standard-32
        pin_memory=True,              # Speed up data transfer to GPU
        persistent_workers=True,      # Keep workers alive between epochs
        prefetch_factor=3,            # Prefetch more batches for T4 throughput
        drop_last=True,               # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size * 2,    # Double batch size for evaluation
        sampler=val_sampler,
        num_workers=max(2, num_workers // 2),  # Fewer workers for validation
        pin_memory=True,
        persistent_workers=True,
    )
    
    return train_loader, val_loader, train_sampler

def prepare_batch_for_fsdp(batch, device):
    """
    Prepare a batch for FSDP training with memory optimizations
    
    Args:
        batch (tuple): Batch from data loader (inputs, targets)
        device (torch.device): Device to move data to
        
    Returns:
        tuple: (inputs, targets) on the correct device
    """
    inputs, targets = batch
    
    # Move to device with non_blocking for asynchronous data transfer
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    
    return inputs, targets