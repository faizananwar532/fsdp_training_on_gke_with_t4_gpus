import os
import argparse
import yaml
import time
import logging
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from optimized_dataloader import create_optimized_data_loaders, prepare_batch_for_fsdp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FSDP_Trainer")

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch FSDP Training on GKE')
    parser.add_argument('--config', type=str, default='/etc/fsdp-config/fsdp_config.yaml', 
                        help='FSDP configuration file')
    parser.add_argument('--data-path', type=str, default='/data', 
                        help='Path to training data')
    parser.add_argument('--output-path', type=str, default='/output',
                        help='Path to save model and checkpoints')
    parser.add_argument('--node-rank', type=int, default=0, 
                        help='Rank of the node')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='Training batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()

def load_config(config_path):
    """Load FSDP configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        # Provide a default config if loading fails
        return {
            "fsdp": {
                "sharding_strategy": "FULL_SHARD",
                "cpu_offload": False,
                "backward_prefetch": "BACKWARD_PRE",
                "mixed_precision": True,
            }
        }

# Define your model here
class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # This is a placeholder model - replace with your actual model
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        return self.layers(x)

def setup_fsdp(fsdp_config):
    """Configure FSDP based on the provided configuration"""
    # Determine sharding strategy (ZeRO-3 is FULL_SHARD)
    if fsdp_config['sharding_strategy'] == 'FULL_SHARD':
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif fsdp_config['sharding_strategy'] == 'SHARD_GRAD_OP':
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    else:
        sharding_strategy = ShardingStrategy.NO_SHARD
    
    # Configure CPU offload if enabled
    cpu_offload = None
    if fsdp_config.get('cpu_offload', False):
        cpu_offload = CPUOffload(offload_params=True)
    
    # Configure backward prefetch
    backward_prefetch = None
    if fsdp_config.get('backward_prefetch') == 'BACKWARD_PRE':
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    elif fsdp_config.get('backward_prefetch') == 'BACKWARD_POST':
        backward_prefetch = BackwardPrefetch.BACKWARD_POST
    
    # Configure mixed precision for T4 GPUs
    mixed_precision = None
    if fsdp_config.get('mixed_precision', False):
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    
    # Return FSDP configuration
    return {
        'sharding_strategy': sharding_strategy,
        'cpu_offload': cpu_offload,
        'backward_prefetch': backward_prefetch,
        'mixed_precision': mixed_precision,
        'device_id': torch.cuda.current_device(),
    }

def train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch, device, args):
    """Train model for one epoch with mixed precision and gradient accumulation"""
    model.train()
    train_sampler = train_loader.sampler
    train_sampler.set_epoch(epoch)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to device
        inputs, targets = prepare_batch_for_fsdp((inputs, targets), device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Scale loss and backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  # More efficient than .zero_grad()
        
        # Calculate metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if batch_idx % 20 == 0 and dist.get_rank() == 0:
            logger.info(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                        f'Loss: {total_loss/(batch_idx+1):.4f} | '
                        f'Acc: {100.*correct/total:.2f}% | '
                        f'Time: {time.time() - start_time:.2f}s')
            start_time = time.time()
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate model performance"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = prepare_batch_for_fsdp((inputs, targets), device)
            
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Gather metrics from all processes
    metrics = torch.tensor([total_loss, correct, total], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_loss, correct, total = metrics.tolist()
    
    avg_loss = total_loss / dist.get_world_size() / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, scaler, epoch, args, best=False):
    """Save model checkpoint with FSDP state dictionary"""
    if dist.get_rank() != 0:
        return  # Only save on master process
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Set checkpoint path
    checkpoint_path = os.path.join(
        args.output_path, 
        f"model_{'best' if best else f'epoch_{epoch}'}.pt"
    )
    
    # Save with FSDP state_dict
    full_state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler else None,
    }
    
    torch.save(full_state_dict, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

# Define the auto_wrap_policy function
def get_auto_wrap_policy(min_num_params=100000):
    """
    Create an auto wrap policy function for FSDP based on parameter size
    
    Args:
        min_num_params (int): Minimum number of parameters for auto wrapping
        
    Returns:
        Callable: Auto wrap policy function
    """
    def custom_auto_wrap_policy(
        module, recurse, nonwrapped_numel, min_params=min_num_params
    ):
        return size_based_auto_wrap_policy(
            module=module,
            recurse=recurse,
            min_num_params=min_params,
            nonwrapped_numel=nonwrapped_numel,
        )
    return custom_auto_wrap_policy

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Setup distributed training
    rank = int(os.environ.get('NODE_RANK', '0'))
    if rank and args.node_rank:  # Parse rank from filename if available
        try:
            hostname = os.environ.get('HOSTNAME', '')
            if '-' in hostname:
                hostname_parts = hostname.split('-')
                if hostname_parts[-1].isdigit():
                    rank = int(hostname_parts[-1])
        except:
            pass
    
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    logger.info(f"Initializing process: rank={rank}, world_size={world_size}")
    logger.info(f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    
    # Initialize process group for distributed training
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # T4-specific optimizations
    torch.backends.cudnn.benchmark = True
    
    # Load FSDP configuration
    fsdp_config = load_config(args.config)['fsdp']
    logger.info(f"Loaded FSDP config: {fsdp_config}")
    
    # Create model
    model = ExampleModel()  # Replace with your actual model
    
    # Setup FSDP wrapping
    fsdp_config_dict = setup_fsdp(fsdp_config)
    
    # Fixed: Use the updated auto_wrap_policy function
    auto_wrap_policy = get_auto_wrap_policy(min_num_params=100000)
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=fsdp_config_dict['sharding_strategy'],
        cpu_offload=fsdp_config_dict['cpu_offload'],
        backward_prefetch=fsdp_config_dict['backward_prefetch'],
        mixed_precision=fsdp_config_dict['mixed_precision'],
        device_id=fsdp_config_dict['device_id'],
    )
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Setup gradient scaler for mixed precision training
    scaler = ShardedGradScaler() if fsdp_config.get('mixed_precision', False) else None
    
    # Create dataloaders
    train_loader, val_loader, train_sampler = create_optimized_data_loaders(
        args.data_path, 
        args.batch_size, 
        world_size, 
        rank,
        num_workers=8  # Adjust based on your CPU cores
    )
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, epoch, device, args
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print metrics on master node
        if rank == 0:
            logger.info(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            save_checkpoint(model, optimizer, scaler, epoch, args)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, optimizer, scaler, epoch, args, best=True)
    
    # Cleanup
    dist.destroy_process_group()
    logger.info("Training completed!")

if __name__ == "__main__":
    main()