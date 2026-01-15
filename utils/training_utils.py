"""
Utility functions for training: logging, checkpointing, schedulers.
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (minimize or maximize metric)
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class WarmupScheduler(_LRScheduler):
    """Learning rate scheduler with linear warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_scheduler: Optional[_LRScheduler] = None,
        min_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_scheduler = base_scheduler
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * factor for base_lr in self.base_lrs]
        elif self.base_scheduler is not None:
            return self.base_scheduler.get_last_lr()
        else:
            # Cosine annealing after warmup
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            return [max(self.min_lr, base_lr * factor.item()) for base_lr in self.base_lrs]


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    config: Config,
    filename: str,
    is_best: bool = False
) -> str:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state (optional)
        epoch: Current epoch
        metrics: Current metrics
        config: Configuration
        filename: Checkpoint filename
        is_best: Whether this is the best model
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'stage1': config.stage1.__dict__ if hasattr(config, 'stage1') else None,
            'stage2': config.stage2.__dict__ if hasattr(config, 'stage2') else None,
        }
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save checkpoint
    checkpoint_path = os.path.join(config.paths.checkpoints_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = os.path.join(config.paths.checkpoints_dir, 'best_' + filename)
        torch.save(checkpoint, best_path)
    
    return checkpoint_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


class TrainingLogger:
    """Logger for training progress and metrics."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.json')
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
            'epoch_times': [],
        }
        
        self.start_time = time.time()
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float
    ):
        """Log metrics for one epoch."""
        epoch_time = time.time() - self.start_time
        
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_metrics'].append(train_metrics)
        self.history['val_metrics'].append(val_metrics)
        self.history['learning_rates'].append(learning_rate)
        self.history['epoch_times'].append(epoch_time)
        
        # Save to file
        self.save()
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Train Metrics: {train_metrics}")
        print(f"  Val Metrics: {val_metrics}")
        print(f"  Learning Rate: {learning_rate:.2e}")
        print(f"  Time: {epoch_time:.1f}s")
    
    def save(self):
        """Save history to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """Get the epoch with best validation metric."""
        if metric == 'val_loss':
            values = self.history['val_loss']
        else:
            values = [m.get(metric, float('inf') if mode == 'min' else float('-inf')) 
                     for m in self.history['val_metrics']]
        
        if mode == 'min':
            return int(np.argmin(values))
        else:
            return int(np.argmax(values))


def get_optimizer(model: nn.Module, config) -> Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Training configuration (Stage1Config or Stage2Config)
        
    Returns:
        Optimizer instance
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )


def get_scheduler(
    optimizer: Optimizer,
    config,
    num_epochs: int
) -> _LRScheduler:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer
        config: Training configuration
        num_epochs: Total number of epochs
        
    Returns:
        Scheduler instance
    """
    warmup_epochs = getattr(config, 'warmup_epochs', 5)
    
    if config.lr_scheduler == 'cosine':
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs
        )
    elif config.lr_scheduler == 'step':
        base_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif config.lr_scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        base_scheduler = None
    
    return WarmupScheduler(
        optimizer, warmup_epochs, num_epochs, base_scheduler
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


# Import numpy for TrainingLogger
import numpy as np
