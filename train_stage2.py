"""
Training script for Stage 2: Gloss-to-English Translation.
Uses Seq2Seq model with attention.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from data.vocabulary import load_vocabularies
from data.dataset import create_dataloaders
from models.stage2_model import GlossToEnglishModel
from utils.metrics import compute_bleu, compute_all_metrics
from utils.training_utils import (
    AverageMeter,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    TrainingLogger,
    get_optimizer,
    get_scheduler,
    count_parameters
)


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    config,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch with gradient accumulation for 8GB VRAM.
    
    Args:
        model: Stage 2 model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        config: Configuration
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    loss_meter = AverageMeter()
    accumulation_steps = config.dataset.gradient_accumulation_steps
    
    # Calculate teacher forcing ratio decay
    tf_ratio = config.stage2.teacher_forcing_ratio
    # Optionally decay teacher forcing ratio
    # tf_ratio = max(0.1, tf_ratio - epoch * 0.02)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    optimizer.zero_grad()  # Zero gradients once at the start
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        gloss_ids = batch['gloss_ids'].to(device)
        sentence_ids = batch['sentence_ids'].to(device)
        gloss_lengths = batch['gloss_lengths'].to(device)
        
        # Forward pass with mixed precision
        if config.mixed_precision and device == 'cuda':
            with autocast():
                loss = model.compute_loss(
                    gloss_ids, gloss_lengths, sentence_ids, tf_ratio
                )
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Update weights only after accumulating enough gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.stage2.gradient_clip_norm
                )
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss = model.compute_loss(
                gloss_ids, gloss_lengths, sentence_ids, tf_ratio
            )
            loss = loss / accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.stage2.gradient_clip_norm
                )
                
                optimizer.step()
                optimizer.zero_grad()
        
        # Update metrics (multiply back for accurate logging)
        loss_meter.update(loss.item() * accumulation_steps, gloss_ids.size(0))
        
        # Clear CUDA cache periodically for memory management
        if config.memory_efficient and (batch_idx + 1) % config.empty_cache_freq == 0:
            torch.cuda.empty_cache()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'tf': f'{tf_ratio:.2f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    # Handle any remaining gradients
    if len(dataloader) % accumulation_steps != 0:
        if config.mixed_precision and device == 'cuda':
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.stage2.gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.stage2.gradient_clip_norm
            )
            optimizer.step()
        optimizer.zero_grad()
    
    # Step scheduler
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    
    return {'loss': loss_meter.avg}


def validate(
    model: nn.Module,
    dataloader,
    english_vocab,
    device: str
) -> Tuple[Dict[str, float], float]:
    """
    Validate the model.
    
    Args:
        model: Stage 2 model
        dataloader: Validation data loader
        english_vocab: English vocabulary for decoding
        device: Device to use
        
    Returns:
        Tuple of (metrics dict, validation loss)
    """
    model.eval()
    
    loss_meter = AverageMeter()
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            gloss_ids = batch['gloss_ids'].to(device)
            sentence_ids = batch['sentence_ids'].to(device)
            gloss_lengths = batch['gloss_lengths'].to(device)
            sentence_texts = batch['sentence_texts']
            
            # Compute loss (no teacher forcing for validation)
            loss = model.compute_loss(
                gloss_ids, gloss_lengths, sentence_ids, teacher_forcing_ratio=0.0
            )
            loss_meter.update(loss.item(), gloss_ids.size(0))
            
            # Decode predictions
            decoded, _ = model.decode_greedy(gloss_ids, gloss_lengths)
            
            # Convert to text
            for i, (seq, ref_text) in enumerate(zip(decoded, sentence_texts)):
                # Decode hypothesis
                hyp_text = english_vocab.decode(seq.tolist(), remove_special=True)
                
                all_references.append(ref_text)
                all_hypotheses.append(hyp_text)
    
    # Compute metrics
    all_metrics = compute_all_metrics(all_references, all_hypotheses)
    
    metrics = {
        'loss': loss_meter.avg,
        'bleu': all_metrics['bleu'],
        'bleu_1': all_metrics['bleu_1'],
        'bleu_4': all_metrics['bleu_4'],
        'rouge_l_f1': all_metrics['rouge_l_f1'],
    }
    
    # Print some examples
    print("\nSample predictions:")
    for i in range(min(3, len(all_references))):
        print(f"  Ref: {all_references[i]}")
        print(f"  Hyp: {all_hypotheses[i]}")
        print()
    
    return metrics, loss_meter.avg


def main(args):
    """Main training function."""
    # Load configuration
    config = get_config()
    
    # Set device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load vocabularies
    try:
        gloss_vocab, english_vocab = load_vocabularies(config)
        print(f"Loaded vocabularies: gloss={len(gloss_vocab)}, english={len(english_vocab)}")
    except FileNotFoundError:
        print("Error: Vocabularies not found. Run train_stage1.py first to build vocabularies.")
        return
    
    # Update config with vocab sizes
    config.update_vocab_sizes(len(gloss_vocab), len(english_vocab))
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config, gloss_vocab, english_vocab, stage=2
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Gradient accumulation steps: {config.dataset.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.dataset.batch_size * config.dataset.gradient_accumulation_steps}")
    
    # Create model
    print("Creating model...")
    model = GlossToEnglishModel(
        config.stage2,
        len(gloss_vocab),
        len(english_vocab),
        gloss_pad_idx=gloss_vocab.pad_idx,
        english_pad_idx=english_vocab.pad_idx,
        english_sos_idx=english_vocab.sos_idx,
        english_eos_idx=english_vocab.eos_idx
    )
    model = model.to(device)
    
    # Print parameter count
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count['total']:,} total, {param_count['trainable']:,} trainable")
    
    # Estimate VRAM usage
    param_memory_mb = (param_count['total'] * 4) / (1024 * 1024)  # FP32
    print(f"Estimated parameter memory: ~{param_memory_mb:.1f} MB")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config.stage2)
    scheduler = get_scheduler(optimizer, config.stage2, config.stage2.num_epochs)
    
    # Mixed precision info
    if config.mixed_precision:
        print("Mixed precision training: ENABLED (FP16)")
    
    # Mixed precision scaler
    scaler = GradScaler() if config.mixed_precision and device == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.stage2.early_stopping_patience,
        mode='max'  # Maximize BLEU
    )
    
    # Training logger
    logger = TrainingLogger(config.paths.logs_dir, 'stage2')
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_bleu = 0.0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(model, args.resume, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch'] + 1
        best_bleu = checkpoint['metrics'].get('bleu', 0.0)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, config.stage2.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, device, epoch
        )
        
        # Validate
        val_metrics, val_loss = validate(model, val_loader, english_vocab, device)
        
        # Update plateau scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        
        # Log metrics
        logger.log_epoch(
            epoch,
            train_metrics['loss'],
            val_loss,
            train_metrics,
            val_metrics,
            optimizer.param_groups[0]['lr']
        )
        
        # Check if best model
        is_best = val_metrics['bleu'] > best_bleu
        if is_best:
            best_bleu = val_metrics['bleu']
            print(f"New best BLEU: {best_bleu:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                config, 'stage2_checkpoint.pt', is_best
            )
        
        # Early stopping
        if early_stopping(val_metrics['bleu']):
            print(f"Early stopping at epoch {epoch}")
            break
    
    print("\nTraining completed!")
    print(f"Best BLEU: {best_bleu:.4f}")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    
    # Load best model
    best_checkpoint = os.path.join(config.paths.checkpoints_dir, 'best_stage2_checkpoint.pt')
    if os.path.exists(best_checkpoint):
        load_checkpoint(model, best_checkpoint, device=device)
    
    test_metrics, _ = validate(model, test_loader, english_vocab, device)
    print(f"Test BLEU: {test_metrics['bleu']:.4f}")
    print(f"Test BLEU-4: {test_metrics['bleu_4']:.4f}")
    print(f"Test ROUGE-L F1: {test_metrics['rouge_l_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 2: Gloss-to-English")
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)
