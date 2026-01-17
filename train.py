"""
Training script for Hybrid Sign-to-Text Model.

Features:
- Hybrid loss: α * CTC + (1-α) * Attention (α decays from 0.5 → 0.2)
- Teacher forcing with decay (1.0 → 0.7)
- AdamW optimizer (lr=1e-4), gradient clipping (max_norm=5)
- Mixed-precision training (AMP) for A100 GPU
- Robust checkpointing (per-epoch + best model)
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config, get_config_from_args, Config
from models.hybrid_model import HybridSignToTextModel
from data.vocabulary import build_vocabularies, save_vocabularies, load_vocabularies, Vocabulary
from data.dataset import create_dataloaders


def setup_logging(config: Config) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(config.paths.logs_dir, 'train.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_format)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    path: str,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    torch.save(checkpoint, path)
    
    if is_best:
        best_path = path.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler = None
) -> Tuple[int, Dict]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('metrics', {})


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: Config,
    epoch: int,
    logger: logging.Logger
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_ce_loss = 0.0
    num_batches = 0
    
    # Get current epoch settings
    ctc_weight = config.training.get_ctc_weight(epoch)
    teacher_forcing = config.training.get_teacher_forcing_ratio(epoch)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        features = batch['features'].to(config.device)
        feature_lengths = batch['feature_lengths'].to(config.device)
        gloss_ids = batch['gloss_ids'].to(config.device)
        gloss_lengths = batch['gloss_lengths'].to(config.device)
        text_ids = batch['text_ids'].to(config.device)
        
        # Mixed precision forward pass
        with autocast(enabled=config.training.use_amp):
            losses = model.compute_loss(
                features, feature_lengths,
                gloss_ids, gloss_lengths,
                text_ids,
                ctc_weight=ctc_weight,
                teacher_forcing_ratio=teacher_forcing
            )
            
            loss = losses['loss']
            
            # Scale loss for gradient accumulation
            if config.training.gradient_accumulation_steps > 1:
                loss = loss / config.training.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.training.max_grad_norm
            )
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Track losses
        total_loss += losses['loss'].item()
        total_ctc_loss += losses['ctc_loss'].item()
        total_ce_loss += losses['ce_loss'].item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{losses["loss"].item():.4f}',
            'ctc': f'{losses["ctc_loss"].item():.4f}',
            'ce': f'{losses["ce_loss"].item():.4f}',
            'α': f'{ctc_weight:.2f}',
            'tf': f'{teacher_forcing:.2f}'
        })
        
        # Logging
        if (batch_idx + 1) % config.training.log_interval == 0:
            logger.info(
                f'Epoch {epoch+1} [{batch_idx+1}/{len(dataloader)}] '
                f'Loss: {losses["loss"].item():.4f} '
                f'CTC: {losses["ctc_loss"].item():.4f} '
                f'CE: {losses["ce_loss"].item():.4f}'
            )
    
    return {
        'train_loss': total_loss / num_batches,
        'train_ctc_loss': total_ctc_loss / num_batches,
        'train_ce_loss': total_ce_loss / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    config: Config,
    epoch: int,
    gloss_vocab: Vocabulary,
    text_vocab: Vocabulary,
    logger: logging.Logger = None,
    num_qualitative_samples: int = 5
) -> Dict[str, float]:
    """
    Validate model with diagnostics:
    - Standard loss/accuracy metrics
    - Shuffle test for visual grounding verification
    - Qualitative predictions logging
    """
    model.eval()
    
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_ce_loss = 0.0
    total_attn_entropy = 0.0
    total_correct = 0
    total_samples = 0
    
    # For qualitative samples
    qualitative_samples = []
    
    # For shuffle test
    shuffle_correct = 0
    shuffle_total = 0
    
    ctc_weight = config.training.get_ctc_weight(epoch)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validation')):
        features = batch['features'].to(config.device)
        feature_lengths = batch['feature_lengths'].to(config.device)
        gloss_ids = batch['gloss_ids'].to(config.device)
        gloss_lengths = batch['gloss_lengths'].to(config.device)
        text_ids = batch['text_ids'].to(config.device)
        
        with autocast(enabled=config.training.use_amp):
            losses = model.compute_loss(
                features, feature_lengths,
                gloss_ids, gloss_lengths,
                text_ids,
                ctc_weight=ctc_weight,
                teacher_forcing_ratio=0.0  # No teacher forcing during validation
            )
        
        total_loss += losses['loss'].item()
        total_ctc_loss += losses['ctc_loss'].item()
        total_ce_loss += losses['ce_loss'].item()
        if 'attn_entropy_loss' in losses:
            total_attn_entropy += losses['attn_entropy_loss'].item()
        
        # Greedy decode for accuracy
        decoded = model.decode_greedy(features, feature_lengths)
        
        # Calculate sequence-level accuracy and collect samples
        for i in range(decoded.size(0)):
            pred_text = text_vocab.decode(decoded[i].tolist())
            target_text = batch['text_texts'][i].lower()
            
            if pred_text.strip() == target_text.strip():
                total_correct += 1
            total_samples += 1
            
            # Collect qualitative samples (first N from first batch)
            if batch_idx == 0 and len(qualitative_samples) < num_qualitative_samples:
                qualitative_samples.append({
                    'reference': target_text,
                    'prediction': pred_text,
                    'match': pred_text.strip() == target_text.strip()
                })
        
        # =================================================================
        # SHUFFLE TEST: Verify visual grounding
        # Shuffle encoder outputs across batch dimension to break 
        # visual-semantic correspondence. Predictions should degrade.
        # =================================================================
        if batch_idx == 0 and features.size(0) > 1:  # Only on first batch with multiple samples
            # Shuffle indices
            batch_size = features.size(0)
            shuffle_idx = torch.randperm(batch_size, device=features.device)
            
            # Create shuffled features (different video for each sample)
            shuffled_features = features[shuffle_idx]
            shuffled_lengths = feature_lengths[shuffle_idx]
            
            # Decode with shuffled encoder outputs
            shuffled_decoded = model.decode_greedy(shuffled_features, shuffled_lengths)
            
            # Check if predictions still match (they shouldn't if model uses visual info)
            for i in range(batch_size):
                shuffled_pred = text_vocab.decode(shuffled_decoded[i].tolist())
                target_text = batch['text_texts'][i].lower()
                
                if shuffled_pred.strip() == target_text.strip():
                    shuffle_correct += 1
                shuffle_total += 1
    
    num_batches = len(dataloader)
    
    # Calculate shuffle test degradation
    original_acc = total_correct / total_samples if total_samples > 0 else 0.0
    shuffle_acc = shuffle_correct / shuffle_total if shuffle_total > 0 else 0.0
    grounding_score = original_acc - shuffle_acc  # Should be positive if visually grounded
    
    # Log qualitative predictions
    if logger:
        logger.info("\n" + "-" * 50)
        logger.info("QUALITATIVE PREDICTIONS (Greedy Decoding):")
        logger.info("-" * 50)
        for idx, sample in enumerate(qualitative_samples):
            status = "CORRECT" if sample['match'] else "WRONG"
            logger.info(f"[{idx+1}] {status}")
            logger.info(f"    REF:  {sample['reference']}")
            logger.info(f"    PRED: {sample['prediction']}")
        
        # Log shuffle test results
        logger.info("\n" + "-" * 50)
        logger.info("VISUAL GROUNDING DIAGNOSTIC (Shuffle Test):")
        logger.info("-" * 50)
        logger.info(f"  Original accuracy: {original_acc:.4f}")
        logger.info(f"  Shuffled accuracy: {shuffle_acc:.4f}")
        logger.info(f"  Grounding score:   {grounding_score:.4f} (should be >> 0)")
        if grounding_score <= 0.05:
            logger.info("  WARNING: Low grounding score suggests decoder ignores encoder!")
    
    return {
        'val_loss': total_loss / num_batches,
        'val_ctc_loss': total_ctc_loss / num_batches,
        'val_ce_loss': total_ce_loss / num_batches,
        'val_attn_entropy': total_attn_entropy / num_batches if num_batches > 0 else 0.0,
        'val_accuracy': original_acc,
        'shuffle_accuracy': shuffle_acc,
        'grounding_score': grounding_score,
    }


def train(config: Config):
    """Main training function."""
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("Hybrid Sign-to-Text Training")
    logger.info("=" * 60)
    
    # Set seed
    set_seed(config.training.seed)
    logger.info(f"Random seed: {config.training.seed}")
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Build or load vocabularies
    vocab_exists = (
        os.path.exists(config.paths.gloss_vocab_path) and
        os.path.exists(config.paths.text_vocab_path)
    )
    
    if vocab_exists:
        logger.info("Loading existing vocabularies...")
        gloss_vocab, text_vocab = load_vocabularies(config)
    else:
        logger.info("Building vocabularies from training data...")
        with open(config.paths.train_split_path, 'r', encoding='utf-8') as f:
            train_samples = json.load(f)
        gloss_vocab, text_vocab = build_vocabularies(train_samples, config)
        save_vocabularies(gloss_vocab, text_vocab, config)
    
    logger.info(f"Gloss vocabulary size: {len(gloss_vocab)}")
    logger.info(f"Text vocabulary size: {len(text_vocab)}")
    
    # Update config with vocab sizes
    config.update_vocab_sizes(len(gloss_vocab), len(text_vocab))
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config, gloss_vocab, text_vocab
    )
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = HybridSignToTextModel(
        input_size=config.model.encoder.input_size,
        conv_channels=config.model.encoder.conv_channels,
        conv_kernel_size=config.model.encoder.conv_kernel_size,
        conv_stride=config.model.encoder.conv_stride,
        gru_hidden_size=config.model.encoder.gru_hidden_size,
        gru_num_layers=config.model.encoder.gru_num_layers,
        gru_dropout=config.model.encoder.gru_dropout,
        decoder_embedding_dim=config.model.decoder.embedding_dim,
        decoder_hidden_size=config.model.decoder.hidden_size,
        decoder_num_layers=config.model.decoder.num_layers,
        decoder_dropout=config.model.decoder.dropout,
        attention_dim=config.model.decoder.attention_dim,
        gloss_vocab_size=len(gloss_vocab),
        text_vocab_size=len(text_vocab),
        gloss_blank_idx=gloss_vocab.blank_idx,
        text_pad_idx=text_vocab.pad_idx,
        text_sos_idx=text_vocab.sos_idx,
        text_eos_idx=text_vocab.eos_idx,
        max_decode_length=config.model.decoder.max_decode_length,
        use_encoder_projection=config.model.encoder.use_encoder_projection,
        encoder_projection_dim=config.model.encoder.encoder_projection_dim,
        attention_entropy_weight=config.training.attention_entropy_weight,
        # Anti-shortcut parameters
        min_eos_step=config.model.decoder.min_eos_step,
        eos_penalty=config.model.decoder.eos_penalty,
        decoder_input_dropout=config.model.decoder.decoder_input_dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=config.training.betas
    )
    
    # Learning rate scheduler
    if config.training.lr_scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.training.warmup_epochs,
            T_mult=2
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    
    # Gradient scaler for AMP
    scaler = GradScaler(enabled=config.training.use_amp)
    
    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Resume from checkpoint if exists
    checkpoint_path = os.path.join(config.paths.checkpoint_dir, 'latest.pt')
    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, metrics = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
        best_val_loss = metrics.get('val_loss', float('inf'))
        start_epoch += 1
    
    # Training loop
    logger.info(f"\nStarting training from epoch {start_epoch + 1}")
    logger.info(f"CTC weight decay: {config.training.ctc_weight_start} → {config.training.ctc_weight_end}")
    logger.info(f"Teacher forcing decay: {config.training.teacher_forcing_start} → {config.training.teacher_forcing_end}")
    
    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, config, epoch, logger
        )
        
        # Validate (includes shuffle test and qualitative predictions)
        val_metrics = validate(
            model, val_loader, config, epoch, gloss_vocab, text_vocab,
            logger=logger, num_qualitative_samples=5
        )
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics}
        
        # Update scheduler
        if config.training.lr_scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_metrics['val_loss'])
        
        # Logging
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"\nEpoch {epoch+1}/{config.training.num_epochs} completed in {epoch_time:.2f}s"
        )
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Val Accuracy: {val_metrics['val_accuracy']:.4f}")
        logger.info(f"  Grounding Score: {val_metrics['grounding_score']:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # Check for best model
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            logger.info(f"  New best model! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Save checkpoint
        if config.training.save_every_epoch:
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                os.path.join(config.paths.checkpoint_dir, f'epoch_{epoch+1}.pt'),
                is_best=False
            )
        
        # Save latest and best
        save_checkpoint(
            model, optimizer, scheduler, epoch, metrics,
            os.path.join(config.paths.checkpoint_dir, 'latest.pt'),
            is_best=is_best
        )
        
        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    logger.info("\nTraining completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    config = get_config_from_args()
    train(config)
