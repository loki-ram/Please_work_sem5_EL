"""
Unified Training Script for ISL-CSLTR Two-Stage Sign Language Translation System.
Combines Stage 1 (Video-to-Gloss) and Stage 2 (Gloss-to-English) training.

Usage:
    python train.py --stage all      # Train both stages sequentially
    python train.py --stage 1        # Train only Stage 1
    python train.py --stage 2        # Train only Stage 2
    python train.py --stage all --device cpu  # Force CPU
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Dict, Tuple, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config, Config
from data.vocabulary import (
    Vocabulary,
    load_vocabularies, 
    build_vocabularies, 
    save_vocabularies
)
from data.dataset import create_dataloaders
from data.split_dataset import load_splits, create_dataset_split, save_splits
from models.stage1_model import VideoToGlossModel
from models.stage2_model import GlossToEnglishModel
from utils.metrics import compute_wer, compute_bleu, compute_all_metrics
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


# =============================================================================
# Stage 1: Video-to-Gloss Training Functions
# =============================================================================

def train_epoch_stage1(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    config: Config,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Train Stage 1 for one epoch with gradient accumulation."""
    model.train()
    
    loss_meter = AverageMeter()
    accumulation_steps = config.dataset.gradient_accumulation_steps
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Stage1 Train]")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        features = batch['features'].to(device)
        feature_lengths = batch['feature_lengths'].to(device)
        gloss_ids = batch['gloss_ids'].to(device)
        gloss_lengths = batch['gloss_lengths'].to(device)
        
        if config.mixed_precision and device == 'cuda':
            with autocast():
                loss = model.compute_loss(
                    features, feature_lengths, gloss_ids, gloss_lengths
                )
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.stage1.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss = model.compute_loss(
                features, feature_lengths, gloss_ids, gloss_lengths
            )
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.stage1.gradient_clip_norm
                )
                optimizer.step()
                optimizer.zero_grad()
        
        loss_meter.update(loss.item() * accumulation_steps, features.size(0))
        
        if config.memory_efficient and (batch_idx + 1) % config.empty_cache_freq == 0:
            torch.cuda.empty_cache()
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    # Handle remaining gradients
    if len(dataloader) % accumulation_steps != 0:
        if config.mixed_precision and device == 'cuda':
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.stage1.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.stage1.gradient_clip_norm)
            optimizer.step()
        optimizer.zero_grad()
    
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    
    return {'loss': loss_meter.avg}


def validate_stage1(
    model: nn.Module,
    dataloader,
    gloss_vocab: Vocabulary,
    device: str
) -> Tuple[Dict[str, float], float]:
    """Validate Stage 1 model."""
    model.eval()
    
    loss_meter = AverageMeter()
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Stage1"):
            features = batch['features'].to(device)
            feature_lengths = batch['feature_lengths'].to(device)
            gloss_ids = batch['gloss_ids'].to(device)
            gloss_lengths = batch['gloss_lengths'].to(device)
            gloss_texts = batch['gloss_texts']
            
            loss = model.compute_loss(features, feature_lengths, gloss_ids, gloss_lengths)
            loss_meter.update(loss.item(), features.size(0))
            
            decoded_seqs, decoded_lens = model.decode_greedy(features, feature_lengths)
            
            for seq, ref_text in zip(decoded_seqs, gloss_texts):
                hyp_text = gloss_vocab.decode(seq, remove_special=True)
                all_references.append(ref_text)
                all_hypotheses.append(hyp_text)
    
    wer_results = compute_wer(all_references, all_hypotheses)
    
    metrics = {
        'loss': loss_meter.avg,
        'wer': wer_results['wer'],
    }
    
    print("\nSample Stage1 predictions:")
    for i in range(min(3, len(all_references))):
        print(f"  Ref: {all_references[i]}")
        print(f"  Hyp: {all_hypotheses[i]}")
        print()
    
    return metrics, loss_meter.avg


# =============================================================================
# Stage 2: Gloss-to-English Training Functions
# =============================================================================

def train_epoch_stage2(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    config: Config,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Train Stage 2 for one epoch with gradient accumulation."""
    model.train()
    
    loss_meter = AverageMeter()
    accumulation_steps = config.dataset.gradient_accumulation_steps
    tf_ratio = config.stage2.teacher_forcing_ratio
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Stage2 Train]")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        gloss_ids = batch['gloss_ids'].to(device)
        sentence_ids = batch['sentence_ids'].to(device)
        gloss_lengths = batch['gloss_lengths'].to(device)
        
        if config.mixed_precision and device == 'cuda':
            with autocast():
                loss = model.compute_loss(gloss_ids, gloss_lengths, sentence_ids, tf_ratio)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.stage2.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss = model.compute_loss(gloss_ids, gloss_lengths, sentence_ids, tf_ratio)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.stage2.gradient_clip_norm
                )
                optimizer.step()
                optimizer.zero_grad()
        
        loss_meter.update(loss.item() * accumulation_steps, gloss_ids.size(0))
        
        if config.memory_efficient and (batch_idx + 1) % config.empty_cache_freq == 0:
            torch.cuda.empty_cache()
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'tf': f'{tf_ratio:.2f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    # Handle remaining gradients
    if len(dataloader) % accumulation_steps != 0:
        if config.mixed_precision and device == 'cuda':
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.stage2.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.stage2.gradient_clip_norm)
            optimizer.step()
        optimizer.zero_grad()
    
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    
    return {'loss': loss_meter.avg}


def validate_stage2(
    model: nn.Module,
    dataloader,
    english_vocab: Vocabulary,
    device: str
) -> Tuple[Dict[str, float], float]:
    """Validate Stage 2 model."""
    model.eval()
    
    loss_meter = AverageMeter()
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Stage2"):
            gloss_ids = batch['gloss_ids'].to(device)
            sentence_ids = batch['sentence_ids'].to(device)
            gloss_lengths = batch['gloss_lengths'].to(device)
            sentence_texts = batch['sentence_texts']
            
            loss = model.compute_loss(gloss_ids, gloss_lengths, sentence_ids, teacher_forcing_ratio=0.0)
            loss_meter.update(loss.item(), gloss_ids.size(0))
            
            decoded, _ = model.decode_greedy(gloss_ids, gloss_lengths)
            
            for seq, ref_text in zip(decoded, sentence_texts):
                hyp_text = english_vocab.decode(seq.tolist(), remove_special=True)
                all_references.append(ref_text)
                all_hypotheses.append(hyp_text)
    
    all_metrics = compute_all_metrics(all_references, all_hypotheses)
    
    metrics = {
        'loss': loss_meter.avg,
        'bleu': all_metrics['bleu'],
        'bleu_1': all_metrics['bleu_1'],
        'bleu_4': all_metrics['bleu_4'],
        'rouge_l_f1': all_metrics['rouge_l_f1'],
    }
    
    print("\nSample Stage2 predictions:")
    for i in range(min(3, len(all_references))):
        print(f"  Ref: {all_references[i]}")
        print(f"  Hyp: {all_hypotheses[i]}")
        print()
    
    return metrics, loss_meter.avg


# =============================================================================
# Main Training Functions
# =============================================================================

def setup_training(args) -> Tuple[Config, str, Vocabulary, Vocabulary]:
    """Setup training: config, device, vocabularies, and splits."""
    config = get_config()
    
    # Set device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Ensure splits exist
    splits_path = os.path.join(config.paths.splits_dir, "train.json")
    if not os.path.exists(splits_path):
        print("Creating dataset splits...")
        splits = create_dataset_split(config, stratify_by_length=True)
        save_splits(splits, config)
    
    # Load or build vocabularies
    try:
        gloss_vocab, english_vocab = load_vocabularies(config)
        print(f"Loaded vocabularies: gloss={len(gloss_vocab)}, english={len(english_vocab)}")
    except FileNotFoundError:
        print("Building vocabularies...")
        splits = load_splits(config)
        gloss_vocab, english_vocab = build_vocabularies(splits['train'], config)
        save_vocabularies(gloss_vocab, english_vocab, config)
        print(f"Built vocabularies: gloss={len(gloss_vocab)}, english={len(english_vocab)}")
    
    # Update config with vocab sizes
    config.update_vocab_sizes(len(gloss_vocab), len(english_vocab))
    
    return config, device, gloss_vocab, english_vocab


def train_stage1(
    config: Config,
    device: str,
    gloss_vocab: Vocabulary,
    english_vocab: Vocabulary,
    resume_checkpoint: Optional[str] = None
) -> str:
    """
    Train Stage 1: Video-to-Gloss Recognition.
    
    Returns:
        Path to best checkpoint
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Video-to-Gloss Recognition (TCN-BiGRU + CTC)")
    print("=" * 60)
    
    # Create data loaders
    print("Creating Stage 1 data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config, gloss_vocab, english_vocab, stage=1
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Effective batch size: {config.dataset.batch_size * config.dataset.gradient_accumulation_steps}")
    
    # Create model
    print("Creating Stage 1 model...")
    use_checkpointing = getattr(config, 'gradient_checkpointing', False)
    model = VideoToGlossModel(config.stage1, len(gloss_vocab), use_checkpointing=use_checkpointing)
    model = model.to(device)
    
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count['total']:,} total, {param_count['trainable']:,} trainable")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config.stage1)
    scheduler = get_scheduler(optimizer, config.stage1, config.stage1.num_epochs)
    scaler = GradScaler() if config.mixed_precision and device == 'cuda' else None
    
    if config.mixed_precision:
        print("Mixed precision training: ENABLED")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.stage1.early_stopping_patience, mode='min')
    logger = TrainingLogger(config.paths.logs_dir, 'stage1')
    
    # Resume if specified
    start_epoch = 0
    best_wer = float('inf')
    
    if resume_checkpoint:
        print(f"Resuming from: {resume_checkpoint}")
        checkpoint = load_checkpoint(model, resume_checkpoint, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch'] + 1
        best_wer = checkpoint['metrics'].get('wer', float('inf'))
    
    # Training loop
    print(f"\nStarting Stage 1 training for {config.stage1.num_epochs} epochs...")
    
    for epoch in range(start_epoch, config.stage1.num_epochs):
        train_metrics = train_epoch_stage1(
            model, train_loader, optimizer, scheduler, scaler, config, device, epoch
        )
        
        val_metrics, val_loss = validate_stage1(model, val_loader, gloss_vocab, device)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics['wer'])
        
        logger.log_epoch(
            epoch, train_metrics['loss'], val_loss,
            train_metrics, val_metrics, optimizer.param_groups[0]['lr']
        )
        
        is_best = val_metrics['wer'] < best_wer
        if is_best:
            best_wer = val_metrics['wer']
            print(f"★ New best WER: {best_wer:.4f}")
        
        if (epoch + 1) % config.save_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                config, 'stage1_checkpoint.pt', is_best
            )
        
        if early_stopping(val_metrics['wer']):
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\nStage 1 Training completed! Best WER: {best_wer:.4f}")
    
    # Test evaluation
    print("\nEvaluating Stage 1 on test set...")
    best_checkpoint = os.path.join(config.paths.checkpoints_dir, 'best_stage1_checkpoint.pt')
    if os.path.exists(best_checkpoint):
        load_checkpoint(model, best_checkpoint, device=device)
    
    test_metrics, _ = validate_stage1(model, test_loader, gloss_vocab, device)
    print(f"Test WER: {test_metrics['wer']:.4f}")
    
    return best_checkpoint


def train_stage2(
    config: Config,
    device: str,
    gloss_vocab: Vocabulary,
    english_vocab: Vocabulary,
    resume_checkpoint: Optional[str] = None
) -> str:
    """
    Train Stage 2: Gloss-to-English Translation.
    
    Returns:
        Path to best checkpoint
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Gloss-to-English Translation (Seq2Seq + Attention)")
    print("=" * 60)
    
    # Create data loaders
    print("Creating Stage 2 data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config, gloss_vocab, english_vocab, stage=2
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Effective batch size: {config.dataset.batch_size * config.dataset.gradient_accumulation_steps}")
    
    # Create model
    print("Creating Stage 2 model...")
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
    
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count['total']:,} total, {param_count['trainable']:,} trainable")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config.stage2)
    scheduler = get_scheduler(optimizer, config.stage2, config.stage2.num_epochs)
    scaler = GradScaler() if config.mixed_precision and device == 'cuda' else None
    
    if config.mixed_precision:
        print("Mixed precision training: ENABLED")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.stage2.early_stopping_patience, mode='max')
    logger = TrainingLogger(config.paths.logs_dir, 'stage2')
    
    # Resume if specified
    start_epoch = 0
    best_bleu = 0.0
    
    if resume_checkpoint:
        print(f"Resuming from: {resume_checkpoint}")
        checkpoint = load_checkpoint(model, resume_checkpoint, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch'] + 1
        best_bleu = checkpoint['metrics'].get('bleu', 0.0)
    
    # Training loop
    print(f"\nStarting Stage 2 training for {config.stage2.num_epochs} epochs...")
    
    for epoch in range(start_epoch, config.stage2.num_epochs):
        train_metrics = train_epoch_stage2(
            model, train_loader, optimizer, scheduler, scaler, config, device, epoch
        )
        
        val_metrics, val_loss = validate_stage2(model, val_loader, english_vocab, device)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        
        logger.log_epoch(
            epoch, train_metrics['loss'], val_loss,
            train_metrics, val_metrics, optimizer.param_groups[0]['lr']
        )
        
        is_best = val_metrics['bleu'] > best_bleu
        if is_best:
            best_bleu = val_metrics['bleu']
            print(f"★ New best BLEU: {best_bleu:.4f}")
        
        if (epoch + 1) % config.save_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                config, 'stage2_checkpoint.pt', is_best
            )
        
        if early_stopping(val_metrics['bleu']):
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\nStage 2 Training completed! Best BLEU: {best_bleu:.4f}")
    
    # Test evaluation
    print("\nEvaluating Stage 2 on test set...")
    best_checkpoint = os.path.join(config.paths.checkpoints_dir, 'best_stage2_checkpoint.pt')
    if os.path.exists(best_checkpoint):
        load_checkpoint(model, best_checkpoint, device=device)
    
    test_metrics, _ = validate_stage2(model, test_loader, english_vocab, device)
    print(f"Test BLEU: {test_metrics['bleu']:.4f}")
    print(f"Test BLEU-4: {test_metrics['bleu_4']:.4f}")
    print(f"Test ROUGE-L F1: {test_metrics['rouge_l_f1']:.4f}")
    
    return best_checkpoint


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Training Script for ISL-CSLTR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --stage all              # Train both stages
  python train.py --stage 1                # Train only Stage 1
  python train.py --stage 2                # Train only Stage 2
  python train.py --stage all --device cpu # Force CPU training
  python train.py --stage 1 --resume checkpoint.pt  # Resume Stage 1
        """
    )
    parser.add_argument(
        '--stage', 
        type=str, 
        choices=['1', '2', 'all'], 
        default='all',
        help='Which stage(s) to train: 1, 2, or all (default: all)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default=None,
        help='Device to use (cuda/cpu). Auto-detected if not specified.'
    )
    parser.add_argument(
        '--resume1', 
        type=str, 
        default=None,
        help='Path to Stage 1 checkpoint to resume from'
    )
    parser.add_argument(
        '--resume2', 
        type=str, 
        default=None,
        help='Path to Stage 2 checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ISL-CSLTR Two-Stage Sign Language Translation System")
    print("=" * 60)
    print(f"Training stage(s): {args.stage}")
    
    # Setup
    config, device, gloss_vocab, english_vocab = setup_training(args)
    
    # Train Stage 1
    if args.stage in ('1', 'all'):
        train_stage1(config, device, gloss_vocab, english_vocab, args.resume1)
    
    # Train Stage 2
    if args.stage in ('2', 'all'):
        train_stage2(config, device, gloss_vocab, english_vocab, args.resume2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
