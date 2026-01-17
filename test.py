"""
Test script for Hybrid Sign-to-Text Model.

Evaluates on test split using attention decoder only (no CTC decoding).
Reports: exact-match accuracy, BLEU, WER.
"""

import os
import sys
import json
import csv
import argparse
import glob
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config, Config
from models.hybrid_model import HybridSignToTextModel
from data.vocabulary import load_vocabularies
from data.dataset import create_dataloaders, HybridDataset, create_collate_fn
from utils.metrics import compute_bleu, compute_wer, compute_exact_match


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Config,
    gloss_vocab_size: int,
    text_vocab_size: int,
    gloss_vocab,
    text_vocab
) -> HybridSignToTextModel:
    """Load model from checkpoint. Accepts a file path or a directory (picks latest_best.pt)."""
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
        gloss_vocab_size=gloss_vocab_size,
        text_vocab_size=text_vocab_size,
        gloss_blank_idx=gloss_vocab.blank_idx,
        text_pad_idx=text_vocab.pad_idx,
        text_sos_idx=text_vocab.sos_idx,
        text_eos_idx=text_vocab.eos_idx,
        max_decode_length=config.model.decoder.max_decode_length,
        use_encoder_projection=config.model.encoder.use_encoder_projection,
        encoder_projection_dim=config.model.encoder.encoder_projection_dim,
        attention_entropy_weight=config.training.attention_entropy_weight,
        min_eos_step=config.model.decoder.min_eos_step,
        eos_penalty=config.model.decoder.eos_penalty,
        decoder_input_dropout=config.model.decoder.decoder_input_dropout
    )
    
    # If directory provided, look for latest_best.pt first, then pick most recent
    if os.path.isdir(checkpoint_path):
        # Prefer latest_best.pt if it exists
        best_checkpoint = os.path.join(checkpoint_path, 'latest_best.pt')
        if os.path.exists(best_checkpoint):
            checkpoint_file = best_checkpoint
            print(f"Using best checkpoint: {checkpoint_file}")
        else:
            patterns = [os.path.join(checkpoint_path, '*' + ext) for ext in ('.pt', '.pth', '.ckpt', '.tar')]
            files = []
            for p in patterns:
                files.extend(glob.glob(p))
            if not files:
                raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_path}")
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            checkpoint_file = files[0]
            print(f"Using checkpoint file: {checkpoint_file}")
    else:
        checkpoint_file = checkpoint_path
    
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    # Support different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    
    return model


@torch.no_grad()
def evaluate(
    model: HybridSignToTextModel,
    dataloader,
    text_vocab,
    device: torch.device
) -> dict:
    """
    Evaluate model on dataset using attention decoder only.
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    all_predictions = []
    all_references = []
    
    for batch in tqdm(dataloader, desc='Evaluating'):
        features = batch['features'].to(device)
        feature_lengths = batch['feature_lengths'].to(device)
        
        # Greedy decode with attention decoder only
        decoded = model.decode_greedy(features, feature_lengths)
        
        # Decode tokens to text
        for i in range(decoded.size(0)):
            pred_text = text_vocab.decode(decoded[i].tolist())
            ref_text = batch['text_texts'][i]
            
            all_predictions.append(pred_text)
            all_references.append(ref_text)
    
    # Compute metrics
    exact_match = compute_exact_match(all_predictions, all_references)
    bleu = compute_bleu(all_predictions, all_references)
    wer = compute_wer(all_predictions, all_references)
    
    return {
        'exact_match': exact_match,
        'bleu': bleu,
        'wer': wer,
        'num_samples': len(all_predictions),
        'predictions': all_predictions,
        'references': all_references
    }


def main():
    parser = argparse.ArgumentParser(description='Test Hybrid Sign-to-Text Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for predictions (JSON)')
    parser.add_argument('--csv-output', type=str, default=None,
                        help='Output file for predictions (CSV)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for evaluation')
    parser.add_argument('--full-dataset', action='store_true', default=True,
                        help='Evaluate on entire dataset (train + val + test) - default: True')
    parser.add_argument('--test-only', action='store_true',
                        help='Evaluate only on test set (overrides --full-dataset)')
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Hybrid Sign-to-Text Model Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    
    # Load vocabularies
    print("\nLoading vocabularies...")
    gloss_vocab, text_vocab = load_vocabularies(config)
    print(f"Gloss vocabulary: {len(gloss_vocab)} tokens")
    print(f"Text vocabulary: {len(text_vocab)} tokens")
    
    # Update config
    config.update_vocab_sizes(len(gloss_vocab), len(text_vocab))
    
    # Handle --test-only flag (overrides --full-dataset)
    use_full_dataset = args.full_dataset and not args.test_only
    
    # Create dataloader
    if use_full_dataset:
        print("\nLoading entire dataset (train + val + test)...")
        # Load all splits
        with open(config.paths.train_split_path, 'r', encoding='utf-8') as f:
            train_samples = json.load(f)
        with open(config.paths.val_split_path, 'r', encoding='utf-8') as f:
            val_samples = json.load(f)
        with open(config.paths.test_split_path, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
        
        print(f"Total samples: {len(train_samples) + len(val_samples) + len(test_samples)} (train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)})")
        
        # Create separate datasets for each split (they look for features in different directories)
        train_dataset = HybridDataset(train_samples, gloss_vocab, text_vocab, config, 'train')
        val_dataset = HybridDataset(val_samples, gloss_vocab, text_vocab, config, 'val')
        test_dataset = HybridDataset(test_samples, gloss_vocab, text_vocab, config, 'test')
        
        print(f"Valid samples: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        
        # Concatenate all datasets
        full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
        print(f"Total valid samples for evaluation: {len(full_dataset)}")
        
        # Track split boundaries for CSV output
        split_info = {
            'train_count': len(train_dataset),
            'val_count': len(val_dataset),
            'test_count': len(test_dataset)
        }
        
        collate_fn = create_collate_fn(gloss_vocab, text_vocab)
        
        eval_loader = DataLoader(
            full_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=collate_fn,
            pin_memory=config.training.pin_memory
        )
        print(f"Total batches: {len(eval_loader)}")
    else:
        print("\nLoading test data...")
        _, _, eval_loader = create_dataloaders(config, gloss_vocab, text_vocab)
        print(f"Test batches: {len(eval_loader)}")
        split_info = None  # No split info for test-only
    
    # Load model
    print("\nLoading model from checkpoint...")
    model = load_model_from_checkpoint(
        args.checkpoint, config,
        len(gloss_vocab), len(text_vocab),
        gloss_vocab, text_vocab
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Evaluate
    eval_set_name = "entire dataset" if use_full_dataset else "test set"
    print(f"\nEvaluating on {eval_set_name}...")
    results = evaluate(model, eval_loader, text_vocab, device)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Exact Match Accuracy: {results['exact_match']:.4f} ({results['exact_match']*100:.2f}%)")
    print(f"BLEU Score: {results['bleu']:.4f}")
    print(f"Word Error Rate (WER): {results['wer']:.4f}")
    
    # Print some examples
    print("\n" + "-" * 60)
    print("SAMPLE PREDICTIONS")
    print("-" * 60)
    num_examples = min(5, len(results['predictions']))
    for i in range(num_examples):
        print(f"\n[{i+1}]")
        print(f"  Reference:  {results['references'][i]}")
        print(f"  Prediction: {results['predictions'][i]}")
        is_correct = results['predictions'][i].strip().lower() == results['references'][i].strip().lower()
        print(f"  Correct: {'✓' if is_correct else '✗'}")
    
    # Save predictions if requested
    if args.output:
        output_data = {
            'metrics': {
                'exact_match': results['exact_match'],
                'bleu': results['bleu'],
                'wer': results['wer'],
                'num_samples': results['num_samples']
            },
            'samples': [
                {'reference': ref, 'prediction': pred}
                for ref, pred in zip(results['references'], results['predictions'])
            ]
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nPredictions saved to: {args.output}")
    
    # Save predictions to CSV if requested
    if args.csv_output:
        with open(args.csv_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'split', 'reference', 'prediction', 'correct'])
            
            for i, (ref, pred) in enumerate(zip(results['references'], results['predictions'])):
                is_correct = pred.strip().lower() == ref.strip().lower()
                
                # Determine which split this sample belongs to
                if split_info:
                    if i < split_info['train_count']:
                        split = 'train'
                    elif i < split_info['train_count'] + split_info['val_count']:
                        split = 'val'
                    else:
                        split = 'test'
                else:
                    split = 'test'
                
                writer.writerow([i + 1, split, ref, pred, is_correct])
        
        print(f"Predictions saved to CSV: {args.csv_output}")
    
    print("\n[OK] Evaluation completed!")


if __name__ == "__main__":
    main()
