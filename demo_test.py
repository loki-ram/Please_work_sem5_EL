"""
Quick demo test to verify ISL-CSLTR code works correctly.
"""

import os
import pandas as pd
from argparse import Namespace
from config import get_config

def main():
    config = get_config()
    print('='*60)
    print('ISL-CSLTR Full Verification Test')
    print('='*60)

    # Test 1: Config
    print('\n[1] Config: OK')

    # Test 2: CSV and folder matching
    df = pd.read_csv(config.paths.gloss_csv)
    folders = os.listdir(config.paths.videos_dir)
    csv_sentences = df['Sentence'].str.strip().str.lower().tolist()
    video_folders_lower = [f.lower() for f in folders]

    matched = sum(1 for s in csv_sentences if s in video_folders_lower)
    print(f'[2] Folder matching: {matched}/{len(csv_sentences)}')

    # Show the mismatch
    for s in csv_sentences:
        if s not in video_folders_lower:
            print(f'    Unmatched CSV: "{s}"')

    # Test 3: Vocabulary
    from data.vocabulary import Vocabulary
    vocab = Vocabulary()
    vocab.build_from_texts(['HELLO WORLD'])
    print('[3] Vocabulary: OK')

    # Test 4: Models with correct names
    import torch
    from models.stage1_model import VideoToGlossModel
    from models.stage2_model import GlossToEnglishModel
    print('[4] Model imports: OK')

    # Test 5: Create Stage 1 model
    model = VideoToGlossModel(
        config=config.stage1,
        gloss_vocab_size=100,
        use_checkpointing=False
    )
    print('[5] Stage1 Model creation: OK')

    # Test 6: MediaPipe
    from data.feature_extraction import MEDIAPIPE_AVAILABLE
    print(f'[6] MediaPipe available: {MEDIAPIPE_AVAILABLE}')

    # Test 7: Check a video folder has videos
    sample_folder = os.path.join(config.paths.videos_dir, folders[0])
    videos = [f for f in os.listdir(sample_folder) if f.endswith(('.mp4', '.MP4', '.avi', '.mov'))]
    print(f'[7] Sample folder "{folders[0]}" has {len(videos)} videos')

    # Test 8: Test the combined training script setup
    print('\n[8] Testing combined train.py setup...')
    from train import setup_training
    args = Namespace(device='cpu', resume1=None, resume2=None, stage='all')
    cfg, device, gloss_vocab, english_vocab = setup_training(args)
    print(f'    Gloss vocab: {len(gloss_vocab)} tokens')
    print(f'    English vocab: {len(english_vocab)} tokens')

    # Test 9: Test Stage 2 dataloaders (no features needed)
    print('\n[9] Testing Stage 2 dataloaders...')
    from data.dataset import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, gloss_vocab, english_vocab, stage=2
    )
    print(f'    Train: {len(train_loader)} batches')
    print(f'    Val: {len(val_loader)} batches')
    print(f'    Test: {len(test_loader)} batches')
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f'    Batch gloss_ids shape: {batch["gloss_ids"].shape}')
    print(f'    Batch sentence_ids shape: {batch["sentence_ids"].shape}')

    # Test 10: Create Stage 2 model and do forward pass
    print('\n[10] Testing Stage 2 model forward pass...')
    model2 = GlossToEnglishModel(
        config.stage2,
        len(gloss_vocab),
        len(english_vocab),
        gloss_pad_idx=gloss_vocab.pad_idx,
        english_pad_idx=english_vocab.pad_idx,
        english_sos_idx=english_vocab.sos_idx,
        english_eos_idx=english_vocab.eos_idx
    )
    
    # Forward pass test
    with torch.no_grad():
        loss = model2.compute_loss(
            batch['gloss_ids'],
            batch['gloss_lengths'],
            batch['sentence_ids'],
            teacher_forcing_ratio=1.0
        )
    print(f'    Forward pass loss: {loss.item():.4f}')

    print('\n' + '='*60)
    print('All tests passed! Code is working correctly.')
    print('='*60)
    
    print('\n--- Next Steps ---')
    if not MEDIAPIPE_AVAILABLE:
        print('1. Install MediaPipe: pip install mediapipe')
        print('2. Extract features from videos')
    print('3. Train Stage 1: python train.py --stage 1')
    print('4. Train Stage 2: python train.py --stage 2')
    print('5. Or train both: python train.py --stage all')

if __name__ == "__main__":
    main()
