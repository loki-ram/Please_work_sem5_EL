"""
Dataset splitting and preprocessing pipeline.
Creates train/val/test splits and extracts features from videos.
"""

import os
import json
import glob
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, Config
from data.feature_extraction import (
    MediaPipeExtractor, 
    FeatureNormalizer, 
    DataAugmenter,
    MEDIAPIPE_AVAILABLE
)


def discover_videos(config: Config) -> List[Dict]:
    """
    Discover all videos in the video root directory.
    
    Expected structure:
        video_root/
            sentence_folder_1/
                video1.mp4
                video2.mp4
            sentence_folder_2/
                video1.mp4
    
    Returns:
        List of sample dictionaries with video info
    """
    video_root = config.paths.video_root
    samples = []
    
    print(f"Discovering videos in: {video_root}")
    
    # Iterate through folders (each folder is a sentence)
    for folder_name in os.listdir(video_root):
        folder_path = os.path.join(video_root, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Find all video files
        video_extensions = ['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        for video_path in video_files:
            video_filename = os.path.basename(video_path)
            
            # The sentence is the folder name (lowercase)
            sentence = folder_name.lower()
            
            # Simple gloss: uppercase words from folder name
            gloss = folder_name.upper()
            
            sample = {
                'folder_name': folder_name,
                'video_path': video_path,
                'video_filename': video_filename,
                'gloss': gloss,
                'sentence': sentence,
                'gloss_length': len(gloss.split())
            }
            samples.append(sample)
    
    print(f"Found {len(samples)} videos in {len(set(s['folder_name'] for s in samples))} folders")
    return samples


def create_splits(
    samples: List[Dict],
    config: Config
) -> Dict[str, List[Dict]]:
    """
    Create train/val/test splits.
    
    Args:
        samples: List of all samples
        config: Configuration object
        
    Returns:
        Dictionary with 'train', 'val', 'test' lists
    """
    random.seed(config.training.seed)
    
    # Shuffle samples
    samples = samples.copy()
    random.shuffle(samples)
    
    n = len(samples)
    train_end = int(n * 0.8)  # 80% train
    val_end = int(n * 0.9)    # 10% val, 10% test
    
    splits = {
        'train': samples[:train_end],
        'val': samples[train_end:val_end],
        'test': samples[val_end:]
    }
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    
    return splits


def save_splits(splits: Dict[str, List[Dict]], config: Config):
    """Save splits to JSON files."""
    os.makedirs(config.paths.splits_dir, exist_ok=True)
    
    for split_name, samples in splits.items():
        split_path = os.path.join(config.paths.splits_dir, f"{split_name}.json")
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"Saved {split_name} split to: {split_path}")


def load_splits(config: Config) -> Dict[str, List[Dict]]:
    """Load splits from JSON files."""
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_path = os.path.join(config.paths.splits_dir, f"{split_name}.json")
        with open(split_path, 'r', encoding='utf-8') as f:
            splits[split_name] = json.load(f)
    return splits


def preprocess_dataset(
    splits: Dict[str, List[Dict]],
    config: Config,
    skip_existing: bool = True
):
    """
    Preprocess entire dataset: extract features and save as .npy files.
    
    Args:
        splits: Dataset splits
        config: Configuration
        skip_existing: Skip already processed files
    """
    if not MEDIAPIPE_AVAILABLE:
        raise RuntimeError("MediaPipe is required for preprocessing")
    
    # Initialize components
    extractor = MediaPipeExtractor(config)
    normalizer = FeatureNormalizer(config)
    augmenter = DataAugmenter(config)
    
    stats = {
        'train': {'processed': 0, 'skipped': 0, 'failed': 0},
        'val': {'processed': 0, 'skipped': 0, 'failed': 0},
        'test': {'processed': 0, 'skipped': 0, 'failed': 0},
    }
    
    for split_name, samples in splits.items():
        print(f"\nProcessing {split_name} split ({len(samples)} samples)...")
        
        # Create output directory
        split_features_dir = os.path.join(config.paths.features_dir, split_name)
        os.makedirs(split_features_dir, exist_ok=True)
        
        for sample in tqdm(samples, desc=f"Processing {split_name}"):
            # Generate feature filename
            video_id = os.path.splitext(sample['video_filename'])[0]
            folder_name = sample['folder_name'].replace('/', '_').replace('\\', '_')
            feature_filename = f"{folder_name}_{video_id}.npy"
            feature_path = os.path.join(split_features_dir, feature_filename)
            
            # Skip if exists
            if skip_existing and os.path.exists(feature_path):
                stats[split_name]['skipped'] += 1
                sample['feature_path'] = feature_path
                continue
            
            try:
                # Extract features
                raw_features, metadata = extractor.extract_video(sample['video_path'])
                
                if len(raw_features) < config.preprocessing.min_video_length:
                    stats[split_name]['failed'] += 1
                    continue
                
                # Normalize
                normalized_features = normalizer.normalize_features(raw_features)
                
                # Add temporal derivatives
                features = normalizer.add_temporal_derivatives(normalized_features)
                
                # Augment training data (optional - save non-augmented for reproducibility)
                # If you want augmentation, uncomment:
                # if split_name == 'train' and config.preprocessing.augment_training:
                #     features = augmenter.augment(features)
                
                # Save features
                np.save(feature_path, features)
                
                # Update sample with feature path
                sample['feature_path'] = feature_path
                sample['num_frames'] = len(features)
                
                stats[split_name]['processed'] += 1
                
            except Exception as e:
                print(f"\nError processing {sample['video_path']}: {e}")
                stats[split_name]['failed'] += 1
    
    extractor.close()
    
    # Print statistics
    print("\n" + "=" * 50)
    print("Preprocessing Statistics:")
    print("=" * 50)
    for split_name, split_stats in stats.items():
        print(f"{split_name:>8}: processed={split_stats['processed']}, "
              f"skipped={split_stats['skipped']}, failed={split_stats['failed']}")
    
    # Save updated splits with feature paths
    for split_name, samples in splits.items():
        split_path = os.path.join(config.paths.splits_dir, f"{split_name}.json")
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)


def main():
    """Main preprocessing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess sign language videos')
    parser.add_argument('--discover', action='store_true',
                        help='Discover videos and create new splits')
    parser.add_argument('--preprocess', action='store_true',
                        help='Extract features from videos')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip already processed files')
    parser.add_argument('--all', action='store_true',
                        help='Run full pipeline (discover + preprocess)')
    
    args = parser.parse_args()
    
    config = get_config()
    
    print("=" * 60)
    print("Sign Language Dataset Preprocessing")
    print("=" * 60)
    print(f"Video root: {config.paths.video_root}")
    print(f"Features dir: {config.paths.features_dir}")
    print(f"Splits dir: {config.paths.splits_dir}")
    
    if args.all:
        args.discover = True
        args.preprocess = True
    
    # Step 1: Discover videos and create splits
    if args.discover:
        print("\n[Step 1] Discovering videos...")
        samples = discover_videos(config)
        
        if len(samples) == 0:
            print("No videos found! Check your video_root path.")
            return
        
        print("\n[Step 2] Creating splits...")
        splits = create_splits(samples, config)
        save_splits(splits, config)
    
    # Step 2: Preprocess videos
    if args.preprocess:
        print("\n[Step 3] Loading splits...")
        try:
            splits = load_splits(config)
        except FileNotFoundError:
            print("Splits not found. Run with --discover first.")
            return
        
        print("\n[Step 4] Preprocessing videos...")
        preprocess_dataset(splits, config, skip_existing=args.skip_existing)
    
    if not args.discover and not args.preprocess:
        print("\nNo action specified. Use --discover, --preprocess, or --all")
        print("\nUsage examples:")
        print("  python preprocess.py --all           # Full pipeline")
        print("  python preprocess.py --discover      # Only create splits")
        print("  python preprocess.py --preprocess    # Only extract features")
    
    print("\n[OK] Done!")


if __name__ == "__main__":
    main()
