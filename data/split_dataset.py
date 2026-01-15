"""
Dataset splitting utility for ISL-CSLTR dataset.
Splits the dataset into training, validation, and test sets.
"""

import os
import json
import random
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, Config


def load_gloss_annotations(csv_path: str) -> pd.DataFrame:
    """
    Load gloss annotations from CSV file.
    
    Expected CSV format (ISL Corpus):
    - Sentence: English sentence
    - SIGN GLOSSES: ISL gloss sequence (space-separated, uppercase)
    
    Args:
        csv_path: Path to the gloss CSV file
        
    Returns:
        DataFrame with standardized columns: folder_name, gloss, sentence
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    # Validate and normalize path to prevent path traversal
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Gloss CSV not found: {csv_path}")
    
    # Validate file extension
    if not csv_path.lower().endswith('.csv'):
        raise ValueError(f"Expected CSV file, got: {csv_path}")
    
    # Load CSV with error handling
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Try alternative encoding
        df = pd.read_csv(csv_path, encoding='latin-1')
    
    # Standardize column names for ISL Corpus format
    column_mapping = {
        # ISL Corpus format
        'Sentence': 'sentence',
        'SIGN GLOSSES': 'gloss',
        # Alternative column names
        'folder': 'folder_name',
        'folder_id': 'folder_name',
        'sentence_id': 'folder_name',
        'id': 'folder_name',
        'gloss_sequence': 'gloss',
        'glosses': 'gloss',
        'isl_gloss': 'gloss',
        'english': 'sentence',
        'text': 'sentence',
        'english_sentence': 'sentence',
        'translation': 'sentence',
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # If folder_name doesn't exist, create it from sentence
    # For ISL Corpus, folder names match the sentence text (lowercased)
    if 'folder_name' not in df.columns:
        # Use the sentence as folder name (matching the actual folder structure)
        df['folder_name'] = df['Sentence'].astype(str).str.strip().str.lower() if 'Sentence' in df.columns else df['sentence'].astype(str).str.strip().str.lower()
    
    # Ensure gloss and sentence columns exist
    required_columns = ['gloss', 'sentence']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    # Clean up data
    df['sentence'] = df['sentence'].astype(str).str.strip().str.lower()
    df['gloss'] = df['gloss'].astype(str).str.strip().str.upper()
    
    # Remove any rows with empty values
    df = df.dropna(subset=['sentence', 'gloss'])
    df = df[df['sentence'].str.len() > 0]
    df = df[df['gloss'].str.len() > 0]
    
    return df


def get_video_files(videos_dir: str, folder_name: str) -> List[str]:
    """
    Get all video files in a sentence folder.
    
    Args:
        videos_dir: Root videos directory
        folder_name: Sentence folder name
        
    Returns:
        List of video file paths
    """
    folder_path = os.path.join(videos_dir, folder_name)
    
    if not os.path.exists(folder_path):
        return []
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    videos = []
    
    for file in os.listdir(folder_path):
        if os.path.splitext(file)[1].lower() in video_extensions:
            videos.append(os.path.join(folder_path, file))
    
    return videos


def create_dataset_split(
    config: Config,
    stratify_by_length: bool = True
) -> Dict[str, List[Dict]]:
    """
    Create train/val/test splits for the dataset.
    
    Args:
        config: Configuration object
        stratify_by_length: Whether to stratify by gloss sequence length
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing sample lists
    """
    random.seed(config.dataset.random_seed)
    
    # Load annotations
    print(f"Loading annotations from: {config.paths.gloss_csv}")
    df = load_gloss_annotations(config.paths.gloss_csv)
    print(f"Found {len(df)} sentence annotations")
    
    # Build samples list with all videos per sentence
    samples = []
    skipped_folders = 0
    
    for idx, row in df.iterrows():
        folder_name = str(row['folder_name'])
        gloss = str(row['gloss'])
        sentence = str(row['sentence'])
        
        # Get all videos for this sentence
        videos = get_video_files(config.paths.videos_dir, folder_name)
        
        if not videos:
            skipped_folders += 1
            continue
        
        # Create sample for each video
        for video_path in videos:
            sample = {
                'folder_name': folder_name,
                'video_path': video_path,
                'video_filename': os.path.basename(video_path),
                'gloss': gloss,
                'sentence': sentence,
                'gloss_length': len(gloss.split()),
            }
            samples.append(sample)
    
    print(f"Total video samples: {len(samples)}")
    print(f"Skipped folders (no videos): {skipped_folders}")
    
    if len(samples) == 0:
        raise ValueError("No valid samples found! Check dataset paths.")
    
    # Group by folder for stratified splitting (keep videos from same sentence together)
    folder_samples = defaultdict(list)
    for sample in samples:
        folder_samples[sample['folder_name']].append(sample)
    
    folders = list(folder_samples.keys())
    
    if stratify_by_length:
        # Sort folders by average gloss length for stratification
        folders_with_length = [(f, folder_samples[f][0]['gloss_length']) for f in folders]
        folders_with_length.sort(key=lambda x: x[1])
        folders = [f for f, _ in folders_with_length]
    else:
        random.shuffle(folders)
    
    # Calculate split indices
    n_folders = len(folders)
    n_train = int(n_folders * config.dataset.train_ratio)
    n_val = int(n_folders * config.dataset.val_ratio)
    
    # For stratified split, take samples evenly from sorted list
    if stratify_by_length:
        indices = list(range(n_folders))
        random.shuffle(indices)
        
        train_indices = set(indices[:n_train])
        val_indices = set(indices[n_train:n_train + n_val])
        test_indices = set(indices[n_train + n_val:])
        
        train_folders = [folders[i] for i in range(n_folders) if i in train_indices]
        val_folders = [folders[i] for i in range(n_folders) if i in val_indices]
        test_folders = [folders[i] for i in range(n_folders) if i in test_indices]
    else:
        train_folders = folders[:n_train]
        val_folders = folders[n_train:n_train + n_val]
        test_folders = folders[n_train + n_val:]
    
    # Collect samples for each split
    splits = {
        'train': [],
        'val': [],
        'test': [],
    }
    
    for folder in train_folders:
        splits['train'].extend(folder_samples[folder])
    for folder in val_folders:
        splits['val'].extend(folder_samples[folder])
    for folder in test_folders:
        splits['test'].extend(folder_samples[folder])
    
    # Shuffle within each split
    for split_name in splits:
        random.shuffle(splits[split_name])
    
    # Print statistics
    print("\n" + "=" * 50)
    print("Dataset Split Statistics:")
    print("=" * 50)
    for split_name, split_samples in splits.items():
        n_samples = len(split_samples)
        n_unique_sentences = len(set(s['folder_name'] for s in split_samples))
        avg_gloss_len = sum(s['gloss_length'] for s in split_samples) / max(n_samples, 1)
        print(f"{split_name:>8}: {n_samples:5} videos, {n_unique_sentences:4} sentences, "
              f"avg gloss length: {avg_gloss_len:.1f}")
    print("=" * 50)
    
    return splits


def save_splits(splits: Dict[str, List[Dict]], config: Config) -> None:
    """
    Save dataset splits to JSON files.
    
    Args:
        splits: Dictionary with split data
        config: Configuration object
    """
    os.makedirs(config.paths.splits_dir, exist_ok=True)
    
    for split_name, samples in splits.items():
        output_path = os.path.join(config.paths.splits_dir, f"{split_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"Saved {split_name} split to: {output_path}")
    
    # Save split summary
    summary = {
        'total_samples': sum(len(s) for s in splits.values()),
        'train_samples': len(splits['train']),
        'val_samples': len(splits['val']),
        'test_samples': len(splits['test']),
        'train_ratio': config.dataset.train_ratio,
        'val_ratio': config.dataset.val_ratio,
        'test_ratio': config.dataset.test_ratio,
        'random_seed': config.dataset.random_seed,
    }
    
    summary_path = os.path.join(config.paths.splits_dir, "split_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved split summary to: {summary_path}")


def load_splits(config: Config) -> Dict[str, List[Dict]]:
    """
    Load dataset splits from JSON files.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with split data
    """
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        split_path = os.path.join(config.paths.splits_dir, f"{split_name}.json")
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}. Run split_dataset.py first.")
        
        with open(split_path, 'r', encoding='utf-8') as f:
            splits[split_name] = json.load(f)
    
    return splits


def main():
    """Main function to create and save dataset splits."""
    config = get_config()
    
    print("Creating dataset splits for ISL-CSLTR...")
    print(f"Dataset root: {config.paths.dataset_root}")
    print(f"Split ratios - Train: {config.dataset.train_ratio}, "
          f"Val: {config.dataset.val_ratio}, Test: {config.dataset.test_ratio}")
    print()
    
    # Create splits
    splits = create_dataset_split(config, stratify_by_length=True)
    
    # Save splits
    save_splits(splits, config)
    
    print("\nDataset splitting completed successfully!")


if __name__ == "__main__":
    main()
