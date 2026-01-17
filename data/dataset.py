"""
Dataset and DataLoader for Hybrid Sign-to-Text training.
Dynamically reconstructs video paths using VIDEO_ROOT + folder_name + video_filename.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Optional, Callable
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, Config
from data.vocabulary import Vocabulary, load_vocabularies


def reconstruct_video_path(sample: Dict, video_root: str) -> str:
    """
    Reconstruct video path from VIDEO_ROOT + folder_name + video_filename.
    Ignores absolute video_path in JSON for OS-agnostic loading.
    
    Args:
        sample: Sample dictionary with folder_name and video_filename
        video_root: Root directory for videos
        
    Returns:
        Reconstructed video path
    """
    folder_name = sample['folder_name']
    video_filename = sample['video_filename']
    return os.path.join(video_root, folder_name, video_filename)


class HybridDataset(Dataset):
    """
    Dataset for hybrid CTC + Attention training.
    Returns (features, gloss_ids, text_ids).
    """
    
    def __init__(
        self,
        samples: List[Dict],
        gloss_vocab: Vocabulary,
        text_vocab: Vocabulary,
        config: Config,
        split: str = 'train',
        features_dir: Optional[str] = None
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of sample dictionaries
            gloss_vocab: Gloss vocabulary (uppercase)
            text_vocab: Text vocabulary (lowercase)
            config: Configuration object
            split: Split name ('train', 'val', 'test')
            features_dir: Directory containing precomputed features
        """
        self.samples = samples
        self.gloss_vocab = gloss_vocab
        self.text_vocab = text_vocab
        self.config = config
        self.split = split
        
        # Features directory
        if features_dir:
            self.features_dir = features_dir
        else:
            self.features_dir = os.path.join(config.paths.features_dir, split)
        
        # Filter samples with valid feature files
        self.valid_samples = []
        for sample in samples:
            feature_path = self._get_feature_path(sample)
            if os.path.exists(feature_path):
                self.valid_samples.append(sample)
        
        print(f"{split} dataset: {len(self.valid_samples)}/{len(samples)} valid samples")
    
    def _get_feature_path(self, sample: Dict) -> str:
        """Get feature file path for a sample."""
        video_id = os.path.splitext(sample['video_filename'])[0]
        folder_name = sample['folder_name'].replace('/', '_').replace('\\', '_')
        feature_filename = f"{folder_name}_{video_id}.npy"
        return os.path.join(self.features_dir, feature_filename)
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - features: (seq_len, feature_dim) tensor
                - gloss_ids: (gloss_len,) tensor
                - text_ids: (text_len,) tensor (with SOS/EOS)
                - feature_length: scalar tensor
                - gloss_length: scalar tensor
                - text_length: scalar tensor
                - gloss_text: original gloss string
                - text_text: original sentence string
        """
        sample = self.valid_samples[idx]
        
        # Load features
        feature_path = self._get_feature_path(sample)
        features = np.load(feature_path)
        
        # Truncate if necessary
        max_len = self.config.preprocessing.max_video_length
        if len(features) > max_len:
            features = features[:max_len]
        
        # Encode gloss (uppercase, no SOS/EOS for CTC)
        gloss_ids = self.gloss_vocab.encode(
            sample['gloss'],
            uppercase=True,
            add_sos=False,
            add_eos=False
        )
        
        # Encode text (lowercase, with SOS/EOS for decoder)
        text_ids = self.text_vocab.encode(
            sample['sentence'],
            lowercase=True,
            add_sos=True,
            add_eos=True
        )
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'gloss_ids': torch.tensor(gloss_ids, dtype=torch.long),
            'text_ids': torch.tensor(text_ids, dtype=torch.long),
            'feature_length': torch.tensor(len(features), dtype=torch.long),
            'gloss_length': torch.tensor(len(gloss_ids), dtype=torch.long),
            'text_length': torch.tensor(len(text_ids), dtype=torch.long),
            'gloss_text': sample['gloss'],
            'text_text': sample['sentence'],
        }


def hybrid_collate_fn(
    batch: List[Dict],
    gloss_pad_idx: int = 0,
    text_pad_idx: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Collate function for hybrid dataloader.
    Pads feature sequences and both targets.
    
    Decoder inputs: <SOS> + text_ids (already included in text_ids from dataset)
    Decoder targets: text_ids + <EOS> (already included in text_ids from dataset)
    
    Args:
        batch: List of sample dictionaries
        gloss_pad_idx: Padding index for gloss
        text_pad_idx: Padding index for text
        
    Returns:
        Batched and padded tensors
    """
    # Extract tensors
    features = [item['features'] for item in batch]
    gloss_ids = [item['gloss_ids'] for item in batch]
    text_ids = [item['text_ids'] for item in batch]
    feature_lengths = torch.stack([item['feature_length'] for item in batch])
    gloss_lengths = torch.stack([item['gloss_length'] for item in batch])
    text_lengths = torch.stack([item['text_length'] for item in batch])
    
    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    gloss_ids_padded = pad_sequence(gloss_ids, batch_first=True, padding_value=gloss_pad_idx)
    text_ids_padded = pad_sequence(text_ids, batch_first=True, padding_value=text_pad_idx)
    
    # Text strings for evaluation
    gloss_texts = [item['gloss_text'] for item in batch]
    text_texts = [item['text_text'] for item in batch]
    
    return {
        'features': features_padded,
        'gloss_ids': gloss_ids_padded,
        'text_ids': text_ids_padded,
        'feature_lengths': feature_lengths,
        'gloss_lengths': gloss_lengths,
        'text_lengths': text_lengths,
        'gloss_texts': gloss_texts,
        'text_texts': text_texts,
    }


def create_collate_fn(gloss_vocab: Vocabulary, text_vocab: Vocabulary) -> Callable:
    """Create collate function with vocabulary padding indices."""
    def collate_fn(batch):
        return hybrid_collate_fn(
            batch,
            gloss_pad_idx=gloss_vocab.pad_idx,
            text_pad_idx=text_vocab.pad_idx
        )
    return collate_fn


def create_dataloaders(
    config: Config,
    gloss_vocab: Vocabulary,
    text_vocab: Vocabulary
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration object
        gloss_vocab: Gloss vocabulary
        text_vocab: Text vocabulary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load splits
    with open(config.paths.train_split_path, 'r', encoding='utf-8') as f:
        train_samples = json.load(f)
    with open(config.paths.val_split_path, 'r', encoding='utf-8') as f:
        val_samples = json.load(f)
    with open(config.paths.test_split_path, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
    
    # Create datasets
    train_dataset = HybridDataset(
        train_samples, gloss_vocab, text_vocab, config, 'train'
    )
    val_dataset = HybridDataset(
        val_samples, gloss_vocab, text_vocab, config, 'val'
    )
    test_dataset = HybridDataset(
        test_samples, gloss_vocab, text_vocab, config, 'test'
    )
    
    # Create collate function
    collate_fn = create_collate_fn(gloss_vocab, text_vocab)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.training.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.training.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.training.pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    config = get_config()
    
    print("=" * 60)
    print("Testing Dataset and DataLoader")
    print("=" * 60)
    
    # Load vocabularies
    try:
        gloss_vocab, text_vocab = load_vocabularies(config)
        print(f"Loaded gloss vocabulary: {len(gloss_vocab)} tokens")
        print(f"Loaded text vocabulary: {len(text_vocab)} tokens")
    except FileNotFoundError:
        print("Vocabularies not found. Please run vocabulary.py first.")
        sys.exit(1)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config, gloss_vocab, text_vocab
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    for batch in train_loader:
        print(f"\n[Batch Shapes]")
        print(f"  Features: {batch['features'].shape}")
        print(f"  Gloss IDs: {batch['gloss_ids'].shape}")
        print(f"  Text IDs: {batch['text_ids'].shape}")
        print(f"  Feature lengths: {batch['feature_lengths']}")
        print(f"  Gloss lengths: {batch['gloss_lengths']}")
        print(f"  Text lengths: {batch['text_lengths']}")
        
        print(f"\n[Sample]")
        print(f"  Gloss: {batch['gloss_texts'][0]}")
        print(f"  Sentence: {batch['text_texts'][0]}")
        print(f"  Gloss IDs[0]: {batch['gloss_ids'][0].tolist()}")
        print(f"  Text IDs[0]: {batch['text_ids'][0].tolist()}")
        break
    
    print("\n[OK] Dataset test completed!")
