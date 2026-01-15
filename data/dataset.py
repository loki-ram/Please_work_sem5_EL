"""
PyTorch Dataset classes for ISL-CSLTR.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Optional, Callable
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, Config
from data.vocabulary import Vocabulary, load_vocabularies


class SignLanguageDataset(Dataset):
    """
    Dataset for sign language video features and gloss/sentence annotations.
    """
    
    def __init__(
        self,
        samples: List[Dict],
        gloss_vocab: Vocabulary,
        english_vocab: Vocabulary,
        config: Config,
        split: str = 'train',
        load_features_to_memory: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of sample dictionaries
            gloss_vocab: Gloss vocabulary
            english_vocab: English vocabulary
            config: Configuration object
            split: Split name ('train', 'val', 'test')
            load_features_to_memory: Whether to preload all features
        """
        self.samples = samples
        self.gloss_vocab = gloss_vocab
        self.english_vocab = english_vocab
        self.config = config
        self.split = split
        self.load_features_to_memory = load_features_to_memory
        
        # Filter samples with valid feature files
        self.valid_samples = []
        for sample in samples:
            feature_path = sample.get('feature_path')
            if feature_path and os.path.exists(feature_path):
                self.valid_samples.append(sample)
        
        print(f"{split} dataset: {len(self.valid_samples)}/{len(samples)} valid samples")
        
        # Preload features if requested
        self.cached_features = {}
        if load_features_to_memory:
            print(f"Loading features to memory...")
            for i, sample in enumerate(self.valid_samples):
                self.cached_features[i] = np.load(sample['feature_path'])
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - features: (seq_len, feature_dim) tensor
                - gloss_ids: (gloss_len,) tensor
                - sentence_ids: (sentence_len,) tensor (with SOS/EOS)
                - feature_length: scalar tensor
                - gloss_length: scalar tensor
                - sentence_length: scalar tensor
        """
        sample = self.valid_samples[idx]
        
        # Load features
        if self.load_features_to_memory:
            features = self.cached_features[idx]
        else:
            features = np.load(sample['feature_path'])
        
        # Truncate if necessary
        max_len = self.config.preprocessing.max_video_length
        if len(features) > max_len:
            features = features[:max_len]
        
        # Encode gloss
        gloss_ids = self.gloss_vocab.encode(sample['gloss'])
        gloss_ids = gloss_ids[:self.config.dataset.max_gloss_length]
        
        # Encode English sentence (with SOS/EOS for decoder)
        sentence_ids = self.english_vocab.encode(
            sample['sentence'],
            add_sos=True,
            add_eos=True
        )
        sentence_ids = sentence_ids[:self.config.dataset.max_sentence_length]
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'gloss_ids': torch.tensor(gloss_ids, dtype=torch.long),
            'sentence_ids': torch.tensor(sentence_ids, dtype=torch.long),
            'feature_length': torch.tensor(len(features), dtype=torch.long),
            'gloss_length': torch.tensor(len(gloss_ids), dtype=torch.long),
            'sentence_length': torch.tensor(len(sentence_ids), dtype=torch.long),
            'gloss_text': sample['gloss'],
            'sentence_text': sample['sentence'],
        }


class Stage2Dataset(Dataset):
    """
    Dataset for Stage 2 (Gloss to English translation).
    Can use either ground truth or predicted glosses.
    """
    
    def __init__(
        self,
        samples: List[Dict],
        gloss_vocab: Vocabulary,
        english_vocab: Vocabulary,
        config: Config,
        predicted_glosses: Optional[Dict[str, str]] = None
    ):
        """
        Initialize Stage 2 dataset.
        
        Args:
            samples: List of sample dictionaries
            gloss_vocab: Gloss vocabulary
            english_vocab: English vocabulary
            config: Configuration object
            predicted_glosses: Optional dict mapping sample_id to predicted gloss
        """
        self.samples = samples
        self.gloss_vocab = gloss_vocab
        self.english_vocab = english_vocab
        self.config = config
        self.predicted_glosses = predicted_glosses
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Use predicted gloss if available, otherwise ground truth
        if self.predicted_glosses and sample.get('video_filename') in self.predicted_glosses:
            gloss_text = self.predicted_glosses[sample['video_filename']]
        else:
            gloss_text = sample['gloss']
        
        # Encode gloss (input)
        gloss_ids = self.gloss_vocab.encode(gloss_text, add_sos=True, add_eos=True)
        
        # Encode English sentence (target)
        sentence_ids = self.english_vocab.encode(
            sample['sentence'],
            add_sos=True,
            add_eos=True
        )
        
        return {
            'gloss_ids': torch.tensor(gloss_ids, dtype=torch.long),
            'sentence_ids': torch.tensor(sentence_ids, dtype=torch.long),
            'gloss_length': torch.tensor(len(gloss_ids), dtype=torch.long),
            'sentence_length': torch.tensor(len(sentence_ids), dtype=torch.long),
            'gloss_text': gloss_text,
            'sentence_text': sample['sentence'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Pads sequences to same length within batch.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched and padded tensors
    """
    # Separate tensors and strings
    features = [item['features'] for item in batch]
    gloss_ids = [item['gloss_ids'] for item in batch]
    sentence_ids = [item['sentence_ids'] for item in batch]
    feature_lengths = torch.stack([item['feature_length'] for item in batch])
    gloss_lengths = torch.stack([item['gloss_length'] for item in batch])
    sentence_lengths = torch.stack([item['sentence_length'] for item in batch])
    
    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    gloss_ids_padded = pad_sequence(gloss_ids, batch_first=True, padding_value=0)
    sentence_ids_padded = pad_sequence(sentence_ids, batch_first=True, padding_value=0)
    
    # Text for evaluation
    gloss_texts = [item['gloss_text'] for item in batch]
    sentence_texts = [item['sentence_text'] for item in batch]
    
    return {
        'features': features_padded,
        'gloss_ids': gloss_ids_padded,
        'sentence_ids': sentence_ids_padded,
        'feature_lengths': feature_lengths,
        'gloss_lengths': gloss_lengths,
        'sentence_lengths': sentence_lengths,
        'gloss_texts': gloss_texts,
        'sentence_texts': sentence_texts,
    }


def collate_fn_stage2(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for Stage 2 DataLoader.
    """
    gloss_ids = [item['gloss_ids'] for item in batch]
    sentence_ids = [item['sentence_ids'] for item in batch]
    gloss_lengths = torch.stack([item['gloss_length'] for item in batch])
    sentence_lengths = torch.stack([item['sentence_length'] for item in batch])
    
    gloss_ids_padded = pad_sequence(gloss_ids, batch_first=True, padding_value=0)
    sentence_ids_padded = pad_sequence(sentence_ids, batch_first=True, padding_value=0)
    
    gloss_texts = [item['gloss_text'] for item in batch]
    sentence_texts = [item['sentence_text'] for item in batch]
    
    return {
        'gloss_ids': gloss_ids_padded,
        'sentence_ids': sentence_ids_padded,
        'gloss_lengths': gloss_lengths,
        'sentence_lengths': sentence_lengths,
        'gloss_texts': gloss_texts,
        'sentence_texts': sentence_texts,
    }


def create_dataloaders(
    config: Config,
    gloss_vocab: Vocabulary,
    english_vocab: Vocabulary,
    stage: int = 1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration object
        gloss_vocab: Gloss vocabulary
        english_vocab: English vocabulary
        stage: 1 for video-to-gloss, 2 for gloss-to-english
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from data.split_dataset import load_splits
    
    # Load processed splits
    splits_dir = config.paths.splits_dir
    
    # Try to load processed splits first (with feature paths)
    train_path = os.path.join(splits_dir, "train_processed.json")
    val_path = os.path.join(splits_dir, "val_processed.json")
    test_path = os.path.join(splits_dir, "test_processed.json")
    
    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            train_samples = json.load(f)
        with open(val_path, 'r', encoding='utf-8') as f:
            val_samples = json.load(f)
        with open(test_path, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
    else:
        # Fall back to original splits
        splits = load_splits(config)
        train_samples = splits['train']
        val_samples = splits['val']
        test_samples = splits['test']
    
    if stage == 1:
        # Create Stage 1 datasets
        train_dataset = SignLanguageDataset(
            train_samples, gloss_vocab, english_vocab, config, 'train'
        )
        val_dataset = SignLanguageDataset(
            val_samples, gloss_vocab, english_vocab, config, 'val'
        )
        test_dataset = SignLanguageDataset(
            test_samples, gloss_vocab, english_vocab, config, 'test'
        )
        collate = collate_fn
    else:
        # Create Stage 2 datasets
        train_dataset = Stage2Dataset(
            train_samples, gloss_vocab, english_vocab, config
        )
        val_dataset = Stage2Dataset(
            val_samples, gloss_vocab, english_vocab, config
        )
        test_dataset = Stage2Dataset(
            test_samples, gloss_vocab, english_vocab, config
        )
        collate = collate_fn_stage2
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        collate_fn=collate,
        pin_memory=config.dataset.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        collate_fn=collate,
        pin_memory=config.dataset.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        collate_fn=collate,
        pin_memory=config.dataset.pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    config = get_config()
    
    print("Testing data loaders...")
    
    # Load vocabularies
    try:
        gloss_vocab, english_vocab = load_vocabularies(config)
    except FileNotFoundError:
        print("Vocabularies not found. Run vocabulary.py first.")
        exit(1)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config, gloss_vocab, english_vocab, stage=1
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    for batch in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Features: {batch['features'].shape}")
        print(f"  Gloss IDs: {batch['gloss_ids'].shape}")
        print(f"  Sentence IDs: {batch['sentence_ids'].shape}")
        print(f"  Feature lengths: {batch['feature_lengths']}")
        print(f"\nSample gloss: {batch['gloss_texts'][0]}")
        print(f"Sample sentence: {batch['sentence_texts'][0]}")
        break
