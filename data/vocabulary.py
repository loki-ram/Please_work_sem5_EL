"""
Vocabulary building for gloss and English tokens.
"""

import os
import json
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, Config


class Vocabulary:
    """
    Vocabulary class for mapping tokens to indices and vice versa.
    """
    
    def __init__(
        self,
        pad_token: str = "<PAD>",
        sos_token: str = "<SOS>",
        eos_token: str = "<EOS>",
        unk_token: str = "<UNK>",
        blank_token: Optional[str] = None  # For CTC
    ):
        """
        Initialize vocabulary with special tokens.
        
        Args:
            pad_token: Padding token
            sos_token: Start of sequence token
            eos_token: End of sequence token
            unk_token: Unknown token
            blank_token: CTC blank token (added at index 0 if provided)
        """
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.blank_token = blank_token
        
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        
        # Add special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        # For CTC, blank must be at index 0
        if self.blank_token:
            self._add_token(self.blank_token)
        
        self._add_token(self.pad_token)
        self._add_token(self.sos_token)
        self._add_token(self.eos_token)
        self._add_token(self.unk_token)
    
    def _add_token(self, token: str) -> int:
        """Add a token to vocabulary."""
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        return self.token2idx[token]
    
    def build_from_texts(
        self,
        texts: List[str],
        min_freq: int = 1,
        max_size: Optional[int] = None,
        tokenizer: Optional[callable] = None
    ):
        """
        Build vocabulary from list of texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency to include token
            max_size: Maximum vocabulary size (None for unlimited)
            tokenizer: Custom tokenizer function (default: split by space)
        """
        if tokenizer is None:
            tokenizer = lambda x: x.strip().split()
        
        # Count tokens
        counter = Counter()
        for text in texts:
            tokens = tokenizer(text)
            counter.update(tokens)
        
        # Filter by frequency and sort by frequency
        filtered_tokens = [
            (token, count) for token, count in counter.items()
            if count >= min_freq
        ]
        filtered_tokens.sort(key=lambda x: (-x[1], x[0]))
        
        # Limit size if specified
        if max_size:
            # Account for special tokens
            num_special = len(self.token2idx)
            max_regular_tokens = max_size - num_special
            filtered_tokens = filtered_tokens[:max_regular_tokens]
        
        # Add tokens to vocabulary
        for token, _ in filtered_tokens:
            self._add_token(token)
    
    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text to list of indices.
        
        Args:
            text: Input text string
            add_sos: Whether to add SOS token at start
            add_eos: Whether to add EOS token at end
            
        Returns:
            List of token indices
        """
        tokens = text.strip().split()
        indices = []
        
        if add_sos:
            indices.append(self.token2idx[self.sos_token])
        
        for token in tokens:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                indices.append(self.token2idx[self.unk_token])
        
        if add_eos:
            indices.append(self.token2idx[self.eos_token])
        
        return indices
    
    def decode(
        self,
        indices: List[int],
        remove_special: bool = True,
        join: bool = True
    ) -> str:
        """
        Decode list of indices to text.
        
        Args:
            indices: List of token indices
            remove_special: Whether to remove special tokens
            join: Whether to join tokens into string
            
        Returns:
            Decoded text string or list of tokens
        """
        tokens = []
        special_tokens = {self.pad_token, self.sos_token, self.eos_token}
        if self.blank_token:
            special_tokens.add(self.blank_token)
        
        for idx in indices:
            # Safe index access with bounds checking
            if isinstance(idx, (int, np.integer)) and idx in self.idx2token:
                token = self.idx2token[idx]
                if remove_special and token in special_tokens:
                    continue
                if token == self.eos_token:
                    break
                tokens.append(token)
            elif not remove_special:
                # If not removing special tokens, add unknown marker for invalid indices
                tokens.append(self.unk_token)
        
        if join:
            return ' '.join(tokens)
        return tokens
    
    def __len__(self) -> int:
        return len(self.token2idx)
    
    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.pad_token]
    
    @property
    def sos_idx(self) -> int:
        return self.token2idx[self.sos_token]
    
    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.eos_token]
    
    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.unk_token]
    
    @property
    def blank_idx(self) -> Optional[int]:
        if self.blank_token:
            return self.token2idx[self.blank_token]
        return None
    
    def save(self, path: str):
        """Save vocabulary to JSON file."""
        data = {
            'token2idx': self.token2idx,
            'pad_token': self.pad_token,
            'sos_token': self.sos_token,
            'eos_token': self.eos_token,
            'unk_token': self.unk_token,
            'blank_token': self.blank_token,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(
            pad_token=data['pad_token'],
            sos_token=data['sos_token'],
            eos_token=data['eos_token'],
            unk_token=data['unk_token'],
            blank_token=data.get('blank_token'),
        )
        
        # Clear and rebuild from saved data
        vocab.token2idx = data['token2idx']
        vocab.idx2token = {int(v): k for k, v in data['token2idx'].items()}
        
        return vocab


def build_vocabularies(
    samples: List[Dict],
    config: Config
) -> Tuple[Vocabulary, Vocabulary]:
    """
    Build gloss and English vocabularies from dataset samples.
    
    Args:
        samples: List of sample dictionaries with 'gloss' and 'sentence' keys
        config: Configuration object
        
    Returns:
        Tuple of (gloss_vocab, english_vocab)
    """
    # Extract texts
    gloss_texts = [sample['gloss'] for sample in samples]
    english_texts = [sample['sentence'] for sample in samples]
    
    # Build gloss vocabulary (with blank for CTC)
    gloss_vocab = Vocabulary(
        pad_token=config.dataset.pad_token,
        sos_token=config.dataset.sos_token,
        eos_token=config.dataset.eos_token,
        unk_token=config.dataset.unk_token,
        blank_token=config.dataset.blank_token,
    )
    gloss_vocab.build_from_texts(
        gloss_texts,
        min_freq=config.dataset.gloss_vocab_min_freq
    )
    
    # Build English vocabulary (no blank needed)
    english_vocab = Vocabulary(
        pad_token=config.dataset.pad_token,
        sos_token=config.dataset.sos_token,
        eos_token=config.dataset.eos_token,
        unk_token=config.dataset.unk_token,
        blank_token=None,
    )
    english_vocab.build_from_texts(
        english_texts,
        min_freq=config.dataset.gloss_vocab_min_freq
    )
    
    return gloss_vocab, english_vocab


def save_vocabularies(
    gloss_vocab: Vocabulary,
    english_vocab: Vocabulary,
    config: Config
):
    """Save vocabularies to files."""
    gloss_path = os.path.join(config.paths.output_dir, "gloss_vocab.json")
    english_path = os.path.join(config.paths.output_dir, "english_vocab.json")
    
    gloss_vocab.save(gloss_path)
    english_vocab.save(english_path)
    
    print(f"Saved gloss vocabulary ({len(gloss_vocab)} tokens) to: {gloss_path}")
    print(f"Saved English vocabulary ({len(english_vocab)} tokens) to: {english_path}")


def load_vocabularies(config: Config) -> Tuple[Vocabulary, Vocabulary]:
    """Load vocabularies from files."""
    gloss_path = os.path.join(config.paths.output_dir, "gloss_vocab.json")
    english_path = os.path.join(config.paths.output_dir, "english_vocab.json")
    
    gloss_vocab = Vocabulary.load(gloss_path)
    english_vocab = Vocabulary.load(english_path)
    
    return gloss_vocab, english_vocab


if __name__ == "__main__":
    from data.split_dataset import load_splits
    
    config = get_config()
    
    print("Building vocabularies...")
    
    # Load splits
    splits = load_splits(config)
    
    # Use training data for vocabulary building
    train_samples = splits['train']
    
    # Build vocabularies
    gloss_vocab, english_vocab = build_vocabularies(train_samples, config)
    
    print(f"\nGloss vocabulary size: {len(gloss_vocab)}")
    print(f"English vocabulary size: {len(english_vocab)}")
    
    # Save vocabularies
    save_vocabularies(gloss_vocab, english_vocab, config)
    
    # Test encoding/decoding
    if train_samples:
        sample = train_samples[0]
        print(f"\nSample gloss: {sample['gloss']}")
        encoded = gloss_vocab.encode(sample['gloss'])
        decoded = gloss_vocab.decode(encoded)
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        
        print(f"\nSample sentence: {sample['sentence']}")
        encoded = english_vocab.encode(sample['sentence'], add_sos=True, add_eos=True)
        decoded = english_vocab.decode(encoded)
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
