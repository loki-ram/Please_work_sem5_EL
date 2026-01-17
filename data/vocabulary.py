"""
Vocabulary building for gloss (uppercase) and text (lowercase) tokens.
Builds two separate word-level vocabularies from the training split only.
"""

import os
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        blank_token: Optional[str] = None
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
        lowercase: bool = False,
        uppercase: bool = False
    ):
        """
        Build vocabulary from list of texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency to include token
            max_size: Maximum vocabulary size
            lowercase: Convert tokens to lowercase
            uppercase: Convert tokens to uppercase
        """
        # Count tokens
        counter = Counter()
        for text in texts:
            tokens = text.strip().split()
            for token in tokens:
                if lowercase:
                    token = token.lower()
                elif uppercase:
                    token = token.upper()
                counter[token] += 1
        
        # Filter by frequency and sort
        filtered = [
            (token, count) for token, count in counter.items()
            if count >= min_freq
        ]
        filtered.sort(key=lambda x: (-x[1], x[0]))
        
        # Limit size
        if max_size:
            num_special = len(self.token2idx)
            filtered = filtered[:max_size - num_special]
        
        # Add tokens
        for token, _ in filtered:
            self._add_token(token)
    
    def encode(
        self,
        text: str,
        add_sos: bool = False,
        add_eos: bool = False,
        lowercase: bool = False,
        uppercase: bool = False
    ) -> List[int]:
        """
        Encode text to list of indices.
        
        Args:
            text: Input text string
            add_sos: Add SOS token at start
            add_eos: Add EOS token at end
            lowercase: Convert tokens to lowercase
            uppercase: Convert tokens to uppercase
            
        Returns:
            List of token indices
        """
        tokens = text.strip().split()
        indices = []
        
        if add_sos:
            indices.append(self.sos_idx)
        
        for token in tokens:
            if lowercase:
                token = token.lower()
            elif uppercase:
                token = token.upper()
            
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                indices.append(self.unk_idx)
        
        if add_eos:
            indices.append(self.eos_idx)
        
        return indices
    
    def decode(
        self,
        indices: List[int],
        remove_special: bool = True,
        stop_at_eos: bool = True
    ) -> str:
        """
        Decode list of indices to text.
        
        Args:
            indices: List of token indices
            remove_special: Remove special tokens
            stop_at_eos: Stop decoding at EOS token
            
        Returns:
            Decoded text string
        """
        special_tokens = {self.pad_token, self.sos_token}
        if self.blank_token:
            special_tokens.add(self.blank_token)
        
        tokens = []
        for idx in indices:
            if isinstance(idx, int) and idx in self.idx2token:
                token = self.idx2token[idx]
                
                if stop_at_eos and token == self.eos_token:
                    break
                
                if remove_special and token in special_tokens:
                    continue
                
                tokens.append(token)
        
        return ' '.join(tokens)
    
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
        
        # Rebuild from saved data
        vocab.token2idx = data['token2idx']
        vocab.idx2token = {int(v): k for k, v in data['token2idx'].items()}
        
        return vocab


def build_vocabularies(
    train_samples: List[Dict],
    config: Config
) -> Tuple[Vocabulary, Vocabulary]:
    """
    Build gloss and text vocabularies from training samples.
    
    Args:
        train_samples: List of training sample dictionaries
        config: Configuration object
        
    Returns:
        Tuple of (gloss_vocab, text_vocab)
    """
    # Extract texts
    gloss_texts = [sample['gloss'] for sample in train_samples]
    english_texts = [sample['sentence'] for sample in train_samples]
    
    # Build gloss vocabulary (uppercase, with blank for CTC)
    gloss_vocab = Vocabulary(
        pad_token=config.vocab.pad_token,
        sos_token=config.vocab.sos_token,
        eos_token=config.vocab.eos_token,
        unk_token=config.vocab.unk_token,
        blank_token=config.vocab.blank_token,
    )
    gloss_vocab.build_from_texts(
        gloss_texts,
        min_freq=config.vocab.min_freq,
        max_size=config.vocab.max_gloss_vocab_size,
        uppercase=True
    )
    
    # Build text vocabulary (lowercase, no blank)
    text_vocab = Vocabulary(
        pad_token=config.vocab.pad_token,
        sos_token=config.vocab.sos_token,
        eos_token=config.vocab.eos_token,
        unk_token=config.vocab.unk_token,
        blank_token=None,
    )
    text_vocab.build_from_texts(
        english_texts,
        min_freq=config.vocab.min_freq,
        max_size=config.vocab.max_text_vocab_size,
        lowercase=True
    )
    
    return gloss_vocab, text_vocab


def save_vocabularies(
    gloss_vocab: Vocabulary,
    text_vocab: Vocabulary,
    config: Config
):
    """Save vocabularies to files."""
    gloss_vocab.save(config.paths.gloss_vocab_path)
    text_vocab.save(config.paths.text_vocab_path)
    
    print(f"Saved gloss vocabulary ({len(gloss_vocab)} tokens) to: {config.paths.gloss_vocab_path}")
    print(f"Saved text vocabulary ({len(text_vocab)} tokens) to: {config.paths.text_vocab_path}")


def load_vocabularies(config: Config) -> Tuple[Vocabulary, Vocabulary]:
    """Load vocabularies from files."""
    gloss_vocab = Vocabulary.load(config.paths.gloss_vocab_path)
    text_vocab = Vocabulary.load(config.paths.text_vocab_path)
    
    return gloss_vocab, text_vocab


if __name__ == "__main__":
    config = get_config()
    
    print("=" * 60)
    print("Building Vocabularies")
    print("=" * 60)
    
    # Load training data
    with open(config.paths.train_split_path, 'r', encoding='utf-8') as f:
        train_samples = json.load(f)
    
    print(f"Loaded {len(train_samples)} training samples")
    
    # Build vocabularies
    gloss_vocab, text_vocab = build_vocabularies(train_samples, config)
    
    print(f"\nGloss vocabulary size: {len(gloss_vocab)}")
    print(f"Text vocabulary size: {len(text_vocab)}")
    
    # Save vocabularies
    save_vocabularies(gloss_vocab, text_vocab, config)
    
    # Test encoding/decoding
    if train_samples:
        sample = train_samples[0]
        
        print(f"\n[Test Sample]")
        print(f"  Gloss: {sample['gloss']}")
        gloss_encoded = gloss_vocab.encode(sample['gloss'], uppercase=True)
        gloss_decoded = gloss_vocab.decode(gloss_encoded)
        print(f"  Encoded: {gloss_encoded}")
        print(f"  Decoded: {gloss_decoded}")
        
        print(f"\n  Sentence: {sample['sentence']}")
        text_encoded = text_vocab.encode(sample['sentence'], add_sos=True, add_eos=True, lowercase=True)
        text_decoded = text_vocab.decode(text_encoded)
        print(f"  Encoded: {text_encoded}")
        print(f"  Decoded: {text_decoded}")
    
    print("\n[OK] Vocabulary building completed!")
