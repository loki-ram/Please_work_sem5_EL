"""
Data preprocessing pipeline for ISL-CSLTR.
Handles normalization, temporal derivatives, and augmentation.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, Config
from data.feature_extraction import MediaPipeExtractor, extract_features_from_video


class FeatureNormalizer:
    """
    Normalize pose and hand features using shoulder-centered coordinates.
    """
    
    def __init__(self, config: Config):
        """
        Initialize normalizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.preprocessing = config.preprocessing
        self.mediapipe = config.mediapipe
        
        # Calculate indices for shoulder landmarks in flattened array
        # Pose landmarks order: [11, 12, 13, 14, 15, 16] = [L_shoulder, R_shoulder, ...]
        # Each landmark has 3 values (x, y, z)
        self.left_shoulder_idx = 0 * 3  # Index 11 -> position 0 in extracted
        self.right_shoulder_idx = 1 * 3  # Index 12 -> position 1 in extracted
        
        # Feature dimensions
        self.num_pose_landmarks = len(self.mediapipe.pose_landmarks_indices)
        self.pose_dim = self.num_pose_landmarks * 3
        self.hand_dim = self.mediapipe.num_hand_landmarks * 3
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply normalization to features.
        
        Args:
            features: Raw features of shape (num_frames, num_features)
            
        Returns:
            Normalized features of same shape
        """
        if len(features) == 0:
            return features
        
        normalized = features.copy()
        
        for i in range(len(normalized)):
            normalized[i] = self._normalize_frame(normalized[i])
        
        return normalized
    
    def _normalize_frame(self, frame_features: np.ndarray) -> np.ndarray:
        """
        Normalize a single frame's features.
        
        Args:
            frame_features: Features for one frame
            
        Returns:
            Normalized features
        """
        normalized = frame_features.copy()
        
        # Extract shoulder positions
        left_shoulder = frame_features[self.left_shoulder_idx:self.left_shoulder_idx + 3]
        right_shoulder = frame_features[self.right_shoulder_idx:self.right_shoulder_idx + 3]
        
        # Calculate center point (midpoint between shoulders)
        if self.preprocessing.normalize_to_shoulder:
            center = (left_shoulder + right_shoulder) / 2
        else:
            center = np.zeros(3)
        
        # Calculate scale factor (inter-shoulder distance)
        if self.preprocessing.use_inter_shoulder_scaling:
            scale = np.linalg.norm(left_shoulder - right_shoulder)
            if scale < 1e-6:  # Avoid division by zero
                scale = 1.0
        else:
            scale = 1.0
        
        # Apply normalization to pose landmarks
        for j in range(self.num_pose_landmarks):
            idx = j * 3
            normalized[idx:idx + 3] = (frame_features[idx:idx + 3] - center) / scale
        
        # Apply normalization to left hand
        hand_start = self.pose_dim
        for j in range(self.mediapipe.num_hand_landmarks):
            idx = hand_start + j * 3
            if np.any(frame_features[idx:idx + 3] != 0):  # Only if hand detected
                normalized[idx:idx + 3] = (frame_features[idx:idx + 3] - center) / scale
        
        # Apply normalization to right hand
        hand_start = self.pose_dim + self.hand_dim
        for j in range(self.mediapipe.num_hand_landmarks):
            idx = hand_start + j * 3
            if np.any(frame_features[idx:idx + 3] != 0):  # Only if hand detected
                normalized[idx:idx + 3] = (frame_features[idx:idx + 3] - center) / scale
        
        return normalized
    
    def add_temporal_derivatives(self, features: np.ndarray) -> np.ndarray:
        """
        Add velocity and acceleration features.
        
        Args:
            features: Normalized features of shape (num_frames, base_features)
            
        Returns:
            Features with derivatives of shape (num_frames, base_features * 3)
        """
        num_frames, base_dim = features.shape
        
        # Initialize output array
        output_dim = base_dim
        if self.preprocessing.use_velocity:
            output_dim += base_dim
        if self.preprocessing.use_acceleration:
            output_dim += base_dim
        
        output = np.zeros((num_frames, output_dim), dtype=np.float32)
        
        # Copy base features
        output[:, :base_dim] = features
        
        idx = base_dim
        
        # Add velocity (first-order derivative)
        if self.preprocessing.use_velocity:
            velocity = np.zeros_like(features)
            velocity[1:] = features[1:] - features[:-1]
            output[:, idx:idx + base_dim] = velocity
            idx += base_dim
        
        # Add acceleration (second-order derivative)
        if self.preprocessing.use_acceleration:
            acceleration = np.zeros_like(features)
            if num_frames > 2:
                acceleration[2:] = features[2:] - 2 * features[1:-1] + features[:-2]
            output[:, idx:idx + base_dim] = acceleration
        
        return output


class DataAugmenter:
    """
    Data augmentation for training data.
    """
    
    def __init__(self, config: Config):
        """
        Initialize augmenter.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.preprocessing = config.preprocessing
    
    def augment(self, features: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply augmentation to features.
        
        Args:
            features: Input features of shape (num_frames, num_features)
            seed: Random seed for reproducibility
            
        Returns:
            Augmented features
        """
        if seed is not None:
            np.random.seed(seed)
        
        augmented = features.copy()
        
        # Temporal scaling
        augmented = self._temporal_scale(augmented)
        
        # Spatial noise
        augmented = self._add_spatial_noise(augmented)
        
        # Landmark dropout
        augmented = self._landmark_dropout(augmented)
        
        return augmented
    
    def _temporal_scale(self, features: np.ndarray) -> np.ndarray:
        """Apply random temporal scaling (speed variation)."""
        scale_min, scale_max = self.preprocessing.temporal_scale_range
        scale = np.random.uniform(scale_min, scale_max)
        
        num_frames = len(features)
        new_num_frames = int(num_frames * scale)
        
        if new_num_frames == num_frames:
            return features
        
        # Interpolate to new length
        old_indices = np.linspace(0, num_frames - 1, num_frames)
        new_indices = np.linspace(0, num_frames - 1, new_num_frames)
        
        augmented = np.zeros((new_num_frames, features.shape[1]), dtype=np.float32)
        for i in range(features.shape[1]):
            augmented[:, i] = np.interp(new_indices, old_indices, features[:, i])
        
        return augmented
    
    def _add_spatial_noise(self, features: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to spatial coordinates."""
        noise = np.random.normal(0, self.preprocessing.spatial_noise_std, features.shape)
        return features + noise.astype(np.float32)
    
    def _landmark_dropout(self, features: np.ndarray) -> np.ndarray:
        """Randomly drop landmarks (set to zero)."""
        if self.preprocessing.dropout_landmarks_prob <= 0:
            return features
        
        augmented = features.copy()
        
        # Create dropout mask (drop entire landmark, i.e., 3 consecutive values)
        num_landmarks = features.shape[1] // 3
        
        for i in range(len(augmented)):
            dropout_mask = np.random.random(num_landmarks) < self.preprocessing.dropout_landmarks_prob
            for j, drop in enumerate(dropout_mask):
                if drop:
                    augmented[i, j*3:(j+1)*3] = 0
        
        return augmented


def preprocess_video(
    video_path: str,
    config: Config,
    extractor: MediaPipeExtractor,
    normalizer: FeatureNormalizer,
    augmenter: Optional[DataAugmenter] = None,
    augment: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Full preprocessing pipeline for a single video.
    
    Args:
        video_path: Path to video file
        config: Configuration object
        extractor: MediaPipe extractor
        normalizer: Feature normalizer
        augmenter: Optional data augmenter
        augment: Whether to apply augmentation
        
    Returns:
        Tuple of (processed features, metadata)
    """
    # Extract raw features
    features, metadata = extract_features_from_video(video_path, config, extractor)
    
    # Check minimum length
    if len(features) < config.preprocessing.min_video_length:
        metadata['skipped'] = True
        metadata['skip_reason'] = 'too_short'
        return features, metadata
    
    # Normalize
    features = normalizer.normalize_features(features)
    
    # Add temporal derivatives
    features = normalizer.add_temporal_derivatives(features)
    
    # Augment if requested
    if augment and augmenter is not None:
        features = augmenter.augment(features)
    
    metadata['processed_shape'] = features.shape
    metadata['skipped'] = False
    
    return features, metadata


def preprocess_dataset(
    config: Config,
    splits: Dict[str, List[Dict]],
    save_features: bool = True
) -> Dict[str, Dict]:
    """
    Preprocess entire dataset and optionally save features.
    
    Args:
        config: Configuration object
        splits: Dataset splits from split_dataset.py
        save_features: Whether to save extracted features to disk
        
    Returns:
        Dictionary with preprocessing statistics
    """
    # Initialize components
    extractor = MediaPipeExtractor(config.mediapipe)
    normalizer = FeatureNormalizer(config)
    augmenter = DataAugmenter(config)
    
    stats = {
        'train': {'processed': 0, 'skipped': 0, 'total_frames': 0},
        'val': {'processed': 0, 'skipped': 0, 'total_frames': 0},
        'test': {'processed': 0, 'skipped': 0, 'total_frames': 0},
    }
    
    for split_name, samples in splits.items():
        print(f"\nProcessing {split_name} split ({len(samples)} samples)...")
        
        # Create output directory
        split_features_dir = os.path.join(config.paths.features_dir, split_name)
        os.makedirs(split_features_dir, exist_ok=True)
        
        # Process each sample
        for sample in tqdm(samples, desc=f"Processing {split_name}"):
            video_path = sample['video_path']
            
            # Generate feature filename
            video_id = os.path.splitext(sample['video_filename'])[0]
            folder_name = sample['folder_name'].replace('/', '_').replace('\\', '_')
            feature_filename = f"{folder_name}_{video_id}.npy"
            feature_path = os.path.join(split_features_dir, feature_filename)
            
            try:
                # Preprocess video (augment only training data)
                augment = (split_name == 'train' and config.preprocessing.augment_training)
                features, metadata = preprocess_video(
                    video_path, config, extractor, normalizer,
                    augmenter if augment else None, augment
                )
                
                if metadata.get('skipped', False):
                    stats[split_name]['skipped'] += 1
                    continue
                
                # Save features
                if save_features:
                    np.save(feature_path, features)
                
                # Update sample with feature path
                sample['feature_path'] = feature_path
                sample['num_frames'] = len(features)
                
                stats[split_name]['processed'] += 1
                stats[split_name]['total_frames'] += len(features)
                
            except Exception as e:
                print(f"\nError processing {video_path}: {str(e)}")
                stats[split_name]['skipped'] += 1
    
    # Save updated splits with feature paths
    if save_features:
        for split_name, samples in splits.items():
            split_path = os.path.join(config.paths.splits_dir, f"{split_name}_processed.json")
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n" + "=" * 50)
    print("Preprocessing Statistics:")
    print("=" * 50)
    for split_name, split_stats in stats.items():
        print(f"{split_name:>8}: processed={split_stats['processed']}, "
              f"skipped={split_stats['skipped']}, "
              f"total_frames={split_stats['total_frames']}")
    print("=" * 50)
    
    return stats


if __name__ == "__main__":
    from data.split_dataset import load_splits, create_dataset_split, save_splits
    
    config = get_config()
    
    print("ISL-CSLTR Data Preprocessing Pipeline")
    print("=" * 50)
    
    # Load or create splits
    try:
        splits = load_splits(config)
        print("Loaded existing dataset splits.")
    except FileNotFoundError:
        print("Creating new dataset splits...")
        splits = create_dataset_split(config)
        save_splits(splits, config)
    
    # Preprocess dataset
    print("\nStarting preprocessing...")
    stats = preprocess_dataset(config, splits, save_features=True)
    
    print("\nPreprocessing completed!")
