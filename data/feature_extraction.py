"""
MediaPipe feature extraction from sign language videos.
Extracts pose and hand landmarks from video frames.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, Config

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not installed. Install with: pip install mediapipe")


class MediaPipeExtractor:
    """Extract pose and hand landmarks using MediaPipe."""
    
    def __init__(self, config: Config):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is required for feature extraction")
        
        self.config = config
        self.mp_config = config.mediapipe
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.mp_config.pose_model_complexity,
            min_detection_confidence=self.mp_config.min_detection_confidence,
            min_tracking_confidence=self.mp_config.min_tracking_confidence
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=self.mp_config.hand_model_complexity,
            min_detection_confidence=self.mp_config.min_detection_confidence,
            min_tracking_confidence=self.mp_config.min_tracking_confidence
        )
        
        self.pose_indices = self.mp_config.pose_landmark_indices
        self.num_hand_landmarks = self.mp_config.num_hand_landmarks
    
    def extract_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract landmarks from a single frame.
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            Feature vector of shape (144,) - base features without derivatives
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract pose landmarks
        pose_results = self.pose.process(rgb_frame)
        
        # Extract hand landmarks
        hand_results = self.hands.process(rgb_frame)
        
        # Initialize feature vector
        # 6 pose * 3 + 21 left_hand * 3 + 21 right_hand * 3 = 144
        features = np.zeros(self.mp_config.base_features_per_frame, dtype=np.float32)
        
        # Extract pose landmarks
        if pose_results.pose_landmarks:
            for i, idx in enumerate(self.pose_indices):
                landmark = pose_results.pose_landmarks.landmark[idx]
                features[i*3:i*3+3] = [landmark.x, landmark.y, landmark.z]
        
        # Extract hand landmarks
        left_hand_start = len(self.pose_indices) * 3
        right_hand_start = left_hand_start + self.num_hand_landmarks * 3
        
        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                handedness = hand_results.multi_handedness[hand_idx].classification[0].label
                
                if handedness == 'Left':
                    start_idx = left_hand_start
                else:
                    start_idx = right_hand_start
                
                for i, landmark in enumerate(hand_landmarks.landmark):
                    features[start_idx + i*3:start_idx + i*3 + 3] = [
                        landmark.x, landmark.y, landmark.z
                    ]
        
        return features
    
    def extract_video(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Extract landmarks from entire video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (features array, metadata dict)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling rate
        target_fps = self.config.preprocessing.target_fps
        if fps > 0:
            sample_rate = max(1, int(fps / target_fps))
        else:
            sample_rate = 1
        
        frames_features = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at target FPS
            if frame_idx % sample_rate == 0:
                features = self.extract_frame(frame)
                frames_features.append(features)
                
                # Limit max frames
                if len(frames_features) >= self.config.preprocessing.max_video_length:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        features_array = np.array(frames_features, dtype=np.float32)
        
        metadata = {
            'video_path': video_path,
            'original_fps': fps,
            'total_frames': total_frames,
            'extracted_frames': len(frames_features),
            'sample_rate': sample_rate,
        }
        
        return features_array, metadata
    
    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()
        self.hands.close()


class FeatureNormalizer:
    """
    Normalize pose and hand features using shoulder-centered coordinates.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessing = config.preprocessing
        self.mediapipe = config.mediapipe
        
        # Shoulder indices in flattened array
        self.left_shoulder_idx = 0 * 3
        self.right_shoulder_idx = 1 * 3
        
        # Feature dimensions
        self.num_pose_landmarks = len(self.mediapipe.pose_landmark_indices)
        self.pose_dim = self.num_pose_landmarks * 3
        self.hand_dim = self.mediapipe.num_hand_landmarks * 3
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply shoulder-center normalization to a single frame."""
        normalized = frame.copy()
        
        # Get shoulder positions
        left_shoulder = frame[self.left_shoulder_idx:self.left_shoulder_idx+3]
        right_shoulder = frame[self.right_shoulder_idx:self.right_shoulder_idx+3]
        
        # Center point
        if self.preprocessing.normalize_to_shoulder:
            center = (left_shoulder + right_shoulder) / 2
        else:
            center = np.zeros(3)
        
        # Scale factor
        if self.preprocessing.use_inter_shoulder_scaling:
            scale = np.linalg.norm(left_shoulder - right_shoulder)
            if scale < 1e-6:
                scale = 1.0
        else:
            scale = 1.0
        
        # Apply normalization to pose landmarks
        for j in range(self.num_pose_landmarks):
            idx = j * 3
            normalized[idx:idx+3] = (frame[idx:idx+3] - center) / scale
        
        # Apply normalization to left hand
        hand_start = self.pose_dim
        for j in range(self.mediapipe.num_hand_landmarks):
            idx = hand_start + j * 3
            if np.any(frame[idx:idx+3] != 0):
                normalized[idx:idx+3] = (frame[idx:idx+3] - center) / scale
        
        # Apply normalization to right hand
        hand_start = self.pose_dim + self.hand_dim
        for j in range(self.mediapipe.num_hand_landmarks):
            idx = hand_start + j * 3
            if np.any(frame[idx:idx+3] != 0):
                normalized[idx:idx+3] = (frame[idx:idx+3] - center) / scale
        
        return normalized
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply normalization to all frames."""
        if len(features) == 0:
            return features
        
        normalized = np.array([self.normalize_frame(f) for f in features])
        return normalized
    
    def add_temporal_derivatives(self, features: np.ndarray) -> np.ndarray:
        """Add velocity and acceleration features."""
        num_frames, base_dim = features.shape
        
        output_dim = base_dim
        if self.preprocessing.use_velocity:
            output_dim += base_dim
        if self.preprocessing.use_acceleration:
            output_dim += base_dim
        
        output = np.zeros((num_frames, output_dim), dtype=np.float32)
        output[:, :base_dim] = features
        
        idx = base_dim
        
        if self.preprocessing.use_velocity:
            velocity = np.zeros_like(features)
            velocity[1:] = features[1:] - features[:-1]
            output[:, idx:idx+base_dim] = velocity
            idx += base_dim
        
        if self.preprocessing.use_acceleration:
            acceleration = np.zeros_like(features)
            if num_frames > 2:
                acceleration[2:] = features[2:] - 2*features[1:-1] + features[:-2]
            output[:, idx:idx+base_dim] = acceleration
        
        return output


class DataAugmenter:
    """Data augmentation for training data."""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessing = config.preprocessing
    
    def augment(self, features: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Apply augmentation to features."""
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
        
        if new_num_frames == num_frames or new_num_frames < 1:
            return features
        
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
        num_landmarks = features.shape[1] // 3
        
        for i in range(len(augmented)):
            dropout_mask = np.random.random(num_landmarks) < self.preprocessing.dropout_landmarks_prob
            for j, drop in enumerate(dropout_mask):
                if drop:
                    augmented[i, j*3:(j+1)*3] = 0
        
        return augmented


if __name__ == "__main__":
    print("Feature Extraction Module")
    print("=" * 50)
    
    if MEDIAPIPE_AVAILABLE:
        print("MediaPipe: Available")
    else:
        print("MediaPipe: NOT AVAILABLE - Install with: pip install mediapipe")
