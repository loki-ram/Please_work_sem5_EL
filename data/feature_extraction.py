"""
MediaPipe feature extraction for ISL-CSLTR videos.
Extracts pose and hand landmarks and converts them to normalized features.
"""

import os
import cv2
import numpy as np
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, Config, MediaPipeConfig


def check_mediapipe_available():
    """Check if MediaPipe is available and raise informative error if not."""
    if not MEDIAPIPE_AVAILABLE:
        raise ImportError(
            "MediaPipe is required for feature extraction but is not installed.\n"
            "Install it with: pip install mediapipe\n"
            "Note: MediaPipe may not be available on all platforms."
        )


@dataclass
class LandmarkData:
    """Container for extracted landmarks from a single frame."""
    pose_landmarks: Optional[np.ndarray] = None  # Shape: (num_landmarks, 3)
    left_hand_landmarks: Optional[np.ndarray] = None  # Shape: (21, 3)
    right_hand_landmarks: Optional[np.ndarray] = None  # Shape: (21, 3)
    is_valid: bool = False


class MediaPipeExtractor:
    """
    Extract pose and hand landmarks from video frames using MediaPipe.
    """
    
    def __init__(self, config: MediaPipeConfig):
        """
        Initialize MediaPipe models.
        
        Args:
            config: MediaPipe configuration
            
        Raises:
            ImportError: If MediaPipe is not installed
        """
        check_mediapipe_available()
        
        self.config = config
        
        # Initialize MediaPipe Holistic for combined pose and hand detection
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=config.pose_model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        
        # Pose landmark indices to extract
        self.pose_indices = config.pose_landmarks_indices
        
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'holistic'):
            self.holistic.close()
    
    def extract_frame_landmarks(self, frame: np.ndarray) -> LandmarkData:
        """
        Extract landmarks from a single frame.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            LandmarkData containing extracted landmarks
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.holistic.process(rgb_frame)
        
        landmark_data = LandmarkData()
        
        # Extract pose landmarks
        if results.pose_landmarks:
            pose_all = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.pose_landmarks.landmark
            ])
            # Extract only specified landmarks
            landmark_data.pose_landmarks = pose_all[self.pose_indices]
        
        # Extract left hand landmarks
        if results.left_hand_landmarks:
            landmark_data.left_hand_landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.left_hand_landmarks.landmark
            ])
        
        # Extract right hand landmarks
        if results.right_hand_landmarks:
            landmark_data.right_hand_landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.right_hand_landmarks.landmark
            ])
        
        # Check if we have at least pose landmarks
        landmark_data.is_valid = landmark_data.pose_landmarks is not None
        
        return landmark_data
    
    def extract_video_landmarks(
        self, 
        video_path: str,
        target_fps: Optional[int] = None,
        max_frames: Optional[int] = None
    ) -> Tuple[List[LandmarkData], Dict]:
        """
        Extract landmarks from all frames of a video.
        
        Args:
            video_path: Path to the video file
            target_fps: Target FPS for resampling (None = use original)
            max_frames: Maximum number of frames to extract
            
        Returns:
            Tuple of (list of LandmarkData, metadata dict)
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened or is corrupted
        """
        # Validate path to prevent path traversal attacks
        video_path = os.path.abspath(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = None
        landmarks_list = []
        metadata = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Validate video properties
            if original_fps <= 0 or total_frames <= 0:
                raise ValueError(f"Invalid video properties: fps={original_fps}, frames={total_frames}")
            
            # Calculate frame sampling
            if target_fps and target_fps < original_fps:
                frame_skip = int(original_fps / target_fps)
            else:
                frame_skip = 1
                target_fps = original_fps
            
            frame_idx = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for target FPS
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue
                
                # Check max frames
                if max_frames and extracted_count >= max_frames:
                    break
                
                # Extract landmarks
                landmarks = self.extract_frame_landmarks(frame)
                landmarks_list.append(landmarks)
                
                frame_idx += 1
                extracted_count += 1
            
            metadata = {
                'video_path': video_path,
                'original_fps': original_fps,
                'target_fps': target_fps,
                'total_original_frames': total_frames,
                'extracted_frames': len(landmarks_list),
                'width': width,
                'height': height,
                'valid_frames': sum(1 for lm in landmarks_list if lm.is_valid),
            }
        
        except Exception as e:
            raise ValueError(f"Error processing video {video_path}: {str(e)}")
        finally:
            # Ensure video capture is always released
            if cap is not None:
                cap.release()
        
        return landmarks_list, metadata
    
    def landmarks_to_array(
        self, 
        landmarks_list: List[LandmarkData],
        fill_missing: bool = True
    ) -> np.ndarray:
        """
        Convert list of LandmarkData to numpy array.
        
        Args:
            landmarks_list: List of LandmarkData from video
            fill_missing: Whether to fill missing landmarks with zeros or interpolate
            
        Returns:
            Array of shape (num_frames, num_features)
            Features: [pose_landmarks, left_hand, right_hand] flattened
        """
        num_frames = len(landmarks_list)
        num_pose = len(self.pose_indices)
        num_hand = self.config.num_hand_landmarks
        
        # Calculate feature dimensions
        # Pose: num_pose * 3
        # Left hand: 21 * 3
        # Right hand: 21 * 3
        pose_dim = num_pose * 3
        hand_dim = num_hand * 3
        total_dim = pose_dim + 2 * hand_dim  # 144 for default config
        
        features = np.zeros((num_frames, total_dim), dtype=np.float32)
        
        for i, lm in enumerate(landmarks_list):
            idx = 0
            
            # Pose landmarks
            if lm.pose_landmarks is not None:
                features[i, idx:idx + pose_dim] = lm.pose_landmarks.flatten()
            idx += pose_dim
            
            # Left hand
            if lm.left_hand_landmarks is not None:
                features[i, idx:idx + hand_dim] = lm.left_hand_landmarks.flatten()
            idx += hand_dim
            
            # Right hand
            if lm.right_hand_landmarks is not None:
                features[i, idx:idx + hand_dim] = lm.right_hand_landmarks.flatten()
        
        if fill_missing:
            features = self._interpolate_missing(features, landmarks_list)
        
        return features
    
    def _interpolate_missing(
        self, 
        features: np.ndarray,
        landmarks_list: List[LandmarkData]
    ) -> np.ndarray:
        """
        Interpolate missing landmarks using neighboring frames.
        
        Args:
            features: Feature array with potential zero rows
            landmarks_list: Original landmark list for validity checking
            
        Returns:
            Interpolated feature array
        """
        num_frames = len(features)
        
        # Find valid frame indices
        valid_mask = np.array([lm.is_valid for lm in landmarks_list])
        
        if not np.any(valid_mask):
            # No valid frames, return zeros
            return features
        
        # For each invalid frame, interpolate from nearest valid frames
        for i in range(num_frames):
            if not valid_mask[i]:
                # Find nearest valid frames
                prev_valid = None
                next_valid = None
                
                for j in range(i - 1, -1, -1):
                    if valid_mask[j]:
                        prev_valid = j
                        break
                
                for j in range(i + 1, num_frames):
                    if valid_mask[j]:
                        next_valid = j
                        break
                
                # Interpolate
                if prev_valid is not None and next_valid is not None:
                    # Linear interpolation
                    alpha = (i - prev_valid) / (next_valid - prev_valid)
                    features[i] = (1 - alpha) * features[prev_valid] + alpha * features[next_valid]
                elif prev_valid is not None:
                    features[i] = features[prev_valid]
                elif next_valid is not None:
                    features[i] = features[next_valid]
        
        return features


def extract_features_from_video(
    video_path: str,
    config: Config,
    extractor: Optional[MediaPipeExtractor] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Extract normalized features from a video file.
    
    Args:
        video_path: Path to the video file
        config: Configuration object
        extractor: Optional pre-initialized extractor
        
    Returns:
        Tuple of (feature array, metadata)
    """
    if extractor is None:
        extractor = MediaPipeExtractor(config.mediapipe)
    
    # Extract landmarks
    landmarks_list, metadata = extractor.extract_video_landmarks(
        video_path,
        target_fps=config.preprocessing.target_fps,
        max_frames=config.preprocessing.max_video_length
    )
    
    # Convert to array
    features = extractor.landmarks_to_array(landmarks_list)
    
    return features, metadata


if __name__ == "__main__":
    # Test feature extraction
    config = get_config()
    
    print("Testing MediaPipe feature extraction...")
    print(f"Pose landmark indices: {config.mediapipe.pose_landmarks_indices}")
    print(f"Expected base features per frame: {config.mediapipe.base_features_per_frame}")
    
    # Initialize extractor
    extractor = MediaPipeExtractor(config.mediapipe)
    
    # Look for a test video
    test_video = None
    if os.path.exists(config.paths.videos_dir):
        for folder in os.listdir(config.paths.videos_dir):
            folder_path = os.path.join(config.paths.videos_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        test_video = os.path.join(folder_path, file)
                        break
            if test_video:
                break
    
    if test_video:
        print(f"\nTesting with video: {test_video}")
        features, metadata = extract_features_from_video(test_video, config, extractor)
        print(f"Extracted features shape: {features.shape}")
        print(f"Metadata: {metadata}")
    else:
        print("\nNo test video found. Please check dataset path.")
