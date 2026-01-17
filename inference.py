"""
Real-time inference for Hybrid Sign-to-Text Model.

Features:
- Capture webcam or video input
- Extract MediaPipe landmarks on-the-fly
- Apply same preprocessing as training
- Run encoder + attention decoder (CTC disabled)
- Greedy decoding until <EOS> or max length
- Display predicted English sentence
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config, Config
from models.hybrid_model import HybridSignToTextModel
from data.vocabulary import load_vocabularies

# Try to import mediapipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not installed. Install with: pip install mediapipe")


class MediaPipeLandmarkExtractor:
    """Extract pose and hand landmarks using MediaPipe."""
    
    def __init__(self, config: Config):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is required for inference")
        
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.mediapipe.pose_model_complexity,
            min_detection_confidence=config.mediapipe.min_detection_confidence,
            min_tracking_confidence=config.mediapipe.min_tracking_confidence
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=config.mediapipe.hand_model_complexity,
            min_detection_confidence=config.mediapipe.min_detection_confidence,
            min_tracking_confidence=config.mediapipe.min_tracking_confidence
        )
        
        self.pose_indices = config.mediapipe.pose_landmark_indices
        self.num_hand_landmarks = config.mediapipe.num_hand_landmarks
    
    def extract_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract landmarks from a single frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract pose landmarks
        pose_results = self.pose.process(rgb_frame)
        
        # Extract hand landmarks
        hand_results = self.hands.process(rgb_frame)
        
        # Initialize feature vector
        # 6 pose * 3 + 21 left_hand * 3 + 21 right_hand * 3 = 144
        features = np.zeros(144, dtype=np.float32)
        
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
    
    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()
        self.hands.close()


class FeaturePreprocessor:
    """Apply preprocessing to extracted landmarks."""
    
    def __init__(self, config: Config):
        self.config = config
        self.left_shoulder_idx = 0 * 3
        self.right_shoulder_idx = 1 * 3
        self.num_pose = len(config.mediapipe.pose_landmark_indices)
        self.num_hand = config.mediapipe.num_hand_landmarks
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply shoulder-center normalization and scaling."""
        normalized = frame.copy()
        
        # Get shoulder positions
        left_shoulder = frame[self.left_shoulder_idx:self.left_shoulder_idx+3]
        right_shoulder = frame[self.right_shoulder_idx:self.right_shoulder_idx+3]
        
        # Center point
        if self.config.preprocessing.normalize_to_shoulder:
            center = (left_shoulder + right_shoulder) / 2
        else:
            center = np.zeros(3)
        
        # Scale factor
        if self.config.preprocessing.use_inter_shoulder_scaling:
            scale = np.linalg.norm(left_shoulder - right_shoulder)
            if scale < 1e-6:
                scale = 1.0
        else:
            scale = 1.0
        
        # Apply normalization to all landmarks
        num_landmarks = len(frame) // 3
        for i in range(num_landmarks):
            idx = i * 3
            if np.any(frame[idx:idx+3] != 0):
                normalized[idx:idx+3] = (frame[idx:idx+3] - center) / scale
        
        return normalized
    
    def add_temporal_derivatives(self, features: np.ndarray) -> np.ndarray:
        """Add velocity and acceleration features."""
        num_frames, base_dim = features.shape
        
        output_dim = base_dim
        if self.config.preprocessing.use_velocity:
            output_dim += base_dim
        if self.config.preprocessing.use_acceleration:
            output_dim += base_dim
        
        output = np.zeros((num_frames, output_dim), dtype=np.float32)
        output[:, :base_dim] = features
        
        idx = base_dim
        
        if self.config.preprocessing.use_velocity:
            velocity = np.zeros_like(features)
            velocity[1:] = features[1:] - features[:-1]
            output[:, idx:idx+base_dim] = velocity
            idx += base_dim
        
        if self.config.preprocessing.use_acceleration:
            acceleration = np.zeros_like(features)
            if num_frames > 2:
                acceleration[2:] = features[2:] - 2*features[1:-1] + features[:-2]
            output[:, idx:idx+base_dim] = acceleration
        
        return output
    
    def process(self, frames: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline."""
        # Normalize each frame
        normalized = np.array([self.normalize_frame(f) for f in frames])
        
        # Add temporal derivatives
        features = self.add_temporal_derivatives(normalized)
        
        return features


class SignToTextInference:
    """Real-time sign language to text inference."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Config,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load vocabularies
        print("Loading vocabularies...")
        self.gloss_vocab, self.text_vocab = load_vocabularies(config)
        print(f"Text vocabulary: {len(self.text_vocab)} tokens")
        
        # Load model
        print("Loading model...")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
        # Initialize extractors
        if MEDIAPIPE_AVAILABLE:
            self.extractor = MediaPipeLandmarkExtractor(config)
        else:
            self.extractor = None
        
        self.preprocessor = FeaturePreprocessor(config)
        
        # Frame buffer for sequence accumulation
        self.frame_buffer = []
        self.min_frames = config.preprocessing.min_video_length
        self.max_frames = config.preprocessing.max_video_length
    
    def _load_model(self, checkpoint_path: str) -> HybridSignToTextModel:
        """Load model from checkpoint."""
        model = HybridSignToTextModel(
            input_size=self.config.model.encoder.input_size,
            conv_channels=self.config.model.encoder.conv_channels,
            conv_kernel_size=self.config.model.encoder.conv_kernel_size,
            conv_stride=self.config.model.encoder.conv_stride,
            gru_hidden_size=self.config.model.encoder.gru_hidden_size,
            gru_num_layers=self.config.model.encoder.gru_num_layers,
            gru_dropout=self.config.model.encoder.gru_dropout,
            decoder_embedding_dim=self.config.model.decoder.embedding_dim,
            decoder_hidden_size=self.config.model.decoder.hidden_size,
            decoder_num_layers=self.config.model.decoder.num_layers,
            decoder_dropout=self.config.model.decoder.dropout,
            attention_dim=self.config.model.decoder.attention_dim,
            gloss_vocab_size=len(self.gloss_vocab),
            text_vocab_size=len(self.text_vocab),
            gloss_blank_idx=self.gloss_vocab.blank_idx,
            text_pad_idx=self.text_vocab.pad_idx,
            text_sos_idx=self.text_vocab.sos_idx,
            text_eos_idx=self.text_vocab.eos_idx,
            max_decode_length=self.config.model.decoder.max_decode_length
        )
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model.to(self.device)
    
    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Add a frame to the buffer.
        
        Returns:
            True if buffer has enough frames for inference
        """
        if self.extractor is None:
            raise RuntimeError("MediaPipe extractor not available")
        
        # Extract landmarks
        landmarks = self.extractor.extract_frame(frame)
        self.frame_buffer.append(landmarks)
        
        # Limit buffer size
        if len(self.frame_buffer) > self.max_frames:
            self.frame_buffer.pop(0)
        
        return len(self.frame_buffer) >= self.min_frames
    
    def predict(self) -> str:
        """
        Run inference on buffered frames.
        
        Returns:
            Predicted English sentence
        """
        if len(self.frame_buffer) < self.min_frames:
            return ""
        
        # Preprocess
        frames = np.array(self.frame_buffer)
        features = self.preprocessor.process(frames)
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        lengths_tensor = torch.tensor([len(features)], dtype=torch.long)
        
        features_tensor = features_tensor.to(self.device)
        lengths_tensor = lengths_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            decoded = self.model.decode_greedy(features_tensor, lengths_tensor)
        
        # Decode to text
        prediction = self.text_vocab.decode(decoded[0].tolist())
        
        return prediction
    
    def reset(self):
        """Clear frame buffer."""
        self.frame_buffer = []
    
    def close(self):
        """Release resources."""
        if self.extractor:
            self.extractor.close()


def run_webcam_inference(inference: SignToTextInference):
    """Run real-time inference on webcam."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\n" + "=" * 60)
    print("Real-time Sign Language Translation")
    print("=" * 60)
    print("Press 'SPACE' to start/stop recording a sign")
    print("Press 'R' to reset buffer")
    print("Press 'Q' to quit")
    print("=" * 60)
    
    recording = False
    prediction = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame
        display_frame = frame.copy()
        
        # Recording indicator
        if recording:
            cv2.putText(display_frame, "RECORDING", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Frames: {len(inference.frame_buffer)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add frame to buffer
            inference.add_frame(frame)
        
        # Display prediction
        if prediction:
            cv2.putText(display_frame, f"Prediction: {prediction}",
                       (10, display_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow('Sign Language Translation', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space - toggle recording
            if recording:
                # Stop recording and predict
                recording = False
                if len(inference.frame_buffer) >= inference.min_frames:
                    print(f"\nProcessing {len(inference.frame_buffer)} frames...")
                    prediction = inference.predict()
                    print(f"Prediction: {prediction}")
                else:
                    print(f"\nNot enough frames ({len(inference.frame_buffer)}/{inference.min_frames})")
            else:
                # Start recording
                inference.reset()
                recording = True
                prediction = ""
                print("\nStarted recording...")
        
        elif key == ord('r'):  # Reset
            inference.reset()
            prediction = ""
            recording = False
            print("\nBuffer reset")
        
        elif key == ord('q'):  # Quit
            break
    
    cap.release()
    cv2.destroyAllWindows()


def run_video_inference(inference: SignToTextInference, video_path: str):
    """Run inference on a video file."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    print(f"\nProcessing video: {video_path}")
    
    # Read all frames
    inference.reset()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        inference.add_frame(frame)
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    
    print(f"\nTotal frames: {len(inference.frame_buffer)}")
    
    if len(inference.frame_buffer) >= inference.min_frames:
        prediction = inference.predict()
        print(f"\n{'='*60}")
        print(f"Predicted Translation: {prediction}")
        print(f"{'='*60}")
    else:
        print(f"\nError: Not enough frames ({len(inference.frame_buffer)}/{inference.min_frames})")


def main():
    parser = argparse.ArgumentParser(description='Real-time Sign Language Translation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--source', type=str, default='webcam',
                        help='Video source: "webcam" or path to video file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    # Create inference engine
    print("=" * 60)
    print("Initializing Sign-to-Text Inference")
    print("=" * 60)
    
    inference = SignToTextInference(
        args.checkpoint,
        config,
        device=args.device
    )
    
    try:
        if args.source == 'webcam':
            run_webcam_inference(inference)
        else:
            run_video_inference(inference, args.source)
    finally:
        inference.close()
    
    print("\n[OK] Inference completed!")


if __name__ == "__main__":
    main()
