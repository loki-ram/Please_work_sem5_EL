"""
End-to-end inference pipeline for sign language translation.
Takes a video input and produces English translation.
"""

import os
import sys
import argparse
import torch
import numpy as np
from typing import Dict, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from data.vocabulary import load_vocabularies
from data.feature_extraction import MediaPipeExtractor
from data.preprocessing import FeatureNormalizer
from models.stage1_model import VideoToGlossModel
from models.stage2_model import GlossToEnglishModel
from utils.training_utils import load_checkpoint


class SignLanguageTranslator:
    """
    End-to-end sign language translation pipeline.
    """
    
    def __init__(
        self,
        config=None,
        stage1_checkpoint: Optional[str] = None,
        stage2_checkpoint: Optional[str] = None,
        device: str = 'cuda'
    ):
        """
        Initialize the translator.
        
        Args:
            config: Configuration object (uses default if None)
            stage1_checkpoint: Path to Stage 1 checkpoint
            stage2_checkpoint: Path to Stage 2 checkpoint
            device: Device to use for inference
        """
        self.config = config or get_config()
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load vocabularies
        self.gloss_vocab, self.english_vocab = load_vocabularies(self.config)
        self.config.update_vocab_sizes(len(self.gloss_vocab), len(self.english_vocab))
        
        # Initialize feature extraction
        self.extractor = MediaPipeExtractor(self.config.mediapipe)
        self.normalizer = FeatureNormalizer(self.config)
        
        # Load models
        self._load_models(stage1_checkpoint, stage2_checkpoint)
    
    def _load_models(
        self,
        stage1_checkpoint: Optional[str],
        stage2_checkpoint: Optional[str]
    ):
        """Load Stage 1 and Stage 2 models."""
        # Stage 1 model
        self.stage1_model = VideoToGlossModel(
            self.config.stage1,
            len(self.gloss_vocab)
        )
        
        if stage1_checkpoint is None:
            stage1_checkpoint = os.path.join(
                self.config.paths.checkpoints_dir,
                'best_stage1_checkpoint.pt'
            )
        
        if os.path.exists(stage1_checkpoint):
            load_checkpoint(self.stage1_model, stage1_checkpoint, device=self.device)
            print(f"Loaded Stage 1 model from: {stage1_checkpoint}")
        else:
            print(f"Warning: Stage 1 checkpoint not found: {stage1_checkpoint}")
        
        self.stage1_model = self.stage1_model.to(self.device)
        self.stage1_model.eval()
        
        # Stage 2 model
        self.stage2_model = GlossToEnglishModel(
            self.config.stage2,
            len(self.gloss_vocab),
            len(self.english_vocab),
            gloss_pad_idx=self.gloss_vocab.pad_idx,
            english_pad_idx=self.english_vocab.pad_idx,
            english_sos_idx=self.english_vocab.sos_idx,
            english_eos_idx=self.english_vocab.eos_idx
        )
        
        if stage2_checkpoint is None:
            stage2_checkpoint = os.path.join(
                self.config.paths.checkpoints_dir,
                'best_stage2_checkpoint.pt'
            )
        
        if os.path.exists(stage2_checkpoint):
            load_checkpoint(self.stage2_model, stage2_checkpoint, device=self.device)
            print(f"Loaded Stage 2 model from: {stage2_checkpoint}")
        else:
            print(f"Warning: Stage 2 checkpoint not found: {stage2_checkpoint}")
        
        self.stage2_model = self.stage2_model.to(self.device)
        self.stage2_model.eval()
    
    def extract_features(self, video_path: str) -> np.ndarray:
        """
        Extract and preprocess features from a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Preprocessed feature array
        """
        # Extract landmarks
        landmarks_list, metadata = self.extractor.extract_video_landmarks(
            video_path,
            target_fps=self.config.preprocessing.target_fps,
            max_frames=self.config.preprocessing.max_video_length
        )
        
        # Convert to array
        features = self.extractor.landmarks_to_array(landmarks_list)
        
        # Normalize
        features = self.normalizer.normalize_features(features)
        
        # Add temporal derivatives
        features = self.normalizer.add_temporal_derivatives(features)
        
        return features
    
    def video_to_gloss(self, features: np.ndarray) -> Tuple[str, list]:
        """
        Stage 1: Convert video features to gloss sequence.
        
        Args:
            features: Preprocessed feature array
            
        Returns:
            Tuple of (gloss string, gloss token IDs)
        """
        with torch.no_grad():
            # Prepare input
            features_tensor = torch.tensor(features, dtype=torch.float32)
            features_tensor = features_tensor.unsqueeze(0).to(self.device)
            lengths = torch.tensor([len(features)]).to(self.device)
            
            # Decode
            decoded_seqs, decoded_lens = self.stage1_model.decode_greedy(
                features_tensor, lengths
            )
            
            # Convert to text
            gloss_ids = decoded_seqs[0]
            gloss_text = self.gloss_vocab.decode(gloss_ids, remove_special=True)
            
            return gloss_text, gloss_ids
    
    def gloss_to_english(self, gloss_text: str) -> Tuple[str, list]:
        """
        Stage 2: Translate gloss sequence to English.
        
        Args:
            gloss_text: Gloss sequence string
            
        Returns:
            Tuple of (English sentence, token IDs)
        """
        with torch.no_grad():
            # Encode gloss
            gloss_ids = self.gloss_vocab.encode(gloss_text, add_sos=True, add_eos=True)
            gloss_tensor = torch.tensor([gloss_ids], dtype=torch.long).to(self.device)
            gloss_lengths = torch.tensor([len(gloss_ids)]).to(self.device)
            
            # Decode
            decoded, _ = self.stage2_model.decode_greedy(gloss_tensor, gloss_lengths)
            
            # Convert to text
            english_ids = decoded[0].tolist()
            english_text = self.english_vocab.decode(english_ids, remove_special=True)
            
            return english_text, english_ids
    
    def translate(self, video_path: str) -> Dict[str, str]:
        """
        Full translation pipeline: Video -> Gloss -> English.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with translation results
        """
        print(f"Processing video: {video_path}")
        
        # Extract features
        print("  Extracting features...")
        features = self.extract_features(video_path)
        print(f"  Extracted {len(features)} frames, {features.shape[1]} features/frame")
        
        # Stage 1: Video to Gloss
        print("  Stage 1: Video -> Gloss...")
        gloss_text, gloss_ids = self.video_to_gloss(features)
        print(f"  Predicted gloss: {gloss_text}")
        
        # Stage 2: Gloss to English
        print("  Stage 2: Gloss -> English...")
        english_text, english_ids = self.gloss_to_english(gloss_text)
        print(f"  Translation: {english_text}")
        
        return {
            'video_path': video_path,
            'num_frames': len(features),
            'gloss': gloss_text,
            'translation': english_text,
            'gloss_ids': gloss_ids,
            'english_ids': english_ids,
        }
    
    def translate_batch(self, video_paths: list) -> list:
        """
        Translate multiple videos.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of translation result dictionaries
        """
        results = []
        for video_path in video_paths:
            try:
                result = self.translate(video_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results.append({
                    'video_path': video_path,
                    'error': str(e)
                })
        return results


def main(args):
    """Main inference function."""
    config = get_config()
    
    # Initialize translator
    translator = SignLanguageTranslator(
        config,
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        device=args.device
    )
    
    # Process videos
    if args.video:
        # Single video
        result = translator.translate(args.video)
        print("\n" + "=" * 60)
        print("Translation Result:")
        print("=" * 60)
        print(f"Video: {result['video_path']}")
        print(f"Frames: {result['num_frames']}")
        print(f"Gloss: {result['gloss']}")
        print(f"English: {result['translation']}")
        
    elif args.video_dir:
        # Directory of videos
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_paths = []
        
        for root, dirs, files in os.walk(args.video_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in video_extensions:
                    video_paths.append(os.path.join(root, file))
        
        print(f"Found {len(video_paths)} videos")
        
        results = translator.translate_batch(video_paths)
        
        # Save results
        import json
        output_path = os.path.join(config.paths.output_dir, 'translations.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    else:
        print("Please provide --video or --video_dir argument")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Translation Inference")
    parser.add_argument('--video', type=str, help='Path to a single video file')
    parser.add_argument('--video_dir', type=str, help='Path to directory of videos')
    parser.add_argument('--stage1_checkpoint', type=str, help='Path to Stage 1 checkpoint')
    parser.add_argument('--stage2_checkpoint', type=str, help='Path to Stage 2 checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    main(args)
