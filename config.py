"""
Configuration file for ISL-CSLTR Two-Stage Sign Language Translation System.
Contains all hyperparameters for data processing, model architecture, and training.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PathConfig:
    """Paths configuration."""
    # Dataset paths
    dataset_root: str = r"D:\Please_work_sem5_EL"
    videos_dir: str = field(init=False)
    gloss_csv: str = field(init=False)
    
    # Output paths
    output_dir: str = r"D:\Please_work_sem5_EL\outputs"
    features_dir: str = field(init=False)
    checkpoints_dir: str = field(init=False)
    logs_dir: str = field(init=False)
    exported_models_dir: str = field(init=False)
    
    # Split files
    splits_dir: str = field(init=False)
    
    def __post_init__(self):
        self.videos_dir = os.path.join(self.dataset_root, "videos")
        self.gloss_csv = os.path.join(self.dataset_root, "ISL Corpus sign glosses.csv")
        
        self.features_dir = os.path.join(self.output_dir, "features")
        self.checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.exported_models_dir = os.path.join(self.output_dir, "exported_models")
        self.splits_dir = os.path.join(self.output_dir, "splits")


@dataclass
class MediaPipeConfig:
    """MediaPipe landmark extraction configuration."""
    # Model complexity (0, 1, or 2) - lower is faster
    pose_model_complexity: int = 1
    hand_model_complexity: int = 1
    
    # Detection confidence thresholds
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Landmark indices to extract
    # Pose landmarks: shoulders (11, 12), elbows (13, 14), wrists (15, 16)
    pose_landmarks_indices: List[int] = field(default_factory=lambda: [11, 12, 13, 14, 15, 16])
    
    # Hand landmarks: all 21 keypoints per hand (0-20)
    num_hand_landmarks: int = 21
    
    # Total features per frame:
    # 6 pose landmarks * 3 (x, y, z) = 18
    # 2 hands * 21 landmarks * 3 (x, y, z) = 126
    # Total = 144 base features
    # With velocity and acceleration: 144 * 3 = 432 features
    base_features_per_frame: int = 144
    features_per_frame: int = 432  # With temporal derivatives


@dataclass
class PreprocessingConfig:
    """Data preprocessing configuration."""
    # Video processing
    target_fps: int = 25  # Target frames per second (reduced for memory)
    max_video_length: int = 150  # Maximum frames (6 seconds at 25fps) - reduced for 8GB VRAM
    min_video_length: int = 10  # Minimum frames
    
    # Normalization
    normalize_to_shoulder: bool = True  # Use shoulder-centered coordinates
    use_inter_shoulder_scaling: bool = True  # Scale by inter-shoulder distance
    
    # Temporal derivatives
    use_velocity: bool = True  # First-order derivative
    use_acceleration: bool = True  # Second-order derivative
    
    # Data augmentation
    augment_training: bool = True
    temporal_scale_range: Tuple[float, float] = (0.8, 1.2)
    spatial_noise_std: float = 0.01
    dropout_landmarks_prob: float = 0.1


@dataclass
class DatasetConfig:
    """Dataset splitting and loading configuration."""
    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Data loading - optimized for 8GB VRAM
    batch_size: int = 4  # Reduced for 8GB VRAM (use gradient accumulation)
    num_workers: int = 2  # Reduced to save system memory
    pin_memory: bool = True
    
    # Gradient accumulation to simulate larger batch
    gradient_accumulation_steps: int = 4  # Effective batch size = 4 * 4 = 16
    
    # Vocabulary
    gloss_vocab_min_freq: int = 1  # Minimum frequency to include in vocab
    max_gloss_length: int = 30  # Maximum gloss sequence length
    max_sentence_length: int = 50  # Maximum English sentence length
    
    # Special tokens
    pad_token: str = "<PAD>"
    sos_token: str = "<SOS>"
    eos_token: str = "<EOS>"
    unk_token: str = "<UNK>"
    blank_token: str = "<BLANK>"  # For CTC


@dataclass
class TCNConfig:
    """Temporal Convolutional Network configuration - optimized for 8GB VRAM."""
    input_size: int = 432  # Features per frame (with derivatives)
    hidden_size: int = 192  # Reduced from 256 for 8GB VRAM
    num_layers: int = 4
    kernel_size: int = 3
    dilation_rates: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dropout: float = 0.3
    use_batch_norm: bool = True
    use_layer_norm: bool = False


@dataclass
class BiGRUConfig:
    """Bidirectional GRU configuration - optimized for 8GB VRAM."""
    input_size: int = 192  # From TCN output (reduced)
    hidden_size: int = 192  # Reduced from 256 for 8GB VRAM
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True


@dataclass
class Stage1Config:
    """Stage 1: Video-to-Gloss model configuration."""
    # Sub-model configs
    tcn: TCNConfig = field(default_factory=TCNConfig)
    bigru: BiGRUConfig = field(default_factory=BiGRUConfig)
    
    # Output projection
    gloss_vocab_size: int = 500  # Will be updated based on dataset
    
    # CTC configuration
    ctc_blank_id: int = 0
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"  # Options: "cosine", "step", "plateau"
    warmup_epochs: int = 5
    num_epochs: int = 100
    gradient_clip_norm: float = 5.0
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_wer"  # Word Error Rate


@dataclass
class Seq2SeqEncoderConfig:
    """Stage 2 Encoder configuration - optimized for 8GB VRAM."""
    embedding_dim: int = 128
    hidden_size: int = 192  # Reduced from 256
    num_layers: int = 1
    dropout: float = 0.2


@dataclass
class Seq2SeqDecoderConfig:
    """Stage 2 Decoder configuration - optimized for 8GB VRAM."""
    embedding_dim: int = 128
    hidden_size: int = 192  # Reduced from 256
    num_layers: int = 1
    dropout: float = 0.2
    use_attention: bool = True
    attention_dim: int = 96  # Reduced from 128


@dataclass
class Stage2Config:
    """Stage 2: Gloss-to-English model configuration."""
    # Sub-model configs
    encoder: Seq2SeqEncoderConfig = field(default_factory=Seq2SeqEncoderConfig)
    decoder: Seq2SeqDecoderConfig = field(default_factory=Seq2SeqDecoderConfig)
    
    # Vocabulary sizes (will be updated based on dataset)
    gloss_vocab_size: int = 500
    english_vocab_size: int = 2000
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: str = "plateau"
    num_epochs: int = 100
    gradient_clip_norm: float = 5.0
    teacher_forcing_ratio: float = 0.5  # Probability of using teacher forcing
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_bleu"
    
    # Inference
    max_decode_length: int = 50
    beam_size: int = 5


@dataclass
class ExportConfig:
    """Model export and quantization configuration."""
    # Export formats
    export_tflite: bool = True
    export_onnx: bool = True
    
    # Quantization
    quantize: bool = True
    quantization_type: str = "int8"  # Options: "int8", "float16", "dynamic"
    
    # ONNX settings
    onnx_opset_version: int = 13
    
    # TFLite settings
    tflite_optimize: bool = True


@dataclass
class Config:
    """Master configuration class."""
    # Sub-configurations
    paths: PathConfig = field(default_factory=PathConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # General settings
    device: str = "cuda"  # Options: "cuda", "cpu", "mps"
    mixed_precision: bool = True  # Use FP16 to save VRAM
    debug_mode: bool = False
    log_interval: int = 10  # Log every N batches
    save_interval: int = 5  # Save checkpoint every N epochs
    
    # Memory optimization for 8GB VRAM
    memory_efficient: bool = True
    empty_cache_freq: int = 50  # Clear CUDA cache every N batches
    gradient_checkpointing: bool = True  # Trade compute for memory
    
    def create_directories(self):
        """Create all necessary directories."""
        dirs = [
            self.paths.output_dir,
            self.paths.features_dir,
            self.paths.checkpoints_dir,
            self.paths.logs_dir,
            self.paths.exported_models_dir,
            self.paths.splits_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def update_vocab_sizes(self, gloss_vocab_size: int, english_vocab_size: int):
        """Update vocabulary sizes after building vocabularies."""
        self.stage1.gloss_vocab_size = gloss_vocab_size
        self.stage2.gloss_vocab_size = gloss_vocab_size
        self.stage2.english_vocab_size = english_vocab_size


# Global configuration instance
def get_config() -> Config:
    """Get the configuration instance."""
    config = Config()
    config.create_directories()
    return config


if __name__ == "__main__":
    # Print configuration for verification
    cfg = get_config()
    print("Configuration loaded successfully!")
    print(f"Dataset root: {cfg.paths.dataset_root}")
    print(f"Output directory: {cfg.paths.output_dir}")
    print(f"Features per frame: {cfg.mediapipe.features_per_frame}")
    print(f"Batch size: {cfg.dataset.batch_size}")
    print(f"Stage 1 epochs: {cfg.stage1.num_epochs}")
    print(f"Stage 2 epochs: {cfg.stage2.num_epochs}")
