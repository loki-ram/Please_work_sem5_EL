"""
Centralized Configuration for Hybrid Sign-to-Text System.
All modules must read configuration exclusively from this file.
Optimized for NVIDIA A100 GPU training and Flutter mobile deployment.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

@dataclass
class PathConfig:
    """All path configurations - modify these for your environment."""
    
    # Video root directory (used to reconstruct paths from JSON)
    video_root: str = "/media/rvcse22/CSERV/kortex_sem5/Videos_Sentence_Level_rams"
    
    # Dataset split files (train.json, val.json, test.json)
    splits_dir: str = "/media/rvcse22/CSERV/kortex_sem5/ramita/kaggle2/Please_work_sem5_EL/outputs/splits"
    
    # Output directories (features, checkpoints, vocab, etc.)
    output_dir: str = "/media/rvcse22/CSERV/kortex_sem5/ramita/kaggle2/Please_work_sem5_EL/outputs"
    
    # Derived paths (set in __post_init__)
    features_dir: str = field(init=False)
    checkpoint_dir: str = field(init=False)
    export_dir: str = field(init=False)
    vocab_dir: str = field(init=False)
    logs_dir: str = field(init=False)
    
    def __post_init__(self):
        self.features_dir = os.path.join(self.output_dir, "features")
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.export_dir = os.path.join(self.output_dir, "exported_models")
        self.vocab_dir = os.path.join(self.output_dir, "vocab")
        self.logs_dir = os.path.join(self.output_dir, "logs")
    
    def create_directories(self):
        """Create all output directories."""
        dirs = [
            self.output_dir,
            self.features_dir,
            self.checkpoint_dir,
            self.export_dir,
            self.vocab_dir,
            self.logs_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    @property
    def gloss_vocab_path(self) -> str:
        return os.path.join(self.vocab_dir, "gloss_vocab.json")
    
    @property
    def text_vocab_path(self) -> str:
        return os.path.join(self.vocab_dir, "text_vocab.json")
    
    @property
    def train_split_path(self) -> str:
        return os.path.join(self.splits_dir, "train.json")
    
    @property
    def val_split_path(self) -> str:
        return os.path.join(self.splits_dir, "val.json")
    
    @property
    def test_split_path(self) -> str:
        return os.path.join(self.splits_dir, "test.json")


# ============================================================================
# VOCABULARY CONFIGURATION
# ============================================================================

@dataclass
class VocabConfig:
    """Vocabulary settings."""
    
    # Special tokens
    pad_token: str = "<PAD>"
    sos_token: str = "<SOS>"
    eos_token: str = "<EOS>"
    unk_token: str = "<UNK>"
    blank_token: str = "<BLANK>"  # For CTC
    
    # Minimum frequency to include in vocabulary
    min_freq: int = 1
    
    # Maximum vocabulary sizes (None for unlimited)
    max_gloss_vocab_size: Optional[int] = None
    max_text_vocab_size: Optional[int] = None


# ============================================================================
# MEDIAPIPE CONFIGURATION
# ============================================================================

@dataclass
class MediaPipeConfig:
    """MediaPipe landmark extraction settings."""
    
    # Model complexity (0, 1, or 2)
    pose_model_complexity: int = 1
    hand_model_complexity: int = 1
    
    # Detection thresholds
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Pose landmarks to extract: shoulders, elbows, wrists
    pose_landmark_indices: List[int] = field(
        default_factory=lambda: [11, 12, 13, 14, 15, 16]
    )
    
    # Hand landmarks (21 per hand)
    num_hand_landmarks: int = 21
    
    # Feature dimensions
    # 6 pose * 3 = 18, 2 hands * 21 * 3 = 126, total base = 144
    # With velocity and acceleration: 144 * 3 = 432
    base_features_per_frame: int = 144
    features_per_frame: int = 432


# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

@dataclass
class PreprocessingConfig:
    """Data preprocessing settings."""
    
    # Video processing
    target_fps: int = 25
    max_video_length: int = 150  # Max frames
    min_video_length: int = 10   # Min frames
    
    # Normalization
    normalize_to_shoulder: bool = True
    use_inter_shoulder_scaling: bool = True
    
    # Temporal derivatives
    use_velocity: bool = True
    use_acceleration: bool = True
    
    # Data augmentation (training only)
    augment_training: bool = True
    temporal_scale_range: Tuple[float, float] = (0.8, 1.2)
    spatial_noise_std: float = 0.01
    dropout_landmarks_prob: float = 0.1


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class EncoderConfig:
    """Shared encoder configuration (Conv1D + BiGRU)."""
    
    # Input features
    input_size: int = 432  # Features per frame (with derivatives)
    
    # Conv1D layers for temporal downsampling
    conv_channels: int = 256
    conv_kernel_size: int = 5
    conv_stride: int = 2
    num_conv_layers: int = 2
    
    # BiGRU layers (increased to 512 for stronger encoder signal)
    gru_hidden_size: int = 512
    gru_num_layers: int = 2
    gru_dropout: float = 0.3
    gru_bidirectional: bool = True
    
    # Encoder output projection (strengthens encoder -> attention signal)
    use_encoder_projection: bool = True
    encoder_projection_dim: int = 512  # Project to this dim before attention
    
    @property
    def encoder_output_size(self) -> int:
        """Output size from BiGRU (2x hidden if bidirectional)."""
        base_size = self.gru_hidden_size * (2 if self.gru_bidirectional else 1)
        return self.encoder_projection_dim if self.use_encoder_projection else base_size


@dataclass
class CTCConfig:
    """CTC head configuration."""
    
    # Blank token index (must be 0 for nn.CTCLoss)
    blank_idx: int = 0
    
    # CTC loss settings
    zero_infinity: bool = True


@dataclass
class DecoderConfig:
    """Attention-based text decoder configuration."""
    
    # Embedding
    embedding_dim: int = 256
    
    # GRU decoder
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.3
    
    # Bahdanau attention
    attention_dim: int = 128
    
    # Decoding
    max_decode_length: int = 50
    
    # Anti-shortcut: Early EOS penalty during training
    min_eos_step: int = 2  # Don't allow EOS before this step
    eos_penalty: float = 5.0  # Subtract from EOS logit during training
    
    # Anti-shortcut: Decoder input dropout (forces reliance on encoder)
    decoder_input_dropout: float = 0.3  # Increased from 0.2 for stronger grounding
    
    # Anti-shortcut: Freeze decoder embeddings for initial epochs
    freeze_embedding_epochs: int = 10  # Freeze for first N epochs, then unfreeze


@dataclass
class ModelConfig:
    """Complete model configuration."""
    
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    ctc: CTCConfig = field(default_factory=CTCConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    
    # Vocabulary sizes (set after building vocabularies)
    gloss_vocab_size: int = 100  # Placeholder
    text_vocab_size: int = 200   # Placeholder


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training settings optimized for A100 GPU."""
    
    # Epochs
    num_epochs: int = 100
    
    # Batch size (A100 can handle larger batches)
    batch_size: int = 32
    gradient_accumulation_steps: int = 2  # Effective batch = 64
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Optimizer: AdamW
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Gradient clipping
    max_grad_norm: float = 5.0
    
    # Learning rate scheduler
    lr_scheduler: str = "cosine"  # Options: "cosine", "step", "plateau"
    warmup_epochs: int = 5
    
    # Hybrid loss weights
    # α * CTC_loss + (1-α) * Attention_loss
    ctc_weight_start: float = 0.5  # α starts at 0.5
    ctc_weight_end: float = 0.2    # α decays to 0.2
    ctc_weight_decay_epochs: int = 50  # Epochs to decay over
    
    # Teacher forcing (MORE AGGRESSIVE DECAY for visual grounding)
    # Reaches ~0.5 by epoch 40, ~0.3 by epoch 60
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.2
    teacher_forcing_decay_epochs: int = 70
    
    # Attention regularization (entropy penalty for grounding)
    attention_entropy_weight: float = 0.01  # Penalize low-entropy (peaked) attention
    use_attention_entropy_reg: bool = True
    
    # Mixed precision (AMP) for A100
    use_amp: bool = True
    
    # Checkpointing
    save_every_epoch: bool = True
    early_stopping_patience: int = 50
    early_stopping_metric: str = "val_loss"
    
    # Logging
    log_interval: int = 10  # Log every N batches
    
    # Random seed
    seed: int = 42
    
    def get_ctc_weight(self, epoch: int) -> float:
        """Get CTC weight (α) for current epoch with linear decay."""
        if epoch >= self.ctc_weight_decay_epochs:
            return self.ctc_weight_end
        
        decay_ratio = epoch / self.ctc_weight_decay_epochs
        return self.ctc_weight_start - decay_ratio * (
            self.ctc_weight_start - self.ctc_weight_end
        )
    
    def get_teacher_forcing_ratio(self, epoch: int) -> float:
        """
        Get teacher forcing ratio with STEPPED schedule for harder grounding.
        Epochs 0-20:  1.0 (full teacher forcing)
        Epochs 21-40: 0.6
        Epochs 41-60: 0.4
        Epochs 61+:   0.25
        """
        if epoch <= 20:
            return 1.0
        elif epoch <= 40:
            return 0.6
        elif epoch <= 60:
            return 0.4
        else:
            return 0.25


# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

@dataclass
class ExportConfig:
    """Model export settings for Flutter deployment."""
    
    # Export formats
    export_onnx: bool = True
    export_tflite: bool = True
    
    # ONNX settings
    onnx_opset_version: int = 13
    
    # TFLite settings
    tflite_quantize: bool = True
    tflite_quantization_type: str = "float16"  # Options: "int8", "float16"
    
    # Fixed input shapes for export
    max_input_length: int = 150  # Max frames
    
    # Inference settings (no beam search for simplicity)
    use_greedy_decoding: bool = True


# ============================================================================
# MASTER CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Master configuration class."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Device
    device: str = "cuda"
    
    # Debug mode
    debug: bool = False
    
    def __post_init__(self):
        """Initialize derived settings."""
        self.paths.create_directories()
    
    def update_vocab_sizes(self, gloss_vocab_size: int, text_vocab_size: int):
        """Update vocabulary sizes after building vocabularies."""
        self.model.gloss_vocab_size = gloss_vocab_size
        self.model.text_vocab_size = text_vocab_size


def get_config() -> Config:
    """Get the configuration instance."""
    return Config()


# ============================================================================
# CLI Configuration Override
# ============================================================================

def get_config_from_args(args=None) -> Config:
    """Get configuration with optional CLI argument overrides."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Sign-to-Text Training")
    
    # Path overrides
    parser.add_argument("--video-root", type=str, help="Video root directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    
    # Training overrides
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    parsed_args = parser.parse_args(args)
    
    config = get_config()
    
    # Apply overrides
    if parsed_args.video_root:
        config.paths.video_root = parsed_args.video_root
    if parsed_args.output_dir:
        config.paths.output_dir = parsed_args.output_dir
        config.paths.__post_init__()
        config.paths.create_directories()
    if parsed_args.epochs:
        config.training.num_epochs = parsed_args.epochs
    if parsed_args.batch_size:
        config.training.batch_size = parsed_args.batch_size
    if parsed_args.lr:
        config.training.learning_rate = parsed_args.lr
    if parsed_args.debug:
        config.debug = True
    if parsed_args.device:
        config.device = parsed_args.device
    
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    
    print("=" * 60)
    print("Hybrid Sign-to-Text Configuration")
    print("=" * 60)
    
    print(f"\n[Paths]")
    print(f"  Video root: {config.paths.video_root}")
    print(f"  Splits dir: {config.paths.splits_dir}")
    print(f"  Output dir: {config.paths.output_dir}")
    print(f"  Checkpoint dir: {config.paths.checkpoint_dir}")
    
    print(f"\n[Model]")
    print(f"  Encoder input: {config.model.encoder.input_size}")
    print(f"  Conv channels: {config.model.encoder.conv_channels}")
    print(f"  GRU hidden: {config.model.encoder.gru_hidden_size}")
    print(f"  Encoder output: {config.model.encoder.encoder_output_size}")
    print(f"  Decoder hidden: {config.model.decoder.hidden_size}")
    
    print(f"\n[Training]")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  CTC weight: {config.training.ctc_weight_start} -> {config.training.ctc_weight_end}")
    print(f"  Teacher forcing: {config.training.teacher_forcing_start} -> {config.training.teacher_forcing_end}")
    print(f"  Use AMP: {config.training.use_amp}")
    
    # Test decay functions
    print(f"\n[Decay Examples]")
    for epoch in [0, 10, 25, 50]:
        ctc_w = config.training.get_ctc_weight(epoch)
        tf = config.training.get_teacher_forcing_ratio(epoch)
        print(f"  Epoch {epoch:3d}: CTC weight={ctc_w:.3f}, Teacher forcing={tf:.3f}")
    
    print("\n[OK] Configuration loaded successfully!")
