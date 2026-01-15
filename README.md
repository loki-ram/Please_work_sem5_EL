# ISL-CSLTR Two-Stage Sign Language Translation System

A two-stage continuous sign language translation system for the ISL-CSLTR (Indian Sign Language - Continuous Sign Language Translation and Recognition) dataset.

**Optimized for 8GB VRAM GPUs** with gradient accumulation, mixed precision training, and gradient checkpointing.

## Architecture Overview

### Stage 1: Video-to-Gloss Recognition
- **MediaPipe** for lightweight pose/hand landmark extraction
- **Shoulder-centered normalization** with inter-shoulder scaling
- **Temporal derivatives** (velocity and acceleration) for motion dynamics
- **TCN (Temporal Convolutional Network)** with dilated convolutions (dilation rates: 1, 2, 4, 8)
- **BiGRU** (Bidirectional GRU) for long-range temporal dependencies
- **CTC Loss** for alignment-free sequence prediction

### Stage 2: Gloss-to-English Translation
- **Sequence-to-Sequence** architecture with single-layer GRU
- **Bahdanau Attention** mechanism
- **Teacher Forcing** during training

### Mobile Deployment
- **ONNX** export with quantization
- **TensorFlow Lite** export with int8 quantization

## 8GB VRAM GPU Optimizations

This project is configured for GPUs with 8GB VRAM (e.g., RTX 3060, RTX 3070, etc.):

| Optimization | Setting | Description |
|--------------|---------|-------------|
| Batch Size | 4 | Reduced from 16 |
| Gradient Accumulation | 4 steps | Effective batch = 16 |
| Hidden Size (TCN) | 192 | Reduced from 256 |
| Hidden Size (BiGRU) | 192 | Reduced from 256 |
| Hidden Size (Seq2Seq) | 192 | Reduced from 256 |
| Mixed Precision | Enabled | FP16 training |
| Gradient Checkpointing | Optional | Trade compute for memory |
| Max Video Length | 150 frames | Reduced from 300 |
| CUDA Cache Clearing | Every 50 batches | Prevents OOM |

### Memory Adjustment Tips

If you still encounter OOM (Out of Memory) errors:

1. **Reduce batch size** in `config.py`:
   ```python
   batch_size: int = 2  # Reduce to 2
   gradient_accumulation_steps: int = 8  # Increase to 8
   ```

2. **Enable gradient checkpointing** in `config.py`:
   ```python
   gradient_checkpointing: bool = True
   ```

3. **Reduce max video length**:
   ```python
   max_video_length: int = 100  # Reduce further
   ```

4. **Reduce model sizes** (will affect accuracy):
   ```python
   hidden_size: int = 128  # In TCN, BiGRU, and Seq2Seq configs
   ```

## Project Structure

```
Please_work_sem5_EL/
├── config.py                 # All hyperparameters and configuration
├── train_stage1.py           # Stage 1 training script
├── train_stage2.py           # Stage 2 training script
├── inference.py              # End-to-end inference pipeline
├── export_models.py          # Model export for mobile deployment
│
├── data/
│   ├── __init__.py
│   ├── split_dataset.py      # Train/val/test splitting
│   ├── feature_extraction.py # MediaPipe landmark extraction
│   ├── preprocessing.py      # Normalization and augmentation
│   ├── vocabulary.py         # Gloss and English vocabularies
│   └── dataset.py            # PyTorch Dataset classes
│
├── models/
│   ├── __init__.py
│   ├── tcn.py                # Temporal Convolutional Network
│   ├── stage1_model.py       # TCN-BiGRU with CTC
│   └── stage2_model.py       # Seq2Seq with attention
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # WER, BLEU, ROUGE-L metrics
│   └── training_utils.py     # Checkpointing, logging, schedulers
│
└── outputs/                  # Created automatically
    ├── features/             # Extracted features
    ├── checkpoints/          # Model checkpoints
    ├── logs/                 # Training logs
    ├── splits/               # Dataset splits
    └── exported_models/      # ONNX/TFLite models
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install mediapipe opencv-python
pip install numpy pandas tqdm

# Optional: For model export
pip install onnx onnxruntime
pip install tensorflow onnx-tf
```

## Dataset Setup

1. Download the ISL-CSLTR dataset
2. Organize the dataset:
   ```
   ISL-CSLTR/
   ├── videos/
   │   ├── sentence_001/
   │   │   ├── signer1.mp4
   │   │   ├── signer2.mp4
   │   │   └── ...
   │   ├── sentence_002/
   │   └── ...
   └── glosses.csv
   ```

3. Update `config.py` with your dataset path:
   ```python
   dataset_root: str = r"D:\Please_work_sem5_EL\ISL-CSLTR"
   ```

4. The `glosses.csv` should have columns:
   - `folder_name`: Name of video folder
   - `gloss`: ISL gloss sequence (space-separated)
   - `sentence`: English sentence

## Usage

### Step 1: Split Dataset
```bash
python data/split_dataset.py
```

### Step 2: Preprocess Videos (Extract Features)
```bash
python data/preprocessing.py
```

### Step 3: Build Vocabularies
```bash
python data/vocabulary.py
```

### Step 4: Train Stage 1 (Video → Gloss)
```bash
python train_stage1.py --device cuda
```

### Step 5: Train Stage 2 (Gloss → English)
```bash
python train_stage2.py --device cuda
```

### Step 6: Run Inference
```bash
# Single video
python inference.py --video path/to/video.mp4

# Directory of videos
python inference.py --video_dir path/to/videos/
```

### Step 7: Export for Mobile
```bash
python export_models.py --stage all
```

## Configuration

All hyperparameters are in `config.py`. Key settings:

### Data Processing
- `target_fps`: 30
- `max_video_length`: 300 frames (10 seconds)
- `normalize_to_shoulder`: True
- `use_velocity`: True
- `use_acceleration`: True

### Stage 1 (TCN-BiGRU)
- TCN: 4 layers, hidden_size=256, dilation=[1,2,4,8]
- BiGRU: 2 layers, hidden_size=256, bidirectional=True
- Learning rate: 1e-3
- Epochs: 100

### Stage 2 (Seq2Seq)
- Encoder: 1-layer GRU, hidden_size=256
- Decoder: 1-layer GRU with attention
- Teacher forcing ratio: 0.5
- Learning rate: 1e-3
- Epochs: 100

### Export
- ONNX opset version: 13
- Quantization: int8

## Metrics

### Stage 1 (Video-to-Gloss)
- **WER (Word Error Rate)**: Lower is better

### Stage 2 (Gloss-to-English)
- **BLEU**: Higher is better (0-1)
- **ROUGE-L F1**: Higher is better (0-1)

## Model Architecture Details

### Feature Extraction (144 base features per frame)
- **Pose landmarks** (6 points × 3 coords = 18): shoulders, elbows, wrists
- **Left hand** (21 points × 3 coords = 63)
- **Right hand** (21 points × 3 coords = 63)
- **With derivatives**: 144 × 3 = 432 features per frame

### TCN Encoder
```
Input: (batch, seq_len, 432)
  ↓ Linear projection
(batch, seq_len, 256)
  ↓ TCN Block (dilation=1)
  ↓ TCN Block (dilation=2)
  ↓ TCN Block (dilation=4)
  ↓ TCN Block (dilation=8)
Output: (batch, seq_len, 256)
```

### BiGRU Encoder
```
Input: (batch, seq_len, 256)
  ↓ BiGRU (2 layers)
Output: (batch, seq_len, 512)
```

### CTC Output
```
Input: (batch, seq_len, 512)
  ↓ Linear projection
Output: (seq_len, batch, vocab_size)  # Log probabilities
```

## Tips for Training

1. **GPU Memory**: Reduce batch_size if OOM
2. **Mixed Precision**: Enabled by default, disable if issues occur
3. **Early Stopping**: Patience=15 epochs
4. **Data Augmentation**: Only applied to training data
5. **Resume Training**: Use `--resume checkpoint.pt`

## License

This project is for educational purposes.

## Acknowledgments

- ISL-CSLTR Dataset
- MediaPipe by Google
- PyTorch
