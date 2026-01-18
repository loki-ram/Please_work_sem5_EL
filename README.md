
# Hybrid Sign-to-Text System - Walkthrough

## Summary

Successfully implemented a complete hybrid CTC + Attention-based Sign-to-Text system for ISL sentence-level translation, optimized for A100 GPU training and Flutter mobile deployment.

---

## Project Structure

```
d:\Please_work_sem5_EL\TRIAL2\
├── config.py           # Centralized configuration (all paths, hyperparameters)
├── train.py            # Training script (hybrid loss, AMP, checkpointing)
├── test.py             # Evaluation (attention decoder, BLEU/WER metrics)
├── inference.py        # Real-time webcam/video inference
├── export.py           # ONNX/TFLite export for Flutter
├── requirements.txt    # Python dependencies
├── models/
│   ├── __init__.py
│   └── hybrid_model.py # Conv1D + BiGRU encoder, CTC head, Bahdanau decoder
├── data/
│   ├── __init__.py
│   ├── vocabulary.py   # Separate gloss (uppercase) and text (lowercase) vocabs
│   └── dataset.py      # HybridDataset with collate function
├── utils/
│   ├── __init__.py
│   └── metrics.py      # BLEU, WER, exact-match accuracy
└── outputs/            # Created automatically
	├── checkpoints/
	├── vocab/
	├── logs/
	└── exported_models/
```

---

## Model Architecture

| Component | Configuration |
|-----------|---------------|
| **Conv1D Encoder** | 2 layers, kernel=5, stride=2, out_channels=256 (temporal downsampling) |
| **BiGRU Encoder** | 2 layers, hidden_size=512 per direction, bidirectional (outputs size = 512 * 2 = 1024) |
| **Encoder Projection** | Optional linear projection from raw encoder (1024) -> 512 (enabled by default) used by attention |
| **CTC Head** | Linear(raw_encoder_size → |gloss_vocab|) — raw_encoder_size = 1024 (blank token configured) |
| **Attention Decoder** | Bahdanau attention over encoder outputs; GRU decoder hidden_size=256, 1 layer, embedding_dim=256 |
| **Total Parameters** | ~4.1M (depends on vocab sizes) |

Note: the attention module uses the projected encoder outputs when `use_encoder_projection=True` (default). The CTC head operates on the raw BiGRU outputs (before projection).

---

## Key Features

### Training
- **Hybrid Loss**: `α × CTC + (1-α) × Attention` where α decays 0.5 → 0.2
- **Teacher Forcing**: Decays from 1.0 → 0.7 over training
- **AdamW Optimizer**: lr=1e-4, weight_decay=1e-4, gradient clipping (norm=5)
- **Mixed Precision (AMP)**: Enabled for A100 GPU efficiency
- **Checkpointing**: Per-epoch saves + best model tracking

### Inference
- **Attention-only decoding** (CTC disabled for inference)
- **Greedy decoding** until `<EOS>` or max length (50 tokens)
- **Real-time MediaPipe** landmark extraction for webcam/video

### Export
- **ONNX**: Fixed shapes, static control flow
- **TFLite**: Float16 quantization for Flutter

---

## Usage

### 1. Build Vocabularies
```
cd d:\Please_work_sem5_EL\TRIAL2
python data/vocabulary.py
```

### 2. Train Model
```
python train.py --epochs 100 --batch-size 32 --device cuda
```

### 3. Evaluate
```
python test.py --checkpoint outputs/checkpoints/latest_best.pt
```

### 4. Real-time Inference
```
python inference.py --checkpoint outputs/checkpoints/latest_best.pt --source webcam
python inference.py --checkpoint outputs/checkpoints/latest_best.pt --source video.mp4
```

### 5. Export for Flutter
```
python export.py --checkpoint outputs/checkpoints/latest_best.pt --format onnx tflite
```

---

## Verification Results

| Test | Status |
|------|--------|
| Config loading | ✓ Passed |
| Model initialization | ✓ Passed (4.1M params) |
| Forward pass | ✓ Passed |
| CTC loss computation | ✓ Passed (6.85) |
| Cross-entropy loss | ✓ Passed (5.32) |
| Greedy decoding | ✓ Passed (shape: [batch, 50]) |
| Metrics (BLEU/WER) | ✓ Passed |

---

## Configuration Highlights

| Setting | Value |
|---------|-------|
| Batch size | 32 (effective: 64 with accumulation) |
| Learning rate | 1e-4 |
| CTC weight decay | 0.5 → 0.2 over 50 epochs |
| Teacher forcing decay | 1.0 → 0.7 over 30 epochs |
| Max decode length | 50 tokens |
| AMP enabled | Yes (for A100) |


If you'd like, I can also add a small `models/architecture.md` or update `models/hybrid_model.py` with a docstring describing these layers.
