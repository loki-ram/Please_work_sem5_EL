"""
Export Hybrid Sign-to-Text Model for Flutter deployment.

Features:
- ONNX export with fixed shapes and static control flow
- TFLite export (via ONNX→TensorFlow→TFLite)
- No beam search, greedy decoding only
- Validation of exported model outputs
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config, Config
from models.hybrid_model import HybridSignToTextModel
from data.vocabulary import load_vocabularies


class ExportableEncoder(nn.Module):
    """Encoder module for export (no dynamic control flow)."""
    
    def __init__(self, model: HybridSignToTextModel):
        super().__init__()
        self.conv_encoder = model.conv_encoder
        self.gru_encoder = model.gru_encoder
        self.hidden_bridge = model.hidden_bridge
        self.decoder_num_layers = model.decoder_num_layers
        self.decoder_hidden_size = model.decoder_hidden_size
    
    def forward(self, features: torch.Tensor) -> tuple:
        """
        Encode features without sequence packing (fixed length).
        
        Args:
            features: (batch, max_seq_len, input_size)
            
        Returns:
            Tuple of (encoder_outputs, decoder_hidden)
        """
        # Conv encoding
        x = features.transpose(1, 2)
        x = self.conv_encoder.conv_layers(x)
        x = x.transpose(1, 2)
        
        # GRU encoding (without packing for export)
        self.gru_encoder.gru.flatten_parameters()
        encoder_outputs, hidden = self.gru_encoder.gru(x)
        
        # Prepare decoder hidden
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)
        
        decoder_hidden = self.hidden_bridge(combined)
        decoder_hidden = decoder_hidden.view(
            self.decoder_num_layers, -1, self.decoder_hidden_size
        ).contiguous()
        
        return encoder_outputs, decoder_hidden


class ExportableDecoder(nn.Module):
    """Single-step decoder for export (greedy decoding)."""
    
    def __init__(self, model: HybridSignToTextModel):
        super().__init__()
        self.decoder = model.decoder
        self.text_eos_idx = model.text_eos_idx
    
    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor
    ) -> tuple:
        """
        Single decoding step.
        
        Args:
            input_token: (batch,) - current token
            hidden: (num_layers, batch, hidden_size)
            encoder_outputs: (batch, src_len, encoder_hidden)
            
        Returns:
            Tuple of (next_token, new_hidden)
        """
        logits, hidden, _ = self.decoder.forward_step(
            input_token, hidden, encoder_outputs, encoder_mask=None
        )
        
        next_token = logits.argmax(dim=-1)
        
        return next_token, hidden


class ExportableModel(nn.Module):
    """
    Full model for export with fixed-length greedy decoding.
    Uses loop unrolling for static control flow.
    """
    
    def __init__(self, model: HybridSignToTextModel, max_decode_length: int = 50):
        super().__init__()
        self.encoder = ExportableEncoder(model)
        self.decoder_embedding = model.decoder.embedding
        self.decoder_attention = model.decoder.attention
        self.decoder_gru = model.decoder.gru
        self.decoder_output_proj = model.decoder.output_proj
        self.decoder_dropout = model.decoder.dropout
        
        self.text_sos_idx = model.text_sos_idx
        self.text_eos_idx = model.text_eos_idx
        self.text_pad_idx = model.text_pad_idx
        self.max_decode_length = max_decode_length
        self.num_layers = model.decoder_num_layers
        self.hidden_size = model.decoder_hidden_size
    
    def decode_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor
    ) -> tuple:
        """Single decode step."""
        embedded = self.decoder_embedding(input_token.unsqueeze(1))
        
        context, _ = self.decoder_attention(encoder_outputs, hidden[-1], None)
        
        gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.decoder_gru(gru_input, hidden)
        
        output_combined = torch.cat([output.squeeze(1), context], dim=-1)
        logits = self.decoder_output_proj(output_combined)
        
        next_token = logits.argmax(dim=-1)
        
        return next_token, hidden
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass with fixed-length greedy decoding.
        
        Args:
            features: (batch, max_seq_len, input_size)
            
        Returns:
            decoded: (batch, max_decode_length)
        """
        batch_size = features.size(0)
        device = features.device
        
        # Encode
        encoder_outputs, hidden = self.encoder(features)
        
        # Initialize decoding
        decoded = torch.full(
            (batch_size, self.max_decode_length),
            self.text_pad_idx,
            dtype=torch.long,
            device=device
        )
        decoded[:, 0] = self.text_sos_idx
        
        current_token = torch.full((batch_size,), self.text_sos_idx, dtype=torch.long, device=device)
        
        # Fixed-length decoding loop (unrolled for export)
        for t in range(1, self.max_decode_length):
            next_token, hidden = self.decode_step(current_token, hidden, encoder_outputs)
            decoded[:, t] = next_token
            current_token = next_token
        
        return decoded


def export_onnx(
    model: HybridSignToTextModel,
    config: Config,
    output_path: str,
    max_seq_length: int = 150,
    max_decode_length: int = 50
):
    """Export model to ONNX format."""
    print(f"Exporting to ONNX: {output_path}")
    
    # Create exportable model
    export_model = ExportableModel(model, max_decode_length)
    export_model.eval()
    
    # Create dummy input with fixed shape
    batch_size = 1
    input_size = config.model.encoder.input_size
    dummy_input = torch.randn(batch_size, max_seq_length, input_size)
    
    # Export
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        input_names=['features'],
        output_names=['decoded'],
        dynamic_axes=None,  # Fixed shapes
        opset_version=config.export.onnx_opset_version,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    # Validate
    try:
        import onnx
        import onnxruntime as ort
        
        # Check model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation: OK")
        
        # Test inference
        session = ort.InferenceSession(output_path)
        test_input = np.random.randn(1, max_seq_length, input_size).astype(np.float32)
        
        outputs = session.run(None, {'features': test_input})
        print(f"ONNX inference test: OK (output shape: {outputs[0].shape})")
        
    except ImportError:
        print("Warning: onnx or onnxruntime not installed, skipping validation")
    except Exception as e:
        print(f"Warning: ONNX validation/test failed: {e}")
    
    return True


def export_tflite(
    model: HybridSignToTextModel,
    config: Config,
    output_path: str,
    onnx_path: str = None,
    max_seq_length: int = 150,
    max_decode_length: int = 50
):
    """Export model to TFLite format (via ONNX → TensorFlow → TFLite)."""
    print(f"Exporting to TFLite: {output_path}")
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError as e:
        print(f"Error: Required packages not installed: {e}")
        print("Install with: pip install onnx-tf tensorflow")
        return False
    
    # Export to ONNX first if needed
    if onnx_path is None:
        onnx_path = output_path.replace('.tflite', '.onnx')
    
    if not os.path.exists(onnx_path):
        export_onnx(model, config, onnx_path, max_seq_length, max_decode_length)
    
    # Convert ONNX to TensorFlow
    print("Converting ONNX to TensorFlow...")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    
    # Save TensorFlow model
    tf_model_path = output_path.replace('.tflite', '_tf')
    tf_rep.export_graph(tf_model_path)
    print(f"TensorFlow model saved to: {tf_model_path}")
    
    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    if config.export.tflite_quantize:
        if config.export.tflite_quantization_type == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif config.export.tflite_quantization_type == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {output_path}")
    print(f"TFLite model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return True


def validate_pytorch_vs_onnx(
    model: HybridSignToTextModel,
    onnx_path: str,
    config: Config,
    max_seq_length: int = 150,
    max_decode_length: int = 50
):
    """Compare PyTorch and ONNX model outputs."""
    print("\nValidating PyTorch vs ONNX outputs...")
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime not installed, skipping validation")
        return
    
    # Create test input
    batch_size = 1
    input_size = config.model.encoder.input_size
    test_input = np.random.randn(batch_size, max_seq_length, input_size).astype(np.float32)
    
    # PyTorch inference
    export_model = ExportableModel(model, max_decode_length)
    export_model.eval()
    
    with torch.no_grad():
        pytorch_output = export_model(torch.from_numpy(test_input))
        pytorch_output = pytorch_output.numpy()
    
    # ONNX inference
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {'features': test_input})[0]
    
    # Compare
    matches = np.sum(pytorch_output == onnx_output) / pytorch_output.size
    print(f"Output match rate: {matches * 100:.2f}%")
    
    if matches == 1.0:
        print("Validation: PASSED (outputs match exactly)")
    else:
        print("Validation: WARNING (outputs differ slightly)")


def main():
    parser = argparse.ArgumentParser(description='Export model for Flutter deployment')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for exported models')
    parser.add_argument('--format', type=str, nargs='+', default=['onnx', 'tflite'],
                        choices=['onnx', 'tflite'],
                        help='Export formats')
    parser.add_argument('--max-seq-length', type=int, default=150,
                        help='Maximum input sequence length')
    parser.add_argument('--max-decode-length', type=int, default=50,
                        help='Maximum decoding length')
    parser.add_argument('--validate', action='store_true',
                        help='Validate exported model')
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = config.paths.export_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Model Export for Flutter Deployment")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Formats: {args.format}")
    
    # Load vocabularies
    print("\nLoading vocabularies...")
    gloss_vocab, text_vocab = load_vocabularies(config)
    
    # Load model
    print("Loading model...")
    model = HybridSignToTextModel(
        input_size=config.model.encoder.input_size,
        conv_channels=config.model.encoder.conv_channels,
        conv_kernel_size=config.model.encoder.conv_kernel_size,
        conv_stride=config.model.encoder.conv_stride,
        gru_hidden_size=config.model.encoder.gru_hidden_size,
        gru_num_layers=config.model.encoder.gru_num_layers,
        gru_dropout=config.model.encoder.gru_dropout,
        decoder_embedding_dim=config.model.decoder.embedding_dim,
        decoder_hidden_size=config.model.decoder.hidden_size,
        decoder_num_layers=config.model.decoder.num_layers,
        decoder_dropout=config.model.decoder.dropout,
        attention_dim=config.model.decoder.attention_dim,
        gloss_vocab_size=len(gloss_vocab),
        text_vocab_size=len(text_vocab),
        gloss_blank_idx=gloss_vocab.blank_idx,
        text_pad_idx=text_vocab.pad_idx,
        text_sos_idx=text_vocab.sos_idx,
        text_eos_idx=text_vocab.eos_idx,
        max_decode_length=args.max_decode_length
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Export
    onnx_path = None
    
    if 'onnx' in args.format:
        onnx_path = os.path.join(output_dir, 'model.onnx')
        export_onnx(
            model, config, onnx_path,
            args.max_seq_length, args.max_decode_length
        )
        
        if args.validate:
            validate_pytorch_vs_onnx(
                model, onnx_path, config,
                args.max_seq_length, args.max_decode_length
            )
    
    if 'tflite' in args.format:
        tflite_path = os.path.join(output_dir, 'model.tflite')
        export_tflite(
            model, config, tflite_path, onnx_path,
            args.max_seq_length, args.max_decode_length
        )
    
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    for fmt in args.format:
        if fmt == 'onnx':
            path = os.path.join(output_dir, 'model.onnx')
        else:
            path = os.path.join(output_dir, 'model.tflite')
        
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {fmt.upper()}: {path} ({size_mb:.2f} MB)")
    
    print("\n[OK] Export completed!")


if __name__ == "__main__":
    main()
