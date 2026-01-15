"""
Model export utilities for mobile deployment.
Exports trained models to TensorFlow Lite and ONNX formats with quantization.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from data.vocabulary import load_vocabularies
from models.stage1_model import VideoToGlossModel
from models.stage2_model import GlossToEnglishModel
from utils.training_utils import load_checkpoint


class Stage1ExportWrapper(nn.Module):
    """
    Wrapper for Stage 1 model to simplify export.
    Removes CTC loss and returns raw log probabilities.
    """
    
    def __init__(self, model: VideoToGlossModel):
        super().__init__()
        self.tcn_encoder = model.tcn_encoder
        self.bigru_encoder = model.bigru_encoder
        self.output_projection = model.output_projection
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            features: (batch, seq_len, feature_dim)
            
        Returns:
            Log probabilities (batch, seq_len, vocab_size)
        """
        # TCN encoding
        tcn_output = self.tcn_encoder(features)
        
        # BiGRU encoding (without packing for export)
        bigru_output, _ = self.bigru_encoder.gru(tcn_output)
        
        # Project to vocabulary
        logits = self.output_projection(bigru_output)
        
        # Log softmax
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        return log_probs


class Stage2ExportWrapper(nn.Module):
    """
    Wrapper for Stage 2 model for export.
    Provides a simplified interface for inference.
    """
    
    def __init__(self, model: GlossToEnglishModel, max_decode_length: int = 50):
        super().__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.max_decode_length = max_decode_length
        self.sos_idx = model.english_sos_idx
        self.eos_idx = model.english_eos_idx
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with greedy decoding.
        
        Args:
            src: Source gloss IDs (batch, src_len)
            
        Returns:
            Decoded token IDs (batch, max_length)
        """
        batch_size = src.size(0)
        device = src.device
        
        # Create dummy lengths (assume full length)
        src_lengths = torch.full((batch_size,), src.size(1), device=device)
        
        # Encode (simplified without packing)
        embedded = self.encoder.dropout(self.encoder.embedding(src))
        encoder_outputs, hidden = self.encoder.gru(embedded)
        
        # Decode
        decoded = torch.full((batch_size, self.max_decode_length), 0, device=device)
        decoded[:, 0] = self.sos_idx
        
        input_token = decoded[:, 0:1]
        
        for t in range(1, self.max_decode_length):
            embedded = self.decoder.dropout(self.decoder.embedding(input_token))
            
            if self.decoder.use_attention:
                context, _ = self.decoder.attention(encoder_outputs, hidden[-1], None)
                gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
            else:
                gru_input = embedded
            
            output, hidden = self.decoder.gru(gru_input, hidden)
            
            if self.decoder.use_attention:
                output_combined = torch.cat([output.squeeze(1), context], dim=-1)
            else:
                output_combined = output.squeeze(1)
            
            logits = self.decoder.output_proj(output_combined)
            next_token = logits.argmax(dim=-1)
            decoded[:, t] = next_token
            input_token = next_token.unsqueeze(1)
        
        return decoded


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    input_names: list,
    output_names: list,
    dynamic_axes: Optional[Dict] = None,
    opset_version: int = 13
):
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        input_shape: Shape of dummy input
        input_names: Names of input tensors
        output_names: Names of output tensors
        dynamic_axes: Dynamic axes specification
        opset_version: ONNX opset version
    """
    model.eval()
    
    # Create dummy input
    if len(input_shape) == 3:  # Stage 1: (batch, seq, features)
        dummy_input = torch.randn(*input_shape)
    else:  # Stage 2: (batch, seq) integer tensor
        dummy_input = torch.randint(0, 100, input_shape)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    print(f"Exported ONNX model to: {output_path}")
    
    # Verify the model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
    except ImportError:
        print("Install onnx package to verify exported model: pip install onnx")
    except Exception as e:
        print(f"ONNX verification warning: {e}")


def quantize_onnx(
    input_path: str,
    output_path: str,
    quantization_type: str = 'int8'
):
    """
    Quantize ONNX model.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save quantized model
        quantization_type: Type of quantization ('int8', 'uint8')
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        if quantization_type == 'int8':
            quant_type = QuantType.QInt8
        else:
            quant_type = QuantType.QUInt8
        
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=quant_type
        )
        
        print(f"Quantized model saved to: {output_path}")
        
        # Compare sizes
        original_size = os.path.getsize(input_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Size reduction: {original_size:.2f}MB -> {quantized_size:.2f}MB "
              f"({(1 - quantized_size/original_size)*100:.1f}% reduction)")
        
    except ImportError:
        print("Install onnxruntime to quantize: pip install onnxruntime")


def export_to_tflite(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    quantize: bool = True
):
    """
    Export model to TensorFlow Lite format.
    
    Args:
        model: PyTorch model
        output_path: Path to save TFLite model
        input_shape: Shape of input tensor
        quantize: Whether to apply int8 quantization
    """
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare
        
        # First export to ONNX
        temp_onnx_path = output_path.replace('.tflite', '_temp.onnx')
        
        model.eval()
        if len(input_shape) == 3:
            dummy_input = torch.randn(*input_shape)
        else:
            dummy_input = torch.randint(0, 100, input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            temp_onnx_path,
            opset_version=13,
            do_constant_folding=True
        )
        
        # Convert ONNX to TensorFlow
        onnx_model = onnx.load(temp_onnx_path)
        tf_rep = prepare(onnx_model)
        
        # Save as SavedModel
        temp_saved_model = output_path.replace('.tflite', '_saved_model')
        tf_rep.export_graph(temp_saved_model)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(temp_saved_model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Exported TFLite model to: {output_path}")
        
        # Cleanup temp files
        os.remove(temp_onnx_path)
        import shutil
        shutil.rmtree(temp_saved_model, ignore_errors=True)
        
    except ImportError as e:
        print(f"TFLite export requires additional packages: {e}")
        print("Install: pip install tensorflow onnx-tf")


def export_stage1(config, device: str = 'cpu'):
    """Export Stage 1 model."""
    print("\n" + "=" * 60)
    print("Exporting Stage 1 Model (Video-to-Gloss)")
    print("=" * 60)
    
    # Load vocabularies
    gloss_vocab, _ = load_vocabularies(config)
    config.update_vocab_sizes(len(gloss_vocab), 0)
    
    # Create and load model
    model = VideoToGlossModel(config.stage1, len(gloss_vocab))
    
    checkpoint_path = os.path.join(config.paths.checkpoints_dir, 'best_stage1_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path, device=device)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Exporting untrained model...")
    
    # Create export wrapper
    export_model = Stage1ExportWrapper(model)
    export_model.eval()
    
    # Define input shape
    batch_size = 1
    seq_len = 100
    feature_dim = config.stage1.tcn.input_size
    input_shape = (batch_size, seq_len, feature_dim)
    
    # Export to ONNX
    if config.export.export_onnx:
        onnx_path = os.path.join(config.paths.exported_models_dir, 'stage1_model.onnx')
        export_to_onnx(
            export_model,
            onnx_path,
            input_shape,
            input_names=['features'],
            output_names=['log_probs'],
            dynamic_axes={
                'features': {0: 'batch', 1: 'seq_len'},
                'log_probs': {0: 'batch', 1: 'seq_len'}
            },
            opset_version=config.export.onnx_opset_version
        )
        
        # Quantize if requested
        if config.export.quantize:
            quantized_path = os.path.join(config.paths.exported_models_dir, 'stage1_model_quantized.onnx')
            quantize_onnx(onnx_path, quantized_path, config.export.quantization_type)
    
    # Export to TFLite
    if config.export.export_tflite:
        tflite_path = os.path.join(config.paths.exported_models_dir, 'stage1_model.tflite')
        export_to_tflite(export_model, tflite_path, input_shape, config.export.quantize)


def export_stage2(config, device: str = 'cpu'):
    """Export Stage 2 model."""
    print("\n" + "=" * 60)
    print("Exporting Stage 2 Model (Gloss-to-English)")
    print("=" * 60)
    
    # Load vocabularies
    gloss_vocab, english_vocab = load_vocabularies(config)
    config.update_vocab_sizes(len(gloss_vocab), len(english_vocab))
    
    # Create and load model
    model = GlossToEnglishModel(
        config.stage2,
        len(gloss_vocab),
        len(english_vocab),
        gloss_pad_idx=gloss_vocab.pad_idx,
        english_pad_idx=english_vocab.pad_idx,
        english_sos_idx=english_vocab.sos_idx,
        english_eos_idx=english_vocab.eos_idx
    )
    
    checkpoint_path = os.path.join(config.paths.checkpoints_dir, 'best_stage2_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path, device=device)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Exporting untrained model...")
    
    # Create export wrapper
    export_model = Stage2ExportWrapper(model, config.stage2.max_decode_length)
    export_model.eval()
    
    # Define input shape
    batch_size = 1
    src_len = 30
    input_shape = (batch_size, src_len)
    
    # Export to ONNX
    if config.export.export_onnx:
        onnx_path = os.path.join(config.paths.exported_models_dir, 'stage2_model.onnx')
        export_to_onnx(
            export_model,
            onnx_path,
            input_shape,
            input_names=['gloss_ids'],
            output_names=['decoded_ids'],
            dynamic_axes={
                'gloss_ids': {0: 'batch', 1: 'src_len'},
                'decoded_ids': {0: 'batch'}
            },
            opset_version=config.export.onnx_opset_version
        )
        
        # Quantize if requested
        if config.export.quantize:
            quantized_path = os.path.join(config.paths.exported_models_dir, 'stage2_model_quantized.onnx')
            quantize_onnx(onnx_path, quantized_path, config.export.quantization_type)


def main(args):
    """Main export function."""
    config = get_config()
    
    # Create export directory
    os.makedirs(config.paths.exported_models_dir, exist_ok=True)
    
    device = args.device if args.device else 'cpu'
    
    if args.stage == 1 or args.stage == 'all':
        export_stage1(config, device)
    
    if args.stage == 2 or args.stage == 'all':
        export_stage2(config, device)
    
    print("\n" + "=" * 60)
    print("Export completed!")
    print(f"Models saved to: {config.paths.exported_models_dir}")
    print("=" * 60)
    
    # Also save vocabularies for deployment
    vocab_export_path = os.path.join(config.paths.exported_models_dir, 'vocabularies')
    os.makedirs(vocab_export_path, exist_ok=True)
    
    import shutil
    gloss_src = os.path.join(config.paths.output_dir, 'gloss_vocab.json')
    english_src = os.path.join(config.paths.output_dir, 'english_vocab.json')
    
    if os.path.exists(gloss_src):
        shutil.copy(gloss_src, vocab_export_path)
    if os.path.exists(english_src):
        shutil.copy(english_src, vocab_export_path)
    
    print(f"Vocabularies copied to: {vocab_export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models for mobile deployment")
    parser.add_argument('--stage', type=str, default='all', 
                       help='Stage to export (1, 2, or all)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for loading models')
    
    args = parser.parse_args()
    
    if args.stage not in ['1', '2', 'all']:
        args.stage = 'all'
    elif args.stage in ['1', '2']:
        args.stage = int(args.stage)
    
    main(args)
