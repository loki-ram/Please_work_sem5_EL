"""
Comprehensive testing script for ISL-CSLTR Two-Stage Sign Language Translation System.
Tests all components: data loading, preprocessing, models, and inference.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import numpy as np
import torch
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, Config, PathConfig, MediaPipeConfig, TCNConfig, BiGRUConfig


class TestConfig(unittest.TestCase):
    """Test configuration module."""
    
    def test_config_creation(self):
        """Test that config can be created."""
        config = get_config()
        self.assertIsInstance(config, Config)
    
    def test_config_paths(self):
        """Test path configuration."""
        config = get_config()
        self.assertTrue(os.path.isabs(config.paths.dataset_root))
        self.assertTrue(os.path.isabs(config.paths.output_dir))
    
    def test_config_hyperparameters(self):
        """Test hyperparameter values are valid."""
        config = get_config()
        
        # Check learning rates are positive
        self.assertGreater(config.stage1.learning_rate, 0)
        self.assertGreater(config.stage2.learning_rate, 0)
        
        # Check batch size is positive
        self.assertGreater(config.dataset.batch_size, 0)
        
        # Check TCN configuration
        self.assertEqual(len(config.stage1.tcn.dilation_rates), config.stage1.tcn.num_layers)
    
    def test_config_update_vocab_sizes(self):
        """Test vocabulary size update."""
        config = get_config()
        config.update_vocab_sizes(100, 200)
        
        self.assertEqual(config.stage1.gloss_vocab_size, 100)
        self.assertEqual(config.stage2.gloss_vocab_size, 100)
        self.assertEqual(config.stage2.english_vocab_size, 200)


class TestVocabulary(unittest.TestCase):
    """Test vocabulary module."""
    
    def setUp(self):
        from data.vocabulary import Vocabulary
        self.Vocabulary = Vocabulary
    
    def test_vocabulary_creation(self):
        """Test vocabulary creation with special tokens."""
        vocab = self.Vocabulary()
        
        self.assertIn("<PAD>", vocab.token2idx)
        self.assertIn("<SOS>", vocab.token2idx)
        self.assertIn("<EOS>", vocab.token2idx)
        self.assertIn("<UNK>", vocab.token2idx)
    
    def test_vocabulary_with_blank(self):
        """Test vocabulary with CTC blank token."""
        vocab = self.Vocabulary(blank_token="<BLANK>")
        
        # Blank should be at index 0
        self.assertEqual(vocab.token2idx["<BLANK>"], 0)
        self.assertEqual(vocab.blank_idx, 0)
    
    def test_vocabulary_build_from_texts(self):
        """Test building vocabulary from texts."""
        vocab = self.Vocabulary()
        texts = [
            "hello world",
            "hello there",
            "world peace"
        ]
        vocab.build_from_texts(texts)
        
        # Check tokens are added
        self.assertIn("hello", vocab.token2idx)
        self.assertIn("world", vocab.token2idx)
        self.assertIn("there", vocab.token2idx)
        self.assertIn("peace", vocab.token2idx)
    
    def test_vocabulary_encode_decode(self):
        """Test encoding and decoding."""
        vocab = self.Vocabulary()
        texts = ["hello world", "test sentence"]
        vocab.build_from_texts(texts)
        
        # Encode
        encoded = vocab.encode("hello world")
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(i, int) for i in encoded))
        
        # Decode
        decoded = vocab.decode(encoded)
        self.assertEqual(decoded, "hello world")
    
    def test_vocabulary_encode_with_special_tokens(self):
        """Test encoding with SOS/EOS tokens."""
        vocab = self.Vocabulary()
        texts = ["hello world"]
        vocab.build_from_texts(texts)
        
        encoded = vocab.encode("hello world", add_sos=True, add_eos=True)
        
        self.assertEqual(encoded[0], vocab.sos_idx)
        self.assertEqual(encoded[-1], vocab.eos_idx)
    
    def test_vocabulary_unknown_token(self):
        """Test unknown token handling."""
        vocab = self.Vocabulary()
        texts = ["hello world"]
        vocab.build_from_texts(texts)
        
        encoded = vocab.encode("hello unknown")
        
        # "unknown" should map to UNK token
        self.assertIn(vocab.unk_idx, encoded)
    
    def test_vocabulary_save_load(self):
        """Test saving and loading vocabulary."""
        vocab = self.Vocabulary(blank_token="<BLANK>")
        vocab.build_from_texts(["hello world", "test sentence"])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            vocab.save(temp_path)
            loaded_vocab = self.Vocabulary.load(temp_path)
            
            self.assertEqual(vocab.token2idx, loaded_vocab.token2idx)
            self.assertEqual(vocab.blank_token, loaded_vocab.blank_token)
        finally:
            os.unlink(temp_path)


class TestCSVLoading(unittest.TestCase):
    """Test CSV loading functionality."""
    
    def test_load_isl_corpus_csv(self):
        """Test loading ISL Corpus CSV format."""
        from data.split_dataset import load_gloss_annotations
        
        config = get_config()
        csv_path = config.paths.gloss_csv
        
        if os.path.exists(csv_path):
            df = load_gloss_annotations(csv_path)
            
            # Check required columns exist
            self.assertIn('sentence', df.columns)
            self.assertIn('gloss', df.columns)
            self.assertIn('folder_name', df.columns)
            
            # Check data is not empty
            self.assertGreater(len(df), 0)
            
            # Check sentences are lowercase
            for sentence in df['sentence'].head():
                self.assertEqual(sentence, sentence.lower())
            
            # Check glosses are uppercase
            for gloss in df['gloss'].head():
                self.assertEqual(gloss, gloss.upper())
            
            print(f"✓ Loaded {len(df)} samples from CSV")
            print(f"  Sample sentence: {df['sentence'].iloc[0]}")
            print(f"  Sample gloss: {df['gloss'].iloc[0]}")
        else:
            self.skipTest(f"CSV file not found: {csv_path}")


class TestTCNModel(unittest.TestCase):
    """Test TCN module."""
    
    def setUp(self):
        from models.tcn import TCNEncoder, TCNBlock, CausalConv1d
        self.TCNEncoder = TCNEncoder
        self.TCNBlock = TCNBlock
        self.CausalConv1d = CausalConv1d
        self.config = get_config()
    
    def test_causal_conv1d(self):
        """Test causal convolution maintains causality."""
        conv = self.CausalConv1d(64, 64, kernel_size=3, dilation=2)
        
        x = torch.randn(2, 64, 100)
        y = conv(x)
        
        # Output should have same length as input
        self.assertEqual(x.shape[2], y.shape[2])
    
    def test_tcn_block(self):
        """Test TCN block."""
        block = self.TCNBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            dilation=2,
            dropout=0.1
        )
        
        x = torch.randn(2, 64, 100)
        y = block(x)
        
        self.assertEqual(y.shape, (2, 128, 100))
    
    def test_tcn_encoder(self):
        """Test full TCN encoder."""
        tcn_config = self.config.stage1.tcn
        encoder = self.TCNEncoder(tcn_config)
        
        batch_size = 4
        seq_len = 100
        x = torch.randn(batch_size, seq_len, tcn_config.input_size)
        
        output = encoder(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, tcn_config.hidden_size))
        print(f"✓ TCN receptive field: {encoder.receptive_field}")


class TestStage1Model(unittest.TestCase):
    """Test Stage 1 (Video-to-Gloss) model."""
    
    def setUp(self):
        from models.stage1_model import VideoToGlossModel
        self.VideoToGlossModel = VideoToGlossModel
        self.config = get_config()
    
    def test_model_creation(self):
        """Test model can be created."""
        model = self.VideoToGlossModel(self.config.stage1, gloss_vocab_size=100)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        print(f"✓ Stage 1 model parameters: {total_params:,}")
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = self.VideoToGlossModel(self.config.stage1, gloss_vocab_size=100)
        model.eval()
        
        batch_size = 2
        seq_len = 50
        feature_dim = self.config.stage1.tcn.input_size
        
        features = torch.randn(batch_size, seq_len, feature_dim)
        lengths = torch.tensor([50, 40])
        
        log_probs = model(features, lengths)
        
        # Output shape: (seq_len, batch, vocab_size)
        self.assertEqual(log_probs.shape[1], batch_size)
        self.assertEqual(log_probs.shape[2], 100)
    
    def test_ctc_loss(self):
        """Test CTC loss computation."""
        model = self.VideoToGlossModel(self.config.stage1, gloss_vocab_size=100)
        
        batch_size = 2
        seq_len = 50
        feature_dim = self.config.stage1.tcn.input_size
        
        features = torch.randn(batch_size, seq_len, feature_dim)
        feature_lengths = torch.tensor([50, 40])
        gloss_ids = torch.randint(1, 100, (batch_size, 10))
        gloss_lengths = torch.tensor([8, 6])
        
        loss = model.compute_loss(features, feature_lengths, gloss_ids, gloss_lengths)
        
        self.assertIsInstance(loss.item(), float)
        self.assertFalse(torch.isnan(loss))
        print(f"✓ Stage 1 CTC loss: {loss.item():.4f}")
    
    def test_greedy_decoding(self):
        """Test greedy decoding."""
        model = self.VideoToGlossModel(self.config.stage1, gloss_vocab_size=100)
        model.eval()
        
        batch_size = 2
        seq_len = 50
        feature_dim = self.config.stage1.tcn.input_size
        
        features = torch.randn(batch_size, seq_len, feature_dim)
        lengths = torch.tensor([50, 40])
        
        with torch.no_grad():
            decoded_seqs, decoded_lens = model.decode_greedy(features, lengths)
        
        self.assertEqual(len(decoded_seqs), batch_size)
        self.assertEqual(len(decoded_lens), batch_size)


class TestStage2Model(unittest.TestCase):
    """Test Stage 2 (Gloss-to-English) model."""
    
    def setUp(self):
        from models.stage2_model import GlossToEnglishModel, Attention
        self.GlossToEnglishModel = GlossToEnglishModel
        self.Attention = Attention
        self.config = get_config()
    
    def test_attention_mechanism(self):
        """Test attention mechanism."""
        attention = self.Attention(
            encoder_hidden_size=256,
            decoder_hidden_size=256,
            attention_dim=128
        )
        
        encoder_outputs = torch.randn(2, 15, 256)  # (batch, src_len, hidden)
        decoder_hidden = torch.randn(2, 256)  # (batch, hidden)
        
        context, weights = attention(encoder_outputs, decoder_hidden)
        
        self.assertEqual(context.shape, (2, 256))
        self.assertEqual(weights.shape, (2, 15))
        
        # Attention weights should sum to 1
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5))
    
    def test_model_creation(self):
        """Test model can be created."""
        model = self.GlossToEnglishModel(
            self.config.stage2,
            gloss_vocab_size=100,
            english_vocab_size=200
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        print(f"✓ Stage 2 model parameters: {total_params:,}")
    
    def test_forward_pass(self):
        """Test forward pass with teacher forcing."""
        model = self.GlossToEnglishModel(
            self.config.stage2,
            gloss_vocab_size=100,
            english_vocab_size=200
        )
        model.eval()
        
        batch_size = 2
        src_len = 15
        tgt_len = 20
        
        src = torch.randint(1, 100, (batch_size, src_len))
        src_lengths = torch.tensor([15, 12])
        tgt = torch.randint(1, 200, (batch_size, tgt_len))
        tgt[:, 0] = 1  # SOS token
        
        with torch.no_grad():
            outputs = model(src, src_lengths, tgt, teacher_forcing_ratio=0.5)
        
        # Output shape: (batch, tgt_len-1, vocab_size)
        self.assertEqual(outputs.shape, (batch_size, tgt_len - 1, 200))
    
    def test_loss_computation(self):
        """Test loss computation."""
        model = self.GlossToEnglishModel(
            self.config.stage2,
            gloss_vocab_size=100,
            english_vocab_size=200
        )
        
        batch_size = 2
        src_len = 15
        tgt_len = 20
        
        src = torch.randint(1, 100, (batch_size, src_len))
        src_lengths = torch.tensor([15, 12])
        tgt = torch.randint(1, 200, (batch_size, tgt_len))
        tgt[:, 0] = 1  # SOS token
        
        loss = model.compute_loss(src, src_lengths, tgt)
        
        self.assertIsInstance(loss.item(), float)
        self.assertFalse(torch.isnan(loss))
        print(f"✓ Stage 2 CE loss: {loss.item():.4f}")
    
    def test_greedy_decoding(self):
        """Test greedy decoding."""
        model = self.GlossToEnglishModel(
            self.config.stage2,
            gloss_vocab_size=100,
            english_vocab_size=200,
            english_sos_idx=1,
            english_eos_idx=2
        )
        model.eval()
        
        batch_size = 2
        src_len = 15
        
        src = torch.randint(1, 100, (batch_size, src_len))
        src_lengths = torch.tensor([15, 12])
        
        with torch.no_grad():
            decoded, attention = model.decode_greedy(src, src_lengths, max_length=30)
        
        self.assertEqual(decoded.shape[0], batch_size)
        self.assertIsNotNone(attention)


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""
    
    def setUp(self):
        from utils.metrics import (
            word_error_rate, compute_wer, 
            bleu_score, compute_bleu,
            rouge_l_score, compute_rouge_l,
            compute_all_metrics
        )
        self.word_error_rate = word_error_rate
        self.compute_wer = compute_wer
        self.bleu_score = bleu_score
        self.compute_bleu = compute_bleu
        self.rouge_l_score = rouge_l_score
        self.compute_rouge_l = compute_rouge_l
        self.compute_all_metrics = compute_all_metrics
    
    def test_wer_perfect_match(self):
        """Test WER for perfect match."""
        ref = ["hello", "world"]
        hyp = ["hello", "world"]
        
        wer = self.word_error_rate(ref, hyp)
        self.assertEqual(wer, 0.0)
    
    def test_wer_complete_mismatch(self):
        """Test WER for complete mismatch."""
        ref = ["hello", "world"]
        hyp = ["goodbye", "everyone"]
        
        wer = self.word_error_rate(ref, hyp)
        self.assertEqual(wer, 1.0)  # 2 substitutions / 2 words
    
    def test_wer_insertions_deletions(self):
        """Test WER with insertions and deletions."""
        ref = ["the", "cat", "sat"]
        hyp = ["the", "cat"]  # 1 deletion
        
        wer = self.word_error_rate(ref, hyp)
        self.assertAlmostEqual(wer, 1/3, places=5)
    
    def test_bleu_perfect_match(self):
        """Test BLEU for perfect match."""
        ref = ["the", "cat", "sat", "on", "the", "mat"]
        hyp = ["the", "cat", "sat", "on", "the", "mat"]
        
        bleu = self.bleu_score(ref, hyp)
        self.assertAlmostEqual(bleu, 1.0, places=5)
    
    def test_bleu_partial_match(self):
        """Test BLEU for partial match."""
        ref = ["the", "cat", "sat", "on", "the", "mat"]
        hyp = ["the", "cat", "on", "mat"]
        
        bleu = self.bleu_score(ref, hyp)
        self.assertGreater(bleu, 0)
        self.assertLess(bleu, 1)
    
    def test_rouge_l_score(self):
        """Test ROUGE-L score."""
        ref = ["the", "cat", "sat", "on", "the", "mat"]
        hyp = ["the", "cat", "on", "the", "mat"]
        
        scores = self.rouge_l_score(ref, hyp)
        
        self.assertIn('precision', scores)
        self.assertIn('recall', scores)
        self.assertIn('f1', scores)
        self.assertGreater(scores['f1'], 0)
    
    def test_compute_all_metrics(self):
        """Test computing all metrics."""
        references = [
            "the cat sat on the mat",
            "hello world"
        ]
        hypotheses = [
            "the cat on mat",
            "hello there world"
        ]
        
        metrics = self.compute_all_metrics(references, hypotheses)
        
        self.assertIn('wer', metrics)
        self.assertIn('bleu', metrics)
        self.assertIn('rouge_l_f1', metrics)
        
        print(f"✓ All metrics computed: WER={metrics['wer']:.3f}, BLEU={metrics['bleu']:.3f}")


class TestTrainingUtils(unittest.TestCase):
    """Test training utilities."""
    
    def setUp(self):
        from utils.training_utils import (
            AverageMeter, EarlyStopping, 
            get_optimizer, count_parameters
        )
        self.AverageMeter = AverageMeter
        self.EarlyStopping = EarlyStopping
        self.get_optimizer = get_optimizer
        self.count_parameters = count_parameters
    
    def test_average_meter(self):
        """Test AverageMeter."""
        meter = self.AverageMeter()
        
        meter.update(1.0)
        meter.update(2.0)
        meter.update(3.0)
        
        self.assertEqual(meter.count, 3)
        self.assertEqual(meter.avg, 2.0)
        self.assertEqual(meter.val, 3.0)
    
    def test_early_stopping_min_mode(self):
        """Test EarlyStopping in min mode."""
        es = self.EarlyStopping(patience=3, mode='min', verbose=False)
        
        # Improving scores
        self.assertFalse(es(1.0))
        self.assertFalse(es(0.9))
        self.assertFalse(es(0.8))
        
        # Stagnating scores
        self.assertFalse(es(0.85))
        self.assertFalse(es(0.82))
        self.assertFalse(es(0.81))
        self.assertTrue(es(0.80))  # Should trigger after patience
    
    def test_early_stopping_max_mode(self):
        """Test EarlyStopping in max mode."""
        es = self.EarlyStopping(patience=2, mode='max', verbose=False)
        
        self.assertFalse(es(0.5))
        self.assertFalse(es(0.6))
        self.assertFalse(es(0.55))  # Not improving
        self.assertFalse(es(0.58))  # Still not improving
        self.assertTrue(es(0.57))  # Trigger
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(100, 50)
        
        counts = self.count_parameters(model)
        
        self.assertIn('total', counts)
        self.assertIn('trainable', counts)
        self.assertEqual(counts['total'], 100 * 50 + 50)  # weights + bias


class TestDataPipeline(unittest.TestCase):
    """Test complete data pipeline."""
    
    def test_full_pipeline_with_mock_data(self):
        """Test vocabulary building and dataset creation with mock data."""
        from data.vocabulary import Vocabulary, build_vocabularies
        
        # Create mock samples
        mock_samples = [
            {'gloss': 'HELLO WORLD', 'sentence': 'hello world'},
            {'gloss': 'HOW ARE YOU', 'sentence': 'how are you'},
            {'gloss': 'THANK YOU', 'sentence': 'thank you'},
        ]
        
        config = get_config()
        
        # Build vocabularies
        gloss_vocab, english_vocab = build_vocabularies(mock_samples, config)
        
        self.assertGreater(len(gloss_vocab), 5)  # Special tokens + words
        self.assertGreater(len(english_vocab), 5)
        
        print(f"✓ Built vocabularies: gloss={len(gloss_vocab)}, english={len(english_vocab)}")


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_stage1_training_step(self):
        """Test a single training step for Stage 1."""
        from models.stage1_model import VideoToGlossModel
        
        config = get_config()
        model = VideoToGlossModel(config.stage1, gloss_vocab_size=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Mock batch
        batch_size = 4
        seq_len = 50
        feature_dim = config.stage1.tcn.input_size
        
        features = torch.randn(batch_size, seq_len, feature_dim)
        feature_lengths = torch.tensor([50, 45, 40, 35])
        gloss_ids = torch.randint(1, 100, (batch_size, 8))
        gloss_lengths = torch.tensor([8, 7, 6, 5])
        
        # Training step
        model.train()
        optimizer.zero_grad()
        loss = model.compute_loss(features, feature_lengths, gloss_ids, gloss_lengths)
        loss.backward()
        optimizer.step()
        
        self.assertFalse(torch.isnan(loss))
        print(f"✓ Stage 1 training step completed, loss={loss.item():.4f}")
    
    def test_stage2_training_step(self):
        """Test a single training step for Stage 2."""
        from models.stage2_model import GlossToEnglishModel
        
        config = get_config()
        model = GlossToEnglishModel(
            config.stage2, 
            gloss_vocab_size=100, 
            english_vocab_size=200
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Mock batch
        batch_size = 4
        src_len = 10
        tgt_len = 15
        
        src = torch.randint(1, 100, (batch_size, src_len))
        src_lengths = torch.tensor([10, 9, 8, 7])
        tgt = torch.randint(1, 200, (batch_size, tgt_len))
        tgt[:, 0] = 1  # SOS
        
        # Training step
        model.train()
        optimizer.zero_grad()
        loss = model.compute_loss(src, src_lengths, tgt)
        loss.backward()
        optimizer.step()
        
        self.assertFalse(torch.isnan(loss))
        print(f"✓ Stage 2 training step completed, loss={loss.item():.4f}")
    
    def test_full_inference_pipeline(self):
        """Test full inference from gloss to English."""
        from data.vocabulary import Vocabulary
        from models.stage2_model import GlossToEnglishModel
        
        config = get_config()
        
        # Create vocabularies
        gloss_vocab = Vocabulary(blank_token="<BLANK>")
        gloss_vocab.build_from_texts(["HELLO WORLD", "HOW ARE YOU", "THANK YOU"])
        
        english_vocab = Vocabulary()
        english_vocab.build_from_texts(["hello world", "how are you", "thank you"])
        
        # Create model
        model = GlossToEnglishModel(
            config.stage2,
            len(gloss_vocab),
            len(english_vocab),
            gloss_pad_idx=gloss_vocab.pad_idx,
            english_pad_idx=english_vocab.pad_idx,
            english_sos_idx=english_vocab.sos_idx,
            english_eos_idx=english_vocab.eos_idx
        )
        model.eval()
        
        # Encode input gloss
        gloss_text = "HELLO WORLD"
        gloss_ids = gloss_vocab.encode(gloss_text, add_sos=True, add_eos=True)
        gloss_tensor = torch.tensor([gloss_ids])
        gloss_lengths = torch.tensor([len(gloss_ids)])
        
        # Decode
        with torch.no_grad():
            decoded, _ = model.decode_greedy(gloss_tensor, gloss_lengths)
        
        # Convert to text
        output_text = english_vocab.decode(decoded[0].tolist())
        
        print(f"✓ Inference test: '{gloss_text}' -> '{output_text}'")


def run_tests(verbosity=2):
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConfig,
        TestVocabulary,
        TestCSVLoading,
        TestTCNModel,
        TestStage1Model,
        TestStage2Model,
        TestMetrics,
        TestTrainingUtils,
        TestDataPipeline,
        TestEndToEnd,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed.")
        for test, traceback in result.failures + result.errors:
            print(f"\nFailed: {test}")
            print(traceback)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for ISL-CSLTR system")
    parser.add_argument('--verbose', '-v', action='count', default=2,
                        help='Increase verbosity (can be used multiple times)')
    parser.add_argument('--test', '-t', type=str, default=None,
                        help='Run specific test class (e.g., TestVocabulary)')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run quick tests only (skip slow tests)')
    
    args = parser.parse_args()
    
    if args.test:
        # Run specific test class
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        test_class = globals().get(args.test)
        if test_class:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
            runner = unittest.TextTestRunner(verbosity=args.verbose)
            runner.run(suite)
        else:
            print(f"Test class '{args.test}' not found.")
            print("Available test classes:")
            for name in dir():
                if name.startswith('Test'):
                    print(f"  - {name}")
    else:
        run_tests(verbosity=args.verbose)
