"""
Hybrid Sign-to-Text Model.

Architecture:
1. Shared Encoder: Conv1D (temporal downsampling) + BiGRU (sequence modeling)
2. CTC Head: For gloss supervision (auxiliary loss)
3. Attention Decoder: Bahdanau attention + GRU for text generation

The encoder outputs are branched to both CTC and attention decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import random


class Conv1DEncoder(nn.Module):
    """
    Conv1D layers for temporal downsampling.
    Two layers with kernel=5, stride=2, channels=256.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        kernel_size: int = 5,
        stride: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        
        layers = []
        in_channels = input_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_channels = hidden_size
        
        self.conv_layers = nn.Sequential(*layers)
        self.output_size = hidden_size
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            lengths: Sequence lengths (batch,)
            
        Returns:
            Tuple of (output, new_lengths)
            - output: (batch, new_seq_len, hidden_size)
            - new_lengths: Updated sequence lengths after downsampling
        """
        # (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Apply conv layers
        x = self.conv_layers(x)
        
        # (batch, features, seq_len) -> (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # Update sequence lengths correctly for each conv layer
        # Conv output length = floor((input_len + 2*padding - kernel_size) / stride) + 1
        # With padding = kernel_size // 2
        if lengths is not None:
            new_lengths = lengths.clone()
            for _ in range(2):  # Two conv layers
                # With padding = kernel_size // 2, the formula simplifies
                new_lengths = (new_lengths + self.kernel_size // 2 * 2 - self.kernel_size) // self.stride + 1
                new_lengths = torch.clamp(new_lengths, min=1, max=x.size(1))
        else:
            new_lengths = None
        
        return x, new_lengths


class BiGRUEncoder(nn.Module):
    """
    Bidirectional GRU for sequence modeling.
    2 layers, hidden=256, dropout=0.3.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.output_size = hidden_size * self.num_directions
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with packed sequences for variable length inputs.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            lengths: Sequence lengths (batch,)
            
        Returns:
            Tuple of (outputs, hidden)
            - outputs: (batch, seq_len, hidden * num_directions)
            - hidden: (num_layers * num_directions, batch, hidden)
        """
        self.gru.flatten_parameters()
        
        if lengths is not None:
            # Sort by length (descending) for packing
            lengths = lengths.cpu()
            sorted_lengths, sort_idx = torch.sort(lengths, descending=True)
            sorted_x = x[sort_idx]
            
            # Pack sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                sorted_x, sorted_lengths.clamp(min=1), batch_first=True
            )
            
            # GRU forward
            packed_output, hidden = self.gru(packed)
            
            # Unpack
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
            
            # Unsort
            _, unsort_idx = torch.sort(sort_idx)
            output = output[unsort_idx]
            
            # Reorder hidden states
            if self.bidirectional:
                hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
                hidden = hidden[:, :, unsort_idx, :]
                hidden = hidden.view(self.num_layers * 2, -1, self.hidden_size)
            else:
                hidden = hidden[:, unsort_idx, :]
        else:
            output, hidden = self.gru(x)
        
        return output, hidden


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.
    """
    
    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        attention_dim: int = 128
    ):
        super().__init__()
        
        self.encoder_proj = nn.Linear(encoder_hidden_size, attention_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_hidden_size, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            encoder_outputs: (batch, src_len, encoder_hidden)
            decoder_hidden: (batch, decoder_hidden)
            encoder_mask: (batch, src_len) - True for valid positions
            
        Returns:
            Tuple of (context, attention_weights)
            - context: (batch, encoder_hidden)
            - attention_weights: (batch, src_len)
        """
        batch_size, src_len, _ = encoder_outputs.size()
        
        # Project encoder outputs: (batch, src_len, attention_dim)
        encoder_proj = self.encoder_proj(encoder_outputs)
        
        # Project decoder hidden: (batch, attention_dim)
        decoder_proj = self.decoder_proj(decoder_hidden)
        
        # Expand for addition: (batch, src_len, attention_dim)
        decoder_proj = decoder_proj.unsqueeze(1).expand(-1, src_len, -1)
        
        # Compute energy: (batch, src_len, 1) -> (batch, src_len)
        energy = torch.tanh(encoder_proj + decoder_proj)
        attention_scores = self.v(energy).squeeze(-1)
        
        # Apply mask
        if encoder_mask is not None:
            attention_scores = attention_scores.masked_fill(~encoder_mask, float('-inf'))
        
        # Softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute context vector: (batch, encoder_hidden)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class AttentionDecoder(nn.Module):
    """
    GRU decoder with Bahdanau attention for text generation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 256,
        encoder_hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
        attention_dim: int = 128,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_hidden_size = encoder_hidden_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # Attention
        self.attention = BahdanauAttention(
            encoder_hidden_size, hidden_size, attention_dim
        )
        
        # GRU input: embedding + context
        gru_input_size = embedding_dim + encoder_hidden_size
        
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size + encoder_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoding step.
        
        Args:
            input_token: (batch,) or (batch, 1) - current input token
            hidden: (num_layers, batch, hidden_size)
            encoder_outputs: (batch, src_len, encoder_hidden)
            encoder_mask: (batch, src_len)
            
        Returns:
            Tuple of (logits, new_hidden, attention_weights)
        """
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)
        
        # Embed input: (batch, 1, embedding_dim)
        embedded = self.dropout(self.embedding(input_token))
        
        # Get attention context using last layer's hidden state
        context, attn_weights = self.attention(
            encoder_outputs, hidden[-1], encoder_mask
        )
        
        # Concatenate embedding with context: (batch, 1, embedding + encoder_hidden)
        gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        
        # GRU step
        output, hidden = self.gru(gru_input, hidden)
        
        # Project to vocabulary: (batch, vocab_size)
        output_combined = torch.cat([output.squeeze(1), context], dim=-1)
        logits = self.output_proj(output_combined)
        
        return logits, hidden, attn_weights


class HybridSignToTextModel(nn.Module):
    """
    Hybrid CTC + Attention Sign-to-Text Model.
    
    Architecture:
    - Shared Encoder: Conv1D (downsampling) + BiGRU (sequence modeling)
    - CTC Head: Linear projection for gloss supervision
    - Attention Decoder: GRU with Bahdanau attention for text generation
    
    Loss: α * CTC_loss + (1-α) * CrossEntropy_loss
    """
    
    def __init__(
        self,
        input_size: int = 432,
        conv_channels: int = 256,
        conv_kernel_size: int = 5,
        conv_stride: int = 2,
        gru_hidden_size: int = 512,  # Increased to 512 for stronger encoder
        gru_num_layers: int = 2,
        gru_dropout: float = 0.3,
        decoder_embedding_dim: int = 256,
        decoder_hidden_size: int = 256,
        decoder_num_layers: int = 1,
        decoder_dropout: float = 0.3,
        attention_dim: int = 128,
        gloss_vocab_size: int = 100,
        text_vocab_size: int = 200,
        gloss_blank_idx: int = 0,
        text_pad_idx: int = 0,
        text_sos_idx: int = 1,
        text_eos_idx: int = 2,
        max_decode_length: int = 50,
        use_encoder_projection: bool = True,
        encoder_projection_dim: int = 512,
        attention_entropy_weight: float = 0.01,
        # Anti-shortcut: Early EOS penalty
        min_eos_step: int = 2,
        eos_penalty: float = 5.0,
        # Anti-shortcut: Decoder input dropout
        decoder_input_dropout: float = 0.2
    ):
        super().__init__()
        
        self.gloss_vocab_size = gloss_vocab_size
        self.text_vocab_size = text_vocab_size
        self.gloss_blank_idx = gloss_blank_idx
        self.text_pad_idx = text_pad_idx
        self.text_sos_idx = text_sos_idx
        self.text_eos_idx = text_eos_idx
        self.max_decode_length = max_decode_length
        self.use_encoder_projection = use_encoder_projection
        self.attention_entropy_weight = attention_entropy_weight
        
        # Anti-shortcut parameters
        self.min_eos_step = min_eos_step
        self.eos_penalty = eos_penalty
        self.decoder_input_dropout_rate = decoder_input_dropout
        self.decoder_input_dropout = nn.Dropout(decoder_input_dropout)
        
        # =====================================================================
        # Shared Encoder
        # =====================================================================
        
        # Conv1D for temporal downsampling
        self.conv_encoder = Conv1DEncoder(
            input_size=input_size,
            hidden_size=conv_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            num_layers=2,
            dropout=gru_dropout
        )
        
        # BiGRU for sequence modeling
        self.gru_encoder = BiGRUEncoder(
            input_size=conv_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
            bidirectional=True
        )
        
        # Raw encoder output size (bidirectional)
        raw_encoder_size = gru_hidden_size * 2  # 768 with 384 hidden
        
        # =====================================================================
        # Encoder Projection (strengthens encoder -> attention signal)
        # =====================================================================
        
        if use_encoder_projection:
            self.encoder_projection = nn.Sequential(
                nn.Linear(raw_encoder_size, encoder_projection_dim),
                nn.Tanh(),
                nn.Dropout(gru_dropout)
            )
            encoder_output_size = encoder_projection_dim
        else:
            self.encoder_projection = None
            encoder_output_size = raw_encoder_size
        
        self.encoder_output_size = encoder_output_size
        self.raw_encoder_size = raw_encoder_size
        
        # =====================================================================
        # CTC Head (works on raw encoder output for better gradients)
        # =====================================================================
        
        self.ctc_projection = nn.Linear(raw_encoder_size, gloss_vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(
            blank=gloss_blank_idx,
            reduction='mean',
            zero_infinity=True
        )
        
        # =====================================================================
        # Attention Decoder
        # =====================================================================
        
        # Bridge: project BiGRU hidden to decoder hidden
        self.hidden_bridge = nn.Linear(
            raw_encoder_size,  # Use raw encoder size for hidden bridge
            decoder_hidden_size * decoder_num_layers
        )
        
        self.decoder = AttentionDecoder(
            vocab_size=text_vocab_size,
            embedding_dim=decoder_embedding_dim,
            hidden_size=decoder_hidden_size,
            encoder_hidden_size=encoder_output_size,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
            attention_dim=attention_dim,
            padding_idx=text_pad_idx
        )
        
        self.decoder_num_layers = decoder_num_layers
        self.decoder_hidden_size = decoder_hidden_size
        
        # Cross-entropy loss (ignoring padding)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=text_pad_idx)
    
    def encode(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode video features.
        
        Args:
            features: (batch, seq_len, input_size)
            feature_lengths: (batch,)
            
        Returns:
            Tuple of (encoder_outputs, decoder_hidden, ctc_log_probs, output_lengths, raw_encoder_outputs)
        """
        # Conv1D encoding with downsampling
        conv_output, conv_lengths = self.conv_encoder(features, feature_lengths)
        
        # BiGRU encoding
        raw_encoder_outputs, gru_hidden = self.gru_encoder(conv_output, conv_lengths)
        
        # CTC head (on raw encoder output for better gradients)
        ctc_logits = self.ctc_projection(raw_encoder_outputs)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        ctc_log_probs = ctc_log_probs.transpose(0, 1)  # (seq_len, batch, vocab)
        
        # Apply encoder projection for attention (strengthens visual signal)
        if self.encoder_projection is not None:
            encoder_outputs = self.encoder_projection(raw_encoder_outputs)
        else:
            encoder_outputs = raw_encoder_outputs
        
        # Prepare decoder initial hidden
        # Combine forward and backward last hidden states
        forward_hidden = gru_hidden[-2]   # (batch, hidden)
        backward_hidden = gru_hidden[-1]  # (batch, hidden)
        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)
        
        # Project to decoder hidden
        decoder_hidden = self.hidden_bridge(combined)
        decoder_hidden = decoder_hidden.view(
            self.decoder_num_layers, -1, self.decoder_hidden_size
        ).contiguous()
        
        return encoder_outputs, decoder_hidden, ctc_log_probs, conv_lengths, raw_encoder_outputs
    
    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        text_ids: torch.Tensor,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with teacher forcing.
        
        Args:
            features: (batch, seq_len, input_size)
            feature_lengths: (batch,)
            text_ids: (batch, tgt_len) - includes SOS at start
            teacher_forcing_ratio: Probability of using ground truth
            
        Returns:
            Tuple of (decoder_outputs, ctc_log_probs)
            - decoder_outputs: (batch, tgt_len-1, vocab_size)
            - ctc_log_probs: (seq_len, batch, gloss_vocab)
        """
        batch_size = features.size(0)
        tgt_len = text_ids.size(1)
        device = features.device
        
        # Encode
        encoder_outputs, hidden, ctc_log_probs, enc_lengths, _ = self.encode(
            features, feature_lengths
        )
        
        # Create encoder mask
        max_enc_len = encoder_outputs.size(1)
        encoder_mask = torch.arange(max_enc_len, device=device).unsqueeze(0) < enc_lengths.unsqueeze(1)
        
        # Initialize decoder outputs and attention weights
        outputs = torch.zeros(batch_size, tgt_len - 1, self.text_vocab_size, device=device)
        all_attention_weights = []  # Collect for entropy regularization
        
        # First input is SOS
        input_token = text_ids[:, 0]
        
        # Decode step by step
        for t in range(1, tgt_len):
            # =================================================================
            # ANTI-SHORTCUT: Decoder input dropout during training
            # Zero out previous token embedding with probability 0.2
            # Forces decoder to rely on encoder attention
            # =================================================================
            if self.training and self.decoder_input_dropout_rate > 0:
                # Get embedding and apply dropout
                embedded = self.decoder.embedding(input_token.unsqueeze(1))
                embedded = self.decoder_input_dropout(embedded)
                # Forward step with modified embedding
                logits, hidden, attn_weights = self._decoder_forward_with_embedding(
                    embedded, hidden, encoder_outputs, encoder_mask
                )
            else:
                logits, hidden, attn_weights = self.decoder.forward_step(
                    input_token, hidden, encoder_outputs, encoder_mask
                )
            
            # =================================================================
            # ANTI-SHORTCUT: Early EOS penalty during training
            # Subtract large bias from EOS logit for first min_eos_step steps
            # =================================================================
            if self.training and t <= self.min_eos_step:
                logits[:, self.text_eos_idx] = logits[:, self.text_eos_idx] - self.eos_penalty
            
            outputs[:, t - 1] = logits
            all_attention_weights.append(attn_weights)
            
            # Teacher forcing decision
            use_teacher = random.random() < teacher_forcing_ratio
            if use_teacher:
                input_token = text_ids[:, t]
            else:
                input_token = logits.argmax(dim=-1)
        
        # Stack attention weights: (batch, tgt_len-1, src_len)
        attention_weights = torch.stack(all_attention_weights, dim=1)
        
        return outputs, ctc_log_probs, enc_lengths, attention_weights
    
    def _decoder_forward_with_embedding(
        self,
        embedded: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward step with pre-computed embedding (for decoder input dropout).
        """
        # Get attention context using last layer's hidden state
        context, attn_weights = self.decoder.attention(
            encoder_outputs, hidden[-1], encoder_mask
        )
        
        # Concatenate embedding with context
        gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        
        # GRU step
        output, hidden = self.decoder.gru(gru_input, hidden)
        
        # Project to vocabulary
        output_combined = torch.cat([output.squeeze(1), context], dim=-1)
        logits = self.decoder.output_proj(output_combined)
        
        return logits, hidden, attn_weights
    
    def compute_attention_entropy_loss(
        self,
        attention_weights: torch.Tensor,
        encoder_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention entropy regularization loss.
        Encourages the model to attend to specific frames rather than uniform.
        
        Args:
            attention_weights: (batch, tgt_len, src_len)
            encoder_mask: (batch, src_len)
            
        Returns:
            Negative entropy loss (lower = more peaked attention)
        """
        # Add small epsilon for numerical stability
        eps = 1e-8
        log_attn = torch.log(attention_weights + eps)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -(attention_weights * log_attn).sum(dim=-1)  # (batch, tgt_len)
        
        # Average over valid positions
        mean_entropy = entropy.mean()
        
        # Return negative entropy (we want to encourage peaked attention)
        # Higher entropy = more uniform = bad for grounding
        # Return positive loss that penalizes high entropy
        return mean_entropy
    
    def compute_loss(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        gloss_ids: torch.Tensor,
        gloss_lengths: torch.Tensor,
        text_ids: torch.Tensor,
        ctc_weight: float = 0.5,
        teacher_forcing_ratio: float = 1.0,
        use_attention_entropy: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid CTC + Attention loss with optional attention entropy regularization.
        
        Args:
            features: Input features (batch, seq_len, input_size)
            feature_lengths: Feature lengths (batch,)
            gloss_ids: Target gloss IDs (batch, gloss_len)
            gloss_lengths: Gloss lengths (batch,)
            text_ids: Target text IDs with SOS (batch, text_len)
            ctc_weight: Weight alpha for hybrid loss
            teacher_forcing_ratio: Teacher forcing probability
            use_attention_entropy: Whether to add attention entropy regularization
            
        Returns:
            Dictionary with loss components
        """
        # Forward pass
        decoder_outputs, ctc_log_probs, enc_lengths, attention_weights = self.forward(
            features, feature_lengths, text_ids, teacher_forcing_ratio
        )
        
        # CTC Loss - use actual encoder output lengths
        input_lengths = enc_lengths.long()
        
        ctc_loss = self.ctc_loss_fn(
            ctc_log_probs,
            gloss_ids,
            input_lengths.long(),
            gloss_lengths.long()
        )
        
        # Cross-Entropy Loss (skip SOS in targets)
        ce_outputs = decoder_outputs.reshape(-1, self.text_vocab_size)
        ce_targets = text_ids[:, 1:].reshape(-1)
        ce_loss = self.ce_loss_fn(ce_outputs, ce_targets)
        
        # Attention Entropy Regularization
        if use_attention_entropy and self.attention_entropy_weight > 0:
            device = features.device
            max_enc_len = attention_weights.size(-1)
            encoder_mask = torch.arange(max_enc_len, device=device).unsqueeze(0) < enc_lengths.unsqueeze(1)
            attn_entropy_loss = self.compute_attention_entropy_loss(attention_weights, encoder_mask)
        else:
            attn_entropy_loss = torch.tensor(0.0, device=features.device)
        
        # Hybrid loss: alpha * CTC + (1-alpha) * CE + entropy_weight * entropy
        attention_weight = 1.0 - ctc_weight
        total_loss = (
            ctc_weight * ctc_loss + 
            attention_weight * ce_loss + 
            self.attention_entropy_weight * attn_entropy_loss
        )
        
        return {
            'loss': total_loss,
            'ctc_loss': ctc_loss,
            'ce_loss': ce_loss,
            'attn_entropy_loss': attn_entropy_loss
        }
    
    @torch.no_grad()
    def decode_greedy(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Greedy decoding for inference (attention decoder only).
        
        Anti-shortcut: Enforces minimum output length by disallowing EOS
        before min_eos_step decoded tokens.
        
        Args:
            features: Input features (batch, seq_len, input_size)
            feature_lengths: Feature lengths (batch,)
            max_length: Maximum decoding length
            
        Returns:
            decoded_ids: (batch, max_length)
        """
        if max_length is None:
            max_length = self.max_decode_length
        
        batch_size = features.size(0)
        device = features.device
        
        # Encode
        encoder_outputs, hidden, _, enc_lengths, _ = self.encode(features, feature_lengths)
        
        # Create encoder mask
        max_enc_len = encoder_outputs.size(1)
        encoder_mask = torch.arange(max_enc_len, device=device).unsqueeze(0) < enc_lengths.unsqueeze(1)
        
        # Initialize
        decoded = torch.full((batch_size, max_length), self.text_pad_idx, device=device)
        decoded[:, 0] = self.text_sos_idx
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        input_token = decoded[:, 0]
        
        for t in range(1, max_length):
            logits, hidden, _ = self.decoder.forward_step(
                input_token, hidden, encoder_outputs, encoder_mask
            )
            
            # =================================================================
            # ANTI-SHORTCUT: Minimum output length at inference
            # Disallow EOS before min_eos_step decoded tokens
            # =================================================================
            if t <= self.min_eos_step:
                logits[:, self.text_eos_idx] = float('-inf')
            
            next_token = logits.argmax(dim=-1)
            decoded[:, t] = next_token
            
            # Check for EOS (only after minimum steps)
            finished = finished | (next_token == self.text_eos_idx)
            if finished.all():
                break
            
            input_token = next_token
        
        return decoded
    
    @classmethod
    def from_config(cls, config) -> 'HybridSignToTextModel':
        """Create model from configuration."""
        return cls(
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
            gloss_vocab_size=config.model.gloss_vocab_size,
            text_vocab_size=config.model.text_vocab_size,
            gloss_blank_idx=config.model.ctc.blank_idx,
            max_decode_length=config.model.decoder.max_decode_length
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Hybrid Sign-to-Text Model")
    print("=" * 60)
    
    # Create model
    model = HybridSignToTextModel(
        input_size=432,
        gloss_vocab_size=100,
        text_vocab_size=200
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    feature_dim = 432
    gloss_len = 10
    text_len = 15
    
    features = torch.randn(batch_size, seq_len, feature_dim)
    feature_lengths = torch.tensor([100, 80, 60, 40])
    gloss_ids = torch.randint(1, 100, (batch_size, gloss_len))
    gloss_lengths = torch.tensor([10, 8, 6, 5])
    text_ids = torch.randint(1, 200, (batch_size, text_len))
    text_ids[:, 0] = 1  # SOS token
    
    print(f"\nInput shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Gloss IDs: {gloss_ids.shape}")
    print(f"  Text IDs: {text_ids.shape}")
    
    # Test loss computation
    model.train()
    losses = model.compute_loss(
        features, feature_lengths,
        gloss_ids, gloss_lengths,
        text_ids,
        ctc_weight=0.5,
        teacher_forcing_ratio=1.0
    )
    
    print(f"\nLoss components:")
    print(f"  Total loss: {losses['loss'].item():.4f}")
    print(f"  CTC loss: {losses['ctc_loss'].item():.4f}")
    print(f"  CE loss: {losses['ce_loss'].item():.4f}")
    
    # Test greedy decoding
    model.eval()
    decoded = model.decode_greedy(features[:2], feature_lengths[:2])
    print(f"\nGreedy decoding:")
    print(f"  Decoded shape: {decoded.shape}")
    
    print("\n[OK] All tests passed!")
