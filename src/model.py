"""
GPT Model Implementation from Scratch.
Implements a decoder-only transformer architecture similar to GPT-2.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import config


class SelfAttention(nn.Module):
    """
    Single head of self-attention.
    Implements the scaled dot-product attention mechanism.
    """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        
        # Lower triangular matrix for causal masking (prevents looking ahead)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B, T, C = x.shape  # Batch, Time (sequence length), Channels (embedding dim)
        
        # Compute key, query, value
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        
        # Compute attention scores (affinities)
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        
        # Apply causal mask (prevent attending to future tokens)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax to get attention weights
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Apply attention to values
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention running in parallel.
    Allows the model to attend to different aspects of the input simultaneously.
    """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # Run all attention heads in parallel and concatenate results
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply projection and dropout
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    A simple MLP applied to each position independently.
    """
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # Expand
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),  # Project back
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single transformer block.
    Combines multi-head attention and feed-forward network with residual connections.
    """
    
    def __init__(self):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config.n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x):
        # Apply self-attention with residual connection (pre-norm formulation)
        x = x + self.sa(self.ln1(x))
        # Apply feed-forward with residual connection
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """
    The main GPT model.
    A decoder-only transformer that predicts the next token in a sequence.
    """
    
    def __init__(self, vocab_size):
        super().__init__()
        
        # Token embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        # Position embedding table
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Language modeling head (projects to vocabulary)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        print(f"Model initialized with {self.count_parameters():,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights with normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        Forward pass through the model.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices for loss calculation (B, T)
        
        Returns:
            logits: Predicted token logits (B, T, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        B, T = idx.shape
        
        # Get token embeddings (B, T, C)
        tok_emb = self.token_embedding_table(idx)
        # Get position embeddings (T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        # Combine token and position embeddings
        x = tok_emb + pos_emb  # (B, T, C)
        
        # Pass through transformer blocks
        x = self.blocks(x)  # (B, T, C)
        
        # Final layer norm
        x = self.ln_f(x)  # (B, T, C)
        
        # Project to vocabulary size
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets are provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
        
        Returns:
            Generated token indices (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -config.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature  # (B, C)
            
            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        
        return idx
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test of the model
    print("Testing GPT model...")
    
    vocab_size = 65  # Typical for character-level
    model = GPTModel(vocab_size)
    
    # Create dummy input
    batch_size = 4
    seq_len = 32
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits, loss = model(idx, targets=idx)
    
    print(f"Input shape: {idx.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss.item() if loss is not None else 'N/A'}")
    
    # Test generation
    generated = model.generate(idx[:1, :10], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
    
    print("\n✓ Model test passed!")
