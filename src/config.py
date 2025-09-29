"""
Configuration file for the LLM model.
Adjust these hyperparameters based on your hardware and requirements.
"""

# ==========================
# MODEL ARCHITECTURE
# ==========================
n_embd = 384        # Embedding dimension (size of each token's vector representation)
n_head = 6          # Number of attention heads (must divide n_embd evenly)
n_layer = 6         # Number of transformer blocks (depth of the model)
block_size = 256    # Maximum context length (number of tokens the model can see)
dropout = 0.2       # Dropout rate for regularization (0.0 = no dropout)

# ==========================
# TRAINING HYPERPARAMETERS
# ==========================
batch_size = 64         # Number of sequences to process in parallel
learning_rate = 3e-4    # Learning rate for the optimizer
max_iters = 5000        # Total number of training iterations
eval_interval = 500     # How often to evaluate on validation set
eval_iters = 200        # Number of iterations for evaluation
warmup_iters = 100      # Learning rate warmup iterations

# ==========================
# DATA
# ==========================
train_split = 0.9       # Fraction of data to use for training (rest is validation)
data_path = 'data/input.txt'  # Path to training data

# ==========================
# GENERATION
# ==========================
temperature = 0.8       # Sampling temperature (higher = more random, lower = more deterministic)
top_k = 200            # Top-k sampling (0 = disabled)
max_new_tokens = 500   # Maximum number of tokens to generate

# ==========================
# SYSTEM
# ==========================
device = 'mps'         # 'mps' for Apple Silicon, 'cuda' for NVIDIA, 'cpu' for CPU
seed = 1337            # Random seed for reproducibility
checkpoint_dir = 'checkpoints'  # Directory to save model checkpoints

# ==========================
# HARDWARE-SPECIFIC CONFIGS
# ==========================
# These are recommended settings for different hardware configurations.
# Uncomment the one that matches your setup:

# --- MacBook Pro M4 16GB (your current setup) ---
# (Already configured above - good for ~10-30M parameter models)

# --- MacBook Pro 32GB RAM ---
# n_embd = 768
# n_head = 12
# n_layer = 12
# block_size = 512
# batch_size = 128

# --- Mac Studio 64GB+ RAM ---
# n_embd = 1024
# n_head = 16
# n_layer = 24
# block_size = 1024
# batch_size = 256

# ==========================
# MODEL SIZE ESTIMATION
# ==========================
def estimate_params():
    """Estimate the number of parameters in the model."""
    vocab_size = 65  # Approximate for character-level tokenizer
    
    # Token + position embeddings
    emb_params = vocab_size * n_embd + block_size * n_embd
    
    # Each transformer block has:
    # - Attention: 4 * n_embd * n_embd (Q, K, V, projection)
    # - FFN: 2 * n_embd * (4 * n_embd) 
    # - Layer norms: ~4 * n_embd
    block_params = (4 * n_embd * n_embd) + (2 * n_embd * 4 * n_embd) + (4 * n_embd)
    
    # Final layer norm and output projection
    final_params = vocab_size * n_embd + 2 * n_embd
    
    total = emb_params + (n_layer * block_params) + final_params
    return total

if __name__ == '__main__':
    params = estimate_params()
    print(f"Estimated model parameters: {params:,} ({params/1e6:.1f}M)")
    print(f"\nCurrent configuration:")
    print(f"  Embedding dim: {n_embd}")
    print(f"  Attention heads: {n_head}")
    print(f"  Transformer layers: {n_layer}")
    print(f"  Context length: {block_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Training iterations: {max_iters}")
