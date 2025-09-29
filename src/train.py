"""
Training script for the GPT model.
This script loads data, initializes the model, and trains it.
"""

import os
import torch
from tqdm import tqdm

import config
from tokenizer import CharTokenizer
from model import GPTModel
from utils import load_data, get_batch, estimate_loss, save_checkpoint, get_lr


def train():
    """Main training function."""
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    
    # Determine device
    if config.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = 'cpu'
    elif config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        device = config.device
    
    print(f"Using device: {device}")
    
    # Load and prepare data
    print(f"\nLoading data from {config.data_path}...")
    text = load_data(config.data_path)
    print(f"Loaded {len(text):,} characters")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = CharTokenizer(text)
    
    # Encode the entire text
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Data encoded to {len(data):,} tokens")
    
    # Split into train and validation
    n = int(config.train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train set: {len(train_data):,} tokens")
    print(f"Val set: {len(val_data):,} tokens")
    
    # Initialize model
    print("\nInitializing model...")
    model = GPTModel(tokenizer.vocab_size)
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {config.max_iters} iterations")
    print(f"{'='*60}\n")
    
    model.train()
    best_val_loss = float('inf')
    
    # Progress bar
    pbar = tqdm(range(config.max_iters), desc="Training")
    
    for iter_num in pbar:
        # Update learning rate
        lr = get_lr(iter_num, config.warmup_iters, config.learning_rate, config.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get a batch of data
        xb, yb = get_batch(train_data, config.batch_size, config.block_size, device)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.6f}'})
        
        # Evaluate and save checkpoint periodically
        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config.eval_iters, device)
            
            print(f"\n[Iter {iter_num}] train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
                save_checkpoint(model, optimizer, iter_num, losses['val'], checkpoint_path)
            
            # Save regular checkpoint
            if iter_num % (config.eval_interval * 2) == 0:
                checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_iter_{iter_num}.pt')
                save_checkpoint(model, optimizer, iter_num, losses['val'], checkpoint_path)
    
    # Save final model
    final_path = os.path.join(config.checkpoint_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, config.max_iters, loss.item(), final_path)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*60}\n")
    
    # Generate a sample
    print("Generating sample text...")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_ids = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=200)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    print("\nSample generation:")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)


if __name__ == '__main__':
    train()
