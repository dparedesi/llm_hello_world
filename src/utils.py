"""
Utility functions for data loading and batch generation.
"""

import torch
import config


def load_data(file_path):
    """
    Load text data from a file.
    
    Args:
        file_path (str): Path to the text file.
    
    Returns:
        str: The text content.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_batch(data, batch_size, block_size, device='cpu'):
    """
    Generate a random batch of training data.
    
    Args:
        data (torch.Tensor): The tokenized data.
        batch_size (int): Number of sequences per batch.
        block_size (int): Length of each sequence.
        device (str): Device to place the tensors on.
    
    Returns:
        tuple: (inputs, targets) where both are (batch_size, block_size) tensors.
    """
    # Generate random starting positions for sequences
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Extract sequences and their targets (shifted by 1)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Move to device
    x, y = x.to(device), y.to(device)
    
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, device):
    """
    Estimate the loss on train and validation sets.
    
    Args:
        model: The model to evaluate.
        train_data (torch.Tensor): Training data.
        val_data (torch.Tensor): Validation data.
        eval_iters (int): Number of iterations to average over.
        device (str): Device to run evaluation on.
    
    Returns:
        dict: Dictionary with 'train' and 'val' loss values.
    """
    out = {}
    model.eval()
    
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, config.batch_size, config.block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out


def save_checkpoint(model, optimizer, iter_num, loss, filepath):
    """
    Save a model checkpoint.
    
    Args:
        model: The model to save.
        optimizer: The optimizer state.
        iter_num (int): Current iteration number.
        loss (float): Current loss value.
        filepath (str): Path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter_num': iter_num,
        'loss': loss,
        'config': {
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_layer': config.n_layer,
            'block_size': config.block_size,
            'dropout': config.dropout,
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load a model checkpoint.
    
    Args:
        filepath (str): Path to the checkpoint file.
        model: The model to load weights into.
        optimizer: Optional optimizer to load state into.
    
    Returns:
        dict: Checkpoint information.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"  Iteration: {checkpoint.get('iter_num', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    
    return checkpoint


def get_lr(iter_num, warmup_iters, learning_rate, max_iters):
    """
    Learning rate schedule with warmup and cosine decay.
    
    Args:
        iter_num (int): Current iteration.
        warmup_iters (int): Number of warmup iterations.
        learning_rate (float): Maximum learning rate.
        max_iters (int): Total number of iterations.
    
    Returns:
        float: Learning rate for this iteration.
    """
    # Linear warmup
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    
    # Cosine decay after warmup
    if iter_num > max_iters:
        return learning_rate * 0.1
    
    decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * decay_ratio)))
    return learning_rate * 0.1 + coeff * (learning_rate - learning_rate * 0.1)


if __name__ == '__main__':
    # Test data loading utilities
    print("Testing data utilities...")
    
    # Test batch generation
    dummy_data = torch.randint(0, 100, (1000,))
    x, y = get_batch(dummy_data, batch_size=4, block_size=8)
    
    print(f"Batch input shape: {x.shape}")
    print(f"Batch target shape: {y.shape}")
    print(f"Sample input: {x[0]}")
    print(f"Sample target: {y[0]}")
    
    # Test learning rate schedule
    lrs = [get_lr(i, 100, 3e-4, 1000) for i in range(0, 1100, 100)]
    print(f"\nLearning rate schedule (every 100 iters): {[f'{lr:.6f}' for lr in lrs]}")
    
    print("\n✓ Utils test passed!")
