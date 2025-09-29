"""
Quick demo script showing end-to-end usage of the LLM.
This script demonstrates:
1. Loading and tokenizing data
2. Creating and inspecting the model
3. Training for a few iterations
4. Generating sample text
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from tokenizer import CharTokenizer
from model import GPTModel
from utils import load_data, get_batch


def quick_demo():
    """Run a quick demonstration of the LLM."""
    
    print("="*80)
    print("LLM HELLO WORLD - QUICK DEMO")
    print("="*80)
    
    # Set device
    device = 'cpu'  # Use CPU for quick demo
    print(f"\nUsing device: {device}")
    
    # Step 1: Load and tokenize data
    print("\n" + "-"*80)
    print("STEP 1: Loading and Tokenizing Data")
    print("-"*80)
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.txt')
    text = load_data(data_path)
    print(f"Loaded {len(text):,} characters")
    print(f"\nFirst 200 characters:")
    print(text[:200])
    
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"\nEncoded to {len(data):,} tokens")
    
    # Step 2: Create model
    print("\n" + "-"*80)
    print("STEP 2: Creating Model")
    print("-"*80)
    
    model = GPTModel(tokenizer.vocab_size)
    print(f"\nModel architecture:")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Embedding dimension: {config.n_embd}")
    print(f"  Number of heads: {config.n_head}")
    print(f"  Number of layers: {config.n_layer}")
    print(f"  Context length: {config.block_size}")
    print(f"  Total parameters: {model.count_parameters():,}")
    
    # Step 3: Quick training demo (just a few iterations)
    print("\n" + "-"*80)
    print("STEP 3: Quick Training Demo (10 iterations)")
    print("-"*80)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    for i in range(10):
        # Get batch
        xb, yb = get_batch(data, batch_size=16, block_size=64, device=device)
        
        # Forward and backward
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if i % 3 == 0:
            print(f"  Iteration {i}: loss = {loss.item():.4f}")
    
    # Step 4: Generate sample text
    print("\n" + "-"*80)
    print("STEP 4: Generating Sample Text (untrained model)")
    print("-"*80)
    
    model.eval()
    prompt = "First Citizen:"
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\nPrompt: '{prompt}'")
    print("\nGenerated text (200 tokens):")
    print("-"*80)
    
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=200, temperature=1.0, top_k=50)
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(generated_text)
    print("-"*80)
    
    # Final notes
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nNOTE: This model is essentially random (only 10 training iterations).")
    print("To train a real model, run: python src/train.py")
    print("This will train for 5000 iterations and produce coherent text.")
    print("\nAfter training, generate text with: python src/generate.py --prompt 'Your prompt'")
    print("="*80 + "\n")


if __name__ == '__main__':
    quick_demo()
