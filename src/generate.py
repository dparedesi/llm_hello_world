"""
Text generation script.
Load a trained model and generate text from a prompt.
"""

import os
import argparse
import torch

import config
from tokenizer import CharTokenizer
from model import GPTModel
from utils import load_data, load_checkpoint


def generate_text(
    prompt="",
    max_tokens=500,
    temperature=0.8,
    top_k=200,
    model_path=None
):
    """
    Generate text using a trained model.
    
    Args:
        prompt (str): Starting text (empty for random generation).
        max_tokens (int): Number of tokens to generate.
        temperature (float): Sampling temperature (higher = more random).
        top_k (int): Top-k sampling parameter (0 = disabled).
        model_path (str): Path to model checkpoint.
    
    Returns:
        str: Generated text.
    """
    # Determine device
    if config.device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    elif config.device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = config.device
    
    print(f"Using device: {device}")
    
    # Load training data to initialize tokenizer
    print("\nLoading tokenizer...")
    text = load_data(config.data_path)
    tokenizer = CharTokenizer(text)
    
    # Initialize model
    print("Initializing model...")
    model = GPTModel(tokenizer.vocab_size)
    
    # Load checkpoint
    if model_path is None:
        # Try to load the best model by default
        model_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
        if not os.path.exists(model_path):
            # Fall back to final model
            model_path = os.path.join(config.checkpoint_dir, 'final_model.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    load_checkpoint(model_path, model)
    model = model.to(device)
    model.eval()
    
    # Encode the prompt
    if prompt:
        print(f"\nPrompt: '{prompt}'")
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
        context = context.unsqueeze(0)  # Add batch dimension
    else:
        print("\nNo prompt provided, generating from scratch...")
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate
    print(f"\nGenerating {max_tokens} tokens...")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    print("\n" + "="*80)
    
    with torch.no_grad():
        generated_ids = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None
        )
    
    # Decode and print
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(generated_text)
    print("="*80 + "\n")
    
    return generated_text


def main():
    """Command-line interface for text generation."""
    parser = argparse.ArgumentParser(description='Generate text using a trained GPT model')
    
    parser.add_argument(
        '--prompt',
        type=str,
        default='',
        help='Starting text for generation (default: empty, random start)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=config.max_new_tokens,
        help=f'Maximum number of tokens to generate (default: {config.max_new_tokens})'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=config.temperature,
        help=f'Sampling temperature (default: {config.temperature})'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=config.top_k,
        help=f'Top-k sampling parameter (default: {config.top_k}, 0 to disable)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model checkpoint (default: best_model.pt or final_model.pt)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1,
        help='Number of independent samples to generate (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Generate multiple samples if requested
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n{'#'*80}")
            print(f"# Sample {i+1}/{args.num_samples}")
            print(f"{'#'*80}\n")
        
        generate_text(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            model_path=args.model
        )


if __name__ == '__main__':
    main()
