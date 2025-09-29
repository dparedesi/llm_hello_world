# LLM: Hello World 🚀

A minimal GPT-style language model built from scratch using PyTorch. This project demonstrates how to build, train, and generate text with a transformer-based language model - perfect for learning and experimentation!

## 🎯 Project Overview

This project implements a decoder-only transformer architecture (similar to GPT-2) from scratch. It's designed to be:
- **Educational**: Clear, readable code with comments
- **Minimal**: ~500 lines of core code
- **Practical**: Actually trains and generates text
- **Optimized for Apple Silicon**: Uses MPS backend for M-series chips

## 🏗️ Architecture

The model implements the following components:

1. **Tokenizer** (`src/tokenizer.py`): Character-level tokenization
2. **Model** (`src/model.py`):
   - Multi-head self-attention
   - Feedforward neural networks
   - Layer normalization
   - Positional embeddings
   - Transformer blocks stacked into GPT
3. **Training** (`src/train.py`): Training loop with gradient descent
4. **Generation** (`src/generate.py`): Text generation with sampling strategies

## 📋 Requirements

- **Hardware**: MacBook Pro M1/M2/M3/M4 with 16GB+ RAM (recommended)
- **Software**: Python 3.9+, PyTorch 2.0+

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your training text in `data/input.txt`. We've included a sample Shakespeare dataset to get started.

### 3. Train the Model

```bash
python src/train.py
```

Training will:
- Load and tokenize your data
- Train the model (progress shown every 100 iterations)
- Save checkpoints to `checkpoints/`
- Take 10-30 minutes depending on your settings

### 4. Generate Text

```bash
python src/generate.py --prompt "Hello, world!" --max_tokens 200
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Model architecture
n_embd = 384          # Embedding dimension
n_head = 6            # Number of attention heads
n_layer = 6           # Number of transformer blocks
block_size = 256      # Context length

# Training
batch_size = 64       # Batch size
learning_rate = 3e-4  # Learning rate
max_iters = 5000      # Training iterations
```

### Recommended Configurations by Hardware

**MacBook Pro 16GB RAM** (your setup):
- Model size: 10-30M parameters
- batch_size: 64
- block_size: 256
- n_layer: 6, n_head: 6, n_embd: 384

**MacBook Pro 32GB RAM**:
- Model size: 50-100M parameters  
- batch_size: 128
- block_size: 512
- n_layer: 12, n_head: 12, n_embd: 768

## 📁 Project Structure

```
llm_hello_world/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── input.txt          # Your training data
├── src/
│   ├── config.py          # Hyperparameters
│   ├── tokenizer.py       # Text tokenization
│   ├── model.py           # Transformer architecture
│   ├── train.py           # Training loop
│   ├── generate.py        # Text generation
│   └── utils.py           # Helper functions
├── checkpoints/           # Saved models
└── examples/
    └── quick_demo.py      # End-to-end example
```

## 🎓 Learning Path

If you're new to LLMs, explore the code in this order:

1. **config.py** - Understand the hyperparameters
2. **tokenizer.py** - See how text becomes numbers
3. **model.py** - Study the transformer architecture
   - Start with `SelfAttention`
   - Then `FeedForward`
   - Then `TransformerBlock`
   - Finally `GPTModel`
4. **train.py** - Learn the training loop
5. **generate.py** - See how text generation works

## 🔬 Experiments to Try

1. **Increase model size**: Double `n_embd`, `n_head`, `n_layer`
2. **Train longer**: Increase `max_iters`
3. **Different data**: Try different text sources
4. **Temperature sampling**: Adjust temperature in generation
5. **Learning rate**: Experiment with different learning rates

## 📊 Expected Results

With default settings on Shakespeare text:
- **Training time**: ~15-20 minutes on M4
- **Model size**: ~10M parameters
- **Loss**: Should drop from ~4.0 to ~1.5
- **Generated text**: Will be coherent but not perfect

The model will learn grammar, structure, and style. Don't expect GPT-4 quality - this is for learning!

## 🐛 Troubleshooting

**Out of Memory?**
- Reduce `batch_size` or `block_size`
- Reduce `n_embd`, `n_head`, or `n_layer`

**Training too slow?**
- Verify MPS (Apple Silicon GPU) is being used
- Check that PyTorch is properly installed

**Poor generation quality?**
- Train longer (increase `max_iters`)
- Increase model size
- Use more training data

## 📚 Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Inspiration for this project
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## 📝 License

MIT License - feel free to use this for learning and experimentation!

## 🙏 Acknowledgments

This project is inspired by Andrej Karpathy's nanoGPT and countless educational resources on transformers.

---

**Happy Learning! 🎉**

If you build something cool with this, share it with the community!
