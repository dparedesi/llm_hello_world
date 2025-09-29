"""
Simple character-level tokenizer.
Converts text to integers and back for model input/output.
"""

class CharTokenizer:
    """
    A simple character-level tokenizer that maps each unique character
    to an integer ID and vice versa.
    """
    
    def __init__(self, text):
        """
        Initialize the tokenizer with a text corpus.
        
        Args:
            text (str): The text corpus to build vocabulary from.
        """
        # Get all unique characters and sort them for consistency
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create mappings from characters to integers and vice versa
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Unique characters: {''.join(chars)}")
    
    def encode(self, text):
        """
        Convert a string to a list of integer tokens.
        
        Args:
            text (str): The text to encode.
            
        Returns:
            list: List of integer token IDs.
        """
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, tokens):
        """
        Convert a list of integer tokens back to a string.
        
        Args:
            tokens (list): List of integer token IDs.
            
        Returns:
            str: The decoded text.
        """
        return ''.join([self.idx_to_char[idx] for idx in tokens])
    
    def __len__(self):
        """Return the vocabulary size."""
        return self.vocab_size


if __name__ == '__main__':
    # Quick test of the tokenizer
    sample_text = "Hello, World! This is a test."
    
    tokenizer = CharTokenizer(sample_text)
    
    # Encode the text
    encoded = tokenizer.encode(sample_text)
    print(f"\nOriginal: {sample_text}")
    print(f"Encoded: {encoded}")
    
    # Decode back
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Verify they match
    assert decoded == sample_text, "Encoding/decoding mismatch!"
    print("\n✓ Tokenizer test passed!")
