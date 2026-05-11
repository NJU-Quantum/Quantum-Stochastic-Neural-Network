"""
Compatibility shim.

Legacy code may import `SimpleTokenizer` from this file.
For the maintained tokenizer implementations, use `lm_tokenizer.py`.
"""

from lm_tokenizer import CharTokenizer, WordPunctTokenizer, BPETokenizer, create_tokenizer


class SimpleTokenizer(WordPunctTokenizer):
    """
    Backward-compatible alias.
    Uses punctuation-aware word tokenization by default.
    """

    pass

