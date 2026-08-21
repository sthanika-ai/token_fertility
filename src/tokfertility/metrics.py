"""Fertility / parity / chars-per-token metrics. Pure functions - no I/O.

Word count is a whitespace split. Hindi, Telugu, Kannada, and Marathi all use
spaces between words, so this is a reasonable proxy - but all four are more
agglutinative than English, so a single orthographic "word" can carry more
morphology. Treat fertility as a relative, not absolute, comparison.
"""

import math


def word_count(text: str) -> int:
    return len(text.split())


def char_count(text: str) -> int:
    return len(text)


def fertility(token_count: int, text: str) -> float:
    """Tokens per word - lower means the tokenizer represents this language
    more efficiently."""
    words = word_count(text)
    return token_count / words if words else math.nan


def chars_per_token(token_count: int, text: str) -> float:
    return char_count(text) / token_count if token_count else math.nan


def parity_vs_english(token_count_lang: int, token_count_en: int) -> float:
    """>1 means this sentence needs proportionally more tokens than its
    English counterpart; 1.0 is parity."""
    return token_count_lang / token_count_en if token_count_en else math.nan
