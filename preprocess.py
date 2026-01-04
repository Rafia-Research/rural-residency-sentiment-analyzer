# preprocess.py
"""
Text preprocessing module for cleaning, language detection, and relevance filtering.
"""

import re
import pandas as pd
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords
from nltk import download

from config import (
    MIN_TEXT_LENGTH, MAX_CHAR_LENGTH, SUPPORTED_LANGUAGES,
    MEDICAL_TERMS, CUSTOM_STOPWORDS
)
from utils import setup_logging

logger = setup_logging()

# Ensure NLTK resources are available
try:
    download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK stopwords: {e}")

# Pre-compile Regex Patterns
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
# HTML entities and Reddit specific characters
REDDIT_FORMATTING = re.compile(r'&[a-z]+;|x200b|\[deleted\]|\[removed\]')
# Keep basic punctuation: . , ! ? '
SPECIAL_CHARS = re.compile(r'[^a-zA-Z0-9\s.,!?\']')
# Multi-whitespace to single
WHITESPACE = re.compile(r'\s+')

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the DataFrame.
    
    Args:
        df: Raw DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned and filtered DataFrame.
    """
    if df.empty:
        return df

    initial_count = len(df)
    
    # 1. Basic cleaning & Language Detection
    # Using apply is slower but necessary for row-by-row logic like language detection
    
    # Pre-calculate to avoid lambda overhead where possible
    df['original_text'] = df['text'].fillna('')
    
    # Filter out empty/deleted immediately to save processing
    df = df[~df['original_text'].isin(['[deleted]', '[removed]', ''])]
    deleted_filter_count = initial_count - len(df)
    logger.info(f"Filtered {deleted_filter_count} rows containing '[deleted]' or '[removed]'")

    # Clean Text
    df['clean_text'] = df['original_text'].apply(clean_text)
    
    # Calculate Length
    df['text_length'] = df['original_text'].str.len()
    
    # Filter Short Text
    df = df[df['text_length'] >= MIN_TEXT_LENGTH]
    length_filter_count = initial_count - deleted_filter_count - len(df)
    logger.info(f"Filtered {length_filter_count} rows shorter than {MIN_TEXT_LENGTH} chars")
    
    # Detect Language
    # Use a cache or just apply directly. For efficiency on large datasets, 
    # we could optimize, but M1 checks out fine for reasonable volumes.
    df['language'] = df['clean_text'].apply(detect_language)
    
    # Filter Language
    df = df[df['language'].isin(SUPPORTED_LANGUAGES)]
    lang_filter_count = initial_count - deleted_filter_count - length_filter_count - len(df)
    logger.info(f"Filtered {lang_filter_count} rows not in {SUPPORTED_LANGUAGES}")

    # Check Medical Relevance
    df['is_relevant'] = df['clean_text'].apply(check_medical_relevance)
    
    # Filter Relevance
    df = df[df['is_relevant'] == True]
    relevance_filter_count = initial_count - deleted_filter_count - length_filter_count - lang_filter_count - len(df)
    logger.info(f"Filtered {relevance_filter_count} rows deemed irrelevant")
    
    # Truncate Long Text (for model input specifically, though we might keep original)
    # The prompt implies we truncate 'clean_text' or maybe a new column?
    # Usually we want full text for manual review but truncated for model.
    # The prompt says: "Truncate longer texts ... Note: Truncation uses character approximation"
    # and "Output columns added: ... clean_text ... original_text"
    # We will truncate 'clean_text' to be safe for the model.
    df['clean_text'] = df['clean_text'].apply(lambda x: truncate_text(x, MAX_CHAR_LENGTH))

    return df

def clean_text(text: str) -> str:
    """
    Clean individual text string.
    
    Args:
        text: Input string.
        
    Returns:
        str: Cleaned string.
    """
    if not isinstance(text, str):
        return ""
        
    # Remove URLs
    text = URL_PATTERN.sub('', text)
    
    # Remove Reddit formatting
    text = REDDIT_FORMATTING.sub('', text)
    
    # Remove special chars (keep basic punctuation)
    text = SPECIAL_CHARS.sub('', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = WHITESPACE.sub(' ', text).strip()
    
    return text

def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from text.
    
    Args:
        text: Input text.
        
    Returns:
        str: Text with stopwords removed.
    """
    if not text:
        return ""
        
    try:
        stops = set(stopwords.words('english'))
        stops.update(CUSTOM_STOPWORDS)
        
        words = text.split()
        filtered_words = [w for w in words if w not in stops]
        return " ".join(filtered_words)
    except Exception:
        # Fallback if NLTK data missing
        return text

def detect_language(text: str) -> str:
    """
    Detect language of the text.
    
    Args:
        text: Input text.
        
    Returns:
        str: ISO language code or 'unknown'.
    """
    if not text or len(text) < 3:
        return "unknown"
        
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def check_medical_relevance(text: str) -> bool:
    """
    Check if text contains any medical keywords.
    
    Args:
        text: Input text (assumed cleaned/lowercase).
        
    Returns:
        bool: True if relevant.
    """
    if not text:
        return False
        
    # Use word boundaries for strict matching
    # We can compile this regex once for efficiency if the list is static
    # Creating a large regex: \b(term1|term2|...)\b
    pattern_str = r'\b(' + '|'.join([re.escape(term) for term in MEDICAL_TERMS]) + r')\b'
    match = re.search(pattern_str, text)
    return bool(match)

def truncate_text(text: str, max_length: int = MAX_CHAR_LENGTH) -> str:
    """
    Smart truncation for long text.
    Keeps beginning and end.
    
    Args:
        text: Input text.
        max_length: Maximum allowed characters.
        
    Returns:
        str: Truncated text.
    """
    if len(text) <= max_length:
        return text
        
    # Keep first 1600 + last 400 = 2000 roughly
    # The prompt specifies "Keep first 1600 characters + last 400 characters"
    
    keep_start = 1600
    keep_end = 400
    
    if max_length != 2000:
        # adjust proportionally if max_length changes via config
        ratio = max_length / 2000
        keep_start = int(1600 * ratio)
        keep_end = int(400 * ratio)

    # Cut carefully - try not to split words?
    # Prompt says: "Never cut in middle of word"
    
    head = text[:keep_start]
    tail = text[-keep_end:]
    
    # Adjust head to end at space
    last_space_head = head.rfind(' ')
    if last_space_head != -1:
        head = head[:last_space_head]
        
    # Adjust tail to start at space
    first_space_tail = tail.find(' ')
    if first_space_tail != -1:
        tail = tail[first_space_tail+1:]
        
    return f"{head} ... {tail}"
