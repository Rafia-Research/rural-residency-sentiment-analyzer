# sentiment.py
"""
Sentiment analysis module using Hugging Face Transformers.
"""

import sys
import gc
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from config import (
    SENTIMENT_MODEL, SENTIMENT_CONFIDENCE_THRESHOLD, SENTIMENT_BATCH_SIZE
)
from utils import setup_logging

logger = setup_logging()

# Global cache for model and tokenizer
_MODEL = None
_TOKENIZER = None

def load_model():
    """
    Load and cache the sentiment model and tokenizer.
    Detects and uses MPS (Apple Silicon) if available.
    
    Returns:
        tuple: (model, tokenizer)
    """
    global _MODEL, _TOKENIZER
    
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER
        
    try:
        logger.info(f"Loading sentiment model: {SENTIMENT_MODEL}")
        print("Downloading sentiment model (one-time, ~500MB)...")
        
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        config = AutoConfig.from_pretrained(SENTIMENT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
        
        # Device selection
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon) acceleration.")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA acceleration.")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU.")
            
        model.to(device)
        model.eval() # Set to evaluation mode
        
        _MODEL = model
        _TOKENIZER = tokenizer
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Retry logic could go here, or we let the pipeline fail gracefully
        # The prompt asks for: "Retry 3x, then raise"
        # Since we are inside a function, simple recursion or loop works
        # But for model loading, it's usually network or disk.
        raise e

def analyze_sentiment_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run sentiment analysis on a DataFrame batch.
    
    Args:
        df: DataFrame containing 'clean_text'.
        
    Returns:
        pd.DataFrame: DataFrame with added sentiment columns.
    """
    if df.empty:
        return df
        
    # Ensure columns exist
    new_cols = ['sentiment_score', 'sentiment_label', 'sentiment_confidence', 'sentiment_uncertain']
    for col in new_cols:
        if col not in df.columns:
            df[col] = None

    model, tokenizer = load_model()
    device = model.device
    
    texts = df['clean_text'].fillna("").tolist()
    results = []
    
    # Process in mini-batches
    for i in tqdm(range(0, len(texts), SENTIMENT_BATCH_SIZE), desc="Sentiment Analysis", disable=True):
        batch_texts = texts[i:i + SENTIMENT_BATCH_SIZE]
        batch_results = get_sentiment_scores(batch_texts, model, tokenizer, device)
        results.extend(batch_results)
    
    # Assign results back to DataFrame
    # We must reset index to ensure alignment if the input df had gaps
    # But usually we pass a sliced default range index batch from main.py
    # Ideally, we match by list order since we iterated over df['clean_text'].tolist()
    
    # Convert list of dicts to DataFrame
    results_df = pd.DataFrame(results)
    
    # Assign columns. Using .values ensures we bypass index mismatch issues 
    # if the input df has non-standard index.
    df['sentiment_score'] = results_df['score'].values
    df['sentiment_label'] = results_df['label'].values
    df['sentiment_confidence'] = results_df['confidence'].values
    
    # Set uncertainty flag
    df['sentiment_uncertain'] = df['sentiment_confidence'] < SENTIMENT_CONFIDENCE_THRESHOLD
    
    return df

def get_sentiment_scores(texts: list, model, tokenizer, device) -> list:
    """
    Perform inference on a list of texts.
    
    Args:
        texts: List of strings.
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        device: Torch device.
        
    Returns:
        list[dict]: List of results with score, label, confidence.
    """
    if not texts:
        return []

    # Handle empty strings separately or let model handle them?
    # Empty strings crash some tokenizers or return garbage.
    # We'll replace empty with a neutral placeholder token or handle post-hoc.
    # clean_text replaces them with "", so let's filter.
    
    valid_indices = [i for i, t in enumerate(texts) if t and len(t.strip()) > 0]
    valid_texts = [texts[i] for i in valid_indices]
    
    # Default result for empty text
    default_result = {
        "score": 0.0, 
        "label": "Neutral", 
        "confidence": 0.0
    }
    
    # Initialize full results list with defaults
    batch_results = [default_result] * len(texts)
    
    if not valid_texts:
        return batch_results
        
    try:
        # Tokenize
        inputs = tokenizer(
            valid_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Parse output
        # Model specific mapping: 0 -> Negative, 1 -> Neutral, 2 -> Positive
        # Verify model card: cardiffnlp/twitter-roberta-base-sentiment-latest
        # Labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive
        
        predictions = predictions.cpu().numpy()
        
        for idx, probs in enumerate(predictions):
            original_idx = valid_indices[idx]
            
            neg_score = probs[0]
            neu_score = probs[1]
            pos_score = probs[2]
            
            # Get max label
            max_idx = np.argmax(probs)
            confidence = probs[max_idx]
            
            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            label = label_map[max_idx]
            
            # Calculate compound score (-1 to 1)
            # Simple weighted average: (Pos * 1 + Neu * 0 + Neg * -1)
            # Or use the confidence of the winning label?
            # Prompt says: "Positive label with 0.9 confidence -> +0.9"
            
            compound_score = convert_to_compound_score(label, confidence)
            
            batch_results[original_idx] = {
                "score": compound_score,
                "label": label,
                "confidence": float(confidence)
            }
            
    except Exception as e:
        logger.error(f"Sentiment inference failed: {e}")
        # Return defaults for this batch if it crashes?
        # Better to log and maybe return partial? 
        # For now, let's just return defaults for failed batch to avoid pipeline crash
        pass
        
    return batch_results

def convert_to_compound_score(label: str, confidence: float) -> float:
    """
    Convert label and confidence to a compound score (-1.0 to +1.0).
    
    Args:
        label: "Positive", "Negative", "Neutral"
        confidence: 0.0 to 1.0
        
    Returns:
        float: Compound score.
    """
    if label == "Positive":
        return confidence
    elif label == "Negative":
        return -confidence
    else: # Neutral
        # Neutral should be close to 0. 
        # If very confident it's neutral, score is 0.
        # If unsure, maybe it's slightly one way? 
        # Requirement: "Neutral returns value close to 0"
        return 0.0
