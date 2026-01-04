# pii.py
"""
PII detection and redaction module using Microsoft Presidio.
"""

import sys
from typing import Tuple, List, Dict
import pandas as pd
from datetime import datetime

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from config import PII_ENTITIES, PII_WHITELIST
from utils import setup_logging

logger = setup_logging()

# Global engines
_ANALYZER = None
_ANONYMIZER = None

def initialize_presidio() -> Tuple[AnalyzerEngine, AnonymizerEngine]:
    """
    Initialize and return Presidio engines.
    
    Returns:
        tuple: (AnalyzerEngine, AnonymizerEngine)
    """
    global _ANALYZER, _ANONYMIZER
    
    if _ANALYZER is None:
        try:
            # Requires 'en_core_web_sm' or similar spaCy model
            _ANALYZER = AnalyzerEngine()
            _ANONYMIZER = AnonymizerEngine()
        except OSError:
            logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise

    return _ANALYZER, _ANONYMIZER

def detect_and_redact_pii(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect and redact PII in the DataFrame.
    
    Args:
        df: Input DataFrame with 'original_text'.
        
    Returns:
        tuple: (processed_df, audit_log_df)
    """
    analyzer, anonymizer = initialize_presidio()
    
    audit_records = []
    redacted_texts = []
    pii_detected_flags = []
    pii_types_list = []
    pii_counts = []
    
    for _, row in df.iterrows():
        post_id = row['id']
        text = row['original_text']
        
        redacted_text, detections = redact_text(text, analyzer, anonymizer)
        
        # Log Audit
        for d in detections:
            audit_entry = create_audit_entry(post_id, d)
            audit_records.append(audit_entry)
            
        # Collect Row Stats
        detected_types = list(set([d.entity_type for d in detections]))
        
        redacted_texts.append(redacted_text)
        pii_detected_flags.append(len(detections) > 0)
        pii_types_list.append(",".join(detected_types) if detected_types else "")
        pii_counts.append(len(detections))
        
    # Update DataFrame
    df['redacted_text'] = redacted_texts
    df['pii_detected'] = pii_detected_flags
    df['pii_types'] = pii_types_list
    df['pii_count'] = pii_counts
    
    # Create Audit DataFrame
    if audit_records:
        audit_df = pd.DataFrame(audit_records)
    else:
        # Empty schema
        audit_df = pd.DataFrame(columns=['post_id', 'entity_type', 'start_position', 'end_position', 'confidence', 'timestamp'])
        
    return df, audit_df

def is_whitelisted(text: str) -> bool:
    """
    Check if text segment matches whitelist.
    
    Args:
        text: Text to check.
        
    Returns:
        bool: True if whitelisted.
    """
    if not text:
        return False
        
    text_lower = text.lower()
    for item in PII_WHITELIST:
        if item.lower() in text_lower:
            return True
            
    return False

def redact_text(text: str, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine) -> Tuple[str, List[RecognizerResult]]:
    """
    Process single text logic: Detect -> Filter Whitelist -> Redact.
    
    Args:
        text: Input text.
        analyzer: Presidio analyzer.
        anonymizer: Presidio anonymizer.
        
    Returns:
        tuple: (redacted_str, list_of_detections)
    """
    if not text or not isinstance(text, str):
        return "", []
        
    # 1. Analyze
    results = analyzer.analyze(
        text=text,
        entities=PII_ENTITIES,
        language='en'
    )
    
    # 2. Filter Whitelisted Results
    valid_results = []
    for res in results:
        entity_text = text[res.start:res.end]
        if not is_whitelisted(entity_text):
            valid_results.append(res)
            
    # 3. Redact
    if not valid_results:
        return text, []
        
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=valid_results,
        operators={
             "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
        }
    )
    
    return anonymized_result.text, valid_results

def create_audit_entry(post_id: str, detection: RecognizerResult) -> dict:
    """
    Create a safe audit log entry without original values.
    
    Args:
        post_id: ID of the post.
        detection: Presidio result object.
        
    Returns:
        dict: Audit entry.
    """
    return {
        "post_id": post_id,
        "entity_type": detection.entity_type,
        "start_position": detection.start,
        "end_position": detection.end,
        "confidence": detection.score,
        "timestamp": datetime.now().isoformat()
    }
