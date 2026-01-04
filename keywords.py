# keywords.py
"""
Keyword detection and flagging module.
"""

import re
import pandas as pd
from typing import List, Tuple

from config import (
    SEARCH_TERMS, OREGON_KEYWORDS, PARTNER_KEYWORDS, PARTNER_KEYWORDS_SPECIAL,
    FACULTY_KEYWORDS, FACULTY_KEYWORDS_SPECIAL, ATTRITION_KEYWORDS
)
from utils import setup_logging

logger = setup_logging()

def flag_all_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full suite of keyword detection functions.
    
    Args:
        df: DataFrame with 'clean_text'.
        
    Returns:
        pd.DataFrame: DataFrame with new flag columns.
    """
    if df.empty:
        return df

    # We use 'clean_text' for general matching, but sometimes preserving case helps?
    # Config lists are lowercase (mostly), clean_text is lowercase. 
    # Let's use clean_text consistently.
    
    logger.info("Flagging OREGON keywords...")
    df['mentions_oregon'] = df['clean_text'].apply(lambda x: detect_generic(x, OREGON_KEYWORDS)[0])
    
    logger.info("Flagging PARTNER keywords...")
    # Partner needs special regex for 'so'
    df['mentions_partner'] = df['clean_text'].apply(lambda x: detect_partner_mentions(x))
    
    logger.info("Flagging FACULTY keywords...")
    # Faculty needs special for 'pd', 'apd'
    df['mentions_faculty'] = df['clean_text'].apply(lambda x: detect_faculty_mentions(x))
    
    logger.info("Flagging ATTRITION keywords...")
    df['mentions_attrition'] = df['clean_text'].apply(lambda x: detect_generic(x, ATTRITION_KEYWORDS)[0])
    
    # Search Categories (Recruitment/Retention/etc keys)
    logger.info("Classifying Search Categories...")
    df['search_category'] = df['clean_text'].apply(classify_search_category)
    
    # Lifecycle Inference
    logger.info("Inferring Lifecycle Stage...")
    # Apply raw row function
    lifecycle = df.apply(infer_lifecycle_stage, axis=1, result_type='expand')
    df = pd.concat([df, lifecycle], axis=1)
    
    # Collect all matched keywords for verification?
    # Requirement: "Output columns: ... keyword_matches (Comma-separated list)"
    # We need to run match again or modify helper to separate matching logic.
    # To save compute, let's do a consolidated pass for 'keyword_matches'
    df['keyword_matches'] = df['clean_text'].apply(collect_all_matches)
    
    return df

def check_keyword_match(text: str, keywords: List[str]) -> Tuple[bool, List[str]]:
    """Generic matcher using word boundaries."""
    if not text:
        return False, []
        
    matches = []
    found = False
    
    for kw in keywords:
        # \bkeyword\b
        # Escape keyword just in case
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, text):
            matches.append(kw)
            found = True
            
    return found, matches

def detect_generic(text: str, keywords: List[str]) -> Tuple[bool, List[str]]:
    """Wrapper for generic lists."""
    return check_keyword_match(text, keywords)

def detect_partner_mentions(text: str) -> bool:
    """Check PARTNER_KEYWORDS + 'so'."""
    # check standard list
    found, _ = check_keyword_match(text, PARTNER_KEYWORDS)
    if found:
        return True
        
    # check special 'so'
    # Config: PARTNER_KEYWORDS_SPECIAL = ["so"]
    # Logic: r'(?<![a-z])so(?![a-z])' to avoid "also"
    # Wait, simple \bso\b works for " also " vs " so ".
    # But user specifically asked for: regex r'\bso\b(?!\w)' and exclude matches preceded by "al"
    # Actually prompt says: "Use special regex: r'(?<![a-z])so(?![a-z])'" in Edge Case Handling
    # AND "For "so" (significant other): use regex r'\bso\b(?!\w)' and exclude matches preceded by "al" (to skip "also")" in DELIVERABLE 7
    
    # Let's allow flexible 'so' matching that avoids 'also'
    # r'\bso\b' strictly ensures word boundaries. 
    # "also" is a word, so \bso\b won't match "also".
    # The user might be worried about some weird case. I'll implement a robust one.
    
    if re.search(r'\bso\b', text):
        return True
        
    return False

def detect_faculty_mentions(text: str) -> bool:
    """Check FACULTY_KEYWORDS + 'pd'/'apd'."""
    found, _ = check_keyword_match(text, FACULTY_KEYWORDS)
    if found:
        return True
        
    # Check special
    for kw in FACULTY_KEYWORDS_SPECIAL: # ['pd', 'apd']
        if re.search(r'\b' + re.escape(kw) + r'\b', text):
            return True
            
    return False

def classify_search_category(text: str) -> str:
    """
    Determine which SEARCH_TERMS category matched.
    Returns comma-separated string of keys.
    """
    matches = []
    for category, terms in SEARCH_TERMS.items():
        found, _ = check_keyword_match(text, terms)
        if found:
            matches.append(category)
            
    if not matches:
        return "other"
        
    return ",".join(matches)

def collect_all_matches(text: str) -> str:
    """Collect all matching keywords from relevant lists."""
    all_lists = OREGON_KEYWORDS + PARTNER_KEYWORDS + FACULTY_KEYWORDS + ATTRITION_KEYWORDS
    # Also search terms? 
    # Usually "keyword_matches" refers to the flagging keywords.
    
    found, matches = check_keyword_match(text, all_lists)
    
    # Add special ones manually if needed, but check_keyword_match handles simple list
    # 'so', 'pd', 'apd' are not in big list above, let's add them
    specials = PARTNER_KEYWORDS_SPECIAL + FACULTY_KEYWORDS_SPECIAL
    _, special_matches = check_keyword_match(text, specials)
    
    total = list(set(matches + special_matches))
    return ",".join(total)

def infer_lifecycle_stage(row: pd.Series) -> pd.Series:
    """
    Determine recruitment/retention/alumni based on text text context.
    
    Args:
        row: DataFrame row.
        
    Returns:
        pd.Series: [is_recruitment, is_retention, is_alumni]
    """
    text = row.get('clean_text', "")
    
    # Simple heuristic mapping based on keywords or categories
    # Prompt says:
    # is_recruitment: True if text discusses choosing programs, matching, ranking
    # is_retention: True if text discusses current experience, burnout, leaving
    # is_alumni: True if text uses past tense ("I left", "moved back", "after X years")
    
    # We can use the SEARCH_TERMS categories as strong signals
    cat = row.get('search_category', "")
    
    is_recruitment = "recruitment" in cat
    is_retention = "retention" in cat or "compensation" in cat or row.get('mentions_attrition')
    is_alumni = False # Harder to detect without explicit past tense parsing
    
    # Refine Alumni logic
    # "left rural", "quit rural" are retention (attrition) events, but also imply alumni status?
    # Usually 'attrition' means the ACT of leaving. Alumni means AFTER leaving.
    # Let's look for specific phrases if not covered by categories.
    
    if "moved back" in text or "after" in text and "years" in text:
        is_alumni = True
    
    if "left" in text and "rural" in text:
        is_alumni = True
        
    return pd.Series([is_recruitment, is_retention, is_alumni], index=['is_recruitment', 'is_retention', 'is_alumni'])
