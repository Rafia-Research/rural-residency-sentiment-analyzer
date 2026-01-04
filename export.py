# export.py
"""
CSV export and aggregation module.
"""

import json
from datetime import datetime, timezone
import pandas as pd
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    OUTPUT_DIR, OUTPUT_TIMEZONE, BACKFILL_MONTHS, 
    SENTIMENT_MODEL, ENABLE_BERTOPIC, ENABLE_ROLLING_AVERAGES, ENABLE_GOOGLE_DRIVE_SYNC
)
from utils import setup_logging, convert_timezone, safe_json_save

logger = setup_logging()

def export_all_csvs(df: pd.DataFrame, aggregations: dict, pii_audit: pd.DataFrame, topic_summary: pd.DataFrame) -> None:
    """
    Export all DataFrames to CSVs in OUTPUT_DIR.
    
    Args:
        df: Main DataFrame.
        aggregations: Dictionary of aggregated DataFrames.
        pii_audit: PII audit log.
        topic_summary: Topic summary.
    """
    ensure_timestamp_cols(df)
    
    # 1. Main Dataset: reddit_sentiment.csv
    logger.info("Exporting reddit_sentiment.csv...")
    df.to_csv(OUTPUT_DIR / "reddit_sentiment.csv", index=False)
    
    # 2. Aggregations: sentiment_by_month.csv
    if 'monthly' in aggregations:
        logger.info("Exporting sentiment_by_month.csv...")
        aggregations['monthly'].to_csv(OUTPUT_DIR / "sentiment_by_month.csv", index=False)
        
    # 3. Negative Keywords: negative_keywords.csv
    logger.info("Extracting negative keywords...")
    neg_keywords = extract_negative_keywords(df)
    neg_keywords.to_csv(OUTPUT_DIR / "negative_keywords.csv", index=False)
    
    # 4. PII Audit: pii_audit_log.csv
    logger.info("Exporting pii_audit_log.csv...")
    if not pii_audit.empty:
        pii_audit.to_csv(OUTPUT_DIR / "pii_audit_log.csv", index=False)
    else:
        # Save empty with header
        pd.DataFrame(columns=['post_id', 'entity_type', 'start_position', 'end_position', 'confidence', 'timestamp']).to_csv(OUTPUT_DIR / "pii_audit_log.csv", index=False)
        
    # 5. Topic Summary: topic_summary.csv
    logger.info("Exporting topic_summary.csv...")
    if not topic_summary.empty:
        topic_summary.to_csv(OUTPUT_DIR / "topic_summary.csv", index=False)
    else:
         pd.DataFrame(columns=['topic_id', 'topic_name', 'top_words', 'post_count', 'avg_sentiment']).to_csv(OUTPUT_DIR / "topic_summary.csv", index=False)

    # 6. Metadata: run_metadata.json
    logger.info("Exporting run_metadata.json...")
    metadata = create_run_metadata(df, pii_audit)
    safe_json_save(metadata, OUTPUT_DIR / "run_metadata.json")

def ensure_timestamp_cols(df: pd.DataFrame):
    """Ensure all required timestamp columns exist."""
    if 'timestamp' not in df.columns:
        return

    # Ensure main timestamp is properly formatted datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
    # Create Pacific time
    tz = pytz.timezone(OUTPUT_TIMEZONE)
    df['timestamp_pacific'] = df['timestamp'].dt.tz_convert(tz)
    
    # Derivative columns
    df['year_month'] = df['timestamp_pacific'].dt.strftime('%Y-%m')
    df['year'] = df['timestamp_pacific'].dt.year
    df['month'] = df['timestamp_pacific'].dt.month
    df['day_of_week'] = df['timestamp_pacific'].dt.day_name()

def calculate_aggregations(df: pd.DataFrame, enable_rolling: bool = False) -> dict:
    """
    Calculate summary stats by month.
    """
    ensure_timestamp_cols(df)
    
    # Group by YYYY-MM
    grouped = df.groupby('year_month')
    
    # Basic Stats
    agg_df = grouped.agg(
        year=('year', 'first'),
        month=('month', 'first'),
        total_posts=('id', 'count'),
        avg_sentiment=('sentiment_score', 'mean'),
        median_sentiment=('sentiment_score', 'median'),
        std_sentiment=('sentiment_score', 'std'),
        oregon_mentions=('mentions_oregon', 'sum'),
        partner_mentions=('mentions_partner', 'sum')
    )
    
    # Count sentiments
    agg_df['positive_count'] = grouped.apply(lambda x: (x['sentiment_label'] == 'Positive').sum(), include_groups=False)
    agg_df['negative_count'] = grouped.apply(lambda x: (x['sentiment_label'] == 'Negative').sum(), include_groups=False)
    agg_df['neutral_count'] = grouped.apply(lambda x: (x['sentiment_label'] == 'Neutral').sum(), include_groups=False)
    
    # Percentage Negative
    agg_df['pct_negative'] = (agg_df['negative_count'] / agg_df['total_posts']) * 100
    
    # Rolling Averages (v2)
    if enable_rolling:
        agg_df = agg_df.sort_index() # Ensure chronological
        agg_df['rolling_3mo_sentiment'] = agg_df['avg_sentiment'].rolling(window=3).mean()
        agg_df['rolling_6mo_sentiment'] = agg_df['avg_sentiment'].rolling(window=6).mean()
        agg_df['rolling_12mo_sentiment'] = agg_df['avg_sentiment'].rolling(window=12).mean()
    else:
        agg_df['rolling_3mo_sentiment'] = None
        agg_df['rolling_6mo_sentiment'] = None
        agg_df['rolling_12mo_sentiment'] = None
        
    agg_df = agg_df.reset_index()
    return {'monthly': agg_df}

def extract_negative_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """
    TF-IDF on negative posts to find top keywords.
    """
    negative_df = df[df['sentiment_label'] == 'Negative']
    
    if len(negative_df) < 5:
        return pd.DataFrame(columns=['keyword', 'tfidf_score', 'frequency', 'category'])
        
    texts = negative_df['clean_text'].dropna().tolist()
    
    # TF-IDF
    try:
        tfidf = TfidfVectorizer(stop_words='english', max_features=50)
        tfidf_matrix = tfidf.fit_transform(texts)
        
        feature_names = tfidf.get_feature_names_out()
        
        # Sum tfidf scores for each term
        sum_scores = tfidf_matrix.sum(axis=0)
        
        data = []
        for col, term in enumerate(feature_names):
            score = sum_scores[0, col]
            # Calculate raw frequency too
            freq = sum([1 for t in texts if term in t]) 
            
            data.append({
                'keyword': term,
                'tfidf_score': score,
                'frequency': freq,
                'category': 'negative_tfidf' # Default
            })
            
        result = pd.DataFrame(data).sort_values('tfidf_score', ascending=False)
        return result
        
    except ValueError: # Empty vocabulary etc
        return pd.DataFrame(columns=['keyword', 'tfidf_score', 'frequency', 'category'])

def create_run_metadata(df: pd.DataFrame, pii_audit: pd.DataFrame) -> dict:
    """
    Generate metadata JSON.
    """
    now = datetime.now(timezone.utc)
    tz = pytz.timezone(OUTPUT_TIMEZONE)
    
    stats = {
        "positive_pct": 0,
        "negative_pct": 0,
        "neutral_pct": 0,
        "oregon_mention_pct": 0,
        "partner_mention_pct": 0,
        "pii_redacted_count": 0
    }
    
    if not df.empty:
        total = len(df)
        stats["positive_pct"] = round((len(df[df['sentiment_label'] == 'Positive']) / total) * 100, 1)
        stats["negative_pct"] = round((len(df[df['sentiment_label'] == 'Negative']) / total) * 100, 1)
        stats["neutral_pct"] = round((len(df[df['sentiment_label'] == 'Neutral']) / total) * 100, 1)
        stats["oregon_mention_pct"] = round((df['mentions_oregon'].sum() / total) * 100, 1)
        stats["partner_mention_pct"] = round((df['mentions_partner'].sum() / total) * 100, 1)
        
        if not pii_audit.empty:
            stats["pii_redacted_count"] = int(pii_audit['post_id'].nunique())
    
    return {
        "run_timestamp": now.isoformat(),
        "run_timestamp_pacific": now.astimezone(tz).isoformat(),
        "records_processed": len(df) if hasattr(df, 'shape') else 0, # Total fetched? Or processed? Main.py tracks processed.
        # "records_after_filtering": ... (Available in main.py, this function sees final DF)
        "records_final": len(df),
        "sentiment_model": SENTIMENT_MODEL,
        "backfill_months": BACKFILL_MONTHS,
        "version": "v1.0",
        "features_enabled": {
            "bertopic": ENABLE_BERTOPIC,
            "rolling_averages": ENABLE_ROLLING_AVERAGES,
            "google_drive_sync": ENABLE_GOOGLE_DRIVE_SYNC
        },
        "statistics": stats
    }
