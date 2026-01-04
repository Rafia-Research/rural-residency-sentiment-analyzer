# main.py
"""
Main orchestration script for Rural Residency Sentiment Analyzer.
Runs the full pipeline: Ingest → Preprocess → Sentiment → PII → Keywords → Topics → Export

Usage:
    python main.py                    # Full pipeline (backfill + incremental)
    python main.py --incremental      # Only new posts since last run
    python main.py --backfill-only    # Initial 24-month backfill
"""

import argparse
import gc
import sys
from datetime import datetime
import pandas as pd

from config import (
    OUTPUT_DIR,
    LOGS_DIR,
    BATCH_SIZE,
    ENABLE_GARBAGE_COLLECTION,
    ENABLE_BERTOPIC,
    ENABLE_ROLLING_AVERAGES,
    MIN_RESULTS_THRESHOLD,
    BACKFILL_MONTHS,
    APIFY_PROXY_GROUPS
)
from utils import setup_logging, log_run, validate_results, ensure_directories
from ingest import fetch_from_apify, load_last_run_timestamp, save_run_timestamp
from preprocess import preprocess_dataframe
from sentiment import analyze_sentiment_batch
from pii import detect_and_redact_pii
from keywords import flag_all_keywords
from topics import extract_topics
from export import export_all_csvs, calculate_aggregations


def run_pipeline(mode: str = "full") -> bool:
    """
    Execute the full data pipeline.
    
    Args:
        mode: "full", "incremental", or "backfill"
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger = setup_logging()
    logger.info(f"Starting pipeline in {mode} mode at {datetime.now()}")
    
    # Ensure output directories exist
    ensure_directories()
    
    try:
        # === 1. DATA INGESTION ===
        logger.info("Step 1/7: Fetching data from Apify...")
        
        if mode == "incremental":
            last_run = load_last_run_timestamp()
            logger.info(f"  Incremental mode: fetching posts since {last_run}")
            df = fetch_from_apify(since_timestamp=last_run)
        else:
            logger.info("  Full/Backfill mode: fetching all historical data")
            df = fetch_from_apify(since_timestamp=None)
        
        # Validate
        if not validate_results(df, MIN_RESULTS_THRESHOLD):
            logger.error(f"Insufficient data returned ({len(df) if df is not None else 0} rows). Aborting to preserve existing data.")
            log_run(success=False, records_processed=0, error="Insufficient data from API")
            return False
        
        logger.info(f"  Fetched {len(df)} records.")
        
        # === 2. PREPROCESSING ===
        logger.info("Step 2/7: Preprocessing text...")
        initial_count = len(df)
        df = preprocess_dataframe(df)
        logger.info(f"  After preprocessing: {len(df)} records remain (filtered {initial_count - len(df)})")
        
        # === 3. SENTIMENT ANALYSIS (Batched for M1 memory) ===
        logger.info("Step 3/7: Running sentiment analysis...")
        
        all_results = []
        total_batches = (len(df) // BATCH_SIZE) + 1
        
        for i in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[i:i + BATCH_SIZE].copy()
            batch = analyze_sentiment_batch(batch)
            all_results.append(batch)
            
            if ENABLE_GARBAGE_COLLECTION:
                gc.collect()
            
            current_batch = (i // BATCH_SIZE) + 1
            logger.info(f"  Processed sentiment batch {current_batch}/{total_batches}")
        
        if all_results:
             df = pd.concat(all_results, ignore_index=True)
        else:
             logger.warning("No records remained for sentiment analysis.")
        
        # === 4. PII DETECTION & REDACTION ===
        logger.info("Step 4/7: Detecting and redacting PII...")
        df, pii_audit = detect_and_redact_pii(df)
        pii_post_count = pii_audit['post_id'].nunique() if not pii_audit.empty else 0
        logger.info(f"  Redacted PII in {pii_post_count} posts.")
        
        # === 5. KEYWORD FLAGGING ===
        logger.info("Step 5/7: Flagging keywords...")
        df = flag_all_keywords(df)
        
        # Log keyword stats
        oregon_count = df['mentions_oregon'].sum() if 'mentions_oregon' in df.columns else 0
        partner_count = df['mentions_partner'].sum() if 'mentions_partner' in df.columns else 0
        logger.info(f"  Oregon mentions: {oregon_count}, Partner mentions: {partner_count}")
        
        # === 6. TOPIC MODELING (v2 only) ===
        if ENABLE_BERTOPIC:
            logger.info("Step 6/7: Extracting topics with BERTopic...")
            df, topic_summary = extract_topics(df)
            logger.info(f"  Extracted {len(topic_summary)} topics.")
        else:
            logger.info("Step 6/7: Topic modeling disabled (v1). Skipping...")
            df["dominant_topic"] = None
            df["topic_keywords"] = None
            topic_summary = pd.DataFrame()
        
        # === 7. AGGREGATIONS & EXPORT ===
        logger.info("Step 7/7: Calculating aggregations and exporting...")
        
        aggregations = calculate_aggregations(df, enable_rolling=ENABLE_ROLLING_AVERAGES)
        export_all_csvs(df, aggregations, pii_audit, topic_summary)
        
        # Save run timestamp for incremental mode
        save_run_timestamp()
        
        logger.info(f"Pipeline complete. Output saved to {OUTPUT_DIR}")
        log_run(success=True, records_processed=len(df))
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        log_run(success=False, records_processed=0, error=str(e))
        return False


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Rural Residency Sentiment Analyzer Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --backfill-only    # First run: fetch 24 months of data
  python main.py --incremental      # Daily runs: fetch only new posts
  python main.py                    # Full refresh
  python main.py --deep-analysis    # Run Gemini deep-dive on existing data
        """
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only fetch new posts since last run"
    )
    parser.add_argument(
        "--backfill-only",
        action="store_true",
        help="Run initial 24-month backfill"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running pipeline"
    )
    parser.add_argument(
        "--deep-analysis",
        action="store_true",
        help="Run Gemini deep-dive on top negative posts (requires GEMINI_API_KEY)"
    )
    parser.add_argument(
        "--deep-n-posts",
        type=int,
        default=100,
        help="Number of posts for deep analysis (default: 100)"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("Configuration validated successfully.")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Backfill months: {BACKFILL_MONTHS}")
        print(f"BERTopic enabled: {ENABLE_BERTOPIC}")
        print(f"Apify Proxy Groups: {APIFY_PROXY_GROUPS if APIFY_PROXY_GROUPS else 'Auto'}")
        sys.exit(0)
    
    # Deep analysis only mode
    if args.deep_analysis:
        from deep_analysis import run_deep_analysis, summarize_insights
        import json
        
        logger = setup_logging()
        logger.info("Running Gemini deep analysis on existing data...")
        
        input_file = OUTPUT_DIR / "reddit_sentiment.csv"
        if not input_file.exists():
            logger.error(f"No data found at {input_file}. Run the pipeline first.")
            sys.exit(1)
        
        df = pd.read_csv(input_file)
        insights = run_deep_analysis(df, n_posts=args.deep_n_posts)
        
        if len(insights) > 0:
            summary = summarize_insights(insights)
            logger.info(f"Deep analysis complete. Summary:\n{json.dumps(summary, indent=2)}")
            sys.exit(0)
        else:
            logger.error("Deep analysis failed.")
            sys.exit(1)
    
    if args.incremental:
        mode = "incremental"
    elif args.backfill_only:
        mode = "backfill"
    else:
        mode = "full"
    
    success = run_pipeline(mode)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

