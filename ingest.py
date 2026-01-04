# ingest.py
"""
Data ingestion module for fetching Reddit data via Apify.
"""

import os
import sys
import json
from datetime import datetime, timezone
import pandas as pd
import requests
import urllib.parse
from requests.exceptions import RequestException
import time

from config import (
    APIFY_TOKEN, APIFY_ACTOR_ID, BACKFILL_MONTHS, 
    MAX_RETRIES, RETRY_DELAY_SECONDS, APIFY_PROXY_GROUPS, APIFY_RUN_TIMEOUT,
    SEARCH_TERMS, SUBREDDITS, OUTPUT_DIR
)
from utils import (
    setup_logging, retry_with_backoff, safe_json_load, safe_json_save, 
    get_last_run_info
)

logger = setup_logging()

@retry_with_backoff(max_retries=MAX_RETRIES, delay=RETRY_DELAY_SECONDS)
def fetch_from_apify(since_timestamp: datetime = None) -> pd.DataFrame:
    """
    Fetch Reddit data from Apify using the 'reddit-scraper' actor.
    
    Args:
        since_timestamp: If provided, fetch posts after this date.
                         If None, performs a backfill based on BACKFILL_MONTHS.
        
    Returns:
        pd.DataFrame: DataFrame containing fetched posts and comments.
    """
    if not APIFY_TOKEN or APIFY_TOKEN == "your-token-here":
        logger.error("APIFY_TOKEN is not set or is using default placeholder.")
        return pd.DataFrame()

    # Construct search queries from all categories in config
    all_search_terms = []
    for category_terms in SEARCH_TERMS.values():
        all_search_terms.extend(category_terms)
    
    # Remove duplicates
    all_search_terms = list(set(all_search_terms))

    # Construct startUrls with explicit subreddit filtering
    # This bypasses the broken 'searches' parameter in the actor
    start_urls = []
    
    # Create the subreddit filter part of the query: (subreddit:A OR subreddit:B...)
    if SUBREDDITS:
        sub_query_part = " OR ".join([f"subreddit:{s}" for s in SUBREDDITS])
        base_query_prefix = f"({sub_query_part}) AND "
    else:
        base_query_prefix = ""
        
    for term in all_search_terms:
        # Full query: (subreddit:sub1 OR ...) AND (term)
        full_query = f"{base_query_prefix}({term})"
        
        # Encode
        encoded_query = urllib.parse.quote(full_query)
        
        # Construct URL with sort=relevance and t=all (or other time if needed)
        # Note: We rely on t=all and filter locally for detailed timestamps
        final_url = f"https://www.reddit.com/search/?q={encoded_query}&sort=relevance&t=all"
        
        start_urls.append(final_url)

    if not start_urls:
         logger.error("No start URLs generated. Check SEARCH_TERMS definition.")
         return pd.DataFrame()

    # Basic input parameters for the Apify Actor (epctex~reddit-scraper)
    # Note: This actor uses 'proxy' not 'proxyConfiguration', and startUrls are strings
    proxy_config = {
        "useApifyProxy": True
    }
    if APIFY_PROXY_GROUPS:
        proxy_config["apifyProxyGroups"] = APIFY_PROXY_GROUPS
        logger.info(f"Using manual proxy groups: {APIFY_PROXY_GROUPS}")
    else:
        logger.info("Using automatic proxy selection (no residential forced).")

    run_input = {
        "startUrls": start_urls,
        "includeComments": True,
        "maxItems": 5000 if since_timestamp is None else 500,  # More for backfill
        "proxy": proxy_config,  # Note: 'proxy' not 'proxyConfiguration' for this actor
        "sort": "relevance",
        "time": "all"
    }

    logger.info(f"DEBUG: Submitting {len(all_search_terms)} search terms to Apify.")
    logger.debug(f"DEBUG: Run input: {json.dumps(run_input, default=str)}")

    # If since_timestamp is provided, we can't easily pass it to all scrapers directly 
    # as a filter, but some support it. For 'epctex/reddit-scraper', it's often best 
    # to fetch and then filter or use relative time limits if available.
    # We will rely on post-fetch filtering for precise timestamp cutoff.
    
    start_url = f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/runs?token={APIFY_TOKEN}"
    
    logger.info(f"Starting Apify actor run for {len(all_search_terms)} search terms...")
    
    try:
        # 1. Start the Actor
        response = requests.post(start_url, json=run_input)
        response.raise_for_status()
        run_data = response.json().get('data', {})
        run_id = run_data.get('id')
        
        if not run_id:
            logger.error("Failed to get run ID from Apify response.")
            return pd.DataFrame()
            
        logger.info(f"Apify run started: {run_id}. Waiting for completion...")
        
        # 2. Wait for finish (Polling) - Using the synchronous 'waitForFinish' endpoint is easier
        # but for long runs, polling is often robust. Let's use the 'waitForFinish=120' param approach
        # or just poll manually. Here we construct the polling URL.
        # Actually, the run endpoint usually returns immediately. We need to poll.
        
        while True:
            # Check status
            status_url = f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/runs/{run_id}?token={APIFY_TOKEN}"
            status_res = requests.get(status_url)
            status_res.raise_for_status()
            status_data = status_res.json().get('data', {})
            status = status_data.get('status')
            
            if status == "SUCCEEDED":
                break
            elif status in ["FAILED", "ABORTED", "TIMED-OUT"]:
                status_message = status_data.get('statusMessage', 'No status message provided')
                logger.error(f"Apify run failed with status: {status}. Message: {status_message}")
                return pd.DataFrame()
            
            # Wait a bit before next poll
            time.sleep(10) 

        logger.info("Apify run succeeded. Fetching dataset...")
        
        # 3. Fetch Dataset
        dataset_id = status_data.get('defaultDatasetId')
        dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={APIFY_TOKEN}&format=json"
        
        data_res = requests.get(dataset_url)
        data_res.raise_for_status()
        items = data_res.json()
        
        if not items:
            logger.warning("Apify returned zero items.")
            return pd.DataFrame()
            
        # 4. Process and Flatten
        flat_data = flatten_comments(items)
        df = pd.DataFrame(flat_data)
        
        # 5. Deduplicate by ID
        if 'id' in df.columns:
            df = df.drop_duplicates(subset=['id'])
            
        # 6. Apply Timestamp Filter (if needed)
        if 'timestamp' in df.columns:
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            
            # Filter null timestamps (impute with current time if critical, or drop? Instructions say: "Use current time, add timestamp_imputed flag")
            df['timestamp_imputed'] = False
            null_mask = df['timestamp'].isnull()
            if null_mask.any():
                logger.warning(f"Found {null_mask.sum()} rows with null timestamps. Imputing with current time.")
                df.loc[null_mask, 'timestamp'] = pd.Timestamp.now(tz='UTC')
                df.loc[null_mask, 'timestamp_imputed'] = True

            if since_timestamp:
                logger.info(f"Filtering for posts after {since_timestamp}")
                # Ensure since_timestamp is consistent timezone-wise
                if since_timestamp.tzinfo is None:
                    since_timestamp = since_timestamp.replace(tzinfo=timezone.utc)
                    
                df = df[df['timestamp'] > since_timestamp]
        
        return df

    except RequestException as e:
        logger.error(f"Network error during Apify interaction: {e}")
        if hasattr(e, 'response') and e.response is not None:
             if e.response.status_code == 401:
                 logger.error("401 Unauthorized: Check your APIFY_TOKEN.")
             elif e.response.status_code == 429:
                 logger.error("429 Rate Limited.")
        raise  # Re-raise for retry decorator

    except Exception as e:
        logger.error(f"Unexpected error in fetch_from_apify: {e}")
        return pd.DataFrame()

def flatten_comments(data: list) -> list:
    """
    Flatten nested Reddit data structure into a list of dictionaries.
    Handles posts and their nested comments.
    
    Args:
        data: List of raw items from Apify (posts with 'comments' field).
        
    Returns:
        list: Flat list of dicts (posts and comments).
    """
    flat_list = []
    
    for item in data:
        # Process the main post
        post_id = item.get('id')
        if not post_id:
            continue
            
        # Extract Post Data
        post_record = {
            "id": post_id,
            "parent_id": None, # Posts don't have a parent post
            "text": (item.get('title', '') + "\n" + item.get('body', '')).strip(),
            "timestamp": item.get('createdAt'), # Apify usually returns 'createdAt'
            "score": item.get('upVotes', 0) - item.get('downVotes', 0),
            "subreddit": item.get('subredditName'),
            "type": "Post",
            "reddit_url": item.get('url'),
            "author": item.get('author')
        }
        flat_list.append(post_record)
        
        # Process Comments (Recursive function to handle arbitrary depth)
        def process_comments(comments_list, parent_id):
            for comment in comments_list:
                comment_id = comment.get('id')
                if not comment_id:
                    continue
                    
                comment_record = {
                    "id": comment_id,
                    "parent_id": parent_id,
                    "text": comment.get('body', '').strip(),
                    "timestamp": comment.get('createdAt'),
                    "score": comment.get('upVotes', 0) - comment.get('downVotes', 0),
                    "subreddit": item.get('subredditName'), # Inherit from post
                    "type": "Comment",
                    "reddit_url": comment.get('url'), # Might be None for deep comments
                    "author": comment.get('author')
                }
                flat_list.append(comment_record)
                
                # Recurse if there are replies
                replies = comment.get('replies', [])
                if replies:
                    process_comments(replies, comment_id)

        # Start comment processing
        if 'comments' in item and isinstance(item['comments'], list):
            process_comments(item['comments'], post_id)
            
    return flat_list

def load_last_run_timestamp() -> datetime:
    """
    Load timestamp of the last successful run.
    
    Returns:
        datetime: Timestamp of last run or None if not found/first run.
    """
    meta = get_last_run_info()
    ts_str = meta.get("run_timestamp")
    
    if ts_str:
        try:
            return pd.to_datetime(ts_str)
        except Exception:
            pass
            
    return None

def save_run_timestamp() -> None:
    """
    Save current UTC timestamp to run_metadata.json.
    """
    
    try:
        current_meta = get_last_run_info()
        current_meta["run_timestamp"] = datetime.now(timezone.utc).isoformat()
        safe_json_save(current_meta, OUTPUT_DIR / "run_metadata.json")
    except Exception as e:
        logger.error(f"Failed to save run timestamp: {e}")
