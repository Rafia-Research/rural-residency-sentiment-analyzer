# utils.py
"""
Utility functions used by all other modules.
"""

import json
import logging
import time
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, Dict, Optional
import functools

import pandas as pd
import pytz

from config import (
    LOG_LEVEL, LOG_FORMAT, PIPELINE_LOG_FILE, RUN_HISTORY_FILE,
    LOGS_DIR, OUTPUT_DIR, OUTPUT_TIMEZONE
)

def setup_logging() -> logging.Logger:
    """
    Configure logging to both console and file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("rural_analyzer")
    logger.setLevel(LOG_LEVEL)
    
    # Check if handlers already exist to avoid duplicate logs
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(LOG_FORMAT)
    
    # File Handler
    log_path = LOGS_DIR / PIPELINE_LOG_FILE
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_run(success: bool, records_processed: int, error: str = None) -> None:
    """
    Append run status to run_history.log in JSON format.
    
    Args:
        success: True if run completed successfully.
        records_processed: Number of records processed.
        error: Error message if failed.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "records_processed": records_processed,
        "error": error
    }
    
    log_path = LOGS_DIR / RUN_HISTORY_FILE
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Failed to write to run history: {e}")

def validate_results(df: Optional[pd.DataFrame], threshold: int) -> bool:
    """
    Validate that the DataFrame has sufficient data.
    
    Args:
        df: DataFrame to check.
        threshold: Minimum number of rows required.
        
    Returns:
        bool: True if valid.
    """
    if df is None:
        return False
    if len(df) < threshold:
        return False
    return True

def convert_timezone(dt: datetime, to_tz: str = OUTPUT_TIMEZONE) -> datetime:
    """
    Convert datetime to specified timezone.
    
    Args:
        dt: Datetime object.
        to_tz: Target timezone string.
        
    Returns:
        datetime: Converted datetime.
    """
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(pytz.timezone(to_tz))

def safe_json_load(path: Path) -> dict:
    """
    Load JSON with error handling.
    
    Args:
        path: Path to JSON file.
        
    Returns:
        dict: Loaded data or empty dict on failure.
    """
    if not path.exists():
        return {}
        
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Failed to load JSON from {path}: {e}")
        return {}

def safe_json_save(data: dict, path: Path) -> bool:
    """
    Save JSON with error handling.
    
    Args:
        data: Dictionary to save.
        path: Destination path.
        
    Returns:
        bool: True if successful.
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Failed to save JSON to {path}: {e}")
        return False

def ensure_directories() -> None:
    """Create output and logs directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

def get_last_run_info() -> dict:
    """
    Load metadata from the last run.
    
    Returns:
        dict: Metadata dictionary.
    """
    return safe_json_load(OUTPUT_DIR / "run_metadata.json")

def calculate_file_hash(path: Path) -> str:
    """
    Calculate MD5 hash of a file for integrity checking.
    
    Args:
        path: Path to file.
        
    Returns:
        str: MD5 hash string.
    """
    if not path.exists():
        return ""
        
    hash_md5 = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Failed to calculate hash for {path}: {e}")
        return ""

def retry_with_backoff(max_retries: int = 3, delay: int = 5) -> Callable:
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries.
        delay: Initial delay in seconds.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            driver_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(driver_delay)
                        driver_delay *= 2  # Exponential backoff
                        logger = setup_logging()
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {driver_delay/2}s...")
            
            # If we get here, all retries failed
            logger = setup_logging()
            logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}. Last error: {last_exception}")
            raise last_exception
        return wrapper
    return decorator
