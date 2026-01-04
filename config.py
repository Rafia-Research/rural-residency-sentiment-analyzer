# config.py
"""
Central configuration for Rural Residency Sentiment Analyzer.
Change values here — not in individual modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === PATHS ===
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# === API CREDENTIALS (Use environment variables in production) ===
APIFY_TOKEN = os.environ.get("APIFY_TOKEN", "your-token-here")
APIFY_ACTOR_ID = "epctex~reddit-scraper"
# Proxy configuration: ["RESIDENTIAL"] for residential, [] or None for auto/datacenter
APIFY_PROXY_GROUPS = ["RESIDENTIAL"] 
APIFY_RUN_TIMEOUT = 300  # Seconds


# === DATA COLLECTION ===
BACKFILL_MONTHS = 24  # Options: 6, 12, 24
SUBREDDITS = ["Residency", "medicalschool", "medicine"]
UPDATE_FREQUENCY_HOURS = 6  # 4x daily

# === SEARCH TERMS (Organized by category for analysis) ===
SEARCH_TERMS = {
    "recruitment": [
        "rural residency",
        "rural rotation",
        "ranked rural",
        "rural vs urban",
        "primary care rural",
        "underserved",
        "NHSC",
        "loan repayment rural",
        "rural match",
    ],
    "partner_family": [
        "spouse rural",
        "partner job rural",
        "wife career",
        "husband job",
        "significant other rural",
        "dual career",
        "family rural",
        "kids rural",
        "relationship rural",
    ],
    "retention": [
        "quit rural",
        "left rural",
        "leaving rural",
        "burned out rural",
        "rural attending",
        "rural faculty",
        "contract rural",
        "moved back city",
        "years rural",
    ],
    "oregon_ohsu": [
        "Oregon residency",
        "OHSU",
        "Portland residency",
        "Pacific Northwest residency",
        "PNW residency",
        "Oregon rural",
    ],
    "compensation": [
        "rural salary",
        "rural vs urban pay",
        "attending pay rural",
        "rural lifestyle",
        "isolation rural",
        "middle of nowhere",
        "student loans rural",
    ],
    "career": [
        "rural fellowship",
        "rural career",
        "scope of practice rural",
        "mid-level rural",
        "autonomy rural",
        "rural mentorship",
    ],
}

# === FEATURE FLAGS (v1 vs v2) ===
ENABLE_BERTOPIC = True            # v1: False, v2: True ✓ ENABLED
ENABLE_ROLLING_AVERAGES = True    # v1: False, v2: True ✓ ENABLED
ENABLE_GOOGLE_DRIVE_SYNC = False  # v1: False, v2: True (requires OAuth setup)
ENABLE_GUI_SCHEDULER = False      # v1: False, v2: True
ENABLE_BIGRAMS = True             # v1: False, v2: True ✓ ENABLED

# === SENTIMENT MODEL ===
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_CONFIDENCE_THRESHOLD = 0.7  # Flag uncertain predictions below this

# === PII DETECTION ===
PII_ACTION = "redact"  # Options: "redact", "flag", "none"
PII_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", "DATE_TIME"]

# === TEXT PROCESSING ===
MIN_TEXT_LENGTH = 15  # Filter out very short posts (characters)
MAX_TEXT_LENGTH = 512  # RoBERTa token limit (truncate longer texts)
# Note: Truncation uses character approximation (avg ~4 chars/token)
MAX_CHAR_LENGTH = 2000  # Character limit for truncation (~500 tokens)

SUPPORTED_LANGUAGES = ["en"]  # English only

# === MEDICAL RELEVANCE FILTER ===
# Posts must contain at least one of these terms to be considered relevant
MEDICAL_TERMS = [
    "residency", "resident", "attending", "physician", "doctor", "md", "do",
    "rotation", "match", "program", "fellowship", "intern", "pgy", "medical school",
    "med school", "clinical", "hospital", "clinic", "patient", "surgery", "medicine",
    "fm", "family medicine", "im", "internal medicine", "pediatrics", "obgyn",
    "psychiatry", "emergency", "radiology", "pathology", "anesthesia", "icu",
    "wards", "step 1", "step 2", "usmle", "comlex", "eras", "nrmp", "soap"
]

# === CUSTOM STOPWORDS ===
CUSTOM_STOPWORDS = [
    "doc", "doctor", "med", "medical", "student", "resident", "residency",
    "hospital", "patient", "people", "get", "like", "one", "would", "know",
    "think", "go", "time", "year", "program", "match", "make", "really",
    "deleted", "removed", "amp", "x200b", "https", "com", "www", "reddit",
    "http", "org", "edu", "just", "also", "much", "even", "well", "back"
]

# === KEYWORD DETECTION LISTS ===
OREGON_KEYWORDS = [
    "oregon", "ohsu", "portland", "pacific northwest", "pnw", 
    "bend", "eugene", "salem", "corvallis", "medford"
]
PARTNER_KEYWORDS = [
    "wife", "husband", "spouse", "partner", "girlfriend", "boyfriend", 
    "fiance", "fiancee", "significant other", "kids", "children", 
    "family", "married", "relationship", "long distance"
]
# Note: "so" (significant other) requires special regex handling - see keywords.py
PARTNER_KEYWORDS_SPECIAL = ["so"]  # Handled separately to avoid "also" matches

FACULTY_KEYWORDS = [
    "attending", "faculty", "program director", "chair", "chief",
    "associate program director", "department chair"
]
# Note: "pd" and "apd" require special handling to avoid false positives
FACULTY_KEYWORDS_SPECIAL = ["pd", "apd"]

ATTRITION_KEYWORDS = [
    "quit", "left", "leaving", "moved", "burned out", "burnout", "resign",
    "resigned", "quitting", "moving back", "transferred"
]

# === PII WHITELIST (False positive prevention) ===
PII_WHITELIST = [
    "dr. house", "grey's anatomy", "scrubs", "er", "the resident",
    "dr. cox", "dr. grey", "dr. shepherd", "meredith", "mcdreamy",
    "house md", "chicago med", "new amsterdam", "the good doctor",
    "doogie howser", "patch adams", "dr. strange"
]

# === BATCH PROCESSING (Memory management for M1) ===
BATCH_SIZE = 500  # Process this many posts at a time
SENTIMENT_BATCH_SIZE = 32  # Hugging Face batch size
ENABLE_GARBAGE_COLLECTION = True  # Call gc.collect() between batches

# === VALIDATION ===
MIN_RESULTS_THRESHOLD = 10  # Alert if Apify returns fewer than this

# === TIMEZONE ===
OUTPUT_TIMEZONE = "America/Los_Angeles"  # Pacific Time for OHSU

# === RETRY LOGIC ===
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# === LOGGING ===
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
PIPELINE_LOG_FILE = "pipeline.log"
RUN_HISTORY_FILE = "run_history.log"
