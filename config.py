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
# Expanded subreddit list to capture rural-specific discussions
SUBREDDITS = [
    "Residency",
    "medicalschool", 
    "medicine",
    "FamilyMedicine",          # FM is the primary rural pipeline
    "emergencymedicine",        # Rural EM is a major topic
    "physicianassistant",       # Rural PA workforce
    "nursepractitioner",        # Rural NP scope issues
    "Noctor",                   # Scope disputes & rural implications
    "premed",                   # Early career thinking about rural
    "whitecoatinvestor",        # Financial decisions about rural
    "locums",                   # Locums as rural alternative
    "IMGreddit",                # IMGs often go rural for visas
]
UPDATE_FREQUENCY_HOURS = 6  # 4x daily

# === SEARCH TERMS (Targeted for "Why They Leave" research) ===
SEARCH_TERMS = {
    # WHY THEY COME (Initial Attraction)
    "recruitment": [
        "rural residency",
        "rural track",
        "rural training track",
        "RTT program",
        "ranked rural",
        "matched rural",
        "underserved area",
        "underserved community",
        "NHSC scholarship",
        "NHSC loan repayment",
        "loan forgiveness rural",
        "PSLF rural",
        "state loan repayment",
        "sign on bonus rural",
        "J1 waiver rural",
        "visa waiver underserved",
        "IMG rural",
    ],
    
    # THE TRAILING SPOUSE PROBLEM (Major attrition driver)
    "partner_family": [
        "spouse won't move",
        "partner hates rural",
        "wife career rural",
        "husband job rural",
        "trailing spouse",
        "dual physician couple rural",
        "two body problem rural",
        "long distance residency",
        "relationship strain residency",
        "divorced during residency",
        "marriage residency",
        "kids in rural",
        "schools rural area",
        "childcare rural",
        "isolated with kids",
        "no family nearby",
        "significant other moved",
    ],
    
    # WHY THEY LEAVE (Direct attrition signals)
    "retention": [
        "quit rural",
        "left rural practice",
        "leaving rural",
        "moved back to city",
        "couldn't stay rural",
        "only stayed 2 years",
        "contract buyout",
        "burned out rural",
        "rural burnout",
        "isolated and burned out",
        "regret going rural",
        "regret rural",
        "wish I stayed urban",
        "should have stayed",
        "never go rural",
        "don't go rural",
        "rural was a mistake",
        "left after loan repayment",
        "finished NHSC and left",
        "3 year commitment rural",
    ],
    
    # ISOLATION & LIFESTYLE (Major complaint)
    "isolation": [
        "middle of nowhere",
        "hours from anything",
        "nothing to do rural",
        "no restaurants",
        "no culture",
        "boring town",
        "small town everyone knows",
        "gossip small town",
        "no privacy rural",
        "dating rural",
        "single in rural",
        "no dating pool",
        "where to meet people rural",
        "friends rural area",
        "lonely rural",
        "isolated physician",
        "only doctor in town",
    ],
    
    # SCOPE & BACKUP (Clinical isolation)
    "clinical_isolation": [
        "no backup rural",
        "only physician",
        "no specialist",
        "transfer patient hours",
        "medevac",
        "critical access hospital",
        "CAH hospital",
        "25 bed hospital",
        "scope creep rural",
        "NP independence rural",
        "mid-level supervision rural",
        "solo practice rural",
        "call every night",
        "1 in 2 call",
        "1 in 3 call",
        "always on call",
        "never off rural",
    ],
    
    # MONEY (Surprisingly NOT enough to retain)
    "compensation": [
        "rural salary not worth",
        "paid more but miserable",
        "money isn't everything rural",
        "golden handcuffs rural",
        "contract trap rural",
        "non-compete rural",
        "can't leave contract",
        "buyout clause",
        "loan repayment trap",
        "stuck for 3 years",
        "RVU rural",
        "productivity rural",
        "admin burden rural",
        "panel size rural",
    ],
    
    # OREGON/PNW SPECIFIC
    "oregon_ohsu": [
        "OHSU rural",
        "Oregon rural residency",
        "Cascades East",
        "Oregon rural track",
        "Eastern Oregon",
        "Klamath Falls residency",
        "Bend Oregon medicine",
        "Portland vs rural Oregon",
        "Pacific Northwest rural",
        "Washington rural",
        "Idaho rural",
        "Montana rural",
    ],
    
    # CAREER TRAJECTORY (Why rural feels like a dead end)
    "career": [
        "rural career dead end",
        "no advancement rural",
        "can't do fellowship rural",
        "stuck in rural",
        "pigeonholed rural",
        "rural on CV",
        "coming back from rural",
        "re-entering urban",
        "transition rural to urban",
        "escape rural",
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
    # Direct leaving language
    "quit", "quitting", "left", "leaving", "moved away", "moving back",
    "burned out", "burnout", "burnt out", "resign", "resigned",
    "transferred", "couldn't stay", "had to leave", "forced to leave",
    # Regret signals
    "regret", "mistake", "worst decision", "should have", "wish I had",
    "never should have", "don't go", "wouldn't recommend",
    # Contract/commitment language
    "finished my contract", "contract ended", "buyout", "non-compete",
    "3 year commitment", "2 years and left", "left after NHSC",
    # Escape language  
    "escape", "get out", "couldn't wait to leave", "counting down",
    "trapped", "stuck", "prison", "sentence",
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
