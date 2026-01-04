# deep_analysis.py
"""
Deep analysis module using Gemini API for qualitative insights.
Processes high-signal posts to extract structured research data.
"""

import os
import json
import time
from typing import List, Dict, Optional
import pandas as pd

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from config import OUTPUT_DIR
from utils import setup_logging

logger = setup_logging()

# Gemini configuration
GEMINI_MODEL = "gemini-1.5-flash"  # Fast and cheap for structured extraction
MAX_POSTS_TO_ANALYZE = 500  # Top N most negative posts
BATCH_SIZE = 10  # Posts per API call (to reduce costs)


def configure_gemini(api_key: str = None) -> bool:
    """
    Configure Gemini API with the provided key.
    
    Args:
        api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
        
    Returns:
        bool: True if configuration successful.
    """
    if not GEMINI_AVAILABLE:
        logger.error("google-generativeai package not installed. Run: pip install google-generativeai")
        return False
        
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        logger.error("GEMINI_API_KEY not set. Add it to your .env file.")
        return False
        
    genai.configure(api_key=key)
    logger.info("Gemini API configured successfully.")
    return True


def create_extraction_prompt(posts: List[Dict]) -> str:
    """
    Create a structured extraction prompt for Gemini.
    
    Args:
        posts: List of post dictionaries with 'id' and 'text' keys.
        
    Returns:
        str: The formatted prompt.
    """
    prompt = """You are a medical workforce researcher analyzing Reddit posts about rural residency and physician retention.

For each post below, extract structured data. Return ONLY valid JSON array.

For each post, provide:
{
    "post_id": "<the post ID>",
    "primary_complaint": "<isolation|money|family|clinical|admin|burnout|career|other>",
    "specific_factor": "<One sentence describing the exact issue>",
    "quotable_line": "<Best direct quote from post, max 100 chars>",
    "mentions_leaving": <true|false>,
    "mentions_rural": <true|false>,
    "mentions_partner": <true|false>,
    "severity": <1-5, where 5 is extremely negative>,
    "actionable_insight": "<What could retain this person? One sentence.>"
}

POSTS TO ANALYZE:
"""
    
    for post in posts:
        prompt += f"\n---\nPOST_ID: {post['id']}\nTEXT: {post['text'][:1500]}\n"
    
    prompt += "\n---\nReturn ONLY the JSON array, no other text."
    
    return prompt


def extract_insights_batch(posts: List[Dict], model) -> List[Dict]:
    """
    Extract insights from a batch of posts using Gemini.
    
    Args:
        posts: List of post dictionaries.
        model: Configured Gemini model.
        
    Returns:
        list: List of extracted insight dictionaries.
    """
    prompt = create_extraction_prompt(posts)
    
    try:
        response = model.generate_content(prompt)
        
        # Parse JSON from response
        response_text = response.text.strip()
        
        # Handle markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        insights = json.loads(response_text)
        return insights
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Gemini response as JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return []


def run_deep_analysis(
    df: pd.DataFrame,
    n_posts: int = MAX_POSTS_TO_ANALYZE,
    sentiment_threshold: float = -0.3
) -> pd.DataFrame:
    """
    Run deep analysis on the most negative posts.
    
    Args:
        df: DataFrame with columns 'id', 'clean_text', 'sentiment_score'.
        n_posts: Number of posts to analyze (will select most negative).
        sentiment_threshold: Only analyze posts below this sentiment score.
        
    Returns:
        pd.DataFrame: DataFrame with extracted insights.
    """
    if not GEMINI_AVAILABLE:
        logger.error("Gemini not available. Skipping deep analysis.")
        return pd.DataFrame()
    
    if not configure_gemini():
        return pd.DataFrame()
    
    # Select most negative posts
    negative_posts = df[df['sentiment_score'] < sentiment_threshold].nsmallest(
        n_posts, 'sentiment_score'
    )
    
    if len(negative_posts) == 0:
        logger.warning("No posts below sentiment threshold for deep analysis.")
        return pd.DataFrame()
    
    logger.info(f"Running deep analysis on {len(negative_posts)} most negative posts...")
    
    # Prepare posts for analysis
    posts_to_analyze = [
        {"id": row['id'], "text": row.get('clean_text', row.get('text', ''))}
        for _, row in negative_posts.iterrows()
    ]
    
    # Initialize model
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    # Process in batches
    all_insights = []
    total_batches = (len(posts_to_analyze) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(posts_to_analyze), BATCH_SIZE):
        batch = posts_to_analyze[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches}...")
        
        insights = extract_insights_batch(batch, model)
        all_insights.extend(insights)
        
        # Rate limiting
        if i + BATCH_SIZE < len(posts_to_analyze):
            time.sleep(1)  # 1 second between batches
    
    # Convert to DataFrame
    insights_df = pd.DataFrame(all_insights)
    
    if len(insights_df) > 0:
        # Save to CSV
        output_path = OUTPUT_DIR / "deep_insights.csv"
        insights_df.to_csv(output_path, index=False)
        logger.info(f"Deep insights saved to {output_path}")
        
        # Log summary
        if 'primary_complaint' in insights_df.columns:
            complaint_counts = insights_df['primary_complaint'].value_counts()
            logger.info(f"Complaint distribution:\n{complaint_counts}")
    
    return insights_df


def summarize_insights(insights_df: pd.DataFrame) -> Dict:
    """
    Generate a summary of the deep analysis insights.
    
    Args:
        insights_df: DataFrame with extracted insights.
        
    Returns:
        dict: Summary statistics and key findings.
    """
    if len(insights_df) == 0:
        return {}
    
    summary = {
        "total_analyzed": len(insights_df),
        "posts_mentioning_leaving": insights_df.get('mentions_leaving', pd.Series()).sum(),
        "posts_mentioning_rural": insights_df.get('mentions_rural', pd.Series()).sum(),
        "posts_mentioning_partner": insights_df.get('mentions_partner', pd.Series()).sum(),
    }
    
    if 'primary_complaint' in insights_df.columns:
        summary["complaint_distribution"] = insights_df['primary_complaint'].value_counts().to_dict()
    
    if 'severity' in insights_df.columns:
        summary["avg_severity"] = insights_df['severity'].mean()
    
    # Get top quotable lines
    if 'quotable_line' in insights_df.columns:
        summary["top_quotes"] = insights_df['quotable_line'].head(10).tolist()
    
    return summary


if __name__ == "__main__":
    # Test with sample data
    import argparse
    
    parser = argparse.ArgumentParser(description="Run deep analysis on sentiment data")
    parser.add_argument("--input", default="output/reddit_sentiment.csv", help="Input CSV file")
    parser.add_argument("--n-posts", type=int, default=50, help="Number of posts to analyze")
    parser.add_argument("--threshold", type=float, default=-0.3, help="Sentiment threshold")
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    print(f"Running deep analysis on top {args.n_posts} negative posts...")
    insights = run_deep_analysis(df, n_posts=args.n_posts, sentiment_threshold=args.threshold)
    
    if len(insights) > 0:
        print(f"\nExtracted {len(insights)} insights.")
        summary = summarize_insights(insights)
        print(f"\nSummary: {json.dumps(summary, indent=2)}")
    else:
        print("No insights extracted.")
