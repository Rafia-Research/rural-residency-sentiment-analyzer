# topics.py
"""
Topic modeling module using BERTopic.
"""

from typing import Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from config import ENABLE_BERTOPIC
from utils import setup_logging

logger = setup_logging()

def extract_topics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract topics from the DataFrame using BERTopic.
    
    Args:
        df: DataFrame with 'clean_text'.
        
    Returns:
        tuple: (processed_df, topic_summary_df)
    """
    if not ENABLE_BERTOPIC:
        logger.info("BERTopic disabled in config. Skipping...")
        df['dominant_topic'] = None
        df['topic_keywords'] = None
        return df, pd.DataFrame()
        
    if df.empty or len(df) < 50: # Check minimum data
        logger.warning("Insufficient data for topic modeling (need > 50 docs). Skipping.")
        df['dominant_topic'] = None
        df['topic_keywords'] = None
        return df, pd.DataFrame()

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN
        
        logger.info("Initializing BERTopic model...")
        
        # 1. Initialize BERTopic
        topic_model = initialize_bertopic(len(df))
        
        texts = df['clean_text'].fillna("").tolist()
        
        # 2. Fit Transform
        logger.info(f"Fitting topics on {len(texts)} documents...")
        topics, probs = topic_model.fit_transform(texts)
        
        # 3. Assign to DataFrame
        df['dominant_topic'] = topics
        
        # 4. Get Topic Info
        freq = topic_model.get_topic_info()
        logger.info(f"Found {len(freq) - 1} topics (excluding outliers).")
        
        # 5. Generate Keywords for each row
        # This can be slow row-by-row. Better to map topic_id -> keywords
        topic_keywords_map = {}
        for topic_id in set(topics):
            if topic_id == -1:
                topic_keywords_map[topic_id] = "outlier"
                continue
                
            top_words = topic_model.get_topic(topic_id)
            if top_words:
                # top_words is list of (word, score)
                words = [w[0] for w in top_words[:10]] 
                topic_keywords_map[topic_id] = ", ".join(words)
            else:
                topic_keywords_map[topic_id] = ""
                
        df['topic_keywords'] = df['dominant_topic'].map(topic_keywords_map)
        
        # 6. Create Summary
        topic_summary = create_topic_summary(df, topic_model)
        
        return df, topic_summary

    except ImportError:
        logger.error("BERTopic not installed. Install requirements.txt.")
        df['dominant_topic'] = None
        df['topic_keywords'] = None
        return df, pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Topic modeling failed: {e}")
        df['dominant_topic'] = None
        df['topic_keywords'] = None
        return df, pd.DataFrame()

def initialize_bertopic(n_docs: int):
    """
    Create and configure BERTopic model.
    """
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Calculate min_topic_size
    min_size = max(10, n_docs // 100)
    
    # UMAP for dimensionality reduction (M1 optimized parameters?)
    # n_neighbors=15 is standard.
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    
    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(min_cluster_size=min_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    # Vectorizer model (remove stopwords here too?)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2)
    
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2", # Fast and good enough
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        verbose=True
    )
    
    return topic_model

def generate_topic_names(topic_model) -> Dict[int, str]:
    """
    Generate human-readable names for topics.
    """
    # Auto-generation or LLM based? 
    # v1 requirements: "Map topic_id to descriptive name based on top keywords"
    # Example: Topic 0 with keywords "spouse, job, partner" -> "Spouse/Partner Employment"
    # We'll use a simple heuristic: "Topic X: kw1, kw2, kw3" or allow BERTopic's custom labels.
    
    # Just return map of topic_id -> "kw1, kw2, kw3" for now as 'topic_name' 
    # unless using LLM (not in scope for v1 w/o API key).
    
    topic_info = topic_model.get_topic_info()
    mapping = {}
    
    for index, row in topic_info.iterrows():
        t_id = row['Topic']
        name = row['Name'] # BERTopic generated name like "0_spouse_job_partner"
        
        # Clean it up: remove ID prefix
        clean_name = "_".join(name.split("_")[1:])
        mapping[t_id] = clean_name.replace("_", ", ").title()
        
    return mapping

def create_topic_summary(df: pd.DataFrame, topic_model) -> pd.DataFrame:
    """
    Create summary stats DataFrame.
    """
    if 'dominant_topic' not in df.columns:
        return pd.DataFrame()
        
    # Group by Topic
    summary = df.groupby('dominant_topic').agg({
        'id': 'count',
        'sentiment_score': 'mean'
    }).rename(columns={'id': 'post_count', 'sentiment_score': 'avg_sentiment'})
    
    summary = summary.reset_index().rename(columns={'dominant_topic': 'topic_id'})
    
    # Add Names and Keywords
    name_map = generate_topic_names(topic_model)
    
    summary['topic_name'] = summary['topic_id'].map(name_map)
    summary['top_words'] = summary['topic_name'] # Using cleaned name as words list
    
    # Handle Outlier -1
    summary.loc[summary['topic_id'] == -1, 'topic_name'] = "Miscellaneous / Outliers"
    
    return summary
