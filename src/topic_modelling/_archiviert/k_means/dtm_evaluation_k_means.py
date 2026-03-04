import os
import json
import gc
import warnings
import itertools
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Set env before other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import numpy as np
import torch
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('gensim').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Download NLTK resources once
nltk.download('stopwords', quiet=True)


# --- CONFIGURATION ---

class Config:
    DATA_PATH = Path("../../data/dataset_popmusic_v8.json")
    # Base path adapted for K-Means
    BASE_PATH = Path("../../bertopic/dtm_k_means")
    EVAL_PATH = BASE_PATH / "evaluation_parameters"
    # Embeddings from original dtm folder (shared)
    EMBEDDINGS_PATH = Path("../../bertopic/dtm/embedding/embeddings.npy")
    
    # Parameter Grid for K-Means
    PARAM_GRID = {
        'n_neighbors': [10,30, 50, 70, 90],
        'n_components': [5],
        'n_clusters': [40,45,50,55,60,65,70,75,80,85],
        'min_df': [1],
        'max_df': [0.5]
    }


def get_custom_stopwords() -> List[str]:
    """Returns a combined list of German NLTK stopwords and custom analysis terms."""
    base_stopwords = stopwords.words('german')
    
    analysis_stopwords = [
        # Substantive
        'song', 'lied', 'text', 'titel', 'refrain', 'strophe', 'vers', 'zeile', 'zeilen',
        'metapher', 'symbol', 'motiv', 'bild', 'wiederholung', 'sprecher', 'lyrische',
        'gegenüberstellung', 'spannungsfeld', 'person', 'interpretation', 'musik', 'klang',
        'mantra', 'frage', 'antwort', 'hintergrund', 'vordergrund', 'atmosphäre',
        # Verben
        'betont', 'fungiert', 'wirkt', 'verdeutlicht', 'suggeriert', 'verstärkt',
        'offenbart', 'symbolisiert', 'verdeutlichen', 'verknüpft', 'schildert',
        'zeichnet', 'dient', 'stellt', 'beschreibt', 'handelt', 'geht', 'zeigt',
        'erzählt', 'entsteht', 'lässt', 'macht',
        # Adjektive
        'wiederholte', 'wiederkehrende', 'ständige', 'eigene', 'eigenen', 'innere',
        'inneren', 'innerer', 'emotionale', 'musikalische',
        # Füllwörter
        'dabei', 'jedoch', 'zugleich', 'fast', 'gleichzeitig', 'insgesamt', 
        'immer', 'oft', 'nie', 'wohl', 'bereits', 'sowohl', 'trotz', 'gegenüber',
        # Englische Lyrics-Stopwords
        'the', 'and', 'you', 'i', 'to', 'a', 'of', 'it', 'my', 'me', 'in', 'on', 
        'is', 'that', 'your', 'we', 'all', 'be', 'for', 'don', 't', 's'
    ]
    
    return base_stopwords + analysis_stopwords


def load_documents(json_path: Path) -> List[str]:
    """Load documents from JSON."""
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
        
    print(f"DEBUG: Lade JSON von {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    
    for item in data:
        if not isinstance(item, dict):
            continue
            
        text = item.get('text', '')
        timestamp = item.get('timestamp', '')
        
        if not text:
            text = item.get('Text', '') or item.get('lyrics', '') or item.get('body', '')
            
        if not text or not isinstance(text, str) or not text.strip():
            continue
            
        if not timestamp:
            continue
            
        try:
            year = int(timestamp[:4])
            if not (1955 <= year <= 2024):
                continue
        except (ValueError, TypeError):
            continue

        documents.append(text.strip())

    print(f"DEBUG: {len(documents)} Dokumente geladen.")
    return documents


def clear_memory():
    """Clear Python GC and CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- METRICS ---

def calculate_coherence(topic_model: BERTopic, tokenized_docs: List[List[str]]) -> float:
    """Calculate c_v coherence score using Gensim."""
    try:
        topics = topic_model.get_topics()
        if not topics:
            return 0.0

        # Create Dictionary
        dictionary = Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=3, no_above=0.85)
        
        if not dictionary:
            return 0.0

        topic_words = []
        for topic_id in topics:
            # K-Means usually has no outliers (-1), but check anyway
            if topic_id == -1:
                continue
            
            words = [w.lower() for w, _ in topic_model.get_topic(topic_id)[:10]]
            valid_words = [w for w in words if w in dictionary.token2id]
            
            if len(valid_words) >= 3:
                topic_words.append(valid_words)

        if not topic_words:
            return 0.0

        cm = CoherenceModel(
            topics=topic_words,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v',
            processes=1
        )
        score = cm.get_coherence()
        return score if not np.isnan(score) else 0.0
        
    except Exception:
        return 0.0


def calculate_diversity(topic_model: BERTopic, top_n: int = 10) -> float:
    """Calculate topic diversity."""
    try:
        topics = topic_model.get_topics()
        all_words = []
        unique_words = set()
        
        for topic_id in topics:
            if topic_id == -1:
                continue
            words = [w for w, _ in topic_model.get_topic(topic_id)[:top_n]]
            all_words.extend(words)
            unique_words.update(words)
        
        if not all_words:
            return 0.0
        
        return len(unique_words) / len(all_words)
    except Exception:
        return 0.0


# --- WORKER FUNCTION ---

def evaluate_single_combination(args) -> Dict[str, Any]:
    """Evaluate a single parameter combination using K-Means."""
    idx, params, documents, embeddings, tokenized_docs = args
    
    start_time = datetime.now()
    
    n_neighbors = params['n_neighbors']
    n_components = params['n_components']
    n_clusters = params['n_clusters']
    min_df = params['min_df']
    max_df = params['max_df']
    
    print(f"[{idx}] Start: neighbors={n_neighbors}, n_clusters={n_clusters}")

    try:
        # 1. Initialize Models
        embedding_model = SentenceTransformer('BAAI/bge-m3', device='cpu')
        
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42,
            n_jobs=1,
            low_memory=False
        )
        
        # K-Means Clustering
        cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        vectorizer = CountVectorizer(
            stop_words=get_custom_stopwords(),
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df
        )
        
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer,
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            representation_model=None,
            language='german',
            calculate_probabilities=False, # Not supported/needed for K-Means in this flow
            verbose=False
        )
        
        # 2. Fit Model
        topics, _ = topic_model.fit_transform(documents, embeddings)
        
        # 3. Calculate Metrics
        coherence = calculate_coherence(topic_model, tokenized_docs)
        diversity = calculate_diversity(topic_model)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Clean up
        del topic_model, umap_model, cluster_model, embedding_model
        clear_memory()
        
        print(f"[{idx}] ✓ Done ({duration:.1f}s) | Clusters: {n_clusters} | "
              f"Coh: {coherence:.4f} | Div: {diversity:.4f}")

        return {
            **params,
            'coherence': coherence,
            'diversity': diversity,
            'time_seconds': duration
        }

    except Exception as e:
        print(f"[{idx}] ✗ Error: {e}")
        clear_memory()
        return {**params, 'error': str(e)}


# --- MAIN ---

def main():
    # Setup
    Config.EVAL_PATH.mkdir(parents=True, exist_ok=True)
    output_file = Config.EVAL_PATH / "results_kmeans.csv"

    logging.info("!!! STARTING NEW FRESH RUN (K-MEANS) !!!")
    
    # 1. Load Data
    documents = load_documents(Config.DATA_PATH)
    logging.info(f"Loaded {len(documents)} documents.")

    # 2. Load Embeddings
    if not Config.EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings not found at {Config.EMBEDDINGS_PATH}")
    
    logging.info("Loading embeddings...")
    embeddings = np.load(Config.EMBEDDINGS_PATH)
    if len(embeddings) != len(documents):
        raise ValueError("Mismatch between documents and embeddings length.")
    
    # 3. Pre-tokenize for Coherence
    logging.info("Pre-tokenizing documents...")
    tokenized_docs = [d.lower().split() for d in documents if d.strip()]

    # 4. Generate Combinations
    combinations = list(itertools.product(
        Config.PARAM_GRID['n_neighbors'],
        Config.PARAM_GRID['n_components'],
        Config.PARAM_GRID['n_clusters'],
        Config.PARAM_GRID['min_df'],
        Config.PARAM_GRID['max_df']
    ))
    
    logging.info(f"Evaluating {len(combinations)} combinations.")

    # 5. Prepare Worker Args
    worker_args = []
    for i, combo in enumerate(combinations):
        params = {
            'n_neighbors': combo[0],
            'n_components': combo[1],
            'n_clusters': combo[2],
            'min_df': combo[3],
            'max_df': combo[4]
        }
        worker_args.append((i+1, params, documents, embeddings, tokenized_docs))

    # 6. Run Evaluation
    results = []
    logging.info("Running sequential evaluation...")
    
    for args in tqdm(worker_args):
        res = evaluate_single_combination(args)
        results.append(res)
        # Intermediate save
        pd.DataFrame(results).to_csv(output_file, index=False)

    logging.info("Done.")

if __name__ == "__main__":
    main()
