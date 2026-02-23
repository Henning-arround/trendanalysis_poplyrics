import os
import json
import gc
import warnings
import itertools
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set
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
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NEU: Gensim Logging auf ERROR setzen, um den "WordOccurrenceAccumulator" Spam zu unterdrücken
# Erklärung: Die hohe Zahl (12 Mio) entsteht durch das Sliding-Window der C_v Metrik (jedes Fenster = 1 Doc).
logging.getLogger('gensim').setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

# Download NLTK resources once
nltk.download('stopwords', quiet=True)


# --- CONFIGURATION ---

class Config:
    DATA_PATH = Path("../../../data/dataset_popmusic_v8.json")
    BASE_PATH = Path("../../../bertopic/dtm")
    EVAL_PATH = BASE_PATH / "evaluation_parameters"
    EMBEDDINGS_PATH = BASE_PATH / "embedding/embeddings.npy"
    
    # NEU: Einstellung für Outlier Reduction
    # Wenn False: Keine Reduktion, Kohärenz wird nur einmal berechnet.
    PERFORM_OUTLIER_REDUCTION = True 
    
    # Parameter Grid
    PARAM_GRID = {
        'n_neighbors': [30,50,70,90],
        'n_components': [5],
        'min_cluster_size': [100,150,200,250],
        'min_samples': [50,90,130,170],
        'cluster_selection_epsilon': [0.0],
        'min_df': [1],
        'max_df': [0.5],
        'cluster_selection_method': ['eom']
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
    """Load documents from JSON with DEBUGGING output."""
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
        
    print(f"DEBUG: Lade JSON von {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"DEBUG: JSON geladen. Datentyp: {type(data)}")
    
    # Fall 1: JSON ist ein Dictionary (oft bei Pandas Export)
    if isinstance(data, dict):
        print("DEBUG: Daten sind ein DICT, keine LISTE. Versuche Werte zu extrahieren...")
        # Versuch, ob es ein Dict von Records ist
        if all(isinstance(v, dict) for v in list(data.values())[:5]):
             data = list(data.values())
        # Versuch, ob das Dict einen Hauptkey hat (z.B. {'data': [...]})
        elif 'data' in data and isinstance(data['data'], list):
             data = data['data']
        else:
             print("WARNUNG: Unbekannte Dict-Struktur!")

    print(f"DEBUG: Untersuche {len(data)} Einträge...")
    
    documents = []
    skipped_count = 0
    empty_text_count = 0
    wrong_type_count = 0
    skipped_year_count = 0
    
    # Wir schauen uns den ersten Eintrag genau an
    if len(data) > 0:
        print(f"DEBUG: Erster Eintrag (Keys): {data[0].keys() if isinstance(data[0], dict) else data[0]}")
    
    for item in data:
        if not isinstance(item, dict):
            wrong_type_count += 1
            continue
            
        # Hier prüfen wir den Key explizit
        text = item.get('text', '')
        timestamp = item.get('timestamp', '')
        
        # Fallback falls Key anders heißt (häufige Fehlerquellen)
        if not text:
            text = item.get('Text', '') or item.get('lyrics', '') or item.get('body', '')
            
        if not text or not isinstance(text, str) or not text.strip():
            empty_text_count += 1
            continue
            
        # Filter by year (1955-2024) - Match logic from create_dtm.py
        if not timestamp:
            skipped_year_count += 1
            continue
            
        try:
            year = int(timestamp[:4])
            if not (1955 <= year <= 2024):
                skipped_year_count += 1
                continue
        except (ValueError, TypeError):
            skipped_year_count += 1
            continue

        documents.append(text.strip())

    print(f"DEBUG: === LADE BERICHT ===")
    print(f" -> Rohdaten Einträge: {len(data)}")
    print(f" -> Erfolgreich geladen: {len(documents)}")
    print(f" -> Übersprungen (kein Dict): {wrong_type_count}")
    print(f" -> Übersprungen (leerer Text/falscher Key): {empty_text_count}")
    print(f" -> Übersprungen (Jahr nicht 1955-2024): {skipped_year_count}")
    
    if len(documents) < 100:
        print("!!! KRITISCHER FEHLER: Fast keine Dokumente geladen. Prüfe Key-Namen in JSON !!!")
        # Zeige Beispiel eines fehlgeschlagenen Items
        if len(data) > 0:
            print(f"Beispiel Roh-Item: {data[0]}")
            
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
        if not topics or (len(topics) == 1 and -1 in topics):
            return 0.0

        # Create Dictionary (optimized parameters)
        dictionary = Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=3, no_above=0.85)
        
        if not dictionary:
            return 0.0

        topic_words = []
        for topic_id in topics:
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
        
    except Exception as e:
        # logging.error(f"Coherence calc error: {e}")
        return 0.0


def calculate_diversity(topic_model: BERTopic, top_n: int = 10) -> float:
    """Calculate topic diversity (percentage of unique words in top N words across all topics)."""
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


def get_topic_stats(topics: List[int], total_docs: int) -> Dict[str, float]:
    """Calculate common topic statistics like coverage and specific topic ratios."""
    n_outliers = sum(1 for t in topics if t == -1)
    
    stats = {
        'n_topics': len(set(topics)) - (1 if -1 in topics else 0),
        'n_outliers': n_outliers,
        'outlier_ratio': n_outliers / len(topics) if topics else 0,
        'topic_coverage': (len(topics) - n_outliers) / total_docs if total_docs else 0,
        'ratio_topic_0': sum(1 for t in topics if t == 0) / len(topics) if topics else 0,
        'ratio_topic_1': sum(1 for t in topics if t == 1) / len(topics) if topics else 0,
    }
    return stats


# --- WORKER FUNCTION ---

def evaluate_single_combination(args) -> Dict[str, Any]:
    """
    Evaluate a single parameter combination.
    Designed for multiprocessing: Model creation happens inside to avoid pickling issues.
    """
    idx, params, documents, embeddings, tokenized_docs = args
    
   
    start_time = datetime.now()
    
    # Unpack params
    n_neighbors = params['n_neighbors']
    n_components = params['n_components']
    min_cluster_size = params['min_cluster_size']
    min_samples = params['min_samples']
    epsilon = params['cluster_selection_epsilon']
    min_df = params['min_df']
    max_df = params['max_df']
    cluster_selection_method = params['cluster_selection_method']
    
    print(f"[{idx}] Start: neighbors={n_neighbors}, min_cluster={min_cluster_size}, min_samples={min_samples}, method={cluster_selection_method}")

    try:
        # 1. Initialize Models
        # Force CPU for Embedding Model in worker (embeddings are pre-computed, only needed for checks)
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
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method=cluster_selection_method,
            prediction_data=True,
            cluster_selection_epsilon=epsilon,
            core_dist_n_jobs=1
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
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            representation_model=None,
            language='german',
            calculate_probabilities=True,
            verbose=False
        )
        
        # 2. Fit Model
        topics, probs = topic_model.fit_transform(documents, embeddings)
        
        # 3. Base Metrics (Before Reduction)
        stats_before = get_topic_stats(topics, len(documents))
        coherence_before = calculate_coherence(topic_model, tokenized_docs)
        diversity_before = calculate_diversity(topic_model)
        
        if Config.PERFORM_OUTLIER_REDUCTION:
            # 4. Outlier Reduction (Probabilities, Threshold 0.03)
            try:
                topics_reduced = topic_model.reduce_outliers(
                    documents, topics, strategy="probabilities", probabilities=probs, threshold=0.03
                )
                topic_model.update_topics(documents, topics=topics_reduced, vectorizer_model=vectorizer)
            except Exception as e:
                print(f"[{idx}] Warn: Reduction failed: {e}")
                topics_reduced = topics

            stats_after = get_topic_stats(topics_reduced, len(documents))
            coherence_after = calculate_coherence(topic_model, tokenized_docs)
            diversity_after = calculate_diversity(topic_model)
        else:
            # Skip reduction
            stats_after = stats_before
            coherence_after = coherence_before
            diversity_after = diversity_before
            
            print(f"[{idx}] Skipped outlier reduction (Config).")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Clean up
        del topic_model, umap_model, hdbscan_model, embedding_model
        clear_memory()
        
        print(f"[{idx}] ✓ Done ({duration:.1f}s) | Topics: {stats_before['n_topics']} -> {stats_after['n_topics']} | "
              f"Coh: {coherence_after:.4f} | Div: {diversity_after:.4f}")

        return {
            **params,
            # Before
            'n_topics_before': stats_before['n_topics'],
            'outlier_ratio_before': stats_before['outlier_ratio'],
            'topic_coverage_before': stats_before['topic_coverage'],
            'coherence_before': coherence_before,
            'diversity_before': diversity_before,
            # After (Main Metric)
            'n_topics_after': stats_after['n_topics'],
            'outlier_ratio_after': stats_after['outlier_ratio'],
            'topic_coverage': stats_after['topic_coverage'],
            'ratio_topic_0': stats_after['ratio_topic_0'],
            'ratio_topic_1': stats_after['ratio_topic_1'],
            'coherence': coherence_after,
            'diversity': diversity_after,
            # Meta
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
    output_file = Config.EVAL_PATH / "results_eom_leaf_1.csv"

    logging.info("!!! STARTING NEW FRESH RUN !!!")
    
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
    
    # 3. Pre-tokenize for Coherence (Speedup)
    logging.info("Pre-tokenizing documents...")
    tokenized_docs = [d.lower().split() for d in documents if d.strip()]

    # 4. Generate Combinations
    combinations = list(itertools.product(
        Config.PARAM_GRID['n_neighbors'],
        Config.PARAM_GRID['n_components'],
        Config.PARAM_GRID['min_cluster_size'],
        Config.PARAM_GRID['min_samples'],
        Config.PARAM_GRID['cluster_selection_epsilon'],
        Config.PARAM_GRID['min_df'],
        Config.PARAM_GRID['max_df'],
        Config.PARAM_GRID['cluster_selection_method']
    ))
    
    # Filter illogical combos (min_samples < min_cluster_size)
    # (neighbors, components, min_cluster, min_samples, epsilon, min_df, max_df, method)
    combinations = [c for c in combinations if c[3] < c[2]]
    
    logging.info(f"Evaluating {len(combinations)} combinations.")

    # 5. Prepare Worker Args
    worker_args = []
    for i, combo in enumerate(combinations):
        params = {
            'n_neighbors': combo[0],
            'n_components': combo[1],
            'min_cluster_size': combo[2],
            'min_samples': combo[3],
            'cluster_selection_epsilon': combo[4],
            'min_df': combo[5],
            'max_df': combo[6],
            'cluster_selection_method': combo[7]
        }
        worker_args.append((i+1, params, documents, embeddings, tokenized_docs))

    # 6. Run Evaluation
    results = []
    logging.info("Running sequential evaluation...")
    
    for args in tqdm(worker_args):
        res = evaluate_single_combination(args)
        results.append(res)
        # Intermediate save after each run
        pd.DataFrame(results).to_csv(output_file, index=False)

    logging.info("Done.")

if __name__ == "__main__":
    main()