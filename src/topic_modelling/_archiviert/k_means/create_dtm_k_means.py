import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
from pathlib import Path
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import numpy as np
import torch

# Download German stopwords if not already present
nltk.download('stopwords', quiet=True)
german_stopwords = stopwords.words('german')

# Define custom stopwords (same as in dtm_evaluation.py)
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

final_stopwords = german_stopwords + analysis_stopwords

# Determine device
def get_device():
    """Determine the best available device for computation."""
    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works
            torch.cuda.init()
            return 'cuda'
        except Exception as e:
            print(f"Warning: CUDA is available but not working properly: {e}")
            print("Falling back to CPU...")
            return 'cpu'
    else:
        print("CUDA not available. Using CPU...")
        return 'cpu'

device = get_device()
print(f"Using device: {device}")

# Configure paths
data_path = "../../../data/dataset_popmusic_v8.json"
# Base path angepasst für K-Means
base_path = "../../../bertopic/dtm_k_means"
model_path = os.path.join(base_path, "models")
# Embeddings aus dem ursprünglichen dtm Ordner laden
embeddings_path = "../../../bertopic/dtm/embedding"
vis_path = os.path.join(base_path, "visualizations", "model_visualisations")

# Create directories
os.makedirs(model_path, exist_ok=True)
# embeddings_path existiert bereits, muss nicht erstellt werden (oder ist egal wenn es existiert)
os.makedirs(embeddings_path, exist_ok=True) 
os.makedirs(vis_path, exist_ok=True)

# Load data from JSON
print(f"Loading data from {data_path}...")
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract documents and timestamps
documents = []
timestamps = []

for entry in data:
    text = entry.get('text', '').strip()
    timestamp = entry.get('timestamp', '')
    
    if text and timestamp:
        # Extract year from timestamp (format: "2022-01-01T00:00:00Z")
        year = int(timestamp[:4])
        
        # Filter by year range (1955-2024 inclusive)
        if 1955 <= year <= 2024:
            documents.append(text)
            timestamps.append(year)

print(f"Loaded {len(documents)} documents with timestamps")
print(f"Year range: {min(timestamps)} - {max(timestamps)}")

# Set up sentence transformer model
embedding_model = SentenceTransformer(
    'BAAI/bge-m3',
    device=device
)

# Load or create embeddings
embeddings_file = os.path.join(embeddings_path, "embeddings.npy")
if os.path.exists(embeddings_file):
    print(f"Loading pre-computed embeddings from {embeddings_file}...")
    embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
else:
    print("Computing embeddings...")
    # Cache leeren und Batch-Size reduzieren um OOM Fehler zu vermeiden
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    embeddings = embedding_model.encode(documents, show_progress_bar=True, batch_size=8)
    
    # Save embeddings
    np.save(embeddings_file, embeddings)
    print(f"Embeddings saved to {embeddings_file}")

# =============================================================================
# PARAMETER - K-Means Version
# =============================================================================
N_NEIGHBORS = 10
N_COMPONENTS = 5
N_CLUSTERS = 85  # Festgelegte Anzahl an Clustern
# =============================================================================

# Train new BERTopic model
print("\nTraining new BERTopic model (K-Means)...")
print(f"Parameters: n_neighbors={N_NEIGHBORS}, n_components={N_COMPONENTS}, "
      f"n_clusters={N_CLUSTERS}")

# Configure UMAP
umap_model = UMAP(
    n_neighbors=N_NEIGHBORS,
    n_components=N_COMPONENTS,
    min_dist=0.0,
    metric='cosine',
    random_state=42,
    n_jobs=1,
    low_memory=False
)

# Configure K-Means
cluster_model = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

# Configure CountVectorizer with German stopwords
vectorizer_model = CountVectorizer(
    stop_words=final_stopwords,
    min_df=1,
    max_df=0.5
)

# Re-initialize embedding model on CPU for BERTopic fitting
print("Switching embedding model to CPU for BERTopic fitting...")
embedding_model = SentenceTransformer(
    'BAAI/bge-m3',
    device='cpu'
)

# Create BERTopic model
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=cluster_model, # K-Means statt HDBSCAN
    vectorizer_model=vectorizer_model,
    ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
    representation_model=None,  
    language='german',
    calculate_probabilities=False, # K-Means unterstützt dies nicht nativ wie HDBSCAN
    verbose=True
)

# Fit the model with pre-computed embeddings
print("Fitting BERTopic model...")
topics, probs = topic_model.fit_transform(documents, embeddings)

# Display basic statistics
print(f"\nStatistics:")
print(f"Number of topics found: {len(set(topics))}")
# K-Means erzeugt keine Outlier (-1), daher entfällt die Outlier-Reduktion

# Save probabilities (if any)
if probs is not None:
    probs_file = os.path.join(model_path, "probabilities.npy")
    np.save(probs_file, probs)
    print(f"Probabilities saved to {probs_file}")

# Get topic info
topic_info = topic_model.get_topic_info()
print("\nTop topics:")
print(topic_info.head(10))

# Save topic info
topic_info.to_csv(os.path.join(model_path, "topic_info.csv"), index=False)
print(f"Topic info saved to {os.path.join(model_path, 'topic_info.csv')}")

# Generate and save visualizations
print("\nGenerating visualizations...")

# 1. Hierarchy
try:
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_html(os.path.join(vis_path, "hierarchy.html"))
    print("Saved hierarchy visualization")
except Exception as e:
    print(f"Could not generate hierarchy visualization: {e}")

# 2. Documents (using UMAP for dimensionality reduction)
try:
    print("Reducing embeddings for document visualization...")
    # Reduce dimensionality of embeddings
    reduced_embeddings = UMAP(n_neighbors=N_NEIGHBORS, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig_docs = topic_model.visualize_documents(documents, reduced_embeddings=reduced_embeddings)
    fig_docs.write_html(os.path.join(vis_path, "documents.html"))
    print("Saved documents visualization")
except Exception as e:
    print(f"Could not generate documents visualization: {e}")

# 3. Barchart
try:
    fig_barchart = topic_model.visualize_barchart(top_n_topics=N_CLUSTERS)
    fig_barchart.write_html(os.path.join(vis_path, "barchart.html"))
    print("Saved barchart visualization")
except Exception as e:
    print(f"Could not generate barchart visualization: {e}")

print(f"\n=== Summary ===")
print(f"Documents processed: {len(documents)}")
print(f"Number of topics: {len(set(topics))}")
print(f"Model saved to: {model_path}")
print(f"Embeddings saved to: {embeddings_path}")
print(f"Visualizations saved to: {vis_path}")
