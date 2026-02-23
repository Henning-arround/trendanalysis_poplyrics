import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Configure paths
data_path = "../../../data/dataset_popmusic_v8.json"
base_path = "../../../bertopic/dtm"
model_path = os.path.join(base_path, "models", "bertopic_model_dtm")
output_dir = os.path.join(base_path, "visualizations", "over_time")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Download German stopwords
nltk.download('stopwords', quiet=True)
german_stopwords = stopwords.words('german')

# Define custom stopwords (same as in create_dtm.py)
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

# Configure CountVectorizer (same as in create_dtm.py)
vectorizer_model = CountVectorizer(
    stop_words=final_stopwords,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.6
)

# Load data to get timestamps (needed for topics_over_time calculation)
print(f"Loading data from {data_path}...")
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

documents = []
timestamps = []

for entry in data:
    text = entry.get('text', '').strip()
    timestamp = entry.get('timestamp', '')
    
    if text and timestamp:
        # Extract year from timestamp (format: "2022-01-01T00:00:00Z")
        year = int(timestamp[:4])
        
        if year >= 1955:
            documents.append(text)
            
            # Create 5-year bins (use start year as integer for BERTopic processing)
            # e.g. 1955 -> 1955 (representing 1955-1959)
            bin_start = (year // 5) * 5
            timestamps.append(bin_start)

print(f"Loaded {len(documents)} documents with timestamps starting from 1955")

# Load the pre-trained model
print(f"Loading BERTopic model from {model_path}...")
topic_model = BERTopic.load(model_path)

# Ensure the loaded model uses the correct vectorizer
topic_model.vectorizer_model = vectorizer_model

# Outlier Reduction
# Wenn du Outlier (Topic -1) reduzieren möchtest, bevor du die Topics over Time berechnest,
# solltest du das hier tun. Das weist Dokumente aus Topic -1 den nächstgelegenen Topics zu.
print("Reducing outliers...")

# Load pre-calculated probabilities
probs_path = os.path.join(base_path, "models", "probabilities.npy")
if os.path.exists(probs_path):
    print(f"Loading probabilities from {probs_path}...")
    probs = np.load(probs_path)
    
    # Filter probabilities to match the filtered documents (since we filtered by year >= 1955)
    # Note: This assumes the probabilities.npy corresponds exactly to the original full dataset order.
    # If the original dataset had entries < 1955, we need to filter the probs array too.
    # Assuming the 'data' list order matches 'probs' index:
    valid_indices = []
    for idx, entry in enumerate(data):
        text = entry.get('text', '').strip()
        timestamp = entry.get('timestamp', '')
        if text and timestamp:
            year = int(timestamp[:4])
            if year >= 1955:
                valid_indices.append(idx)
    
    # If the loaded probs match the original full data size, filter them
    if len(probs) > len(documents):
        print("Warning: Filtering probabilities to match filtered documents...")
        probs = probs[valid_indices]

    # Load embeddings to speed up transform
    embeddings_path = os.path.join(base_path, "embedding", "embeddings.npy")
    embeddings = None
    if os.path.exists(embeddings_path):
        print(f"Loading embeddings from {embeddings_path}...")
        embeddings = np.load(embeddings_path)
        # Filter embeddings if necessary
        if len(embeddings) > len(documents):
            print("Warning: Filtering embeddings to match filtered documents...")
            embeddings = embeddings[valid_indices]

    
    topics, _ = topic_model.transform(documents, embeddings=embeddings) 

    # Wir nutzen die probabilities Strategie mit einem Threshold von 0.03
    new_topics = topic_model.reduce_outliers(documents, topics, probabilities=probs, strategy="probabilities", threshold=0.03)
    topic_model.update_topics(documents, topics=new_topics, vectorizer_model=vectorizer_model)
else:
    print("Warning: probabilities.npy not found. Skipping outlier reduction or recalculating...")


print("Calculating topics over time...")
topics_over_time = topic_model.topics_over_time(documents, timestamps)

# Load LLM topic info for global representations and categories
llm_info_path = os.path.join(base_path, "models", "topic_info_llm.csv")
print(f"Loading LLM topic info from {llm_info_path}...")
llm_topic_info = pd.read_csv(llm_info_path)

# Create mappings
if 'Custom_Label' in llm_topic_info.columns:
    # Use Custom_Label if available, fallback to LLM_Label for NaN values
    llm_labels = dict(zip(llm_topic_info['Topic'], llm_topic_info['Custom_Label'].fillna(llm_topic_info['LLM_Label'])))
else:
    llm_labels = dict(zip(llm_topic_info['Topic'], llm_topic_info['LLM_Label']))


# Convert Timestamp column to string range format (e.g., "1955-1959")
topics_over_time["Timestamp"] = topics_over_time["Timestamp"].apply(lambda x: f"{int(x)}-{int(x)+4}")

# Add global representation (LLM Label) and Category to the DataFrame
topics_over_time['Global_Representation'] = topics_over_time['Topic'].map(llm_labels)


# Save to CSV
output_file = os.path.join(base_path, "models", "topics_over_time.csv")
topics_over_time.to_csv(output_file, index=False)
print(f"Topics over time data saved to {output_file}")
