import os
import json
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import matplotlib as mpl

# Configure plot style for LaTeX
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'text.usetex': True
})

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DTM_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../bertopic/dtm"))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../data"))

MODEL_PATH = os.path.join(DTM_PATH, "models/bertopic_model_dtm")
JSON_PATH = os.path.join(DATA_PATH, "dataset_popmusic_v8.json")
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../visualisations/topic_model/modell_visualisierungen"))

# --- 1. Load Data ---
print("Loading Data...")
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    all_docs = json.load(f)
df_docs = pd.DataFrame(all_docs)

# --- Filter Documents EXACTLY like create_dtm.py ---
# Step 1 & 2: Handle missing/empty values
df_docs['text'] = df_docs['text'].fillna('').str.strip()
df_docs['timestamp'] = df_docs['timestamp'].fillna('')

# Step 3: Filter out empty text or empty timestamp
df_docs = df_docs[(df_docs['text'] != '') & (df_docs['timestamp'] != '')]

# Step 4: Parse Year
def parse_year(date_str):
    try:
        return int(str(date_str)[:4])
    except:
        return None

df_docs['Year'] = df_docs['timestamp'].apply(parse_year)

# Step 5: Filter Year Range (1955-2024)
df_docs = df_docs[(df_docs['Year'] >= 1955) & (df_docs['Year'] <= 2024)].reset_index(drop=True)

print(f"Filtered Documents: {len(df_docs)}")
docs = df_docs['text'].tolist()

# --- 2. Load Model ---
print("Loading Model...")
topic_model = BERTopic.load(MODEL_PATH)

# Align docs if necessary (safety check)
if len(docs) != len(topic_model.topics_):
    print(f"CRITICAL WARNING: Doc count ({len(docs)}) != Model Topic count ({len(topic_model.topics_)}). Truncating docs.")
    min_len = min(len(docs), len(topic_model.topics_))
    docs = docs[:min_len]

# --- 2.1 Apply Vectorizer (Stopwords) ---
print("Applying Vectorizer and Stopwords...")
nltk.download('stopwords', quiet=True)
german_stopwords = stopwords.words('german')

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

vectorizer_model = CountVectorizer(
    stop_words=final_stopwords,
    ngram_range=(1, 2),
    min_df=7,
    max_df=0.4
)

# Update the model with the vectorizer to ensure hierarchy uses correct stopwords
topic_model.update_topics(docs, vectorizer_model=vectorizer_model)

# --- 2.2 Load Custom Labels ---
print("Loading Custom Labels...")
topic_info_path = os.path.join(os.path.dirname(MODEL_PATH), "topic_info_llm.csv")
if os.path.exists(topic_info_path):
    df_labels = pd.read_csv(topic_info_path)
    label_mapping = {}
    for _, row in df_labels.iterrows():
        topic = int(row['Topic'])
        
        # Check Custom_Label first
        if 'Custom_Label' in row and pd.notna(row['Custom_Label']) and str(row['Custom_Label']).strip() != "":
            label = str(row['Custom_Label']).strip()
        # Check LLM_Label
        elif 'LLM_Label' in row and pd.notna(row['LLM_Label']) and str(row['LLM_Label']).strip() != "":
            label = str(row['LLM_Label']).strip()
        else:
            # Fallback to default name if available or generated ID
            label = row['Name'] if 'Name' in row else f"Topic {topic}"
            
        label_mapping[topic] = label
        
    topic_model.set_topic_labels(label_mapping)
    print("Custom labels set.")
else:
    print(f"Warning: {topic_info_path} not found. Using default labels.")

# --- 3. Generate Hierarchy ---
print("Generating Hierarchical Topics...")
hierarchical_topics = topic_model.hierarchical_topics(docs)

# --- 4. Visualize ---
print("Creating Visualization...")
fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics, custom_labels=True)

# --- 5. Save ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "hierarchy.html")
fig.write_html(output_path)
print(f"Hierarchical visualization saved to: {output_path}")

output_path_svg = os.path.join(OUTPUT_DIR, "hierarchy.svg")
fig.write_image(output_path_svg)
print(f"Hierarchical visualization (SVG) saved to: {output_path_svg}")

output_path_png = os.path.join(OUTPUT_DIR, "hierarchy.png")
fig.write_image(output_path_png, scale=3)
print(f"Hierarchical visualization (High-Res PNG) saved to: {output_path_png}")
