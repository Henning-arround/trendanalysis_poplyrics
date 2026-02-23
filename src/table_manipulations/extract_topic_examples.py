import os
import json
import pandas as pd
import numpy as np
from bertopic import BERTopic
import random
import re

# Setup Paths (Relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Models and Data from ../../../bertopic/dtm/models/ relative to src/visualisation_scripts/
MODEL_PATH = os.path.join(SCRIPT_DIR, "../../bertopic/dtm/models/bertopic_model_dtm")
PROBS_PATH = os.path.join(SCRIPT_DIR, "../../bertopic/dtm/models/probabilities.npy")
TOPIC_INFO_PATH = os.path.join(SCRIPT_DIR, "../../bertopic/dtm/models/topic_info_llm.csv")
DATA_PATH = os.path.join(SCRIPT_DIR, "../../data/dataset_popmusic_v8.json")
DNB_PATH = os.path.join(SCRIPT_DIR, "../../data/dataset_popmusic_DNB.json")
OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, "../../visualisations/examples")

def sanitize_filename(name):
    """
    Sanitizes a string to be safe for filenames.
    Replaces non-alphanumeric characters (except space, hyphen, underscore) with nothing or underscore.
    """
    if pd.isna(name):
        return "unknown"
    # Remove characters that are unsafe for filenames
    s = str(name).strip()
    return re.sub(r'[\\/*?:"<>|]', "", s)

def load_topic_labels(path):
    if not os.path.exists(path):
        print(f"Warning: Labels file not found at {path}")
        return {}
    
    try:
        df_labels = pd.read_csv(path)
        label_map = {}
        for _, row in df_labels.iterrows():
            topic_id = int(row['Topic'])
            custom = row.get('Custom_Label')
            llm = row.get('LLM_Label')
            
            # Prefer Custom_Label, then LLM_Label
            if pd.notna(custom) and str(custom).strip() != "":
                label = str(custom).strip()
            elif pd.notna(llm) and str(llm).strip() != "":
                label = str(llm).strip()
            else:
                label = row.get('Name', f"Topic {topic_id}")[:20] # Fallback
            
            label_map[topic_id] = label
        return label_map
    except Exception as e:
        print(f"Error loading labels: {e}")
        return {}

def main():
    print("Starting extraction of examples...")

    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 2. Filter Data (1955-2024) to match model training
    print("Filtering data (1955-2024)...")
    if 'timestamp' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[
            (df['timestamp_dt'].dt.year >= 1955) & 
            (df['timestamp_dt'].dt.year <= 2024)
        ]
        # Reset index to align with arrays
        df.reset_index(drop=True, inplace=True)
    else:
        print("Error: 'timestamp' column missing.")
        return

    # 3. Load Model
    print(f"Loading BERTopic model from {MODEL_PATH}...")
    try:
        topic_model = BERTopic.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Load Probabilities
    print(f"Loading probabilities from {PROBS_PATH}...")
    if os.path.exists(PROBS_PATH):
        probs = np.load(PROBS_PATH)
    else:
        print(f"Error: Probabilities file not found at {PROBS_PATH}")
        return

    # 5. Check Alignment
    n_docs = len(df)
    n_topics_assigned = len(topic_model.topics_)
    n_probs = len(probs)

    print(f"Docs: {n_docs}, Topics: {n_topics_assigned}, Probs: {n_probs}")

    if n_docs != n_topics_assigned:
        print("Error: Mismatch between dataframe length and model topics.")
        return
    
    if n_docs != n_probs:
        print("Error: Mismatch between dataframe length and probabilities.")
        # Attempt to handle if probs shape is (N, K)
        if probs.shape[0] == n_docs:
            pass # OK
        else:
            return

    # 6. Assign Topics and Probabilities to DataFrame
    df['Topic'] = topic_model.topics_
    
    # Handle prob shape
    if probs.ndim == 1:
        df['Probability'] = probs
    else:
        # If it's a matrix (N_docs, N_topics), we generally want the probability of the assigned topic
        # But usually HDBSCAN-based BERTopic returns 1D array of probabilities for the assigned topic
        # If probs is 2D, we assume probabilities[i] is the prob vector for doc i.
        # We take the probability of the assigned topic.
        # However, typically 'probabilities.npy' saved from `model.probs_` is 1D if HDBSCAN was used, 
        # or 2D if calculate_probabilities=True.
        # Let's handle 2D case by extracting prob of assigned topic:
        model_probs = []
        for i, topic in enumerate(df['Topic']):
            if topic == -1:
                model_probs.append(0.0)
            else:
                # Assuming topic index matches column index if topics are 0..K-1
                # But topic -1 is outlier. Columns are usually just topic 0..K-1.
                # If topics are mapped, this is complex.
                # Simplification: Use max probability if 2D
                model_probs.append(probs[i].max())
        df['Probability'] = model_probs

    # 7. Create Periods
    start_year = 1955
    df['year'] = df['timestamp_dt'].dt.year
    df['period_start'] = start_year + ((df['year'] - start_year) // 5) * 5
    df['period_end'] = df['period_start'] + 4
    df['period_label'] = df.apply(lambda x: f"{int(x['period_start'])}-{int(x['period_end'])}", axis=1)

    # 8. Load Labels
    label_map = load_topic_labels(TOPIC_INFO_PATH)

    # Load DNB Data for Lyrics
    print(f"Loading lyrics database from {DNB_PATH}...")
    lyrics_map = {}
    try:
        with open(DNB_PATH, 'r', encoding='utf-8') as f:
            dnb_data = json.load(f)
        
        # Create lookup map (Artist, Title) -> Lyrics
        for entry in dnb_data:
            a = str(entry.get('Künstler', '')).strip()
            t = str(entry.get('Titel', '')).strip()
            l = entry.get('Songtext', '')
            if pd.notna(l) and str(l).strip() != "":
                lyrics_map[(a, t)] = str(l)
        print(f"Loaded lyrics for {len(lyrics_map)} songs.")
    except Exception as e:
        print(f"Error loading DNB data: {e}")
        # Continue without lyrics if fail, or return? User wants them specifically.
        # But we can proceed with empty lyrics.
    
    # 9. Iterate and Extract
    print(f"Writing examples to {OUTPUT_BASE_DIR}...")
    
    # Filter for 100% probability
    # 'im selben ordner wie im modell liegen auch die probabilities.npy' -> User implies these are the probs to use.
    # User said "topic zuweisung von 100 Prozent also 1.00".
    # Using np.isclose just in case of float precision, or strict equality given '1.00' request.
    # HDBSCAN often outputs EXACT 1.0.
    
    high_conf_df = df[df['Probability'] >= 0.99999]
    print(f"Found {len(high_conf_df)} documents with ~100% probability.")

    periods = sorted(df['period_label'].unique())
    topics = sorted([t for t in df['Topic'].unique() if t != -1]) # Ignore outliers for folders

    for period in periods:
        print(f"Processing Period: {period}")
        period_base_dir = os.path.join(OUTPUT_BASE_DIR, period)
        
        # Get data for this period
        period_df = high_conf_df[high_conf_df['period_label'] == period]
        
        for topic in topics:
            topic_docs = period_df[period_df['Topic'] == topic]
            
            if len(topic_docs) == 0:
                continue

            # Get Label
            label_text = label_map.get(topic, "")
            folder_name = f"T{topic}: {label_text}"
            
            # Sanitize folder name
            folder_name = sanitize_filename(folder_name)
            
            topic_dir = os.path.join(period_base_dir, folder_name)
            
            # Sample 5 docs
            n_sample = min(5, len(topic_docs))
            # Use sample without random_state to be truly random as requested or fixed? 
            # I'll use a fixed seed for reproducibility but changing every run if desired.
            # User capped "RANDOM", I will use sample() without seed.
            sampled_docs = topic_docs.sample(n=n_sample)

            if not sampled_docs.empty:
                os.makedirs(topic_dir, exist_ok=True)

            for idx, row in sampled_docs.iterrows():
                artist = row.get('Künstler', 'Unknown Artist')
                title = row.get('Titel', 'Unknown Title')
                
                # Sanitize components
                safe_artist = sanitize_filename(artist)
                safe_title = sanitize_filename(title)
                
                filename_base = f"{safe_artist} - {safe_title}"
                
                # Lyrics: Look up from DNB map
                lyrics_key = (str(artist).strip(), str(title).strip())
                lyrics_text = lyrics_map.get(lyrics_key, "")
                
                # Fallback to current row if empty (though logic says it should be in DNB map optimally)
                if not lyrics_text or str(lyrics_text).strip() == "":
                     lyrics_text = row.get('Songtext', '')
                     if pd.isna(lyrics_text): lyrics_text = ""
                
                lyrics_path = os.path.join(topic_dir, f"{filename_base} - Lyrics.txt")
                with open(lyrics_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(str(lyrics_text))
                
                # Interpretation
                interp_text = row.get('text', '') # "Attribute 'text'" based on user prompt
                if pd.isna(interp_text): interp_text = ""

                interp_path = os.path.join(topic_dir, f"{filename_base} - Interpretation.txt")
                with open(interp_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(str(interp_text))

    print("Done.")

if __name__ == "__main__":
    main()
