import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
from bertopic import BERTopic
from umap import UMAP
import plotly.io as pio

# Helper for loading data (replicates logic from create_dtm.py)
def load_data(data_path):
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    # Replicate filtering logic from create_dtm.py to ensure alignment with embeddings
    for entry in data:
        text = entry.get('text', '').strip()
        timestamp = entry.get('timestamp', '')
        
        if text and timestamp:
            try:
                year = int(timestamp[:4])
                if 1955 <= year <= 2024:
                    documents.append(text)
            except ValueError:
                continue
    
    print(f"Loaded {len(documents)} documents.")
    return documents

def main():
    # Paths
    base_path = "../../../bertopic/dtm"
    data_path = "../../../data/dataset_popmusic_v8.json"
    
    model_path = os.path.join(base_path, "models", "bertopic_model_dtm")
    embeddings_file = os.path.join(base_path, "embedding", "embeddings.npy")
    output_path = os.path.join(base_path, "visualizations", "model_visualisations")
    
    os.makedirs(output_path, exist_ok=True)

    # 1. Load Data (for visualization context)
    documents = load_data(data_path)

    # 2. Load Embeddings
    if os.path.exists(embeddings_file):
        print(f"Loading embeddings from {embeddings_file}...")
        embeddings = np.load(embeddings_file)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Verify alignment
        if len(documents) != embeddings.shape[0]:
            print(f"WARNING: Number of documents ({len(documents)}) does not match embeddings ({embeddings.shape[0]}).")
            print("Visualization might be misaligned. Checking filtering logic...")
            # If lengths differ, we might need to rely on the model without external embeddings or check filtering again.
            # strict replication of create_dtm logic should work.
    else:
        print("Embeddings file not found. Calculating new embeddings (this might take a while)...")
        # Fallback or exit. Better to exit as re-calculating without the original model's embedding model might be inconsistent
        print("Please ensure embeddings.npy exists.")
        return

    # 3. Load Model
    print(f"Loading BERTopic model from {model_path}...")
    topic_model = BERTopic.load(model_path)

    # 4. Create Visualization (Intertopic Distance Map)
    print("Generating visualization (Intertopic Distance Map)...")
    
    fig = topic_model.visualize_topics(custom_labels=True)

    # Style updates for "Masterarbeit" (clean look)
    fig.update_layout(
        title={
            'text': "Intertopic Distance Map",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        font=dict(
            family="Arial, sans-serif", # Standard generic font
            size=12,
            color="black"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes to look cleaner
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')

    # 5. Save

    html_file = os.path.join(output_path, "intertopic_distance_map.html")
    svg_file = os.path.join(output_path, "intertopic_distance_map.svg")
    
    print(f"Saving to {html_file} and {svg_file}...")
    fig.write_html(html_file)
    fig.write_image(svg_file)
    
   
if __name__ == "__main__":
    main()
