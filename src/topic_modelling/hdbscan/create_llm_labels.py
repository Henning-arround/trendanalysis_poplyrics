import pandas as pd
import os
import sys
import ast
import time
import json
import random
import numpy as np
from bertopic import BERTopic
from openai import OpenAI
from requests.exceptions import Timeout, RequestException

def create_prompt(keywords, docs):
    """Erstelle den Labeling-Prompt"""
    # Dokumente formatieren
    formatted_docs_list = []
    for d in docs[:10]:  # 10 Dokumente
        # Zeilenumbrüche entfernen für sauberes Format, aber vollen Text behalten
        clean_doc = str(d).replace('\n', ' ')
        formatted_docs_list.append(clean_doc)
    
    formatted_docs = "\n- ".join(formatted_docs_list)
    
    prompt = f"""Kontext: Topic Modeling Analyse von Songtexten.
Gegeben sind Keywords und Beispieltexte, die ein semantisches Cluster bilden.
Wichtig: Die Reihenfolge der Keywords und Beispiele ist zufällig und impliziert keine Gewichtung.

Thema Keywords: {", ".join(keywords)}

Beispiele:
{formatted_docs}

Aufgabe: Erstelle ein kurzes, prägnantes deutsches Label (max 3 bis 4 Wörter), das das gemeinsame Thema der Texte zusammenfasst. Neutral und keine ausgefallenen narrativen Erzählungen.
Antworte ausschließlich mit dem Label."""
    return prompt

def create_llm_labels():
    # Define paths
    base_dir = os.path.dirname(__file__)
    bertopic_dir = os.path.join(base_dir, "../../../bertopic/dtm/models")
    csv_path = os.path.join(bertopic_dir, "topic_info.csv")
    model_path = os.path.join(bertopic_dir, "bertopic_model_dtm") 
    probs_path = os.path.join(bertopic_dir, "probabilities.npy")
    data_path = os.path.join(base_dir, "../../../data/dataset_popmusic_v8.json")

    print(f"Loading data from {data_path}...")
    documents = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for entry in data:
            text = entry.get('text', '').strip()
            timestamp = entry.get('timestamp', '')
            if text and timestamp:
                year = int(timestamp[:4])
                if 1955 <= year <= 2024:
                    documents.append(text)
        print(f"Loaded {len(documents)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        documents = []

    probs = None
    if os.path.exists(probs_path):
        print(f"Loading probabilities from {probs_path}...")
        probs = np.load(probs_path)
    else:
        print("Warning: Probabilities file not found.")

    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Skip model loading for now to focus on labels
    topic_model = None
    if os.path.exists(model_path):
        try:
            topic_model = BERTopic.load(model_path)
        except Exception as e:
            print(f"Warning: Could not load BERTopic model directly: {e}")

    # API-Konfiguration
    my_api_key = "#######################"
    client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=my_api_key)
    
    
    model_name = "openai/gpt-oss-120b"
    
    max_retries = 3
    retry_delay = 2
    new_labels = []

    print(f"Generating LLM labels using model: {model_name}...")

    topics = None
    if topic_model:
        topics = topic_model.topics_

    for index, row in df.iterrows():
        topic_id = row['Topic']
        
        # Parse string representations of lists back to actual lists
        try:
            keywords = ast.literal_eval(row['Representation'])
            docs = ast.literal_eval(row['Representative_Docs'])
        except Exception as e:
            print(f"Error parsing row {index}: {e}")
            keywords = []
            docs = []
            
        # Try to sample high probability docs
        if topics is not None and probs is not None and len(documents) > 0:
            try:
                # Filter indices for current topic
                current_topic_indices = [i for i, t in enumerate(topics) if t == topic_id]
                
                high_prob_indices = []
                for i in current_topic_indices:
                    # Handle both 1D and 2D probs
                    p = 0.0
                    if len(probs.shape) == 1:
                        p = probs[i]
                    else:
                        # Assuming 2D, take max prob (which should correspond to assigned topic)
                        p = np.max(probs[i])
                    
                    if p >= 0.9999: # Probability ~ 1.00
                        high_prob_indices.append(i)
                
                if high_prob_indices:
                    # Select 10 random documents (or fewer if not enough available)
                    if len(high_prob_indices) >= 10:
                        selected_indices = random.sample(high_prob_indices, 10)
                    else:
                        selected_indices = high_prob_indices
                    
                    # Override docs
                    docs = [documents[i] for i in selected_indices]
            except Exception as e:
                print(f"Warning: Could not sample high-prob docs for topic {topic_id}: {e}")
        
        prompt = create_prompt(keywords, docs)
        label = ""
        filename = f"Topic {topic_id}"

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                    temperature=0
                )
                
                content = response.choices[0].message.content
                if content:
                    label = content.strip().replace('"', '').replace("Label:", "").strip()
                    # Erste Zeile nehmen, falls Modell mehr schreibt
                    label = label.split('\n')[0].strip()
                
                if not label:
                    label = f"Topic_{topic_id}"
                    print(f"Warnung: Leere Antwort für {filename}")
                break
                
            except (ConnectionError, Timeout, RequestException) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"\nNetzwerkfehler bei {filename}. Warte {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"\nFehler bei {filename}: {e}")
                    label = f"Topic_{topic_id}"
            except Exception as e:
                print(f"\nUnerwarteter Fehler bei {filename}: {e}")
                label = f"Topic_{topic_id}"
                break
        
        new_labels.append(label)
        print(f"Topic {topic_id}: {label}")

    # Save to CSV
    df['LLM_Label'] = new_labels
    output_csv = os.path.join(bertopic_dir, "topic_info_llm.csv")
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")

    # Update Model if loaded
    if topic_model:
        # Map Topic ID to Label
        label_mapping = {row['Topic']: label for row, label in zip(df.to_dict('records'), new_labels)}
        topic_model.set_topic_labels(label_mapping)
        
        save_path = os.path.join(bertopic_dir, "bertopic_model_llm")
        topic_model.save(save_path)
        print(f"Updated model saved to {save_path}")

if __name__ == "__main__":
    create_llm_labels()