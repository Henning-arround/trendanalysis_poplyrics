import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from tqdm import tqdm
from openai import OpenAI
from requests.exceptions import RequestException, ConnectionError, Timeout

# API-Konfiguration
my_api_key = "sk-JbSIfkSGPUcXx7OONQMZyw"
client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=my_api_key)
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"

# Konfiguration
BASE_DIRS = [
    "../../llm_as_a_judge/interpretations",
    "../../llm_as_a_judge/interpretations_prompt_engineering"
]
OUTPUT_DIR = "../../llm_as_a_judge/results/cosine_similarity_api"

# API Rate Limiting
MAX_TEXTS_PER_REQUEST = 100  # API-Limit pro Request
REQUEST_DELAY = 0.5  # Sekunden zwischen Requests
MAX_RETRIES = 5

def get_embeddings_batch(texts, max_retries=MAX_RETRIES):
    """Holt Embeddings für eine Liste von Texten via API mit Retry-Logik."""
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
            
        except (ConnectionError, Timeout, RequestException) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"\n  Netzwerkfehler (Versuch {attempt + 1}/{max_retries}). Warte {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n  Fehler nach {max_retries} Versuchen: {e}")
                raise
        except Exception as e:
            print(f"\n  Unerwarteter Fehler: {e}")
            raise
    
    return None

def get_embeddings_for_texts(texts, batch_size=MAX_TEXTS_PER_REQUEST):
    """Holt Embeddings für viele Texte in Batches."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = get_embeddings_batch(batch)
        
        if embeddings is not None:
            all_embeddings.append(embeddings)
        
        # Rate limiting zwischen Batches
        if i + batch_size < len(texts):
            time.sleep(REQUEST_DELAY)
    
    if all_embeddings:
        return np.vstack(all_embeddings)
    return None

def load_interpretations(base_dir):
    """Lädt alle Interpretationen gruppiert nach Modell und Song-ID."""
    base_path = Path(base_dir)
    interpretations = defaultdict(lambda: defaultdict(list))
    
    print(f"  Durchsuche: {base_path}")
    
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        files_found = 0
        files_skipped = 0
        
        for file_path in model_dir.glob("*.txt"):
            parts = file_path.stem.split("_")
            
            try:
                interp_index = parts.index("interpretation")
            except ValueError:
                files_skipped += 1
                continue
            
            song_id = "_".join(parts[:interp_index])
            
            if interp_index + 1 < len(parts):
                try:
                    iteration = int(parts[interp_index + 1])
                except ValueError:
                    print(f"    WARNUNG: Konnte Iteration nicht parsen: {file_path.name}")
                    files_skipped += 1
                    continue
            else:
                print(f"    WARNUNG: Keine Iteration gefunden: {file_path.name}")
                files_skipped += 1
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                print(f"    WARNUNG: Leere Datei: {file_path.name}")
                files_skipped += 1
                continue
            
            interpretations[model_name][song_id].append({
                'iteration': iteration,
                'text': text,
                'file': file_path.name
            })
            files_found += 1
        
        if files_found > 0:
            unique_songs = len(interpretations[model_name])
            print(f"    Modell '{model_name}': {files_found} Dateien geladen, {unique_songs} verschiedene Songs")
            
            incomplete_songs = []
            for song_id, interps in interpretations[model_name].items():
                if len(interps) != 10:
                    incomplete_songs.append(f"{song_id} ({len(interps)} Interpretationen)")
            
            if incomplete_songs:
                print(f"    WARNUNG: Songs mit weniger als 10 Interpretationen:")
                for s in incomplete_songs[:5]:
                    print(f"      - {s}")
                if len(incomplete_songs) > 5:
                    print(f"      ... und {len(incomplete_songs) - 5} weitere")
        
        if files_skipped > 0:
            print(f"    {files_skipped} Dateien übersprungen (falsches Format oder leer)")
    
    return interpretations

def compute_similarity_for_group(texts):
    """Berechnet Cosine Similarity für eine Gruppe von Texten via API."""
    if len(texts) < 2:
        return None
    
    # Embeddings via API erstellen
    embeddings = get_embeddings_for_texts(texts)
    
    if embeddings is None:
        return None
    
    # Cosine Similarity Matrix berechnen
    similarity_matrix = cosine_similarity(embeddings)
    
    # Nur obere Dreiecksmatrix (ohne Diagonale) für pairwise similarities
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    pairwise_similarities = similarity_matrix[upper_triangle_indices]
    
    return {
        'matrix': similarity_matrix,
        'pairwise': pairwise_similarities,
        'mean': np.mean(pairwise_similarities),
        'std': np.std(pairwise_similarities),
        'min': np.min(pairwise_similarities),
        'max': np.max(pairwise_similarities),
        'median': np.median(pairwise_similarities)
    }

def analyze_similarities(interpretations, experiment_name):
    """Analysiert Similarities für alle Modelle und Songs."""
    results = []
    detailed_results = {}
    
    for model_name, songs in interpretations.items():
        print(f"  Verarbeite Modell: {model_name}")
        model_results = []
        detailed_results[model_name] = {}
        
        songs_analyzed = 0
        songs_skipped = 0
        
        # Progress bar für Songs
        for song_id, interpretations_list in tqdm(songs.items(), desc=f"    Songs ({model_name})", leave=False):
            interpretations_list.sort(key=lambda x: x['iteration'])
            texts = [item['text'] for item in interpretations_list]
            
            if len(texts) < 2:
                print(f"    WARNUNG: Song {song_id} hat nur {len(texts)} Interpretation(en), überspringe")
                songs_skipped += 1
                continue
            
            if len(texts) != 10:
                print(f"    INFO: Song {song_id} hat {len(texts)} statt 10 Interpretationen")
            
            sim_data = compute_similarity_for_group(texts)
            
            if sim_data:
                results.append({
                    'experiment': experiment_name,
                    'embedding_model': EMBEDDING_MODEL,
                    'llm_model': model_name,
                    'song_id': song_id,
                    'n_interpretations': len(texts),
                    'mean_similarity': sim_data['mean'],
                    'std_similarity': sim_data['std'],
                    'min_similarity': sim_data['min'],
                    'max_similarity': sim_data['max'],
                    'median_similarity': sim_data['median']
                })
                
                model_results.append(sim_data['mean'])
                detailed_results[model_name][song_id] = sim_data
                songs_analyzed += 1
        
        if model_results:
            print(f"    {songs_analyzed} Songs analysiert, {songs_skipped} übersprungen, Ø Similarity: {np.mean(model_results):.4f}")
    
    return pd.DataFrame(results), detailed_results

def create_comparison_boxplots(df, output_dir):
    """Erstellt Vergleichs-Boxplots."""
    output_path = Path(output_dir)
    
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x='llm_model', y='mean_similarity', hue='experiment')
    plt.title(f'Cosine Similarity: Standard vs. Prompt Engineering\nEmbedding Model: {EMBEDDING_MODEL}')
    plt.xlabel('LLM Model')
    plt.ylabel('Mean Cosine Similarity')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Experiment')
    plt.tight_layout()
    plt.savefig(output_path / 'comparison_experiments_api.png', dpi=300)
    plt.close()
    print(f"  ✓ Erstellt: comparison_experiments_api.png")

def create_summary_statistics(df, output_dir):
    """Erstellt zusammenfassende Statistiken."""
    output_path = Path(output_dir)
    
    summary = df.groupby(['experiment', 'llm_model']).agg({
        'mean_similarity': ['mean', 'std', 'min', 'max', 'median'],
        'song_id': 'count'
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'song_id_count': 'n_songs'})
    
    summary.to_csv(output_path / 'summary_statistics_api.csv')
    
    with open(output_path / 'summary_statistics_api.txt', 'w') as f:
        f.write(f"=== COSINE SIMILARITY ANALYSIS (API: {EMBEDDING_MODEL}) ===\n\n")
        f.write(summary.to_string())
        f.write("\n\n")
    
    return summary

def main():
    print("Starte Similarity-Analyse mit API-Embeddings...")
    print(f"Embedding-Modell: {EMBEDDING_MODEL}")
    print(f"Erwartete Struktur: 100 Songs × 10 Interpretationen = 1000 Dateien pro Modell\n")
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    all_detailed = {}
    
    for base_dir in BASE_DIRS:
        experiment_name = Path(base_dir).name.replace("interpretations", "").strip("_") or "standard"
        print(f"\nAnalysiere: {experiment_name}")
        
        interpretations = load_interpretations(base_dir)
        
        if not interpretations:
            print(f"  Keine Interpretationen gefunden in {base_dir}")
            continue
        
        df, detailed = analyze_similarities(interpretations, experiment_name)
        all_results.append(df)
        all_detailed[experiment_name] = detailed
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        print("\n=== INTERPRETATIONEN PRO SONG ===")
        interp_counts = combined_df.groupby(['experiment', 'llm_model', 'n_interpretations']).size()
        print(interp_counts)
        
        combined_df.to_csv(output_path / 'all_results_api.csv', index=False)
        
        print("\n=== ERSTELLE VISUALISIERUNGEN ===")
        create_comparison_boxplots(combined_df, output_path)
        
        summary = create_summary_statistics(combined_df, output_path)
        print("\n=== SUMMARY ===")
        print(summary)
        
        simplified_detailed = {}
        for exp, models in all_detailed.items():
            simplified_detailed[exp] = {}
            for model, songs in models.items():
                simplified_detailed[exp][model] = {
                    song_id: {
                        'mean': float(data['mean']),
                        'std': float(data['std']),
                        'min': float(data['min']),
                        'max': float(data['max']),
                        'median': float(data['median'])
                    }
                    for song_id, data in songs.items()
                }
        
        with open(output_path / 'detailed_results_api.json', 'w') as f:
            json.dump(simplified_detailed, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Ergebnisse gespeichert in: {output_path}")
        print(f"  - all_results_api.csv: Alle Einzelergebnisse")
        print(f"  - summary_statistics_api.csv: Zusammenfassung")
        print(f"  - detailed_results_api.json: Detaillierte Ergebnisse pro Song")
        print(f"  - comparison_experiments_api.png: Vergleichs-Boxplot")
        print(f"{'='*60}")
    else:
        print("Keine Ergebnisse zum Speichern gefunden.")

if __name__ == "__main__":
    main()
