import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import warnings

# Unterdrücke die sentence-transformers Warnung
warnings.filterwarnings('ignore', message='No sentence-transformers model found')

# Konfiguration
BASE_DIRS = [
    "../../llm_as_a_judge/interpretations",
    "../../llm_as_a_judge/interpretations_prompt_engineering"
]
OUTPUT_DIR = "../../llm_as_a_judge/results/cosine_similarity"

# Mehrere Embedding-Modelle zum Vergleich
EMBEDDING_MODELS = {
    # Top-Empfehlung für Deutsch:
    'gbert-large': 'deutsche-telekom/gbert-large-paraphrase-cosine',
    
    # Multilingual, sehr gut für Deutsch:
    'multilingual-e5-large': 'intfloat/multilingual-e5-large',
    
    # Speziell für deutschen Semantic Search:
    'gbert-base': 'PM-AI/bi-encoder_msmarco_bert-base_german',
    
    # Cross-lingual Deutsch-Englisch:
    'cross-de': 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer',
    
    # Falls du auch Vergleich willst:
    'paraphrase-multi': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
}

# GPU-Konfiguration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Batch-Größe für GPU-Verarbeitung

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
            # Parse filename: 
            # Format 1: "724_interpretation_1.txt"
            # Format 2: "30008_chartsurfer_interpretation_10.txt"
            parts = file_path.stem.split("_")
            
            # Finde den Index von "interpretation"
            try:
                interp_index = parts.index("interpretation")
            except ValueError:
                files_skipped += 1
                continue
            
            # Song-ID ist alles vor "interpretation"
            song_id = "_".join(parts[:interp_index])
            
            # Iteration ist der Teil nach "interpretation"
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
            
            # Prüfe welche Songs nicht 10 Interpretationen haben
            incomplete_songs = []
            for song_id, interps in interpretations[model_name].items():
                if len(interps) != 10:
                    incomplete_songs.append(f"{song_id} ({len(interps)} Interpretationen)")
            
            if incomplete_songs:
                print(f"    WARNUNG: Songs mit weniger als 10 Interpretationen:")
                for s in incomplete_songs[:5]:  # Zeige max. 5 Beispiele
                    print(f"      - {s}")
                if len(incomplete_songs) > 5:
                    print(f"      ... und {len(incomplete_songs) - 5} weitere")
        
        if files_skipped > 0:
            print(f"    {files_skipped} Dateien übersprungen (falsches Format oder leer)")
    
    return interpretations

def compute_similarity_for_group(texts, embedding_model):
    """Berechnet Cosine Similarity für eine Gruppe von Texten."""
    if len(texts) < 2:
        return None
    
    # Embeddings erstellen mit Batch-Verarbeitung auf GPU
    embeddings = embedding_model.encode(
        texts, 
        show_progress_bar=False,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    # ✅ NEU: Vektoren normalisieren (L2-Norm = 1)
    embeddings = normalize(embeddings, norm='l2')
    
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

def analyze_similarities(interpretations, embedding_model, experiment_name, embedding_model_name):
    """Analysiert Similarities für alle Modelle und Songs."""
    results = []
    detailed_results = {}
    
    for model_name, songs in interpretations.items():
        print(f"  Verarbeite Modell: {model_name}")
        model_results = []
        detailed_results[model_name] = {}
        
        songs_analyzed = 0
        songs_skipped = 0
        
        for song_id, interpretations_list in songs.items():
            # Sortiere nach Iteration
            interpretations_list.sort(key=lambda x: x['iteration'])
         
            texts = [item['text'] for item in interpretations_list]
            
            # Mindestens 2 Interpretationen benötigt für Similarity
            if len(texts) < 2:
                print(f"    WARNUNG: Song {song_id} hat nur {len(texts)} Interpretation(en), überspringe")
                songs_skipped += 1
                continue
            
            # Info wenn nicht genau 10 Interpretationen (aber trotzdem verarbeiten)
            if len(texts) != 10:
                print(f"    INFO: Song {song_id} hat {len(texts)} statt 10 Interpretationen, wird trotzdem verarbeitet")
            
            sim_data = compute_similarity_for_group(texts, embedding_model)
            
            if sim_data:
                results.append({
                    'experiment': experiment_name,
                    'embedding_model': embedding_model_name,
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

def create_visualizations(df, output_dir, experiment_name):
    """Erstellt Visualisierungen der Ergebnisse."""
    output_path = Path(output_dir)
    
    # 1. Boxplot: Similarity-Verteilung pro Modell
    plt.figure(figsize=(12, 6))
    df_exp = df[df['experiment'] == experiment_name]
    sns.boxplot(data=df_exp, x='model', y='mean_similarity')
    plt.title(f'Cosine Similarity Distribution - {experiment_name}')
    plt.xlabel('Modell')
    plt.ylabel('Mean Cosine Similarity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / f'boxplot_{experiment_name}.png', dpi=300)
    plt.close()
    
    # 2. Vergleich zwischen Experimenten
    if len(df['experiment'].unique()) > 1:
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df, x='model', y='mean_similarity', hue='experiment')
        plt.title('Cosine Similarity: Standard vs. Prompt Engineering')
        plt.xlabel('Modell')
        plt.ylabel('Mean Cosine Similarity')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Experiment')
        plt.tight_layout()
        plt.savefig(output_path / 'comparison_experiments.png', dpi=300)
        plt.close()
    
    # 3. Heatmap für Modell-Vergleich
    pivot_table = df.pivot_table(
        values='mean_similarity',
        index='model',
        columns='experiment',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': 'Mean Cosine Similarity'})
    plt.title('Mean Cosine Similarity by Model and Experiment')
    plt.tight_layout()
    plt.savefig(output_path / 'heatmap_overview.png', dpi=300)
    plt.close()

def create_comparison_boxplots(df, output_dir):
    """Erstellt Vergleichs-Boxplots für alle Embedding-Modelle."""
    output_path = Path(output_dir)
    
    # Für jedes Embedding-Modell separater Plot
    for emb_model in df['embedding_model'].unique():
        df_emb = df[df['embedding_model'] == emb_model]
        
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df_emb, x='llm_model', y='mean_similarity', hue='experiment')
        plt.title(f'Cosine Similarity: Standard vs. Prompt Engineering\nEmbedding Model: {emb_model}')
        plt.xlabel('LLM Model')
        plt.ylabel('Mean Cosine Similarity')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Experiment')
        plt.tight_layout()
        plt.savefig(output_path / f'comparison_experiments_{emb_model}.png', dpi=300)
        plt.close()
        print(f"  ✓ Erstellt: comparison_experiments_{emb_model}.png")
    
    # Gesamtvergleich: Alle Embedding-Modelle zusammen
    if len(df['embedding_model'].unique()) > 1:
        plt.figure(figsize=(16, 8))
        
        # Kombiniere LLM-Modell und Embedding-Modell für x-Achse
        df['combined'] = df['llm_model'] + '\n(' + df['embedding_model'] + ')'
        
        sns.boxplot(data=df, x='combined', y='mean_similarity', hue='experiment')
        plt.title('Cosine Similarity: Comparison across all Embedding Models')
        plt.xlabel('LLM Model (Embedding Model)')
        plt.ylabel('Mean Cosine Similarity')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Experiment')
        plt.tight_layout()
        plt.savefig(output_path / 'comparison_experiments_all_embeddings.png', dpi=300)
        plt.close()
        print(f"  ✓ Erstellt: comparison_experiments_all_embeddings.png")

def create_summary_statistics(df, output_dir):
    """Erstellt zusammenfassende Statistiken."""
    output_path = Path(output_dir)
    
    # Gruppierte Statistiken
    summary = df.groupby(['experiment', 'embedding_model', 'llm_model']).agg({
        'mean_similarity': ['mean', 'std', 'min', 'max', 'median'],
        'song_id': 'count'
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'song_id_count': 'n_songs'})
    
    # Als CSV speichern
    summary.to_csv(output_path / 'summary_statistics.csv')
    
    # Als formatierte Tabelle
    with open(output_path / 'summary_statistics.txt', 'w') as f:
        f.write("=== COSINE SIMILARITY ANALYSIS ===\n\n")
        f.write(summary.to_string())
        f.write("\n\n")
    
    return summary

def main():
    print("Starte Similarity-Analyse...")
    print(f"Erwartete Struktur: 100 Songs × 10 Interpretationen = 1000 Dateien pro Modell\n")
    
    # GPU-Info
    if torch.cuda.is_available():
        print(f"GPU verfügbar: {torch.cuda.get_device_name(0)}")
        print(f"GPU Speicher: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Verwende Device: {DEVICE}")
        print(f"Batch-Größe: {BATCH_SIZE}\n")
    else:
        print(f"Keine GPU verfügbar, verwende CPU\n")
    
    # Output-Verzeichnis erstellen
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lade alle Embedding-Modelle
    print("Lade Embedding-Modelle...")
    embedding_models = {}
    for model_short_name, model_name in EMBEDDING_MODELS.items():
        print(f"  - {model_short_name}: {model_name}")
        try:
            embedding_models[model_short_name] = SentenceTransformer(model_name, device=DEVICE)
            print(f"    ✓ Erfolgreich geladen")
        except Exception as e:
            print(f"    ✗ FEHLER: {e}")
            print(f"    Überspringe {model_short_name}")
    
    if not embedding_models:
        print("\n✗ Keine Embedding-Modelle konnten geladen werden!")
        return
    
    print(f"\n✓ {len(embedding_models)} Embedding-Modelle erfolgreich geladen\n")
    
    all_results = []
    all_detailed = {}
    
    # Für jedes Embedding-Modell
    for emb_model_name, embedding_model in embedding_models.items():
        print(f"\n{'='*60}")
        print(f"Verwende Embedding-Modell: {emb_model_name}")
        print(f"{'='*60}")
        
        all_detailed[emb_model_name] = {}
        
        # Für jeden Basis-Ordner
        for base_dir in BASE_DIRS:
            experiment_name = Path(base_dir).name.replace("interpretations", "").strip("_") or "standard"
            print(f"\nAnalysiere: {experiment_name}")
            
            # Interpretationen laden
            interpretations = load_interpretations(base_dir)
            
            if not interpretations:
                print(f"  Keine Interpretationen gefunden in {base_dir}")
                continue
            
            # Similarities berechnen
            df, detailed = analyze_similarities(interpretations, embedding_model, experiment_name, emb_model_name)
            all_results.append(df)
            all_detailed[emb_model_name][experiment_name] = detailed
    
    # Alle Ergebnisse kombinieren
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Statistik über Anzahl Interpretationen
        print("\n=== INTERPRETATIONEN PRO SONG ===")
        interp_counts = combined_df.groupby(['experiment', 'embedding_model', 'llm_model', 'n_interpretations']).size()
        print(interp_counts)
        
        # Alle Ergebnisse speichern
        combined_df.to_csv(output_path / 'all_results.csv', index=False)
        
        # Erstelle nur Vergleichs-Boxplots
        print("\n=== ERSTELLE VISUALISIERUNGEN ===")
        create_comparison_boxplots(combined_df, output_path)
        
        # Summary-Statistiken
        summary = create_summary_statistics(combined_df, output_path)
        print("\n=== SUMMARY ===")
        print(summary)
        
        # Detaillierte Ergebnisse als JSON
        simplified_detailed = {}
        for emb_model, experiments in all_detailed.items():
            simplified_detailed[emb_model] = {}
            for exp, models in experiments.items():
                simplified_detailed[emb_model][exp] = {}
                for model, songs in models.items():
                    simplified_detailed[emb_model][exp][model] = {
                        song_id: {
                            'mean': float(data['mean']),
                            'std': float(data['std']),
                            'min': float(data['min']),
                            'max': float(data['max']),
                            'median': float(data['median'])
                        }
                        for song_id, data in songs.items()
                    }
        
        with open(output_path / 'detailed_results.json', 'w') as f:
            json.dump(simplified_detailed, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Ergebnisse gespeichert in: {output_path}")
        print(f"  - all_results.csv: Alle Einzelergebnisse")
        print(f"  - summary_statistics.csv: Zusammenfassung")
        print(f"  - detailed_results.json: Detaillierte Ergebnisse pro Song")
        print(f"  - comparison_experiments_*.png: Vergleichs-Boxplots")
        print(f"{'='*60}")
    else:
        print("Keine Ergebnisse zum Speichern gefunden.")

if __name__ == "__main__":
    main()
