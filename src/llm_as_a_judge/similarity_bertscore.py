import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from bert_score import BERTScorer

# Konfiguration
BASE_DIRS = [
    "../../llm_as_a_judge/interpretations",
    "../../llm_as_a_judge/interpretations_prompt_engineering"
]
OUTPUT_DIR = "../../llm_as_a_judge/results/bertscore_similarity"


BERTSCORE_MODEL = 'microsoft/deberta-xlarge-mnli'  # Deutsches Modell
BERTSCORE_LANG = 'de'  # Deutsch

# Globaler BERTScorer (wird einmal initialisiert)
SCORER = None

def initialize_scorer():
    """Initialisiert den BERTScorer einmalig."""
    global SCORER
    if SCORER is None:
        print(f"\nLade BERTScore Modell: {BERTSCORE_MODEL}...")
        print("Dies geschieht nur einmal zu Beginn der Analyse.\n")
        SCORER = BERTScorer(
            model_type=BERTSCORE_MODEL,
            lang=BERTSCORE_LANG,
            rescale_with_baseline=False,  # Kein Baseline-Rescaling für deutsches Modell
            device='cuda' if os.system('nvidia-smi') == 0 else 'cpu'
        )
        print("✓ BERTScore Modell erfolgreich geladen!\n")
    return SCORER

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

def compute_bertscore_for_group(texts, scorer):
    """Berechnet pairwise BERTScore für eine Gruppe von Texten."""
    if len(texts) < 2:
        return None
    
    n = len(texts)
    similarity_matrix = np.zeros((n, n))
    
    # Pairwise BERTScore mit dem bereits geladenen Scorer
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                # BERTScore zwischen zwei Interpretationen
                P, R, F1 = scorer.score(
                    [texts[i]], 
                    [texts[j]]
                )
                
                # F1-Score verwenden (Balance aus Precision und Recall)
                f1_score = F1.item()
                similarity_matrix[i][j] = f1_score
                similarity_matrix[j][i] = f1_score
    
    # Nur obere Dreiecksmatrix (ohne Diagonale) für pairwise similarities
    upper_triangle_indices = np.triu_indices(n, k=1)
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

def analyze_similarities(interpretations, experiment_name, scorer):
    """Analysiert BERTScore für alle Modelle und Songs."""
    results = []
    detailed_results = {}
    
    for model_name, songs in interpretations.items():
        print(f"  Verarbeite Modell: {model_name}")
        model_results = []
        detailed_results[model_name] = {}
        
        songs_analyzed = 0
        songs_skipped = 0
        
        for song_idx, (song_id, interpretations_list) in enumerate(songs.items(), 1):
            # Sortiere nach Iteration
            interpretations_list.sort(key=lambda x: x['iteration'])
         
            texts = [item['text'] for item in interpretations_list]
            
            # Mindestens 2 Interpretationen benötigt für BERTScore
            if len(texts) < 2:
                print(f"    WARNUNG: Song {song_id} hat nur {len(texts)} Interpretation(en), überspringe")
                songs_skipped += 1
                continue
            
            # Info wenn nicht genau 10 Interpretationen (aber trotzdem verarbeiten)
            if len(texts) != 10:
                print(f"    INFO: Song {song_id} hat {len(texts)} statt 10 Interpretationen, wird trotzdem verarbeitet")
            
            # Fortschrittsanzeige (alle 10 Songs)
            if song_idx % 10 == 0:
                print(f"    Fortschritt: {song_idx}/{len(songs)} Songs verarbeitet...")
            
            bertscore_data = compute_bertscore_for_group(texts, scorer)
            
            if bertscore_data:
                results.append({
                    'experiment': experiment_name,
                    'llm_model': model_name,
                    'song_id': song_id,
                    'n_interpretations': len(texts),
                    'mean_bertscore': bertscore_data['mean'],
                    'std_bertscore': bertscore_data['std'],
                    'min_bertscore': bertscore_data['min'],
                    'max_bertscore': bertscore_data['max'],
                    'median_bertscore': bertscore_data['median']
                })
                
                model_results.append(bertscore_data['mean'])
                detailed_results[model_name][song_id] = bertscore_data
                songs_analyzed += 1
        
        if model_results:
            print(f"    ✓ {songs_analyzed} Songs analysiert, {songs_skipped} übersprungen, Ø BERTScore: {np.mean(model_results):.4f}")
    
    return pd.DataFrame(results), detailed_results

def create_visualizations(df, output_dir, experiment_name):
    """Erstellt Visualisierungen der Ergebnisse."""
    output_path = Path(output_dir)
    
    # 1. Boxplot: BERTScore-Verteilung pro Modell
    plt.figure(figsize=(12, 6))
    df_exp = df[df['experiment'] == experiment_name]
    sns.boxplot(data=df_exp, x='llm_model', y='mean_bertscore')
    plt.title(f'BERTScore Similarity Distribution - {experiment_name}')
    plt.xlabel('Modell')
    plt.ylabel('Mean BERTScore Similarity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / f'boxplot_{experiment_name}.png', dpi=300)
    plt.close()
    
    # 2. Vergleich zwischen Experimenten
    if len(df['experiment'].unique()) > 1:
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df, x='llm_model', y='mean_bertscore', hue='experiment')
        plt.title('BERTScore Similarity: Standard vs. Prompt Engineering')
        plt.xlabel('Modell')
        plt.ylabel('Mean BERTScore Similarity')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Experiment')
        plt.tight_layout()
        plt.savefig(output_path / 'comparison_experiments.png', dpi=300)
        plt.close()
    
    # 3. Heatmap für Modell-Vergleich
    pivot_table = df.pivot_table(
        values='mean_bertscore',
        index='llm_model',
        columns='experiment',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'Mean BERTScore Similarity'})
    plt.title('Mean BERTScore Similarity by Model and Experiment')
    plt.tight_layout()
    plt.savefig(output_path / 'heatmap_overview.png', dpi=300)
    plt.close()

def create_comparison_boxplots(df, output_dir):
    """Erstellt Vergleichs-Boxplots."""
    output_path = Path(output_dir)
    
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x='llm_model', y='mean_bertscore', hue='experiment')
    plt.title('BERTScore Similarity: Standard vs. Prompt Engineering')
    plt.xlabel('LLM Model')
    plt.ylabel('Mean BERTScore Similarity')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Experiment')
    plt.tight_layout()
    plt.savefig(output_path / 'comparison_experiments.png', dpi=300)
    plt.close()
    print(f"  ✓ Erstellt: comparison_experiments.png")

def create_similarity_heatmaps(detailed_results, output_dir):
    """Erstellt Heatmaps für einzelne Songs."""
    output_path = Path(output_dir) / 'heatmaps'
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n=== ERSTELLE SIMILARITY HEATMAPS (Beispiele) ===")
    
    # Erstelle Heatmaps für ersten Song jedes Modells als Beispiel
    for exp, models in detailed_results.items():
        for model, songs in models.items():
            # Nimm ersten Song als Beispiel
            if songs:
                song_id = list(songs.keys())[0]
                data = songs[song_id]
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    data['matrix'],
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlGn',
                    vmin=0,
                    vmax=1,
                    xticklabels=[f"I{i+1}" for i in range(len(data['matrix']))],
                    yticklabels=[f"I{i+1}" for i in range(len(data['matrix']))],
                    cbar_kws={'label': 'BERTScore Similarity'}
                )
                plt.title(f'Semantic Similarity Matrix\n{model} - {exp} - Song {song_id}')
                plt.tight_layout()
                
                filename = f'heatmap_{exp}_{model}_{song_id}.png'
                plt.savefig(output_path / filename, dpi=300)
                plt.close()
                
                print(f"  ✓ Erstellt: {filename}")

def create_summary_statistics(df, output_dir):
    """Erstellt zusammenfassende Statistiken."""
    output_path = Path(output_dir)
    
    # Gruppierte Statistiken
    summary = df.groupby(['experiment', 'llm_model']).agg({
        'mean_bertscore': ['mean', 'std', 'min', 'max', 'median'],
        'song_id': 'count'
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'song_id_count': 'n_songs'})
    
    # Als CSV speichern
    summary.to_csv(output_path / 'summary_statistics.csv')
    
    # Als formatierte Tabelle
    with open(output_path / 'summary_statistics.txt', 'w') as f:
        f.write("=== BERTSCORE SIMILARITY ANALYSIS ===\n\n")
        f.write(f"Model: {BERTSCORE_MODEL}\n")
        f.write(f"Language: {BERTSCORE_LANG}\n")
        f.write(f"Metric: F1-Score (Balance aus Precision und Recall)\n")
        f.write(f"Interpretation: Hoch = semantisch ähnlich, Niedrig = semantisch divers\n\n")
        f.write(summary.to_string())
        f.write("\n\n")
    
    return summary

def main():
    print("Starte BERTScore Similarity Analyse...")
    print(f"BERTScore Model: {BERTSCORE_MODEL}")
    print(f"Language: {BERTSCORE_LANG}")
    print(f"Erwartete Struktur: 100 Songs × 10 Interpretationen = 1000 Dateien pro Modell\n")
    
    # Initialisiere BERTScorer einmalig
    scorer = initialize_scorer()
    
    # Output-Verzeichnis erstellen
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    all_detailed = {}
    
    # Für jeden Basis-Ordner
    for base_dir in BASE_DIRS:
        experiment_name = Path(base_dir).name.replace("interpretations", "").strip("_") or "standard"
        print(f"\nAnalysiere: {experiment_name}")
        
        # Interpretationen laden
        interpretations = load_interpretations(base_dir)
        
        if not interpretations:
            print(f"  Keine Interpretationen gefunden in {base_dir}")
            continue
        
        # BERTScore berechnen (mit bereits geladenem Scorer)
        df, detailed = analyze_similarities(interpretations, experiment_name, scorer)
        all_results.append(df)
        all_detailed[experiment_name] = detailed
    
    # Alle Ergebnisse kombinieren
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Statistik über Anzahl Interpretationen
        print("\n=== INTERPRETATIONEN PRO SONG ===")
        interp_counts = combined_df.groupby(['experiment', 'llm_model', 'n_interpretations']).size()
        print(interp_counts)
        
        # Alle Ergebnisse speichern
        combined_df.to_csv(output_path / 'all_results.csv', index=False)
        
        # Erstelle Visualisierungen
        print("\n=== ERSTELLE VISUALISIERUNGEN ===")
        create_comparison_boxplots(combined_df, output_path)
        
        # Erstelle Beispiel-Heatmaps
        create_similarity_heatmaps(all_detailed, output_path)
        
        # Summary-Statistiken
        summary = create_summary_statistics(combined_df, output_path)
        print("\n=== SUMMARY ===")
        print(summary)
        
        # Detaillierte Ergebnisse als JSON
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
        
        with open(output_path / 'detailed_results.json', 'w') as f:
            json.dump(simplified_detailed, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Ergebnisse gespeichert in: {output_path}")
        print(f"  - all_results.csv: Alle Einzelergebnisse")
        print(f"  - summary_statistics.csv: Zusammenfassung")
        print(f"  - detailed_results.json: Detaillierte Ergebnisse pro Song")
        print(f"  - comparison_experiments.png: Vergleichs-Boxplot")
        print(f"  - heatmaps/: Beispiel-Similarity-Matrizen")
        print(f"{'='*60}")
    else:
        print("Keine Ergebnisse zum Speichern gefunden.")

if __name__ == "__main__":
    main()
