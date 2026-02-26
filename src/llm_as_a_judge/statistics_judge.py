import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Dict, List
import numpy as np

# Configure plot style for LaTeX
# Set style first to avoid overriding rcParams
sns.set_style("whitegrid")
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

def load_json_files(base_dir: str) -> List[Dict]:
    """Lädt alle JSON-Dateien aus dem angegebenen Verzeichnis."""
    data = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Warnung: Verzeichnis {base_dir} existiert nicht")
        return data
    
    for model_dir in base_path.iterdir():
        if model_dir.is_dir():
            for json_file in model_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        content['methode'] = 'prompt_engineering' if 'prompt_engineering' in str(base_path) else 'standard'
                        content['model_folder'] = model_dir.name
                        data.append(content)
                except Exception as e:
                    print(f"Fehler beim Laden von {json_file}: {e}")
    
    return data

def normalize_model_name(model_name: str) -> str:
    """Normalisiert Modellnamen zu einheitlichen Bezeichnungen."""
    model_name_lower = model_name.lower()
    
    if 'deepseek' in model_name_lower or 'r1' in model_name_lower:
        return 'deepseek'
    elif 'gpt' in model_name_lower or 'openai' in model_name_lower:
        return 'openai'
    elif 'llama' in model_name_lower:
        return 'llama'
    else:
        return None

def extract_scores(data: List[Dict]) -> pd.DataFrame:
    """Extrahiert Scores aus den JSON-Daten - verwendet die Durchschnittswerte aus der Zusammenfassung."""
    records = []
    skipped_count = 0
    
    for entry in data:
        judge_model_raw = entry.get('judge_modell', 'unknown')
        interpretations_model_raw = entry.get('interpretations_modell', 'unknown')
        
        judge_model = normalize_model_name(judge_model_raw)
        interpretations_model = normalize_model_name(interpretations_model_raw)
        
        # Überspringe Einträge mit unbekannten Modellen
        if judge_model is None or interpretations_model is None:
            print(f"Warnung: Unbekanntes Modell übersprungen - Judge: '{judge_model_raw}', Interpretations: '{interpretations_model_raw}'")
            skipped_count += 1
            continue
        
        methode = entry.get('methode', 'unknown')
        song = entry.get('song', 'unknown')
        
        # Verwende die Durchschnittswerte aus der Zusammenfassung
        zusammenfassung = entry.get('zusammenfassung', {})
        
        if zusammenfassung:
            records.append({
                'judge_model': judge_model,
                'interpretations_model': interpretations_model,
                'methode': methode,
                'song': song,
                'halluzination_score': zusammenfassung.get('durchschnitt_halluzination'),
                'tiefe_score': zusammenfassung.get('durchschnitt_tiefe'),
                'beste_interpretation': zusammenfassung.get('beste_interpretation')
            })
        else:
            print(f"Warnung: Keine Zusammenfassung gefunden für Song: {song}")
    
    if skipped_count > 0:
        print(f"\nInsgesamt {skipped_count} JSON-Datei(en) mit unbekannten Modellen übersprungen.\n")
    
    return pd.DataFrame(records)

def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet Statistiken gruppiert nach Modell und Methode."""
    stats = df.groupby(['interpretations_model', 'methode']).agg({
        'halluzination_score': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'tiefe_score': ['mean', 'median', 'std', 'min', 'max', 'count']
    }).round(3)
    
    return stats

def create_summary(df: pd.DataFrame, stats: pd.DataFrame) -> str:
    """Erstellt eine textuelle Zusammenfassung der Ergebnisse."""
    summary = []
    summary.append("=" * 80)
    summary.append("ZUSAMMENFASSUNG DER LLM-AS-A-JUDGE EVALUATION")
    summary.append("=" * 80)
    summary.append(f"\nGesamtanzahl der bewerteten Songs: {len(df)}")
    summary.append(f"Anzahl der verschiedenen Songs: {df['song'].nunique()}")
    summary.append(f"Bewertende Modelle: {', '.join(sorted(df['judge_model'].unique()))}")
    summary.append(f"Bewertete Modelle: {', '.join(sorted(df['interpretations_model'].unique()))}")
    
    summary.append("\n" + "-" * 80)
    summary.append("DURCHSCHNITTLICHE SCORES NACH METHODE:")
    summary.append("-" * 80)
    
    for methode in ['standard', 'prompt_engineering']:
        subset = df[df['methode'] == methode]
        if len(subset) > 0:
            summary.append(f"\n{methode.upper()}:")
            summary.append(f"  Anzahl Songs: {len(subset)}")
            summary.append(f"  Halluzinationsrate: {subset['halluzination_score'].mean():.3f} (±{subset['halluzination_score'].std():.3f})")
            summary.append(f"  Interpretationstiefe: {subset['tiefe_score'].mean():.3f} (±{subset['tiefe_score'].std():.3f})")
    
    summary.append("\n" + "-" * 80)
    summary.append("DURCHSCHNITTLICHE SCORES NACH MODELL:")
    summary.append("-" * 80)
    
    for model in sorted(df['interpretations_model'].unique()):
        subset = df[df['interpretations_model'] == model]
        summary.append(f"\n{model.upper()}:")
        for methode in ['standard', 'prompt_engineering']:
            subset_methode = subset[subset['methode'] == methode]
            if len(subset_methode) > 0:
                summary.append(f"  {methode}:")
                summary.append(f"    Anzahl: {len(subset_methode)}")
                summary.append(f"    Halluzinationsrate: {subset_methode['halluzination_score'].mean():.3f} (±{subset_methode['halluzination_score'].std():.3f})")
                summary.append(f"    Interpretationstiefe: {subset_methode['tiefe_score'].mean():.3f} (±{subset_methode['tiefe_score'].std():.3f})")
    
    summary.append("\n" + "=" * 80)

    # Zusätzliche Tabellenansicht für "konkrete Werte"
    summary.append("\n" + "=" * 80)
    summary.append(f"{'Modell':<15} | {'Methode':<20} | {'Halluzinationsrate':<20} | {'Interpretationstiefe':<20}")
    summary.append("-" * 80)
    
    for model in sorted(df['interpretations_model'].unique()):
        for methode in ['standard', 'prompt_engineering']:
            subset = df[(df['interpretations_model'] == model) & (df['methode'] == methode)]
            if len(subset) > 0:
                halluzination = round(subset['halluzination_score'].mean(), 3)
                tiefe = round(subset['tiefe_score'].mean(), 3)
                summary.append(f"{model:<15} | {methode:<20} | {halluzination:<20} | {tiefe:<20}")
    
    summary.append("=" * 80)
    
    return "\n".join(summary)

def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Erstellt Boxplots für die Scores."""
    # sns.set_style("whitegrid")  # Already set globally
    
    # Sortiere Modelle für konsistente Darstellung
    model_order = sorted(df['interpretations_model'].unique())
    
    # Boxplots für Halluzination Score und Tiefe Score
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Halluzination Score nach Methode
    sns.boxplot(data=df, x='methode', y='halluzination_score', ax=axes[0, 0])
    axes[0, 0].set_title('Halluzinationsrate nach Methode', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Methode', fontsize=12)
    axes[0, 0].set_ylabel('Halluzinationsrate', fontsize=12)
    axes[0, 0].set_ylim(0, 5.5)
    
    # Tiefe Score nach Methode
    sns.boxplot(data=df, x='methode', y='tiefe_score', ax=axes[0, 1])
    axes[0, 1].set_title('Interpretationstiefe nach Methode', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Methode', fontsize=12)
    axes[0, 1].set_ylabel('Interpretationstiefe', fontsize=12)
    axes[0, 1].set_ylim(0, 5.5)
    
    # Halluzination Score nach Modell
    sns.boxplot(data=df, x='interpretations_model', y='halluzination_score', hue='methode', 
                ax=axes[1, 0], order=model_order)
    axes[1, 0].set_title('Halluzinationsrate nach Modell und Methode', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Modell', fontsize=12)
    axes[1, 0].set_ylabel('Halluzinationsrate', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=0)
    axes[1, 0].set_ylim(0, 5.5)
    axes[1, 0].legend(title='Methode', loc='lower right')
    
    # Tiefe Score nach Modell
    sns.boxplot(data=df, x='interpretations_model', y='tiefe_score', hue='methode', 
                ax=axes[1, 1], order=model_order)
    axes[1, 1].set_title('Interpretationstiefe nach Modell und Methode', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Modell', fontsize=12)
    axes[1, 1].set_ylabel('Interpretationstiefe', fontsize=12)
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].set_ylim(0, 5.5)
    axes[1, 1].legend(title='Methode', loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Zusätzliche Visualisierung: Vergleich beider Scores nebeneinander
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Beide Scores nach Modell (ohne Methoden-Unterteilung)
    sns.boxplot(data=df, x='interpretations_model', y='halluzination_score', 
                ax=axes[0], order=model_order)
    axes[0].set_title('Halluzinationsrate nach Modell (Gesamt)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Modell', fontsize=12)
    axes[0].set_ylabel('Halluzinationsrate', fontsize=12)
    axes[0].set_ylim(0, 5.5)
    
    sns.boxplot(data=df, x='interpretations_model', y='tiefe_score', 
                ax=axes[1], order=model_order)
    axes[1].set_title('Interpretationstiefe nach Modell (Gesamt)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Modell', fontsize=12)
    axes[1].set_ylabel('Interpretationstiefe', fontsize=12)
    axes[1].set_ylim(0, 5.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot_modelle_gesamt.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualisierungen gespeichert in {output_dir}")

def main():
    # Pfade definieren
    base_dir = Path(__file__).parent.parent.parent
    eval_dir_standard = base_dir / "llm_as_a_judge" / "evaluations_two_dimensions"
    eval_dir_prompt_eng = base_dir / "llm_as_a_judge" / "evaluations_prompt_engineering_two_dimensions"
    output_dir = base_dir / "llm_as_a_judge" / "results" / "llm_judge"
    
    # Output-Verzeichnis erstellen
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Lade JSON-Dateien...")
    data_standard = load_json_files(eval_dir_standard)
    data_prompt_eng = load_json_files(eval_dir_prompt_eng)
    
    all_data = data_standard + data_prompt_eng
    print(f"Insgesamt {len(all_data)} JSON-Dateien geladen")
    
    if not all_data:
        print("Keine Daten gefunden. Bitte Pfade überprüfen.")
        return
    
    print("Extrahiere Scores...")
    df = extract_scores(all_data)
    
    if df.empty:
        print("Keine gültigen Daten gefunden. Überprüfen Sie die Modellnamen und JSON-Struktur.")
        return
    
    print(f"Extrahierte Einträge: {len(df)}")
    print(f"Modelle: {df['interpretations_model'].unique()}")
    print(f"Methoden: {df['methode'].unique()}")
    
    print("\nBerechne Statistiken...")
    stats = calculate_statistics(df)
    
    # CSV speichern
    print("Speichere CSV...")
    df.to_csv(output_dir / 'alle_scores.csv', index=False, encoding='utf-8')
    stats.to_csv(output_dir / 'statistiken.csv', encoding='utf-8')
    
    # Zusammenfassung erstellen
    print("Erstelle Zusammenfassung...")
    summary = create_summary(df, stats)
    with open(output_dir / 'zusammenfassung.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print("\n" + summary)
    
    # Visualisierungen erstellen
    print("\nErstelle Visualisierungen...")
    create_visualizations(df, output_dir)
    
    print(f"\nAlle Ergebnisse wurden in {output_dir} gespeichert:")
    print(f"  - alle_scores.csv: Alle Song-Scores (Durchschnitte)")
    print(f"  - statistiken.csv: Aggregierte Statistiken")
    print(f"  - zusammenfassung.txt: Textuelle Zusammenfassung")
    print(f"  - boxplots_scores.png: Boxplots der Scores")
    print(f"  - boxplot_modelle_gesamt.png: Boxplots nach Modellen")

if __name__ == "__main__":
    main()
