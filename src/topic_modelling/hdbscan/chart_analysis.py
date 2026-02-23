import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Paths
MODEL_PATH = "../../../bertopic/dtm/models/bertopic_model_dtm"
DATA_PATH = "../../../data/dataset_popmusic_v8.json"
VIZ_PATH = "../../../bertopic/dtm/visualizations/chart_analysis"
TOPIC_LABELS_PATH = "../../../bertopic/dtm/models/topics_over_time.csv"

# Create visualization directory if not exists
Path(VIZ_PATH).mkdir(parents=True, exist_ok=True)


def load_topic_labels_map():
    """Load topic labels from CSV."""
    if not os.path.exists(TOPIC_LABELS_PATH):
        print(f"Warning: Topic labels file not found at {TOPIC_LABELS_PATH}")
        return {}
    
    try:
        df_labels = pd.read_csv(TOPIC_LABELS_PATH)
        # Drop duplicates to get one label per topic
        # Use Global_Representation column
        if 'Global_Representation' in df_labels.columns:
            df_unique = df_labels[['Topic', 'Global_Representation']].drop_duplicates('Topic')
            return dict(zip(df_unique['Topic'], df_unique['Global_Representation']))
        else:
            print("Warning: 'Global_Representation' column not found in topic labels CSV")
            return {}
    except Exception as e:
        print(f"Error loading topic labels: {e}")
        return {}


def load_data():
    """Load dataset and determine chart songs."""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Filter logic matching visualize_tot.py to ensure index alignment
    # Step 1 & 2: Handle missing/empty values
    df['text'] = df['text'].fillna('').str.strip()
    df['timestamp'] = df['timestamp'].fillna('')

    # Step 3: Filter out empty text or empty timestamp
    df = df[(df['text'] != '') & (df['timestamp'] != '')]

    # Step 4: Parse Year
    def parse_year(date_str):
        try:
            return int(str(date_str)[:4])
        except:
            return None

    df['year'] = df['timestamp'].apply(parse_year)

    # Step 5: Filter Year Range (1955-2024)
    df = df[(df['year'] >= 1955) & (df['year'] <= 2024)].reset_index(drop=True)
            
    # Determine if chart song: non-empty "Erste Notierung" means chart song
    df['is_chart'] = df['Erste Notierung'].apply(
        lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0
    )
    
    return df


def create_time_bins(df):
    """Create 5-year bins based on year."""
    # Year is already parsed and filtered in load_data
    df['year'] = df['year'].astype(int)
    
    # Create 5-year bins from 1950 to 2024
    # 1950-1954, 1955-1959, 1960-1964, ..., 2020-2024
    bin_starts = list(range(1950, 2025, 5))
    bin_edges = bin_starts + [2030]  # End edge muss > 2024 sein, damit 2024 inkludiert wird
    bin_labels = [f"{bin_starts[i]}-{bin_starts[i]+4}" for i in range(len(bin_starts))]
    
    df['period'] = pd.cut(df['year'], bins=bin_edges, labels=bin_labels, include_lowest=True, right=False)
    
    return df


def chi_square_test_per_topic_period(df):
    """
    Führe Fisher's Exact Test für jedes Topic in jeder 5-Jahres-Periode durch.
    (HINWEIS: Funktionsname aus Kompatibilität beibehalten, nutzt aber Fisher)
    
    Kreuztabelle pro Topic:
                    Chart   Non-Chart
    In Topic          a         b
    Not in Topic      c         d
    
    H0: Topic-Zugehörigkeit ist unabhängig von Chart-Erfolg
    """
    df_filtered = df[df['topic'] != -1].copy()
    
    results = []
    periods = sorted(df_filtered['period'].dropna().unique())
    topics = sorted(df_filtered['topic'].unique())
    
    for period in periods:
        period_data = df_filtered[df_filtered['period'] == period]
        n_total = len(period_data)
        n_chart = period_data['is_chart'].sum()
        
        # Globale Chart-Rate in dieser Periode
        chart_rate = n_chart / n_total if n_total > 0 else 0
        
        for topic in topics:
            # Kreuztabelle erstellen
            in_topic = period_data['topic'] == topic
            
            a = ((in_topic) & (period_data['is_chart'] == 1)).sum()  # In Topic & Chart
            b = ((in_topic) & (period_data['is_chart'] == 0)).sum()  # In Topic & Non-Chart
            
            n_in_topic = a + b
            
            # --- FIX: Ratenberechnung VOR dem Filter ---
            if n_in_topic > 0:
                observed_chart_rate = a / n_in_topic
                rate_difference = observed_chart_rate - chart_rate
            else:
                observed_chart_rate = 0.0
                rate_difference = 0.0 # Keine Abweichung messbar
            
            # Filter: Mindestens 10 Dokumente im Topic für statistische Aussagekraft
            # Trotzdem speichern wir die korrekten Raten (oben berechnet)
            if n_in_topic < 10:
                results.append({
                    'period': period,
                    'topic': topic,
                    'in_topic_chart': a,
                    'in_topic_non_chart': b,
                    'not_in_topic_chart': np.nan,
                    'not_in_topic_non_chart': np.nan,
                    'n_in_topic': n_in_topic,
                    'n_total': n_total,
                    'observed_chart_rate': observed_chart_rate, # KORRIGIERT: Echter Wert
                    'expected_chart_rate': chart_rate,
                    'rate_difference': rate_difference,         # KORRIGIERT: Echter Wert
                    'chi2': np.nan,
                    'p_value': 1.0, # Nicht signifikant, da zu klein
                    'phi': 0,
                    'direction': "neutral"
                })
                continue

            c = ((~in_topic) & (period_data['is_chart'] == 1)).sum()  # Not in Topic & Chart
            d = ((~in_topic) & (period_data['is_chart'] == 0)).sum()  # Not in Topic & Non-Chart
            
            # Kreuztabelle
            contingency_table = np.array([[a, b], [c, d]])
            
            # --- FIX: IMMER Fisher's Exact Test nutzen ---
            # Wegen unbalancierten Klassen (Charts sehr selten) ist Chi2 ungenau
            odds_ratio_fisher, p_value = stats.fisher_exact(contingency_table)
            chi2 = np.nan # Chi2 ist hier nicht relevant
            
            expected_chart_rate = chart_rate
            
            # Effektstärke: Phi-Koeffizient (nur informativ, basierend auf implizitem Chi2)
            # Kann bei Fisher auch über Cramers V / Phi angenähert werden
            # Berechnung manuell für Reporting: chi2_stat = N * phi^2
            # Hier nutzen wir eine Annäherung oder lassen es 0, wenn kein Chi2 berechnet wurde.
            # Um Konsistenz zu wahren: Wir berechnen Chi2 nur für Phi, nutzen aber p-Wert von Fisher.
            try:
                 chi2_stat, _, _, _ = stats.chi2_contingency(contingency_table)
                 n = contingency_table.sum()
                 phi = np.sqrt(chi2_stat / n)
            except:
                 phi = 0
            
            # Richtung des Effekts
            if observed_chart_rate > expected_chart_rate:
                direction = "overrepresented"
            elif observed_chart_rate < expected_chart_rate:
                direction = "underrepresented"
            else:
                direction = "neutral"
            
            results.append({
                'period': period,
                'topic': topic,
                'in_topic_chart': a,
                'in_topic_non_chart': b,
                'not_in_topic_chart': c,
                'not_in_topic_non_chart': d,
                'n_in_topic': n_in_topic,
                'n_total': n_total,
                'observed_chart_rate': observed_chart_rate,
                'expected_chart_rate': expected_chart_rate,
                'rate_difference': rate_difference,
                'chi2': chi2,     # NaN, da Fisher genutzt
                'p_value': p_value,
                'phi': phi,
                'direction': direction
            })
    
    results_df = pd.DataFrame(results)
    
    # Benjamini-Hochberg Korrektur
    rejected, p_adjusted, _, _ = multipletests(results_df['p_value'], alpha=0.05, method='fdr_bh')
    results_df['p_adjusted'] = p_adjusted
    results_df['significant'] = rejected
    
    n_periods = len(periods)
    print(f"\nStatistische Tests (Fisher's Exact Test):")
    print(f"  Zeitperioden: {n_periods} (5-Jahres-Bins von 1955-2024)")
    print(f"  Gesamt: {len(results_df)} Tests")
    print(f"  Signifikant (BH-korrigiert p < 0.05): {results_df['significant'].sum()}")
    
    return results_df


def print_crosstab_example(df, topic, period, topic_info):
    """Zeige eine beispielhafte Kreuztabelle."""
    period_data = df[(df['period'] == period) & (df['topic'] != -1)]
    
    if len(period_data) == 0:
        print(f"\nKeine Daten für Periode {period} gefunden.")
        return
    
    crosstab = pd.crosstab(
        period_data['topic'] == topic,
        period_data['is_chart'],
        rownames=['In Topic'],
        colnames=['Is Chart']
    )
    
    # Prüfe ob beide Spalten vorhanden sind
    if crosstab.shape[1] < 2:
        print(f"\nKreuztabelle für Topic {topic} in Periode {period} hat nicht genug Varianz.")
        return
    
    crosstab.index = ['Nicht in Topic', 'In Topic']
    crosstab.columns = ['Non-Chart', 'Chart']
    
    label_map = load_topic_labels_map()
    if topic in label_map:
        name = label_map[topic]
    else:
        topic_name = topic_info[topic_info['Topic'] == topic]['Name'].values
        name = topic_name[0] if len(topic_name) > 0 else f"Topic {topic}"
    
    print(f"\n=== Kreuztabelle: T{topic}: {name} ({period}) ===")
    print(crosstab)
    print(f"\nFisher's Exact Test:")
    odds, p = stats.fisher_exact(crosstab.iloc[::-1, ::-1]) # Revert index for standard odds interpretation
    print(f"  Odds Ratio = {odds:.2f}, p-Value = {p:.4f}")


def visualize_results(results_df, topic_info):
    """Visualisiere die Ergebnisse."""
    
    sig_topics = results_df[results_df['significant']]['topic'].unique()
    print(f"\nTopics mit signifikanten Ergebnissen: {len(sig_topics)}")
    
    if len(sig_topics) == 0:
        print("Keine signifikanten Ergebnisse gefunden!")
        return
    
    # Topic Labels
    label_map = load_topic_labels_map()
    topic_labels = {}
    for topic in sig_topics:
        if topic in label_map:
            name = label_map[topic]
            topic_labels[topic] = f"T{topic}: {name}"
        elif topic in topic_info['Topic'].values:
            name = topic_info[topic_info['Topic'] == topic]['Name'].values[0]
            topic_labels[topic] = f"T{topic}: {name}"
        else:
            topic_labels[topic] = f"Topic {topic}"
    
    plot_data = results_df[results_df['topic'].isin(sig_topics)].copy()
    plot_data['topic_label'] = plot_data['topic'].map(topic_labels)
    
    # Sortiere nach Anzahl signifikanter Perioden
    topic_sig_counts = plot_data.groupby('topic')['significant'].sum().sort_values(ascending=False)
    sorted_topics = topic_sig_counts.index.tolist()
    sorted_labels = [topic_labels[t] for t in sorted_topics]
    
    n_periods = len(results_df['period'].unique())
    
    # === HEATMAP & BAR CHARTS ===
    # Prepare data for bar charts (Period stats)
    period_stats = results_df[['period', 'n_total', 'expected_chart_rate']].drop_duplicates().sort_values('period')
    periods = period_stats['period'].tolist()
    
    # Calculate height ratios
    heatmap_height = max(8, len(sig_topics) * 0.4)
    bar_chart_height = 3
    total_height = heatmap_height + bar_chart_height * 2 + 4
    
    fig, (ax_freq, ax_rate, ax_heatmap) = plt.subplots(
        3, 1, figsize=(20, total_height), 
        gridspec_kw={'height_ratios': [bar_chart_height, bar_chart_height, heatmap_height], 'hspace': 0.35}
    )
    
    # --- Top: Bar Chart (Absolute Frequency) ---
    x_indices = np.arange(len(periods))
    
    ax_freq.bar(x_indices + 0.5, period_stats['n_total'], color='steelblue', alpha=0.7, width=0.8)
    ax_freq.set_ylabel('Anzahl Songs', fontsize=10)
    ax_freq.set_title('Absolute Häufigkeit der Songs pro Zeitperiode', fontsize=11, pad=5)
    ax_freq.grid(True, alpha=0.3, axis='y')
    ax_freq.set_xlim(0, len(periods))
    
    # X-Achse
    ax_freq.set_xticks([i + 0.5 for i in range(len(periods))])
    ax_freq.set_xticklabels([str(p) for p in periods], rotation=45, ha='right', fontsize=8)
    
    # Add value labels
    for i, val in enumerate(period_stats['n_total']):
        if i % 3 == 0:
            ax_freq.text(i + 0.5, val + max(period_stats['n_total']) * 0.02, f'{int(val)}', 
                        ha='center', va='bottom', fontsize=7, rotation=90)
    
    # --- Middle: Bar Chart (Relative Chart Share) ---
    chart_rates = period_stats['expected_chart_rate'] * 100
    colors = ['forestgreen' if r > chart_rates.mean() else 'gray' for r in chart_rates]
    
    ax_rate.bar(x_indices + 0.5, chart_rates, color=colors, alpha=0.7, width=0.8)
    ax_rate.axhline(y=chart_rates.mean(), color='red', linestyle='--', linewidth=1.5, 
                    label=f'Durchschnitt: {chart_rates.mean():.1f}%')
    ax_rate.set_ylabel('Chart-Anteil (%)', fontsize=10)
    ax_rate.set_title('Relativer Anteil der Chart-Songs pro Zeitperiode', fontsize=11, pad=5)
    ax_rate.grid(True, alpha=0.3, axis='y')
    ax_rate.set_xlim(0, len(periods))
    ax_rate.set_ylim(0, max(chart_rates) * 1.15)
    ax_rate.legend(loc='upper right', fontsize=9)
    
    # X-Achse
    ax_rate.set_xticks([i + 0.5 for i in range(len(periods))])
    ax_rate.set_xticklabels([str(p) for p in periods], rotation=45, ha='right', fontsize=8)

    # --- Bottom: Heatmap ---
    pivot_diff = plot_data.pivot(index='topic_label', columns='period', values='rate_difference')
    pivot_sig = plot_data.pivot(index='topic_label', columns='period', values='significant')
    
    # Ensure columns are sorted same as periods
    pivot_diff = pivot_diff.reindex(columns=periods)
    pivot_sig = pivot_sig.reindex(columns=periods)
    
    pivot_diff = pivot_diff.reindex(sorted_labels)
    pivot_sig = pivot_sig.reindex(sorted_labels)
    
    mask = ~pivot_sig.fillna(False).astype(bool)
    
    # Annotation nur für signifikante Werte
    annot_data = (pivot_diff * 100).round(1)
    annot_strings = annot_data.copy()
    for col in annot_strings.columns:
        for idx in annot_strings.index:
            if mask.loc[idx, col]:
                annot_strings.loc[idx, col] = ''
            else:
                val = annot_data.loc[idx, col]
                annot_strings.loc[idx, col] = f'{val:+.0f}'
    
    sns.heatmap(pivot_diff * 100, ax=ax_heatmap, cmap='RdBu_r', center=0,
                mask=mask,
                annot=annot_strings, fmt='',
                cbar_kws={'label': 'Differenz zur erwarteten Chart-Rate (%-Punkte)', 'shrink': 0.8},
                linewidths=0.5,
                vmin=-15, vmax=15,
                annot_kws={'fontsize': 7})
    
    ax_heatmap.set_title('Chart-Repräsentation pro Topic und Periode\n'
                 '(Nur signifikante Werte, Fisher-Test, BH-korrigiert p < 0.05)', 
                 fontsize=11, pad=10)
    ax_heatmap.set_xlabel('Periode (5-Jahres-Bins)', fontsize=10)
    ax_heatmap.set_ylabel('Topic', fontsize=10)
    
    # X-Achse
    ax_heatmap.set_xticks([i + 0.5 for i in range(len(periods))])
    ax_heatmap.set_xticklabels([str(p) for p in periods], rotation=45, ha='right', fontsize=8)
    
    plt.setp(ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{VIZ_PATH}/chart_analysis_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # === SUMMARY PLOT ===
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(sig_topics) * 0.3)))
    
    sig_only = plot_data[plot_data['significant']]
    effect_by_topic = sig_only.groupby('topic').agg({
        'phi': 'mean',
        'rate_difference': 'mean',
        'significant': 'sum'
    }).reset_index()
    effect_by_topic['topic_label'] = effect_by_topic['topic'].map(topic_labels)
    effect_by_topic = effect_by_topic.sort_values('rate_difference', ascending=True)
    
    colors = ['forestgreen' if x > 0 else 'crimson' for x in effect_by_topic['rate_difference']]
    axes[0].barh(effect_by_topic['topic_label'], effect_by_topic['rate_difference'] * 100, 
                 color=colors, alpha=0.7)
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[0].set_xlabel('Mittlere Differenz zur erwarteten Chart-Rate (%-Punkte)')
    axes[0].set_title('Durchschnittliche Über-/Unterrepräsentation\n(nur signifikante Perioden)')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    effect_by_topic_sorted = effect_by_topic.sort_values('significant', ascending=True)
    axes[1].barh(effect_by_topic_sorted['topic_label'], effect_by_topic_sorted['significant'], 
                 color='steelblue', alpha=0.7)
    axes[1].set_xlabel(f'Anzahl signifikanter Perioden (von {n_periods})')
    axes[1].set_title('Konsistenz des Effekts über Zeit')
    axes[1].set_xlim(0, n_periods)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{VIZ_PATH}/chart_analysis_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # === ZEITVERLÄUFE ===
    n_topics = len(sorted_topics)
    n_cols = 3
    n_rows = (n_topics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    
    periods = sorted(plot_data['period'].unique())
    
    for idx, topic in enumerate(sorted_topics):
        ax = axes[idx]
        topic_data = plot_data[plot_data['topic'] == topic].sort_values('period')
        label = topic_labels[topic]
        
        # Sammle alle Werte für dynamische Y-Achse
        all_values = []
        
        for i, (_, row) in enumerate(topic_data.iterrows()):
            val = row['rate_difference'] * 100
            all_values.append(val)
            if row['significant']:
                color = 'forestgreen' if row['rate_difference'] > 0 else 'crimson'
                ax.bar(i, val, color=color, alpha=0.8, edgecolor='black', width=0.8)
            else:
                ax.bar(i, val, color='gray', alpha=0.3, width=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # X-Achse
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels([str(p) for p in periods], rotation=45, ha='right', fontsize=8)
        
        ax.set_ylabel('Δ Chart-Rate (%)')
        ax.set_title(f'{label}\n({topic_data["significant"].sum()}/{n_periods} sig.)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Dynamische Y-Achse
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_limit_min = min(-5, y_min - 3)
            y_limit_max = max(5, y_max + 3)
            ax.set_ylim(y_limit_min, y_limit_max)
        else:
            ax.set_ylim(-10, 10)
    
    for idx in range(len(sorted_topics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Chart-Repräsentation über Zeit (5-Jahres-Perioden)\n'
                 '(Farbig = signifikant, Grün = überrepräsentiert, Rot = unterrepräsentiert)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{VIZ_PATH}/chart_analysis_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualisierungen gespeichert in {VIZ_PATH}")


def debug_topic_alignment(df, topic_model):
    """
    Debug helper: Selects a random topic and shows:
    1. Representative documents stored in the model.
    2. Random documents from the dataframe assigned to this topic.
    """
    print(f"\n{'='*20} DEBUG: TOPIC ALIGNMENT CHECK {'='*20}")
    
    # Filter valid topics
    unique_topics = list(set(df['topic'].unique()))
    if -1 in unique_topics:
        unique_topics.remove(-1)
        
    if not unique_topics:
        print("No valid topics found.")
        return

    # Pick a random topic
    random_topic = np.random.choice(unique_topics)
    
    # Get Topic Info
    try:
        topic_info = topic_model.get_topic_info(random_topic)
        if not topic_info.empty:
            name_str = topic_info['Name'].values[0]
        else:
            name_str = f"Topic {random_topic}"
    except:
        name_str = f"Topic {random_topic}"

    print(f"Checking Topic {random_topic}: {name_str}")
    
    # 1. Documents from BERTopic Model (Representative Docs)
    print(f"\n--- 3 Representative Documents from BERTopic Model ---")
    try:
        rep_docs = topic_model.get_representative_docs(random_topic)
        if rep_docs:
            for i, doc in enumerate(rep_docs[:3]):
                clean_doc = doc.replace('\n', ' ').strip()
                print(f"[{i+1}] {clean_doc[:150]}...")
        else:
            print("(No representative documents found in model)")
    except Exception as e:
        print(f"(Error retrieving representative docs: {e})")

    # 2. Documents from JSON (Assigned in DF)
    print(f"\n--- 3 Random Documents Assigned from JSON (Dataframe) ---")
    assigned_docs = df[df['topic'] == random_topic]
    if len(assigned_docs) > 0:
        sample_size = min(3, len(assigned_docs))
        samples = assigned_docs.sample(sample_size)
        for i, (_, row) in enumerate(samples.iterrows()):
            clean_text = str(row['text']).replace('\n', ' ').strip()
            print(f"[{i+1}] (Idx {row.name}) {clean_text[:150]}...")
    else:
        print("(No documents assigned to this topic in DataFrame)")
    
    print("="*60 + "\n")


def main():
    print("Loading BERTopic model...")
    topic_model = BERTopic.load(MODEL_PATH)
    
    print("Loading dataset...")
    df = load_data()
    
    print(f"\nDatensatz:")
    print(f"  Gesamt: {len(df)} Songs")
    print(f"  Chart-Songs: {df['is_chart'].sum()} ({df['is_chart'].mean()*100:.1f}%)")
    print(f"  Non-Chart-Songs: {len(df) - df['is_chart'].sum()} ({(1-df['is_chart'].mean())*100:.1f}%)")
    
    topics = topic_model.topics_
    if topics is None:
        print("Topics not stored in model, extracting...")
        doc_info = topic_model.get_document_info(df['text'].tolist())
        topics = doc_info['Topic'].tolist()
    
    # Load probabilities
    probs = None
    probs_path = os.path.join(os.path.dirname(MODEL_PATH), "probabilities.npy")
    if os.path.exists(probs_path):
        print(f"Loading probabilities from {probs_path}...")
        probs = np.load(probs_path)

    # Align lengths of df, topics, and probs
    min_len = len(df)
    if topics is not None:
        min_len = min(min_len, len(topics))
    if probs is not None:
        min_len = min(min_len, len(probs))
        
    if len(df) > min_len:
        print(f"Truncating data to {min_len} entries to ensure alignment.")
        df = df.iloc[:min_len].copy()
    if topics is not None and len(topics) > min_len:
        topics = topics[:min_len]
    if probs is not None and len(probs) > min_len:
        probs = probs[:min_len]
        
    # Outlier reduction
    if probs is not None:
        print("Reducing outliers (threshold=0.03)...")
        try:
            new_topics = topic_model.reduce_outliers(
                df['text'].tolist(), 
                topics, 
                strategy="probabilities", 
                probabilities=probs, 
                threshold=0.03
            )
            topics = new_topics
            topic_model.topics_ = new_topics
        except Exception as e:
            print(f"Warning: Outlier reduction failed: {e}")
    
    df['topic'] = topics
    print(f"  Topics: {len(set(topics))} (ohne Outlier-Topic -1)")
    
    n_outliers = (df['topic'] == -1).sum()
    print(f"  Outliers (Topic -1): {n_outliers} ({n_outliers/len(df)*100:.1f}%)")
    
    print("\nErstelle 5-Jahres-Bins (1950-2024)...")
    df = create_time_bins(df)
    print(f"  Dokumente nach Zeitfilterung: {len(df)}")
    print(f"  Perioden: {df['period'].nunique()}")
    
    # Debug Test randomly checking alignment
    debug_topic_alignment(df, topic_model)
    
    topic_info = topic_model.get_topic_info()
    
    # Beispiel-Kreuztabelle
    example_topic = df[df['topic'] != -1]['topic'].mode().values[0]
    # Wähle eine Periode aus der Mitte des Datensatzes
    available_periods = sorted(df['period'].dropna().unique())
    example_period = available_periods[len(available_periods) // 2]  # Mittlere Periode
    print_crosstab_example(df, example_topic, example_period, topic_info)
    
    print("\n" + "="*60)
    print("FISHER'S EXACT TESTS PRO TOPIC UND 5-JAHRES-PERIODE")
    print("="*60)
    results_df = chi_square_test_per_topic_period(df)
    
    print("\nErstelle Visualisierungen...")
    visualize_results(results_df, topic_info)
    
    results_df.to_csv(f"{VIZ_PATH}/chi_square_results.csv", index=False)
    print(f"\nErgebnisse gespeichert: {VIZ_PATH}/chi_square_results.csv")
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)
    
    label_map = load_topic_labels_map()
    
    n_periods = len(results_df['period'].unique())
    sig_results = results_df[results_df['significant']]
    
    overrep = sig_results[sig_results['direction'] == 'overrepresented']
    overrep_topics = overrep.groupby('topic').agg({
        'significant': 'sum',
        'rate_difference': 'mean'
    }).sort_values('significant', ascending=False)
    
    print(f"\n📈 Topics ÜBERREPRÄSENTIERT bei Chart-Songs:")
    for topic in overrep_topics.head(10).index:
        n_sig = int(overrep_topics.loc[topic, 'significant'])
        diff = overrep_topics.loc[topic, 'rate_difference'] * 100
        if topic in label_map:
            name = label_map[topic]
        else:
            name_vals = topic_info[topic_info['Topic'] == topic]['Name'].values
            name = name_vals[0] if len(name_vals) > 0 else "Unknown"
        print(f"  T{int(topic)}: +{diff:.1f}% (in {n_sig}/{n_periods} Perioden sig.) - {name}")
    
    underrep = sig_results[sig_results['direction'] == 'underrepresented']
    underrep_topics = underrep.groupby('topic').agg({
        'significant': 'sum',
        'rate_difference': 'mean'
    }).sort_values('significant', ascending=False)
    
    print(f"\n📉 Topics UNTERREPRÄSENTIERT bei Chart-Songs:")
    for topic in underrep_topics.head(10).index:
        n_sig = int(underrep_topics.loc[topic, 'significant'])
        diff = underrep_topics.loc[topic, 'rate_difference'] * 100
        if topic in label_map:
            name = label_map[topic]
        else:
            name_vals = topic_info[topic_info['Topic'] == topic]['Name'].values
            name = name_vals[0] if len(name_vals) > 0 else "Unknown"
        print(f"  T{int(topic)}: {diff:.1f}% (in {n_sig}/{n_periods} Perioden sig.) - {name}")


if __name__ == "__main__":
    main()