import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
from bertopic import BERTopic
import torch
from hdbscan import HDBSCAN

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
    'text.usetex': False
})

# Determine device
def get_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            return 'cuda'
        except Exception as e:
            print(f"CUDA not working: {e}, using CPU")
            return 'cpu'
    return 'cpu'

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths relative to src/visualisation_scripts/topic_model/
    data_path = os.path.join(current_dir, '../../../data/dataset_popmusic_v8.json')
    embeddings_path = os.path.join(current_dir, '../../../bertopic/dtm/embedding/embeddings.npy')
    model_path = os.path.join(current_dir, '../../../bertopic/dtm/models/bertopic_model_dtm')
    probs_path = os.path.join(current_dir, '../../../bertopic/dtm/models/probabilities.npy')
    
    # Output path
    output_path = os.path.join(current_dir, '../../../visualisations/topic_model/outlier_analysis')
    os.makedirs(output_path, exist_ok=True)

    # Check input files
    for p in [data_path, str(model_path), probs_path]: # Embeddings not strictly needed for probability strategy
        if not os.path.exists(p):
            print(f"Error: Path not found at {p}")
            return

    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    timestamps = []
    genres_raw = []

    for entry in data:
        text = entry.get('text', '').strip()
        timestamp = entry.get('timestamp', '')
        genre = entry.get('Genre Genius', '')
        
        if text and timestamp:
            year = int(timestamp[:4])
            
            # Filter by year range (1955-2024 inclusive) to match create_dtm.py and embeddings
            if 1955 <= year <= 2024:
                documents.append(text)
                timestamps.append(year)
                genres_raw.append(genre)

    print(f"Loaded {len(documents)} documents")
    print(f"Year range: {min(timestamps)} - {max(timestamps)}")

    # Load embeddings (Optional for checking, but keeping consistent with original)
    if os.path.exists(embeddings_path):
        print(f"Loading embeddings from {embeddings_path}...")
        embeddings = np.load(embeddings_path)
        print(f"Embeddings shape: {embeddings.shape}")
    else:
        print(f"Embeddings file not found at {embeddings_path}, skipping load (not needed for probability-based reduction).")

    # Parameters
    OUTLIER_THRESHOLD = 0.03

    # Load BERTopic model
    print(f"\nLoading BERTopic model from {model_path}...")
    topic_model = BERTopic.load(model_path)

    # Load probabilities
    print(f"Loading probabilities from {probs_path}...")
    probs = np.load(probs_path)

    # === Analysis BEFORE outlier reduction ===
    print("\n" + "="*60)
    print("BEFORE OUTLIER REDUCTION")
    print("="*60)

    # Use HDBSCAN labels as the "before" state (raw clusters before any reduction)
    topics = topic_model.hdbscan_model.labels_

    # Calculate statistics
    n_topics_before = len(set(topics)) - (1 if -1 in topics else 0)
    outlier_mask_before = [t == -1 for t in topics]
    n_outliers_before = sum(outlier_mask_before)

    print(f"Number of topics (HDBSCAN clusters): {n_topics_before}")
    print(f"Total outliers: {n_outliers_before} ({n_outliers_before/len(documents):.2%})")

    # Analyze outliers by 5-year intervals
    min_year = min(timestamps)
    max_year = max(timestamps)
    start_year = (min_year // 5) * 5
    end_year = ((max_year // 5) + 1) * 5

    time_bins = []
    bin_labels = []
    for y in range(start_year, end_year, 5):
        time_bins.append((y, y+4))
        bin_labels.append(f"{y}-{y+4}")

    def get_bin_index(year):
        return (year - start_year) // 5

    # Aggregate counts by bin
    outliers_by_bin_before = Counter()
    total_by_bin = Counter()

    for i, (ts, is_outlier) in enumerate(zip(timestamps, outlier_mask_before)):
        bin_idx = get_bin_index(ts)
        if 0 <= bin_idx < len(bin_labels):
            label = bin_labels[bin_idx]
            total_by_bin[label] += 1
            if is_outlier:
                outliers_by_bin_before[label] += 1

    print("\nOutliers by 5-year intervals (before reduction):")
    for label in bin_labels:
        count = outliers_by_bin_before[label]
        total = total_by_bin[label]
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label}: {count:4d} / {total:4d} ({pct:5.1f}%)")

    # === Visualization BEFORE outlier reduction ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Absolute outliers by bin
    ax1 = axes[0, 0]
    outlier_counts_before = [outliers_by_bin_before[l] for l in bin_labels]
    non_outlier_counts_before = [total_by_bin[l] - outliers_by_bin_before[l] for l in bin_labels]

    ax1.bar(bin_labels, non_outlier_counts_before, label='Topic zugewiesen', color='steelblue', alpha=0.8)
    ax1.bar(bin_labels, outlier_counts_before, bottom=non_outlier_counts_before, label='Outliers', color='coral', alpha=0.8)
    ax1.set_xlabel('Zeitraum')
    ax1.set_ylabel('Anzahl Dokumente')
    ax1.set_title('Dokumente nach Zeitraum - VOR Outlier Reduction')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Percentage of outliers by bin
    ax2 = axes[0, 1]
    outlier_pct_before = [outliers_by_bin_before[l] / total_by_bin[l] * 100 if total_by_bin[l] > 0 else 0 for l in bin_labels]
    ax2.bar(bin_labels, outlier_pct_before, color='coral', alpha=0.8)
    ax2.axhline(y=n_outliers_before/len(documents)*100, color='red', linestyle='--', label=f'Gesamt: {n_outliers_before/len(documents)*100:.1f}%')
    ax2.set_xlabel('Zeitraum')
    ax2.set_ylabel('Outlier Anteil (%)')
    ax2.set_title('Outlier Anteil nach Zeitraum - VOR Reduction')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)

    # === Apply outlier reduction ===
    print("\n" + "="*60)
    print(f"APPLYING OUTLIER REDUCTION (threshold={OUTLIER_THRESHOLD})")
    print("="*60)

    # Apply reduction to the "before" topics (HDBSCAN labels) using the loaded probabilities
    new_topics = topic_model.reduce_outliers(
        documents, 
        topics, 
        strategy="probabilities", 
        probabilities=probs,
        threshold=OUTLIER_THRESHOLD
    )

    # === Analysis AFTER outlier reduction ===
    print("\n" + "="*60)
    print("AFTER OUTLIER REDUCTION")
    print("="*60)

    n_topics_after = len(set(new_topics)) - (1 if -1 in new_topics else 0)
    outlier_mask_after = [t == -1 for t in new_topics]
    n_outliers_after = sum(outlier_mask_after)

    print(f"Number of topics: {n_topics_after}")
    print(f"Total outliers: {n_outliers_after} ({n_outliers_after/len(documents):.2%})")

    # Analyze outliers by bin after reduction
    outliers_by_bin_after = Counter()
    for i, (ts, is_outlier) in enumerate(zip(timestamps, outlier_mask_after)):
        bin_idx = get_bin_index(ts)
        if 0 <= bin_idx < len(bin_labels):
            label = bin_labels[bin_idx]
            if is_outlier:
                outliers_by_bin_after[label] += 1

    print("\nOutliers by 5-year intervals (after reduction):")
    for label in bin_labels:
        count = outliers_by_bin_after[label]
        total = total_by_bin[label]
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label}: {count:4d} / {total:4d} ({pct:5.1f}%)")

    # === Visualization AFTER outlier reduction ===
    # Plot 3: Absolute outliers by bin (after)
    ax3 = axes[1, 0]
    outlier_counts_after = [outliers_by_bin_after[l] for l in bin_labels]
    non_outlier_counts_after = [total_by_bin[l] - outliers_by_bin_after[l] for l in bin_labels]

    ax3.bar(bin_labels, non_outlier_counts_after, label='Topic zugewiesen', color='steelblue', alpha=0.8)
    ax3.bar(bin_labels, outlier_counts_after, bottom=non_outlier_counts_after, label='Outliers', color='coral', alpha=0.8)
    ax3.set_xlabel('Zeitraum')
    ax3.set_ylabel('Anzahl Dokumente')
    ax3.set_title('Dokumente nach Zeitraum - NACH Outlier Reduction')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Percentage of outliers by bin (after)
    ax4 = axes[1, 1]
    outlier_pct_after = [outliers_by_bin_after[l] / total_by_bin[l] * 100 if total_by_bin[l] > 0 else 0 for l in bin_labels]
    ax4.bar(bin_labels, outlier_pct_after, color='coral', alpha=0.8)
    ax4.axhline(y=n_outliers_after/len(documents)*100, color='red', linestyle='--', label=f'Gesamt: {n_outliers_after/len(documents)*100:.1f}%')
    ax4.set_xlabel('Zeitraum')
    ax4.set_ylabel('Outlier Anteil (%)')
    ax4.set_title('Outlier Anteil nach Zeitraum - NACH Reduction')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'outlier_analysis_comparison.png'), dpi=300, bbox_inches='tight')
    # plt.show() # Commented out for automated execution
    print(f"\nVisualization saved to {os.path.join(output_path, 'outlier_analysis_comparison.png')}")

    # === Additional comparison visualization ===
    fig2, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(bin_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, outlier_pct_before, width, label='Vor Reduction', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, outlier_pct_after, width, label='Nach Reduction', color='seagreen', alpha=0.8)

    ax.set_xlabel('Zeitraum')
    ax.set_ylabel('Outlier Anteil (%)')
    ax.set_title('Vergleich Outlier-Anteil: Vor vs Nach Reduction')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'outlier_comparison_by_year.png'), dpi=300, bbox_inches='tight')
    # plt.show() # Commented out
    print(f"Comparison saved to {os.path.join(output_path, 'outlier_comparison_by_year.png')}")

    # === Outlier Ratio Over Time (Standalone) ===
    plt.figure(figsize=(12, 6))
    outlier_pct = [outliers_by_bin_after[l] / total_by_bin[l] * 100 if total_by_bin[l] > 0 else 0 for l in bin_labels]
    
    plt.bar(bin_labels, outlier_pct, color='lightcoral', edgecolor='black', width=0.7)
    
    # Add trend line (Average)
    avg_outlier_pct = n_outliers_after / len(documents) * 100
    plt.axhline(y=avg_outlier_pct, color='darkred', linestyle='--', linewidth=2, label=f'Durchschnitt: {avg_outlier_pct:.1f}%')
    
    plt.xlabel('Zeitraum', fontsize=12)
    plt.ylabel('Anteil Outlier (%)', fontsize=12)
    plt.title('Anteil der Outlier-Dokumente pro 5-Jahres-Zeitraum', fontsize=16)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    y_max = max(outlier_pct) if outlier_pct else 0
    plt.ylim(0, y_max * 1.15) # Add space for labels
    
    for i, v in enumerate(outlier_pct):
        plt.text(i, v + (y_max * 0.02), f'{v:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    output_path_ratio = os.path.join(output_path, 'outlier_ratio_over_time.png')
    plt.savefig(output_path_ratio, dpi=300)
    print(f"Saved {output_path_ratio}")

    # === Probability Distribution Visualization ===
    print("\n" + "="*60)
    print("PROBABILITY DISTRIBUTION ANALYSIS")
    print("="*60)

    # Ensure probs is a numpy array
    probs = np.array(probs)

    # Calculate max probability for each document
    if probs.ndim > 1:
        max_probs = np.max(probs, axis=1)
    else:
        max_probs = probs

    fig3, ax_prob = plt.subplots(figsize=(14, 7))

    # Create finer bins (0.01 steps)
    bins = np.arange(0, 1.01, 0.01)

    # Plot histogram
    counts, bins, patches = ax_prob.hist(max_probs, bins=bins, color='mediumpurple', edgecolor='black', alpha=0.7)

    # Add vertical line for threshold
    ax_prob.axvline(x=OUTLIER_THRESHOLD, color='red', linestyle='-', linewidth=2, label=f'Threshold ({OUTLIER_THRESHOLD})')

    # Highlight the outlier region
    ax_prob.axvspan(0, OUTLIER_THRESHOLD, alpha=0.2, color='red', label='Outlier-Bereich')

    # Set Y limit to accommodate labels
    y_max = max(counts)
    ax_prob.set_ylim(0, y_max * 1.15)
    
    # Add labels on top of bars
    y_offset = y_max * 0.02

    for count, rect in zip(counts, patches):
        if count > 0:
            height = rect.get_height()
            if rect.get_x() <= 0.1 or height > y_max * 0.02:
                ax_prob.text(rect.get_x() + rect.get_width()/2., height + y_offset,
                        f'{int(count)}',
                        ha='center', va='bottom', fontsize=7, rotation=90)

    ax_prob.set_xlabel('Wahrscheinlichkeit der Top-Topic-Zuweisung', labelpad=10)
    ax_prob.set_ylabel('Anzahl Dokumente')
    ax_prob.set_title(f'Verteilung der Top-Topic-Zuweisungswahrscheinlichkeiten (Threshold: {OUTLIER_THRESHOLD})')

    # Set x-ticks to show 0.03 clearly
    ticks = np.arange(0, 1.05, 0.05)
    if not any(np.isclose(ticks, OUTLIER_THRESHOLD)):
        ticks = np.sort(np.append(ticks, OUTLIER_THRESHOLD))

    ax_prob.set_xticks(ticks)
    tick_labels = []
    for t in ticks:
        if np.isclose(t, OUTLIER_THRESHOLD):
            tick_labels.append(f'{t:.2f} (Thresh)')
        else:
            tick_labels.append(f'{t:.2f}')

    ax_prob.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)

    for label in ax_prob.get_xticklabels():
        if "Thresh" in label.get_text():
            label.set_color('red')
            label.set_fontweight('bold')

    ax_prob.grid(axis='y', alpha=0.3, linestyle='--')
    ax_prob.set_xlim(0, 1.0)
    ax_prob.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'assignment_probability_distribution.png'), dpi=300, bbox_inches='tight')
    # plt.show() # Commented out
    print(f"Probability distribution saved to {os.path.join(output_path, 'assignment_probability_distribution.png')}")

    # === Outlier Genre Analysis ===
    print("\n" + "="*60)
    print("OUTLIER GENRE ANALYSIS")
    print("="*60)

    def process_genres(genre_str):
        if not genre_str or not isinstance(genre_str, str):
            return []
        # Split by ; only (logic from viz_genres.py)
        parts = genre_str.split(';')
        parts = [g.strip() for g in parts if g.strip()]
        return parts

    outlier_genres = []
    # Identify outliers based on new_topics (after reduction)
    for idx, topic in enumerate(new_topics):
        if topic == -1:
            # Get the raw genre string corresponding to this document
            g_str = genres_raw[idx]
            # Process and add to list
            outlier_genres.extend(process_genres(g_str))

    if outlier_genres:
        # Convert to pandas Series for easier handling analogous to viz_genres.py
        genre_outlier_counts = pd.Series(outlier_genres).value_counts()
        top_20_outlier_genres = genre_outlier_counts.head(20)
        
        print(f"Found {len(outlier_genres)} total genre tags in {n_outliers_after} outlier documents.")
        print("Top 20 Genres in Outliers:")
        print(top_20_outlier_genres)

        # 1. Bar Chart: Top 20 Genres (Absolute) - Style from viz_genres.py
        plt.figure(figsize=(12, 8))
        top_20_outlier_genres.plot(kind='bar', color='skyblue')
        plt.title('Top 20 Genres (Outlier Topic -1)', fontsize=16)
        plt.xlabel('Genre', fontsize=12)
        plt.ylabel('Anzahl der Songs', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path_bar = os.path.join(output_path, 'outlier_genres_top20_bar.png')
        plt.savefig(output_path_bar, dpi=300)
        print(f"Saved {output_path_bar}")

        # 2. Pie Chart: Top 20 Genres + "Andere" (Relative) - Style from viz_genres.py
        top_20_sum = top_20_outlier_genres.sum()
        total_sum = genre_outlier_counts.sum()
        other_sum = total_sum - top_20_sum
        
        pie_data = top_20_outlier_genres.copy()
        pie_data['Andere'] = other_sum
        
        # Define colors: use tab20 for the top 20 genres, and gray for 'Andere'
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i/20) for i in range(20)]
        colors.append('lightgray') # Distinct color for 'Andere'

        plt.figure(figsize=(12, 12))
        
        def make_autopct(values):
            def my_autopct(pct):
                # Only show percentage if it's big enough to avoid clutter
                if pct > 2:
                    return f'{pct:.1f}%'
                return ''
            return my_autopct

        plt.pie(pie_data, labels=pie_data.index, autopct=make_autopct(pie_data), startangle=140, textprops={'fontsize': 10}, colors=colors)
        plt.title('Verteilung der Genres im Outlier Topic (Top 20 + Andere)', fontsize=16)
        plt.tight_layout()
        
        output_path_pie = os.path.join(output_path, "outlier_genres_distribution_pie.png")
        plt.savefig(output_path_pie, dpi=300)
        print(f"Saved {output_path_pie}")

    else:
        print("No outlier genres found to visualize.")

    # === Summary ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Documents processed: {len(documents)}")
    print(f"\nBEFORE reduction:")
    print(f"  Topics: {n_topics_before}")
    print(f"  Outliers: {n_outliers_before} ({n_outliers_before/len(documents):.2%})")
    print(f"\nAFTER reduction (threshold={OUTLIER_THRESHOLD}):")
    print(f"  Topics: {n_topics_after}")
    print(f"  Outliers: {n_outliers_after} ({n_outliers_after/len(documents):.2%})")
    print(f"\nReduction: {n_outliers_before - n_outliers_after} documents reassigned ({(n_outliers_before - n_outliers_after)/n_outliers_before*100:.1f}% of original outliers)")

if __name__ == "__main__":
    main()
