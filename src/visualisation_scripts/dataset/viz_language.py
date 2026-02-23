import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np

import json

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

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../../data/dataset_popmusic_v8.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "../../../visualisations/dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Extract year from timestamp
    # Convert timestamp to datetime and extract year
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['Jahr'] = df['timestamp'].dt.year
    
    return df

def main():
    df = load_data(DATA_PATH)
    
    # Filter for valid languages
    df = df.dropna(subset=['Sprache Lyrics'])
    
    # --- 1. Pie Chart: Language Distribution ---
    language_counts = df['Sprache Lyrics'].value_counts()
    total_count = language_counts.sum()
    
    # Determine threshold for "Other" (e.g., keep top 10 or those > 1%)
    # Let's keep top 10 for readability
    top_n = 10
    if len(language_counts) > top_n:
        top_languages = language_counts.head(top_n)
        other_count = language_counts.iloc[top_n:].sum()
        pie_data = top_languages.copy()
        pie_data['Andere'] = other_count
    else:
        pie_data = language_counts

    plt.figure(figsize=(10, 10))
    
    def make_autopct(values):
        def my_autopct(pct):
            if pct > 1: # Only show if > 1%
                return f'{pct:.1f}%'
            return ''
        return my_autopct

    # Use a colormap
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(pie_data))]
    
    plt.pie(pie_data, labels=pie_data.index, autopct=make_autopct(pie_data), startangle=140, colors=colors)
    plt.title('Verteilung der Sprachen (Lyrics)', fontsize=16)
    plt.tight_layout()
    
    output_path_pie = os.path.join(OUTPUT_DIR, "language_distribution_pie.png")
    plt.savefig(output_path_pie, dpi=300)
    print(f"Saved {output_path_pie}")

    # --- 2. Bar Chart: Language Share over Time (5-year steps) ---
    # Filter years 1955 to 2024
    df_time = df[(df['Jahr'] >= 1955) & (df['Jahr'] <= 2024)].copy()
    
    # Create 5-year bins
    # Bins: 1955, 1960, 1965, ... 2025 (right edge exclusive usually, or we define labels)
    bins = range(1955, 2030, 5) # 1955, 1960, ..., 2025
    labels = [f"{y}-{y+4}" for y in range(1955, 2025, 5)]
    
    df_time['time_bin'] = pd.cut(df_time['Jahr'], bins=bins, labels=labels, right=False)
    
    # We want percentage of each language per bin
    # Group by bin and language
    counts_per_bin = df_time.groupby(['time_bin', 'Sprache Lyrics'], observed=False).size().unstack(fill_value=0)
    
    # Normalize to percentages
    percentages_per_bin = counts_per_bin.div(counts_per_bin.sum(axis=1), axis=0) * 100
    
    # To make the chart readable, we should probably only show the top languages overall, 
    # and group the rest into "Other".
    # Let's use the same top languages from the pie chart (excluding 'Other' if we added it manually)
    top_langs_list = language_counts.head(top_n).index.tolist()
    
    # Filter columns: Keep top languages, sum others
    # Identify columns that are NOT in top_langs_list
    other_cols = [col for col in percentages_per_bin.columns if col not in top_langs_list]
    
    # Create a new DataFrame for plotting
    plot_df = percentages_per_bin[top_langs_list].copy()
    if other_cols:
        plot_df['Andere'] = percentages_per_bin[other_cols].sum(axis=1)
    
    # Plot stacked bar chart
    ax = plot_df.plot(kind='bar', stacked=True, colormap='tab20', figsize=(15, 8), width=0.8)
    
    # Add percentage labels to every 3rd bar
    for container in ax.containers:
        for i, bar in enumerate(container):
            if i % 3 == 0:  # Every 3rd bar
                height = bar.get_height()
                if height > 2:  # Only label if > 2% to avoid clutter
                    ax.annotate(f'{height:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                                xytext=(0, 0),
                                textcoords="offset points",
                                ha='center', va='center', fontsize=8)
    
    plt.title('Sprachanteil pro 5-Jahres-Zeitraum (1955-2024)', fontsize=16)
    plt.xlabel('Zeitraum', fontsize=12)
    plt.ylabel('Anteil (%)', fontsize=12)
    plt.legend(title='Sprache', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path_bar = os.path.join(OUTPUT_DIR, "language_distribution_over_time.png")
    plt.savefig(output_path_bar, dpi=300)
    print(f"Saved {output_path_bar}")

if __name__ == "__main__":
    main()
