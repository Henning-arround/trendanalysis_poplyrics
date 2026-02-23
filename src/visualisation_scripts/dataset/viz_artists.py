import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import json
import os

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
    return pd.DataFrame(data)

def main():
    df = load_data(DATA_PATH)
    
    top_artists = df['Künstler'].value_counts().head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_artists.values, y=top_artists.index, palette='viridis')
    
    plt.title('Top 20 Künstler mit den meisten Songs')
    plt.xlabel('Anzahl der Songs')
    plt.ylabel('Künstler')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "top_20_artists.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

    # Second plot: Distribution of songs per artist
    songs_per_artist = df['Künstler'].value_counts()
    mean_songs = songs_per_artist.mean()
    std_songs = songs_per_artist.std()

    plt.figure(figsize=(10, 6))
    sns.histplot(songs_per_artist, kde=True, bins=50)
    plt.axvline(mean_songs, color='r', linestyle='--', label=f'Durchschnitt: {mean_songs:.2f}')
    plt.axvline(mean_songs + std_songs, color='g', linestyle=':', label=f'Standardabweichung: {std_songs:.2f}')
    plt.axvline(mean_songs - std_songs, color='g', linestyle=':')
    
    plt.title('Verteilung der Anzahl der Songs pro Künstler')
    plt.xlabel('Anzahl der Songs')
    plt.ylabel('Anzahl der Künstler')
    plt.legend()
    plt.tight_layout()
    
    output_path_dist = os.path.join(OUTPUT_DIR, "songs_per_artist_distribution.png")
    plt.savefig(output_path_dist, dpi=300)
    print(f"Saved {output_path_dist}")

if __name__ == "__main__":
    main()
