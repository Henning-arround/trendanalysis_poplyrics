import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    
    # Extract year from timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['year'] = df['timestamp'].dt.year
    
    # Drop rows without a valid year or artist
    df = df.dropna(subset=['year', 'Künstler'])
    df['year'] = df['year'].astype(int)
    
    # Determine if a song is a rap song
    def is_rap_song(genre_str):
        if not genre_str or not isinstance(genre_str, str):
            return False
        genre_lower = genre_str.lower()
        return 'rap' in genre_lower or 'deutschrap' in genre_lower or 'hip-hop' in genre_lower 
        
    df['is_rap_song'] = df['Genre Genius'].apply(is_rap_song)
    
    # Aggregate by artist to determine if they are a rap artist
    artist_stats = df.groupby('Künstler').agg(
        total_songs=('is_rap_song', 'count'),
        rap_songs=('is_rap_song', 'sum')
    )
    artist_stats['rap_percentage'] = artist_stats['rap_songs'] / artist_stats['total_songs']
    artist_stats['is_rap_artist'] = artist_stats['rap_percentage'] >= 0.10
    
    # Map back to main dataframe
    df = df.merge(artist_stats[['is_rap_artist']], left_on='Künstler', right_index=True)
    
    # Determine chart status
    df['is_chart'] = df['Erste Notierung'].apply(lambda x: True if x and str(x).strip() != "" else False)
    
    # Group by year, is_rap_artist, is_chart, and Künstler to get song counts per artist per year
    yearly_artist_counts = df.groupby(['year', 'is_rap_artist', 'is_chart', 'Künstler']).size().reset_index(name='song_count')
    
    # Now calculate the average number of songs per artist for each year and group
    yearly_avg = yearly_artist_counts.groupby(['year', 'is_rap_artist', 'is_chart'])['song_count'].mean().unstack(level=['is_rap_artist', 'is_chart'])
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot for Chart Songs
    if (True, True) in yearly_avg.columns:
        ax1.plot(yearly_avg.index, yearly_avg[(True, True)], label='Rap-Künstler:innen', color='orange', linewidth=2)
    if (False, True) in yearly_avg.columns:
        ax1.plot(yearly_avg.index, yearly_avg[(False, True)], label='Nicht-Rap-Künstler:innen', color='blue', linewidth=2)
        
    ax1.set_title('In Charts: Durchschnittliche Anzahl an Chart-Songs pro aktiven Künstler:innen pro Jahr')
    ax1.set_ylabel('Ø Anzahl Chart-Songs')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot for Non-Chart Songs
    if (True, False) in yearly_avg.columns:
        ax2.plot(yearly_avg.index, yearly_avg[(True, False)], label='Rap-Künstler:innen', color='orange', linewidth=2)
    if (False, False) in yearly_avg.columns:
        ax2.plot(yearly_avg.index, yearly_avg[(False, False)], label='Nicht-Rap-Künstler:innen', color='blue', linewidth=2)
        
    ax2.set_title('Nicht in Charts: Durchschnittliche Anzahl an Nicht-Chart Songs pro aktiven Künstler:innen pro Jahr')
    ax2.set_xlabel('Jahr')
    ax2.set_ylabel('Ø Anzahl Nicht-Chart-Songs')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "avg_songs_rap_vs_nonrap.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
