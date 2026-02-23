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
    
    # Process genres
    def process_genres(genre_str):
        if not genre_str or not isinstance(genre_str, str):
            return []
        # Split by ; only
        parts = genre_str.split(';')
        parts = [g.strip() for g in parts if g.strip()]
        return parts

    df['genre_list'] = df['Genre Genius'].apply(process_genres)
    
    # Flatten list to count all genres
    all_genres = [genre for sublist in df['genre_list'] for genre in sublist]
    if not all_genres:
        print("No genres found.")
        return
        
    genre_counts = pd.Series(all_genres).value_counts()
    top_20_genres = genre_counts.head(20)
    
    # 1. Bar Chart: Top 20 Genres (Absolute)
    plt.figure(figsize=(12, 8))
    top_20_genres.plot(kind='bar', color='skyblue')
    plt.title('Top 20 Genres (Absolute Anzahl)', fontsize=16)
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Anzahl der Songs', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path_bar = os.path.join(OUTPUT_DIR, "top_20_genres_bar.png")
    plt.savefig(output_path_bar, dpi=300)
    print(f"Saved {output_path_bar}")
    
    # 2. Pie Chart: Top 20 Genres + "Andere" (Relative)
    top_20_sum = top_20_genres.sum()
    total_sum = genre_counts.sum()
    other_sum = total_sum - top_20_sum
    
    pie_data = top_20_genres.copy()
    pie_data['Andere'] = other_sum
    
    # Define colors: use tab20 for the top 20 genres, and gray for 'Andere'
    # tab20 has 20 distinct colors. We map them to the first 20 slices.
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
    plt.title('Verteilung der Genres (Top 20 + Andere)', fontsize=16)
    plt.tight_layout()
    
    output_path_pie = os.path.join(OUTPUT_DIR, "genres_distribution_pie.png")
    plt.savefig(output_path_pie, dpi=300)
    print(f"Saved {output_path_pie}")

if __name__ == "__main__":
    main()
