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
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['year'] = df['timestamp'].dt.year
    
    # Filter for range 1955 to 2024
    df_filtered = df[(df['year'] >= 1955) & (df['year'] <= 2024)].copy()
    
    # Create 5-year bins
    bins = list(range(1955, 2030, 5))
    labels = [f"{b}-{b+4}" for b in bins[:-1]]
    df_filtered['time_bin'] = pd.cut(df_filtered['year'], bins=bins, labels=labels, right=False)
    
    # Determine chart status
    df_filtered['is_chart'] = df_filtered['Erste Notierung'].apply(lambda x: True if x and str(x).strip() != "" else False)
    
    # Group by bin and chart status
    counts = df_filtered.groupby(['time_bin', 'is_chart'], observed=False).size().unstack(fill_value=0)
    
    # Plot
    plt.figure(figsize=(14, 8))
    counts.plot(kind='bar', stacked=True, color=['lightgray', 'skyblue'], ax=plt.gca())
    
    plt.title('Anzahl der Songs in 5-Jahres-Intervallen (1955-2024)')
    plt.xlabel('Zeitraum')
    plt.ylabel('Anzahl der Songs')
    plt.legend(['Nicht in Charts', 'In Charts'], title='Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "songs_per_5year_bin.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
