import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import textwrap

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

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to input CSV
    csv_path = os.path.join(current_dir, '../../../bertopic/dtm/models/topics_over_time.csv')
    
    # Path to output directory
    output_dir = os.path.join(current_dir, '../../../visualisations/topic_model')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading data from: {os.path.abspath(csv_path)}")
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Calculate Relative Frequency
    # Total songs per bin
    bin_totals = df.groupby('Timestamp')['Frequency'].sum().reset_index()
    bin_totals.rename(columns={'Frequency': 'Total_Freq_In_Bin'}, inplace=True)

    # Merge back to original df
    df = df.merge(bin_totals, on='Timestamp')
    df['Relative_Frequency'] = df['Frequency'] / df['Total_Freq_In_Bin']

    # Filter for Topic 
    topic_id = 13
    topic_df = df[df['Topic'] == topic_id].copy()

    if topic_df.empty:
        print(f"No data found for Topic {topic_id}")
        return

    # Sort by Timestamp just in case
    # Assuming Timestamp format "YYYY-YYYY"
    topic_df['Start_Year'] = topic_df['Timestamp'].str.split('-').str[0].astype(int)
    topic_df = topic_df.sort_values('Start_Year')

    # Get Global Representation for Title
    global_rep = topic_df['Global_Representation'].iloc[0] if 'Global_Representation' in topic_df.columns else f"Topic {topic_id}"

    # Prepare Visualization
    fig, ax = plt.subplots(figsize=(16, 10))

    years = topic_df['Start_Year'].tolist()
    rel_frequencies = topic_df['Relative_Frequency'].tolist()
    timestamps = topic_df['Timestamp'].tolist()
    words_list = topic_df['Words'].tolist()

    # Plot the Trend Line (Relative Frequency)
    ax.plot(years, rel_frequencies, marker='o', color='#2c3e50', linewidth=2, label='Relative Frequenz')
    
    # Customize Axes
    ax.set_xlabel('Zeitraum', fontsize=12)
    ax.set_ylabel('Relative Häufigkeit (Anteil am Korpus)', fontsize=12)
    ax.set_title(f'Entwicklung von Topic {topic_id}: "{global_rep}"', fontsize=16, pad=20)
    
    # Set X ticks
    ax.set_xticks(years)
    ax.set_xticklabels(timestamps, rotation=45, ha='right')

    # Improve Y-axis limit to give space for annotations
    max_freq = max(rel_frequencies)
    ax.set_ylim(0, max_freq * 2)  # Make more room at the top

    # Add Words annotations
    for i, (year, freq, words_str) in enumerate(zip(years, rel_frequencies, words_list)):
        # Parse top 5 words
        words = [w.strip() for w in words_str.split(',')]
        top_words = words[:5]
        words_display = "\n".join(top_words)
        
        # Stagger annotations to prevent overlap - 4 levels with larger spacing
        mod = i % 2
        base_offset = 40
        step = 90
        offset = base_offset + (mod * step)
        
        # Add a text box
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bdc3c7')
        
        ax.annotate(words_display, 
                    xy=(year, freq), 
                    xytext=(0, offset), 
                    textcoords='offset points',
                    ha='center', 
                    va='bottom',
                    bbox=props,
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#7f8c8d'))

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_filename = f'topic_{topic_id}_development.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
