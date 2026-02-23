import pandas as pd
import os

# Define file paths
charts_file = '../data/charts_with_language_updated_v2.csv'
genius_file = '../data/missing_source_lyrics.csv'
output_file = '../data/charts_with_language_updated_v3.csv'

# Create the data directory if it doesn't exist
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the datasets with proper CSV settings
try:
    charts_df = pd.read_csv(charts_file, sep=',', quotechar='"', escapechar='\\')
    genius_df = pd.read_csv(genius_file, sep=',', quotechar='"', escapechar='\\')
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    exit()

# Check if both dataframes have the required columns
if 'Index' not in charts_df.columns or 'Quelle Lyrics' not in charts_df.columns:
    print("Error: 'Index' or 'Quelle Lyrics' column not found in charts file")
    exit()

if 'Index' not in genius_df.columns or 'Quelle Lyrics' not in genius_df.columns:
    print("Error: 'Index' or 'Quelle Lyrics' column not found in genius file")
    exit()

# ✅ Optimiert: Verwende merge() für bessere Performance
# Temporäre Spalte erstellen für das Update
genius_df_temp = genius_df.rename(columns={'Quelle Lyrics': 'Quelle Lyrics_new'})

# Merge mit left join
charts_df = charts_df.merge(genius_df_temp[['Index', 'Quelle Lyrics_new']], 
                           on='Index', how='left')

# Update nur die Zeilen, wo neue Werte vorhanden sind
charts_df.loc[charts_df['Quelle Lyrics_new'].notna(), 'Quelle Lyrics'] = \
    charts_df.loc[charts_df['Quelle Lyrics_new'].notna(), 'Quelle Lyrics_new']

# Temporäre Spalte entfernen
charts_df = charts_df.drop('Quelle Lyrics_new', axis=1)

# Save the updated dataframe with proper CSV settings
charts_df.to_csv(output_file, index=False, sep=',', quotechar='"', quoting=1)

print(f"Updated table saved to {output_file}")