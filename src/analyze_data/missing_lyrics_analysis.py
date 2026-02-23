import pandas as pd
import os
from collections import Counter
import re

# Define file paths
input_file = '../data/charts_with_language_updated.csv'
output_file = 'missing_lyrics_analysis.txt'

# Load the dataset
try:
    df = pd.read_csv(input_file, sep=',', quotechar='"', escapechar='\\')
    print(f"Loaded {len(df)} rows from {input_file}")
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    exit()

# Filter rows where "Quelle Lyrics" is empty or NaN
missing_lyrics = df[df['Quelle Lyrics'].isna() | (df['Quelle Lyrics'] == '') | (df['Quelle Lyrics'].str.strip() == '')]

print(f"Found {len(missing_lyrics)} entries without lyrics source")

# Extract relevant columns
analysis_data = missing_lyrics[['Titel', 'Künstler', 'Quelle Lyrics']].copy()

# Write only entries to file
with open(output_file, 'w', encoding='utf-8') as f:
    for idx, row in analysis_data.iterrows():
        f.write(f"{row['Künstler']} - {row['Titel']}\n")

# Analyze patterns for terminal output
title_symbols = []
artist_symbols = []
common_words_title = []
common_words_artist = []

for title in analysis_data['Titel'].dropna():
    title_str = str(title)
    symbols = re.findall(r'[&/\-\+\(\)\[\]\"\'\.,:;!?]', title_str)
    title_symbols.extend(symbols)
    words = re.findall(r'\b\w+\b', title_str.lower())
    common_words_title.extend(words)

for artist in analysis_data['Künstler'].dropna():
    artist_str = str(artist)
    symbols = re.findall(r'[&/\-\+\(\)\[\]\"\'\.,:;!?]', artist_str)
    artist_symbols.extend(symbols)
    words = re.findall(r'\b\w+\b', artist_str.lower())
    common_words_artist.extend(words)

# Count occurrences
title_symbol_counts = Counter(title_symbols)
artist_symbol_counts = Counter(artist_symbols)
title_word_counts = Counter(common_words_title)
artist_word_counts = Counter(common_words_artist)

# Print statistics to terminal
print("\n" + "=" * 50)
print("PATTERN ANALYSIS STATISTICS")
print("=" * 50)

print("\nMOST COMMON SYMBOLS IN TITLES:")
for symbol, count in title_symbol_counts.most_common(10):
    print(f"'{symbol}': {count} times")

print("\nMOST COMMON SYMBOLS IN ARTISTS:")
for symbol, count in artist_symbol_counts.most_common(10):
    print(f"'{symbol}': {count} times")

print("\nMOST COMMON WORDS IN TITLES:")
for word, count in title_word_counts.most_common(15):
    print(f"'{word}': {count} times")

print("\nMOST COMMON WORDS IN ARTISTS:")
for word, count in artist_word_counts.most_common(15):
    print(f"'{word}': {count} times")

# Special pattern checks
feat_pattern = analysis_data[analysis_data['Titel'].str.contains('feat|ft\.', case=False, na=False) | 
                            analysis_data['Künstler'].str.contains('feat|ft\.', case=False, na=False)]
and_pattern = analysis_data[analysis_data['Titel'].str.contains(' and ', case=False, na=False) | 
                           analysis_data['Künstler'].str.contains(' and ', case=False, na=False)]
ampersand_pattern = analysis_data[analysis_data['Titel'].str.contains('&', na=False) | 
                                 analysis_data['Künstler'].str.contains('&', na=False)]
slash_pattern = analysis_data[analysis_data['Titel'].str.contains('/', na=False) | 
                             analysis_data['Künstler'].str.contains('/', na=False)]

print("\n" + "=" * 50)
print("SPECIAL PATTERN CHECKS")
print("=" * 50)
print(f"Entries with 'feat' or 'ft.': {len(feat_pattern)}")
print(f"Entries with ' and ': {len(and_pattern)}")
print(f"Entries with '&': {len(ampersand_pattern)}")
print(f"Entries with '/': {len(slash_pattern)}")

print(f"\nEntries saved to {output_file}")
