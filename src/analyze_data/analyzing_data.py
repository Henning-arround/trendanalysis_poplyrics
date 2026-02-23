import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Create Visualizations directory if it doesn't exist
vis_dir = "Visualizations"
os.makedirs(vis_dir, exist_ok=True)

# Load the CSV file
df = pd.read_csv("../data/charts_with_language_updated.csv")

print("Dataset Overview:")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())

# Filter rows that have entries in "Quelle Lyrics" column
df_with_lyrics = df[df['Quelle Lyrics'].notna() & (df['Quelle Lyrics'] != '')]

print(f"\nRows with 'Quelle Lyrics' entries: {len(df_with_lyrics)}")

# Convert 'Jahr' to numeric, handling any non-numeric values
df_with_lyrics['Jahr'] = pd.to_numeric(df_with_lyrics['Jahr'], errors='coerce')

# Remove rows where Jahr is NaN and filter for years 1925-2025
df_with_lyrics = df_with_lyrics[df_with_lyrics['Jahr'].notna()]
df_with_lyrics = df_with_lyrics[(df_with_lyrics['Jahr'] >= 1925) & (df_with_lyrics['Jahr'] <= 2025)]

print(f"Rows with valid 'Jahr' (1925-2025) and 'Quelle Lyrics': {len(df_with_lyrics)}")

# Create 10-year bins with fixed range 1925-2025
bins = range(1925, 2026, 10)  # 1925-1934, 1935-1944, ..., 2015-2024
df_with_lyrics['Jahr_Bin'] = pd.cut(df_with_lyrics['Jahr'], bins=bins, right=False, include_lowest=True)

print(f"\nYear range (filtered): 1925 - 2025")
print(f"Bins created: {len(bins)-1} bins")

# Separate data based on Chartentry
df_with_chartentry = df_with_lyrics[df_with_lyrics['Chartentry'].notna() & (df_with_lyrics['Chartentry'] != '')]
df_without_chartentry = df_with_lyrics[df_with_lyrics['Chartentry'].isna() | (df_with_lyrics['Chartentry'] == '')]

print(f"\nWith 'Quelle Lyrics' AND 'Chartentry': {len(df_with_chartentry)}")
print(f"With 'Quelle Lyrics' but WITHOUT 'Chartentry': {len(df_without_chartentry)}")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Analyse der Zeilen mit "Quelle Lyrics" Einträgen (1925-2025)', fontsize=16)

# Plot 1: With Chartentry
counts_with_chart = df_with_chartentry['Jahr_Bin'].value_counts().sort_index()
bin_labels = [f"{interval.left}-{interval.right-1}" for interval in counts_with_chart.index]
axes[0, 0].bar(range(len(counts_with_chart)), counts_with_chart.values, alpha=0.7, color='blue')
axes[0, 0].set_title('Mit "Quelle Lyrics" UND "Chartentry"')
axes[0, 0].set_xlabel('10-Jahres Bins')
axes[0, 0].set_ylabel('Anzahl Zeilen')
axes[0, 0].set_xticks(range(len(counts_with_chart)))
axes[0, 0].set_xticklabels(bin_labels, rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Without Chartentry
counts_without_chart = df_without_chartentry['Jahr_Bin'].value_counts().sort_index()
bin_labels_without = [f"{interval.left}-{interval.right-1}" for interval in counts_without_chart.index]
axes[0, 1].bar(range(len(counts_without_chart)), counts_without_chart.values, alpha=0.7, color='red')
axes[0, 1].set_title('Mit "Quelle Lyrics" aber OHNE "Chartentry"')
axes[0, 1].set_xlabel('10-Jahres Bins')
axes[0, 1].set_ylabel('Anzahl Zeilen')
axes[0, 1].set_xticks(range(len(counts_without_chart)))
axes[0, 1].set_xticklabels(bin_labels_without, rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: All with Quelle Lyrics
counts_all = df_with_lyrics['Jahr_Bin'].value_counts().sort_index()
bin_labels_all = [f"{interval.left}-{interval.right-1}" for interval in counts_all.index]
axes[1, 0].bar(range(len(counts_all)), counts_all.values, alpha=0.7, color='green')
axes[1, 0].set_title('Alle Zeilen mit "Quelle Lyrics"')
axes[1, 0].set_xlabel('10-Jahres Bins')
axes[1, 0].set_ylabel('Anzahl Zeilen')
axes[1, 0].set_xticks(range(len(counts_all)))
axes[1, 0].set_xticklabels(bin_labels_all, rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Comparison plot
all_bins = sorted(set(counts_with_chart.index) | set(counts_without_chart.index))
with_chart_values = [counts_with_chart.get(bin, 0) for bin in all_bins]
without_chart_values = [counts_without_chart.get(bin, 0) for bin in all_bins]
comparison_labels = [f"{interval.left}-{interval.right-1}" for interval in all_bins]

x = np.arange(len(all_bins))
width = 0.35

axes[1, 1].bar(x - width/2, with_chart_values, width, label='Mit Chartentry', alpha=0.7, color='blue')
axes[1, 1].bar(x + width/2, without_chart_values, width, label='Ohne Chartentry', alpha=0.7, color='red')
axes[1, 1].set_title('Vergleich: Mit vs. Ohne Chartentry')
axes[1, 1].set_xlabel('10-Jahres Bins')
axes[1, 1].set_ylabel('Anzahl Zeilen')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(comparison_labels, rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(vis_dir, 'quelle_lyrics_analysis_1925_2025.png'), dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {vis_dir}/ directory")

plt.show()

# Print detailed statistics
print("\n" + "="*50)
print("DETAILLIERTE STATISTIKEN (1925-2025)")
print("="*50)

print(f"\nGesamtanzahl Zeilen im Dataset: {len(df)}")
print(f"Zeilen mit 'Quelle Lyrics' (1925-2025): {len(df_with_lyrics)}")
print(f"Prozentsatz mit 'Quelle Lyrics' (1925-2025): {len(df_with_lyrics)/len(df)*100:.2f}%")

print(f"\nVon den Zeilen mit 'Quelle Lyrics':")
print(f"  - Mit Chartentry: {len(df_with_chartentry)} ({len(df_with_chartentry)/len(df_with_lyrics)*100:.2f}%)")
print(f"  - Ohne Chartentry: {len(df_without_chartentry)} ({len(df_without_chartentry)/len(df_with_lyrics)*100:.2f}%)")

print(f"\nVerteilung nach 10-Jahres Bins (alle mit 'Quelle Lyrics'):")
for bin_interval, count in counts_all.items():
    print(f"  {bin_interval.left}-{bin_interval.right-1}: {count} Zeilen")

print(f"\nVerteilung nach 10-Jahres Bins (mit 'Quelle Lyrics' UND 'Chartentry'):")
for bin_interval, count in counts_with_chart.items():
    print(f"  {bin_interval.left}-{bin_interval.right-1}: {count} Zeilen")

print(f"\nVerteilung nach 10-Jahres Bins (mit 'Quelle Lyrics' OHNE 'Chartentry'):")
for bin_interval, count in counts_without_chart.items():
    print(f"  {bin_interval.left}-{bin_interval.right-1}: {count} Zeilen")
