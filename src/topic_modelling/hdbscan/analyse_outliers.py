import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import torch

# Download German stopwords
nltk.download('stopwords', quiet=True)
german_stopwords = stopwords.words('german')

# Define custom stopwords (same as in create_dtm.py)
analysis_stopwords = [
    # Substantive
    'song', 'lied', 'text', 'titel', 'refrain', 'strophe', 'vers', 'zeile', 'zeilen',
    'metapher', 'symbol', 'motiv', 'bild', 'wiederholung', 'sprecher', 'lyrische',
    'gegenüberstellung', 'spannungsfeld', 'person', 'interpretation', 'musik', 'klang',
    'mantra', 'frage', 'antwort', 'hintergrund', 'vordergrund', 'atmosphäre',
    # Verben
    'betont', 'fungiert', 'wirkt', 'verdeutlicht', 'suggeriert', 'verstärkt',
    'offenbart', 'symbolisiert', 'verdeutlichen', 'verknüpft', 'schildert',
    'zeichnet', 'dient', 'stellt', 'beschreibt', 'handelt', 'geht', 'zeigt',
    'erzählt', 'entsteht', 'lässt', 'macht',
    # Adjektive
    'wiederholte', 'wiederkehrende', 'ständige', 'eigene', 'eigenen', 'innere',
    'inneren', 'innerer', 'emotionale', 'musikalische',
    # Füllwörter
    'dabei', 'jedoch', 'zugleich', 'fast', 'gleichzeitig', 'insgesamt', 
    'immer', 'oft', 'nie', 'wohl', 'bereits', 'sowohl', 'trotz', 'gegenüber',
    # Englische Lyrics-Stopwords
    'the', 'and', 'you', 'i', 'to', 'a', 'of', 'it', 'my', 'me', 'in', 'on', 
    'is', 'that', 'your', 'we', 'all', 'be', 'for', 'don', 't', 's'
]

final_stopwords = german_stopwords + analysis_stopwords

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

device = get_device()
print(f"Using device: {device}")

# Configure paths
data_path = "../../../data/dataset_popmusic_v8.json"
embeddings_path = "../../../bertopic/dtm/embedding/embeddings.npy"
output_path = "../../../bertopic/dtm/outlier_analysis"
model_path = "../../../bertopic/dtm/models/bertopic_model_dtm"
probs_path = "../../../bertopic/dtm/models/probabilities.npy"

os.makedirs(output_path, exist_ok=True)

# Load data
print(f"Loading data from {data_path}...")
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

documents = []
timestamps = []

for entry in data:
    text = entry.get('text', '').strip()
    timestamp = entry.get('timestamp', '')
    
    if text and timestamp:
        year = int(timestamp[:4])
        
        # Filter by year range (1955-2024 inclusive) to match create_dtm.py and embeddings
        if 1955 <= year <= 2024:
            documents.append(text)
            timestamps.append(year)

print(f"Loaded {len(documents)} documents")
print(f"Year range: {min(timestamps)} - {max(timestamps)}")

# Load embeddings
print(f"Loading embeddings from {embeddings_path}...")
embeddings = np.load(embeddings_path)
print(f"Embeddings shape: {embeddings.shape}")

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

n_topics_before = len(set(topics)) - (1 if -1 in topics else 0)
outlier_mask_before = [t == -1 for t in topics]
n_outliers_before = sum(outlier_mask_before)

print(f"Number of topics (HDBSCAN clusters): {n_topics_before}")
print(f"Total outliers: {n_outliers_before} ({n_outliers_before/len(documents):.2%})")

# Analyze outliers by 5-year intervals
# Create bins: 1955-1959, 1960-1964, etc.
min_year = min(timestamps)
max_year = max(timestamps)
# Round down to nearest 5 for start
start_year = (min_year // 5) * 5
# Round up to nearest 5 for end
end_year = ((max_year // 5) + 1) * 5

time_bins = []
bin_labels = []
for y in range(start_year, end_year, 5):
    time_bins.append((y, y+4))
    bin_labels.append(f"{y}-{y+4}")

# Helper to find bin index for a year
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

ax1.bar(bin_labels, non_outlier_counts_before, label='Assigned to Topic', color='steelblue', alpha=0.8)
ax1.bar(bin_labels, outlier_counts_before, bottom=non_outlier_counts_before, label='Outliers', color='coral', alpha=0.8)
ax1.set_xlabel('Time Period')
ax1.set_ylabel('Number of Documents')
ax1.set_title('Documents by Period - BEFORE Outlier Reduction')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Percentage of outliers by bin
ax2 = axes[0, 1]
outlier_pct_before = [outliers_by_bin_before[l] / total_by_bin[l] * 100 if total_by_bin[l] > 0 else 0 for l in bin_labels]
ax2.bar(bin_labels, outlier_pct_before, color='coral', alpha=0.8)
ax2.axhline(y=n_outliers_before/len(documents)*100, color='red', linestyle='--', label=f'Overall: {n_outliers_before/len(documents)*100:.1f}%')
ax2.set_xlabel('Time Period')
ax2.set_ylabel('Outlier Percentage (%)')
ax2.set_title('Outlier Percentage by Period - BEFORE Reduction')
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
# We don't update the model here to avoid modifying the loaded model state, we just analyze the result
# topic_model.update_topics(documents, topics=new_topics, vectorizer_model=vectorizer_model)

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

ax3.bar(bin_labels, non_outlier_counts_after, label='Assigned to Topic', color='steelblue', alpha=0.8)
ax3.bar(bin_labels, outlier_counts_after, bottom=non_outlier_counts_after, label='Outliers', color='coral', alpha=0.8)
ax3.set_xlabel('Time Period')
ax3.set_ylabel('Number of Documents')
ax3.set_title('Documents by Period - AFTER Outlier Reduction')
ax3.legend()
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Percentage of outliers by bin (after)
ax4 = axes[1, 1]
outlier_pct_after = [outliers_by_bin_after[l] / total_by_bin[l] * 100 if total_by_bin[l] > 0 else 0 for l in bin_labels]
ax4.bar(bin_labels, outlier_pct_after, color='coral', alpha=0.8)
ax4.axhline(y=n_outliers_after/len(documents)*100, color='red', linestyle='--', label=f'Overall: {n_outliers_after/len(documents)*100:.1f}%')
ax4.set_xlabel('Time Period')
ax4.set_ylabel('Outlier Percentage (%)')
ax4.set_title('Outlier Percentage by Period - AFTER Reduction')
ax4.legend()
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'outlier_analysis_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"\nVisualization saved to {os.path.join(output_path, 'outlier_analysis_comparison.png')}")

# === Additional comparison visualization ===
fig2, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(bin_labels))
width = 0.35

bars1 = ax.bar(x - width/2, outlier_pct_before, width, label='Before Reduction', color='coral', alpha=0.8)
bars2 = ax.bar(x + width/2, outlier_pct_after, width, label='After Reduction', color='seagreen', alpha=0.8)

ax.set_xlabel('Time Period')
ax.set_ylabel('Outlier Percentage (%)')
ax.set_title('Outlier Percentage Comparison: Before vs After Reduction')
ax.set_xticks(x)
ax.set_xticklabels(bin_labels)
ax.legend()
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'outlier_comparison_by_year.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"Comparison saved to {os.path.join(output_path, 'outlier_comparison_by_year.png')}")

# === Probability Distribution Visualization ===
print("\n" + "="*60)
print("PROBABILITY DISTRIBUTION ANALYSIS")
print("="*60)

# Ensure probs is a numpy array
probs = np.array(probs)

# Calculate max probability for each document
# If probs is 2D (n_samples, n_topics), take max along axis 1 (confidence of top topic)
# If probs is 1D (n_samples,), use as is
if probs.ndim > 1:
    max_probs = np.max(probs, axis=1)
else:
    max_probs = probs

fig3, ax_prob = plt.subplots(figsize=(14, 7))

# Create finer bins (0.01 steps) to see details around low probabilities
bins = np.arange(0, 1.01, 0.01)

# Plot histogram
counts, bins, patches = ax_prob.hist(max_probs, bins=bins, color='mediumpurple', edgecolor='black', alpha=0.7)

# Add vertical line for threshold
ax_prob.axvline(x=OUTLIER_THRESHOLD, color='red', linestyle='-', linewidth=2, label=f'Threshold ({OUTLIER_THRESHOLD})')

# Highlight the outlier region
ax_prob.axvspan(0, OUTLIER_THRESHOLD, alpha=0.2, color='red', label='Outlier Region')

# Add labels on top of bars - only for bars with counts to avoid clutter, and rotate them
for count, rect in zip(counts, patches):
    if count > 0:
        height = rect.get_height()
        # Only label if it's near the threshold or significant enough to matter visually
        if rect.get_x() <= 0.1 or height > max(counts) * 0.02:
            ax_prob.text(rect.get_x() + rect.get_width()/2., height + 5,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=7, rotation=90)

ax_prob.set_xlabel('Probability of Top Topic Assignment')
ax_prob.set_ylabel('Number of Documents')
ax_prob.set_title(f'Distribution of Top Topic Assignment Probabilities (Threshold: {OUTLIER_THRESHOLD})')

# Set x-ticks to show 0.03 clearly. 
# We use major ticks every 0.05, but ensure 0.03 is marked specifically
ticks = np.arange(0, 1.05, 0.05)
# Add the threshold to ticks if not present close enough
if not any(np.isclose(ticks, OUTLIER_THRESHOLD)):
    ticks = np.sort(np.append(ticks, OUTLIER_THRESHOLD))

ax_prob.set_xticks(ticks)
# Format tick labels, highlighting the threshold
tick_labels = []
for t in ticks:
    if np.isclose(t, OUTLIER_THRESHOLD):
        tick_labels.append(f'{t:.2f} (Thresh)')
    else:
        tick_labels.append(f'{t:.2f}')

ax_prob.set_xticklabels(tick_labels, rotation=45, fontsize=9)

# Color the tick label for threshold red
for label in ax_prob.get_xticklabels():
    if "Thresh" in label.get_text():
        label.set_color('red')
        label.set_fontweight('bold')

ax_prob.grid(axis='y', alpha=0.3, linestyle='--')
ax_prob.set_xlim(0, 1.0)
ax_prob.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'assignment_probability_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"Probability distribution saved to {os.path.join(output_path, 'assignment_probability_distribution.png')}")

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
