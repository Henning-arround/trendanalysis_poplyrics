import pandas as pd
import shutil
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)

# Paths
csv_path = Path("../../data/dataset_popmusic_v8.csv")
source_dir = Path("../../data/dataset_popmusic_lyrics")
target_dir = Path("../../llm_as_a_judge/samples")

# Create target directory if it doesn't exist
target_dir.mkdir(parents=True, exist_ok=True)

# Read CSV
df = pd.read_csv(csv_path)

# Filter for rows where lyrics files exist (keep Index as string to include chartsurfer)
df_filtered = df[['Index', 'Sprache Lyrics', 'Jahr']].dropna()
df_filtered['Index'] = df_filtered['Index'].astype(str)

# Calculate language distribution
total_count = len(df_filtered)
lang_counts = df_filtered['Sprache Lyrics'].value_counts()
print(f"Total files: {total_count}")
print(f"Language distribution:\n{lang_counts}")

# Calculate year distribution
year_counts = df_filtered['Jahr'].value_counts()
print(f"\nYear distribution:\n{year_counts.sort_index()}")

# Required indices (both "de")
required_indices = ['724', '73012']

# Separate by language and remove required indices from available pool
df_de = df_filtered[df_filtered['Sprache Lyrics'] == 'de']
df_en = df_filtered[df_filtered['Sprache Lyrics'] == 'en']

df_de_available = df_de[~df_de['Index'].isin(required_indices)]
df_en_available = df_en.copy()

# Calculate target counts (18 remaining after 2 required)
remaining_count = 98  # 100 total - 2 required
de_proportion = len(df_de) / total_count
en_proportion = len(df_en) / total_count

de_target = round(remaining_count * de_proportion) - len(required_indices)
en_target = remaining_count - de_target

print(f"\nTarget sample size: 100")
print(f"Required DE indices: {required_indices}")
print(f"Additional DE needed: {de_target}")
print(f"EN needed: {en_target}")

# Stratified sampling by year within each language
def stratified_sample_by_year(df, n, random_state):
    if n <= 0:
        return pd.DataFrame()
    if len(df) <= n:
        return df
    
    # Calculate year proportions
    year_proportions = df['Jahr'].value_counts(normalize=True)
    samples = []
    
    for year, proportion in year_proportions.items():
        year_df = df[df['Jahr'] == year]
        n_year = max(1, round(n * proportion))  # At least 1 from each year if possible
        
        if len(year_df) <= n_year:
            samples.append(year_df)
        else:
            samples.append(year_df.sample(n=n_year, random_state=random_state))
    
    result = pd.concat(samples)
    
    # Adjust if we have too many or too few
    if len(result) > n:
        result = result.sample(n=n, random_state=random_state)
    elif len(result) < n:
        remaining_df = df[~df['Index'].isin(result['Index'])]
        if len(remaining_df) > 0:
            additional = remaining_df.sample(n=min(n - len(result), len(remaining_df)), random_state=random_state)
            result = pd.concat([result, additional])
    
    return result

# Sample with year stratification
sampled_de = stratified_sample_by_year(df_de_available, de_target, random_state=42)
sampled_en = stratified_sample_by_year(df_en_available, en_target, random_state=42)

# Combine all selected indices
selected_indices = required_indices + sampled_de['Index'].tolist() + sampled_en['Index'].tolist()

print(f"\nSelected {len(selected_indices)} files:")
# Sort with special handling for mixed types
def sort_key(x):
    try:
        return (0, int(x))
    except:
        return (1, x)

print(sorted(selected_indices, key=sort_key))

# Copy files
copied_count = 0
missing_files = []

for idx in selected_indices:
    source_file = source_dir / f"{idx}.txt"
    target_file = target_dir / f"{idx}.txt"
    
    if source_file.exists():
        shutil.copy2(source_file, target_file)
        copied_count += 1
    else:
        missing_files.append(idx)

print(f"\nSuccessfully copied {copied_count} files to {target_dir}")
if missing_files:
    print(f"Warning: {len(missing_files)} files not found: {missing_files}")

# Save list of selected indices
selected_df = df_filtered[df_filtered['Index'].isin(selected_indices)].copy()
selected_df = selected_df.sort_values('Index', key=lambda x: x.map(sort_key))
output_csv = target_dir / "sample_metadata.csv"
selected_df.to_csv(output_csv, index=False)
print(f"\nSample metadata saved to {output_csv}")

# Print sample statistics
print("\nSample statistics:")
print(f"Language distribution in sample:\n{selected_df['Sprache Lyrics'].value_counts()}")
print(f"\nYear distribution in sample:\n{selected_df['Jahr'].value_counts().sort_index()}")
