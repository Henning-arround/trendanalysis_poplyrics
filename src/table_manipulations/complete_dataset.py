import pandas as pd
from pathlib import Path
import re
import os

# Define file paths
data_dir = Path(__file__).parent.parent.parent / "data"
first_half_file = data_dir / "charts_offizielle_charts_minimal_first_half.csv"
second_half_file = data_dir / "charts_offizielle_charts_minimal.csv"
language_file = data_dir / "charts_with_language_updated_v5.csv"
chartsurfer_file = data_dir / "charts_chartsurfer_1954_1977_v5.csv"
lyrics_chartsurfer_dir = data_dir / "lyrics_chartsurfer"
output_file = data_dir / "dataset_popmusic.csv"

# Read both CSV files
df_first = pd.read_csv(first_half_file)
df_second = pd.read_csv(second_half_file)

# Concatenate the dataframes
df_complete = pd.concat([df_first, df_second], ignore_index=True)

# Read the language file with chart information
df_language = pd.read_csv(language_file)

# Extract date from "Chartentry" column (remove brackets and content after)
df_language['Erste Notierung'] = df_language['Chartentry'].apply(
    lambda x: re.sub(r'\s*\(.*\)', '', str(x)).strip() if pd.notna(x) else None
)

# Extract date from "Letzte Chartposition" column
df_language['Letzte Notierung'] = df_language['Letzte Chartposition'].apply(
    lambda x: re.sub(r'\s*\(.*\)', '', str(x)).strip() if pd.notna(x) else None
)

# Extract highest position (lowest number) from "Chartverlauf"
def get_highest_position(chartverlauf):
    if pd.isna(chartverlauf):
        return None
    positions = str(chartverlauf).split(',')
    numbers = []
    for pos in positions:
        pos = pos.strip()
        if pos.lower() != 'null' and pos.isdigit():
            numbers.append(int(pos))
    return min(numbers) if numbers else None

df_language['Höchstposition'] = df_language['Chartverlauf'].apply(get_highest_position)

# Count number of weeks at position 1
def count_number_one_weeks(chartverlauf):
    if pd.isna(chartverlauf):
        return None
    positions = str(chartverlauf).split(',')
    count = sum(1 for pos in positions if pos.strip() == '1')
    return count if count > 0 else None

df_language['Nr. 1 Wochen'] = df_language['Chartverlauf'].apply(count_number_one_weeks)

# Count Top-10 weeks (positions 1-10)
def count_top10_weeks(chartverlauf):
    if pd.isna(chartverlauf):
        return None
    positions = str(chartverlauf).split(',')
    count = 0
    for pos in positions:
        pos = pos.strip()
        if pos.isdigit() and 1 <= int(pos) <= 10:
            count += 1
    return count if count > 0 else None

df_language['Top-10 Wochen'] = df_language['Chartverlauf'].apply(count_top10_weeks)

# Rename "Anzahl Wochen" to "Wochen Gesamt"
df_language['Wochen Gesamt'] = df_language['Anzahl Wochen']

# Merge the enriched columns into df_complete based on index
columns_to_add = ['Index', 'Erste Notierung', 'Letzte Notierung', 'Höchstposition', 
                  'Nr. 1 Wochen', 'Top-10 Wochen', 'Wochen Gesamt']
df_complete = df_complete.merge(
    df_language[columns_to_add], 
    on='Index', 
    how='left'
)

# Read chartsurfer data
df_chartsurfer = pd.read_csv(chartsurfer_file)

# Filter to only include rows where "Quelle Genius" has a value
if 'Quelle Genius' in df_chartsurfer.columns:
    df_chartsurfer = df_chartsurfer[df_chartsurfer['Quelle Genius'].notna() & (df_chartsurfer['Quelle Genius'] != '')]

# Add suffix to Index column in chartsurfer data
df_chartsurfer['Index'] = df_chartsurfer['Index'].astype(str) + '_chartsurfer'

# Rename lyrics files in lyrics_chartsurfer directory
if lyrics_chartsurfer_dir.exists():
    for file in lyrics_chartsurfer_dir.glob('*.txt'):
        old_name = file.stem  # filename without extension
        
        # Remove double suffix if it exists
        if old_name.endswith('_chartsurfer_chartsurfer'):
            new_name = old_name.replace('_chartsurfer_chartsurfer', '_chartsurfer') + '.txt'
            new_path = lyrics_chartsurfer_dir / new_name
            file.rename(new_path)
            print(f"Fixed double suffix: {file.name} -> {new_name}")
        # Add suffix only if it doesn't exist
        elif not old_name.endswith('_chartsurfer'):
            new_name = f"{old_name}_chartsurfer.txt"
            new_path = lyrics_chartsurfer_dir / new_name
            file.rename(new_path)
            print(f"Renamed: {file.name} -> {new_name}")

# Prepare chartsurfer dataframe with same columns as df_complete
df_chartsurfer_processed = pd.DataFrame()

# Get column names from df_complete
existing_columns = df_complete.columns.tolist()

# Map chartsurfer columns to dataset_popmusic columns
for col in existing_columns:
    if col in df_chartsurfer.columns:
        df_chartsurfer_processed[col] = df_chartsurfer[col]
    elif col == 'Quelle':
        if 'Quelle Genius' in df_chartsurfer.columns:
            df_chartsurfer_processed[col] = df_chartsurfer['Quelle Genius']
        else:
            df_chartsurfer_processed[col] = None
    else:
        df_chartsurfer_processed[col] = None

# Add additional columns from chartsurfer
if 'MusicBrainz Release Date' in df_chartsurfer.columns:
    df_chartsurfer_processed['Release Date external'] = df_chartsurfer['MusicBrainz Release Date']
    
if 'Release Date Genius' in df_chartsurfer.columns:
    df_chartsurfer_processed['Release Date Genius'] = df_chartsurfer['Release Date Genius']

# Concatenate df_complete with chartsurfer data
df_complete = pd.concat([df_complete, df_chartsurfer_processed], ignore_index=True)

# Save to new CSV file
df_complete.to_csv(output_file, index=False, quoting=1)

print(f"Combined dataset created with {len(df_complete)} rows")
print(f"Saved to: {output_file}")
