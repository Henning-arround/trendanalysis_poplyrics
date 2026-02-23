import pandas as pd
import re
from fuzzywuzzy import fuzz
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def remove_parentheses(text):
    """Remove text in parentheses from a string."""
    return re.sub(r'\([^)]*\)', '', text).strip()

def process_artist_group(args):
    """Process a single artist group to find duplicates."""
    artist, group, title_col, threshold = args
    indices = group.index.tolist()
    kept_indices = set()
    skip_indices = set()
    
    for i, idx1 in enumerate(indices):
        if idx1 in skip_indices:
            continue
            
        # Get title without parentheses
        title1 = remove_parentheses(str(group.loc[idx1, title_col]))
        duplicate_group = [idx1]
        
        # Compare with remaining titles
        for idx2 in indices[i+1:]:
            if idx2 in skip_indices:
                continue
                
            title2 = remove_parentheses(str(group.loc[idx2, title_col]))
            
            # Fuzzy match
            similarity = fuzz.ratio(title1.lower(), title2.lower())
            
            if similarity >= threshold:
                # Found a duplicate
                duplicate_group.append(idx2)
                skip_indices.add(idx2)
        
        # Keep the one with the shortest title (original with parentheses)
        shortest_idx = min(duplicate_group, key=lambda idx: len(str(group.loc[idx, title_col])))
        kept_indices.add(shortest_idx)
    
    return list(kept_indices)

def find_duplicates(df, artist_col='Künstler', title_col='Titel', threshold=90):
    """Find and remove duplicate songs using fuzzy matching."""
    # Prepare arguments for parallel processing
    groups = [(artist, group, title_col, threshold) 
              for artist, group in df.groupby(artist_col)]
    
    num_workers = cpu_count()
    chunksize = 1
    
    print(f"Using {num_workers} workers with chunksize {chunksize}")
    
    # Process in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_artist_group, groups, chunksize=chunksize),
            total=len(groups),
            desc="Processing artists"
        ))
    
    # Flatten results
    rows_to_keep = [idx for result in results for idx in result]
    
    return df.loc[sorted(rows_to_keep)]

# Read the CSV file
input_path = Path(__file__).parent.parent.parent / 'data' / 'charts_chartsurfer_1954_1977_v2.csv'
df = pd.read_csv(input_path)

print(f"Original number of rows: {len(df)}")

# Remove duplicates
df_cleaned = find_duplicates(df, threshold=90)

print(f"Number of rows after removing duplicates: {len(df_cleaned)}")
print(f"Removed {len(df) - len(df_cleaned)} duplicate entries")

# Save the cleaned data
output_path = Path(__file__).parent.parent.parent / 'data' / 'charts_chartsurfer_1954_1977_v2_without_duplicates.csv'
df_cleaned.to_csv(output_path, index=False)

print(f"Saved cleaned data to: {output_path}")
