import pandas as pd
import shutil
import re
import csv
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def clean_lyrics(text):
    """Clean lyrics by removing unwanted patterns."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Handle first line if it contains "X Contributor(s)" pattern
        if i == 0 and re.search(r'\d+\s+Contributors?', line, re.IGNORECASE):
            # Find "lyrics" (case insensitive) and keep everything after it
            lyrics_match = re.search(r'lyrics', line, re.IGNORECASE)
            if lyrics_match:
                line = line[lyrics_match.end():].strip()
                if not line:  # If nothing left after "lyrics", skip this line
                    continue
            else:
                continue  # If no "lyrics" found, skip entire line
            
        # Skip lines that contain only square brackets with content
        if re.match(r'^\[.*\]$', line):
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def copy_and_clean_single_file(index, source_paths, output_path):
    """Copy and clean a single lyrics file from source directories to output."""
    filename = f"{index}.txt"
    
    for source_path in source_paths:
        source_file = source_path / filename
        if source_file.exists():
            try:
                # Read original file
                with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Clean the lyrics
                cleaned_content = clean_lyrics(content)
                
                # Count tokens (split by whitespace)
                tokens = cleaned_content.split()
                
                # Only save if at least 5 tokens
                if len(tokens) >= 5:
                    destination_file = output_path / filename
                    with open(destination_file, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    return index, True, len(tokens)
                else:
                    return index, False, len(tokens)
                    
            except Exception as e:
                print(f"Error processing {index}: {e}")
                return index, False, 0
    
    return index, False, 0

def copy_dataset_lyrics(csv_path, source_dirs, output_dir, max_workers=8):
    """Copy and clean lyrics files that match the Index column in the CSV."""
    # Read CSV and get Index column
    df = pd.read_csv(csv_path)
    indices = df['Index'].tolist()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert source directories to Path objects
    source_paths = [Path(dir) for dir in source_dirs]
    
    copied_count = 0
    not_found = []
    too_short = []
    valid_indices = []
    
    # Use ThreadPoolExecutor for parallel copying
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(copy_and_clean_single_file, index, source_paths, output_path): index
            for index in indices
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Copying and cleaning lyrics"):
            index, found, token_count = future.result()
            if found:
                copied_count += 1
                valid_indices.append(index)
            elif token_count > 0:
                too_short.append(index)
            else:
                not_found.append(index)
    
    return copied_count, not_found, too_short, valid_indices

def main():
    csv_path = "../../data/dataset_popmusic_v4.csv"
    source_dirs = [
        "../../data/lyrics_chartsurfer",
        "../../data/lyrics_offiziellecharts_revisited"
    ]
    output_dir = "../../data/dataset_popmusic_lyrics"
    output_csv = "../../data/dataset_popmusic_v5.csv"
    
    print("Copying and cleaning lyrics files for dataset...")
    copied_count, not_found, too_short, valid_indices = copy_dataset_lyrics(
        csv_path, source_dirs, output_dir, max_workers=8
    )
    
    print(f"\nSuccessfully copied and cleaned {copied_count} files")
    print(f"Files not found: {len(not_found)}")
    print(f"Files too short (< 5 tokens): {len(too_short)}")
    
    # Read original CSV and filter to keep only valid indices
    print("\nUpdating CSV file...")
    df = pd.read_csv(csv_path)
    df_filtered = df[df['Index'].isin(valid_indices)]
    
    # Save filtered CSV with quoting=1 (QUOTE_ALL)
    df_filtered.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"Original CSV entries: {len(df)}")
    print(f"Filtered CSV entries: {len(df_filtered)}")
    print(f"Removed entries: {len(df) - len(df_filtered)}")
    print(f"\nNew CSV saved to: {output_csv}")
    print(f"Cleaned lyrics saved to: {output_dir}")

if __name__ == "__main__":
    main()
