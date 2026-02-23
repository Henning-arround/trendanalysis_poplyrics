import pandas as pd
import json
import os
from pathlib import Path
import numpy as np

def create_dnb_dataset():
    # Define paths
    csv_path = "../../data/dataset_popmusic_v8.csv"
    lyrics_folder = "../../data/dataset_popmusic_lyrics"
    output_path = "../../data/dataset_popmusic_DNB.json"
    
    # Get the directory of this script to resolve relative paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / csv_path
    lyrics_folder = script_dir / lyrics_folder
    output_path = script_dir / output_path
    
    # Read the CSV file
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convert DataFrame to list of dictionaries
    records = df.to_dict(orient='records')
    
    # Clean up values: convert NaN to empty string, floats to int where appropriate
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = ""
            elif isinstance(value, float) and value == int(value):
                record[key] = int(value)
    
    # Process each record and add lyrics
    print(f"Processing {len(records)} records...")
    lyrics_found = 0
    lyrics_not_found = 0
    
    for record in records:
        index_value = str(record.get('Index', ''))
        lyrics_file_path = lyrics_folder / f"{index_value}.txt"
        
        # Try to read the lyrics file
        if lyrics_file_path.exists():
            try:
                with open(lyrics_file_path, 'r', encoding='utf-8') as f:
                    record['Songtext'] = f.read()
                lyrics_found += 1
            except Exception as e:
                print(f"Error reading {lyrics_file_path}: {e}")
                record['Songtext'] = ""
                lyrics_not_found += 1
        else:
            record['Songtext'] = ""
            lyrics_not_found += 1
    
    print(f"Lyrics found: {lyrics_found}")
    print(f"Lyrics not found: {lyrics_not_found}")
    
    # Write to JSON file
    print(f"Writing JSON to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print("Done!")
    return records

if __name__ == "__main__":
    create_dnb_dataset()
