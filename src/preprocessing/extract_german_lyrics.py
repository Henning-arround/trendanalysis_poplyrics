import pandas as pd
import os
import shutil
from pathlib import Path

def prepare_german_lyrics():
    # Read the CSV file
    csv_path = "../../data/charts_with_language_updated_v4.csv"
    df = pd.read_csv(csv_path)
    
    # Filter rows where "Spracherkennung Lyrics" is "de"
    german_lyrics = df[df["Spracherkennung Lyrics"] == "de"]
    
    # Get the indices
    german_indices = german_lyrics["Index"].tolist()
    
    # Source and target directories
    source_dir = Path("../../data/lyrics")
    target_dir = Path("../../data/lyrics_german")
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files with matching indices
    copied_count = 0
    for index in german_indices:
        source_file = source_dir / f"{index}.txt"
        target_file = target_dir / f"{index}.txt"
        
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            copied_count += 1
        else:
            print(f"Warning: File {source_file} not found")
    
    print(f"Copied {copied_count} German lyrics files to {target_dir}")
    print(f"Total German entries found: {len(german_indices)}")

if __name__ == "__main__":
    prepare_german_lyrics()
