import pandas as pd
import os
from langdetect import detect, LangDetectException
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial

def detect_language_from_file(index, lyrics_folder):
    """Detect language from a lyrics file given its index."""
    file_path = os.path.join(lyrics_folder, f"{index}.txt")
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    return detect(text)
        return "unknown"
    except (LangDetectException, Exception):
        return "unknown"

def process_batch(indices, lyrics_folder):
    """Process a batch of indices."""
    return [detect_language_from_file(idx, lyrics_folder) for idx in indices]

def main():
    # File paths
    input_csv = "../../data/dataset_popmusic_v5.csv"
    output_csv = "../../data/dataset_popmusic_v6.csv"
    lyrics_folder = "../../data/dataset_popmusic_lyrics"
    
    # Read the CSV
    print("Reading CSV file...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")
    
    # Get indices
    indices = df['Index'].tolist()
    
    # Parallelize language detection
    print(f"\nDetecting languages using {cpu_count()} cores...")
    num_processes = cpu_count()
    
    # Create partial function with lyrics_folder
    detect_func = partial(detect_language_from_file, lyrics_folder=lyrics_folder)
    
    # Use multiprocessing with tqdm
    with Pool(processes=num_processes) as pool:
        languages = list(tqdm(
            pool.imap(detect_func, indices),
            total=len(indices),
            desc="Processing files"
        ))
    
    # Add language column to dataframe
    df['Sprache Lyrics'] = languages
    
    # Save to CSV with quoting=1
    print(f"\nSaving to {output_csv}...")
    df.to_csv(output_csv, index=False, quoting=1)
    print("Saved successfully!")
    
    # Display statistics
    print("\n" + "="*50)
    print("LANGUAGE STATISTICS")
    print("="*50)
    language_counts = df['Sprache Lyrics'].value_counts()
    print(f"\nTotal languages detected: {len(language_counts)}")
    print(f"\nLanguage distribution:")
    for lang, count in language_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {lang}: {count} ({percentage:.2f}%)")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
