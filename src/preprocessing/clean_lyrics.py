import os
import re
from pathlib import Path
from tqdm import tqdm

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

def process_lyrics_files(input_dir, output_dir):
    """Process all lyrics files and save cleaned versions."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    
    # Get list of files for progress bar
    files = list(input_path.glob("*.txt"))
    
    for file_path in tqdm(files, desc="Processing lyrics"):
        try:
            # Read original file with utf-8, ignoring errors
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Clean the lyrics
            cleaned_content = clean_lyrics(content)
            
            # Count tokens (split by whitespace)
            tokens = cleaned_content.split()
            
            # Only save if at least 5 tokens
            if len(tokens) >= 5:
                # Write cleaned file to output directory (always in UTF-8)
                output_file = output_path / file_path.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                processed_count += 1
            else:
                skipped_count += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return processed_count, skipped_count

def main():
    input_dir = "../../data/lyrics_german/"
    output_dir = "../../data/lyrics_german_cleaned/"
    
    print("Cleaning lyrics files...")
    processed_count, skipped_count = process_lyrics_files(input_dir, output_dir)
    print(f"Successfully processed {processed_count} files")
    print(f"Skipped {skipped_count} files (less than 5 tokens)")
    print(f"Cleaned files saved to: {output_dir}")

if __name__ == "__main__":
    main()
