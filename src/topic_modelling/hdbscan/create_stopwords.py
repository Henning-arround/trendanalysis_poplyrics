import json
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

def process_chunk(texts):
    """Helper function to process a chunk of texts in a separate process."""
    stop_words = set(stopwords.words('german'))
    chunk_words = []
    for text in texts:
        if not text:
            continue
        tokens = word_tokenize(text)
        filtered_tokens = [
            word.lower() for word in tokens 
            if word.isalpha() and word.lower() not in stop_words
        ]
        chunk_words.extend(filtered_tokens)
    return chunk_words

def get_top_words(json_path, top_n=100):
    # Load German stopwords
    all_words = []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Loaded {len(data)} entries from {json_path}")

        # Extract texts list for parallel processing
        texts = [entry.get("text", "") for entry in data]

        # Determine number of processes and chunk size
        num_processes = cpu_count()
        chunk_size = max(1, len(texts) // num_processes)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

        print(f"Processing with {num_processes} cores...")

        # Process chunks in parallel with progress bar
        with Pool(num_processes) as pool:
            results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Tokenizing & Filtering"))
        
        # Flatten results
        for res in results:
            all_words.extend(res)
            
        # Count frequencies
        word_counts = Counter(all_words)
        
        # Get top N words
        top_words = word_counts.most_common(top_n)
        
        return top_words

    except FileNotFoundError:
        print(f"File not found: {json_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    # Path relative to this script location
    # Script is in src/topic modelling/
    # Data is in data/ (two levels up from src)
    file_path = os.path.join(os.path.dirname(__file__), "../../../data/dataset_popmusic_v8.json")
    
    top_100 = get_top_words(file_path)
    
    # Extract just the words for the list
    stopwords_list = [word for word, count in top_100]
    
    # Define output path
    output_path = os.path.join(os.path.dirname(__file__), "../../../bertopic/dtm/stopwords/stopwords.txt")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write as a python list string so it can be copied directly into code
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(stopwords_list))

    print(f"Stopwords saved as list to {output_path}")
    
    print("Top 100 words (excluding German stopwords):")
    for word, count in top_100:
        print(f"{word}: {count}")
