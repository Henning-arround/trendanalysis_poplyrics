import pandas as pd
import os
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from multiprocessing import Pool, cpu_count
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_language_from_file(lyrics_path):
    """
    Detect language from lyrics file.
    Returns language code or None if detection fails.
    """
    try:
        if not os.path.exists(lyrics_path):
            logger.warning(f"Lyrics file not found: {lyrics_path}")
            return None
            
        with open(lyrics_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            
        if not content:
            logger.warning(f"Empty lyrics file: {lyrics_path}")
            return None
        
        # Check if lyrics have more than 5 tokens
        tokens = content.split()
        if len(tokens) <= 5:
            logger.info(f"Lyrics too short ({len(tokens)} tokens): {lyrics_path}")
            return None
            
        # Detect language
        detected_lang = detect(content)
        return detected_lang
        
    except LangDetectException as e:
        logger.warning(f"Language detection failed for {lyrics_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing {lyrics_path}: {e}")
        return None

def process_row(args):
    """
    Process a single row for language detection.
    """
    index_value, quelle_lyrics = args
    
    if pd.isna(quelle_lyrics) or quelle_lyrics == '':
        return index_value, None
        
    lyrics_path = f"../data/lyrics/{index_value}.txt"
    detected_lang = detect_language_from_file(lyrics_path)
    
    return index_value, detected_lang

def main():
    # Read CSV file
    charts_file = "../data/charts_with_language_updated_v3.csv"
    
    logger.info("Reading CSV file...")
    df = pd.read_csv(charts_file, sep=',', quotechar='"', escapechar='\\')
    
    logger.info(f"Total rows in CSV: {len(df)}")
    
    # Filter rows with entries in "Quelle Lyrics"
    filtered_df = df[df['Quelle Lyrics'].notna() & (df['Quelle Lyrics'] != '')]
    logger.info(f"Rows with lyrics source: {len(filtered_df)}")
    
    # Prepare data for parallel processing
    process_data = [(row['Index'], row['Quelle Lyrics']) for _, row in filtered_df.iterrows()]
    
    # Parallel processing
    logger.info("Starting parallel language detection...")
    num_processes = min(cpu_count(), len(process_data))
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_row, process_data),
            total=len(process_data),
            desc="Processing lyrics"
        ))
    
    # Create language detection mapping
    language_map = {index_val: lang for index_val, lang in results}
    
    # Create new dataframe with updated column
    df_updated = df.copy()
    
    # Update existing "Spracherkennung Lyrics" column
    df_updated['Spracherkennung Lyrics'] = df_updated['Index'].map(language_map)
    
    # Save updated CSV
    output_file = "../data/charts_with_language_updated_v4.csv"
    df_updated.to_csv(output_file, sep=',', quotechar='"', escapechar='\\', index=False, quoting=1)
    
    logger.info(f"Updated CSV saved to: {output_file}")
    
    # Print statistics
    detected_languages = df_updated['Spracherkennung Lyrics'].value_counts()
    logger.info("Language detection statistics:")
    for lang, count in detected_languages.items():
        logger.info(f"  {lang}: {count}")
    
    logger.info(f"Total processed: {len(df_updated)}")
    logger.info(f"Successfully detected: {df_updated['Spracherkennung Lyrics'].notna().sum()}")

if __name__ == "__main__":
    main()
