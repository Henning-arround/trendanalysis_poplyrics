import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from requests.exceptions import RequestException, ConnectionError, Timeout
from langchain_openai import ChatOpenAI

# API-Konfiguration
my_api_key = "sk-JbSIfkSGPUcXx7OONQMZyw"
client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=my_api_key)
model_name = "openai/gpt-oss-120b"

# Pfade definieren
lyrics_dir = "../../data/dataset_popmusic_lyrics"
output_dir = "../../data/dataset_popmusic_interpretations_revisited"
csv_path = "../../data/dataset_popmusic_v8.csv"
log_file = "../../data/interpretation_progress.json"

# Output-Verzeichnis erstellen
os.makedirs(output_dir, exist_ok=True)

def create_prompt(titel, künstler, jahr, songtext):
    """Erstelle den Interpretations-Prompt"""
    prompt = f"""Analysiere und interpretiere den Song {titel} von {künstler} aus dem Jahr {jahr}. 
Verfasse deine Interpretation als zusammenhängenden Fließtext mit maximal 300 Wörtern. Verwende keine Aufzählungen, Stichpunkte oder Zwischenüberschriften. Gib ausschließlich die Interpretation aus, ohne einleitende oder abschließende Kommentare.
Songtext:
{songtext}"""
    return prompt

def load_progress():
    """Lade bereits verarbeitete Songs aus Log-Datei"""
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_progress(processed_songs):
    """Speichere Fortschritt in Log-Datei"""
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(list(processed_songs), f, indent=2)

def process_song(row, processed_songs):
    """Verarbeitet einen einzelnen Songtext"""
    index = row.name
    filename = f"{index}.txt"
    output_filename = f"{index}_interpretation.txt"
    lyrics_path = os.path.join(lyrics_dir, filename)
    
    # Prüfen ob bereits verarbeitet
    if output_filename in processed_songs:
        return output_filename, True
    
    # Prüfen ob Songtext existiert
    if not os.path.exists(lyrics_path):
        print(f"\nSongtext nicht gefunden: {filename}")
        return filename, False
    
    # Metadaten extrahieren
    titel = row['Titel']
    künstler = row['Künstler']
    jahr = row['Jahr']
    
    # Songtext laden
    with open(lyrics_path, 'r', encoding='utf-8') as f:
        songtext = f.read()
    
    # Prompt erstellen
    prompt = create_prompt(titel, künstler, jahr, songtext)
    
    # Retry-Logik für API-Anfrage
    max_retries = 8
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                temperature=0
            )
            
            interpretation = response.choices[0].message.content
            break
            
        except (ConnectionError, Timeout, RequestException) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"\nNetzwerkfehler bei {filename} (Versuch {attempt + 1}/{max_retries}). Warte {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\nFehler bei {filename} nach {max_retries} Versuchen: {e}")
                raise
        except Exception as e:
            print(f"\nUnerwarteter Fehler bei {filename}: {e}")
            raise
    
    # Interpretation speichern
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(interpretation)
    
    return output_filename, True

def get_llm_model():
    # Placeholder: Replace with your actual API key or ensure it's in env variables
    api_key = os.getenv("OPENAI_API_KEY", "sk-YOUR_API_KEY_HERE")
    
    # Initialize the LLM (e.g., GPT-3.5 or GPT-4)
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        openai_api_key=api_key
    )
    return llm

def main():
    # Fortschritt laden
    processed_songs = load_progress()
    print(f"Bereits verarbeitet: {len(processed_songs)} Songs")
    
    # CSV-Datei laden
    df = pd.read_csv(csv_path)
    df = df.set_index('Index')
    
    # Liste aller verfügbaren Songtexte
    available_lyrics = set(f for f in os.listdir(lyrics_dir) if f.endswith(".txt"))
    
    # Songs filtern, die noch nicht verarbeitet wurden
    songs_to_process = [(idx, row) for idx, row in df.iterrows() 
                        if f"{idx}.txt" in available_lyrics and f"{idx}_interpretation.txt" not in processed_songs]
    
    print(f"Zu verarbeiten: {len(songs_to_process)} Songs")
    
    if not songs_to_process:
        print("Alle Songs wurden bereits verarbeitet!")
        return
    
    # Parallel verarbeiten
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_song, row, processed_songs): (idx, row) 
                   for idx, row in songs_to_process}
        
        for future in tqdm(as_completed(futures), total=len(songs_to_process), 
                          desc="Interpretiere Songtexte"):
            try:
                filename, success = future.result()
                if success:
                    processed_songs.add(filename)
                    # Regelmäßig Fortschritt speichern
                    if len(processed_songs) % 10 == 0:
                        save_progress(processed_songs)
            except Exception as e:
                idx, row = futures[future]
                print(f"\nFehler bei Index {idx}: {e}")
    
    # Finalen Fortschritt speichern
    save_progress(processed_songs)
    print(f"\nFertig! Insgesamt verarbeitet: {len(processed_songs)} Songs")

if __name__ == "__main__":
    main()
