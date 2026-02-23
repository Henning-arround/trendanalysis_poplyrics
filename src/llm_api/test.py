import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from requests.exceptions import RequestException, ConnectionError, Timeout
from umap import UMAP
from bertopic import BERTopic

my_api_key = "sk-JbSIfkSGPUcXx7OONQMZyw"

client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=my_api_key)

model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# Pfade definieren
input_dir = "../../data/lyrics_german_cleaned"
output_dir = "../../data/lyrics_german_cleaned_llm_intepretation"
csv_path = "../../data/charts_with_language_updated_v4.csv"

# Output-Verzeichnis erstellen, falls es nicht existiert
os.makedirs(output_dir, exist_ok=True)

# CSV-Datei laden
df = pd.read_csv(csv_path)
df = df.set_index('Index')

def process_song(filename):
    """Verarbeitet einen einzelnen Songtext"""
    input_path = os.path.join(input_dir, filename)
    
    # Index aus Dateinamen extrahieren (z.B. "567.txt" -> 567)
    song_index = int(filename.replace(".txt", ""))
    
    # Künstler und Titel aus DataFrame extrahieren
    if song_index in df.index:
        kuenstler = df.loc[song_index, 'Künstler']
        titel = df.loc[song_index, 'Titel']
    else:
        kuenstler = "Unbekannt"
        titel = "Unbekannt"
    
    # Songtext laden
    with open(input_path, 'r', encoding='utf-8') as f:
        songtext = f.read()
    
    # Prompt für das Modell
    prompt = f"Interpretiere den folgenden Songtext {titel} von {kuenstler}. Gib dabei nur die Intepretation aus, keine weitere Ausgabe. Nutze Maximal 400 Wörter. Hier der Songtext: {songtext}"
    
    # Retry-Logik für API-Anfrage
    max_retries = 8
    retry_delay = 2  # Startverzögerung in Sekunden
    
    for attempt in range(max_retries):
        try:
            # Anfrage an das Modell
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name
            )
            
            # Interpretation extrahieren
            interpretation = response.choices[0].message.content
            break  # Erfolgreich, Schleife verlassen
            
        except (ConnectionError, Timeout, RequestException) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponentielles Backoff
                print(f"\nNetzwerkfehler bei {filename} (Versuch {attempt + 1}/{max_retries}). Warte {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\nFehler bei {filename} nach {max_retries} Versuchen: {e}")
                raise
        except Exception as e:
            print(f"\nUnerwarteter Fehler bei {filename}: {e}")
            raise
    
    # Output-Dateiname erstellen
    output_filename = filename.replace(".txt", "_llm.txt")
    output_path = os.path.join(output_dir, output_filename)
    
    # Interpretation speichern
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(interpretation)
    
    return filename

# Liste aller Textdateien erstellen
txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

# Parallel verarbeiten mit 5 Threads
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(process_song, filename): filename for filename in txt_files}
    
    for future in tqdm(as_completed(futures), total=len(txt_files), desc="Interpretiere Songtexte"):
        try:
            future.result()
        except Exception as e:
            print(f"\nFehler bei {futures[future]}: {e}")

print("\nAlle Songtexte wurden interpretiert!")

