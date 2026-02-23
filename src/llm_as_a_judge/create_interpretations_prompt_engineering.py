import os
import pandas as pd
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

# API Setup
my_api_key = "############"
client = OpenAI(
    base_url="https://llm.scads.ai/v1", 
    api_key=my_api_key,
    timeout=httpx.Timeout(30.0, connect=10.0)  # 30 Sekunden Timeout für Anfragen
)

# Pfade
SAMPLES_DIR = Path("../../llm_as_a_judge/samples")
CSV_PATH = Path("../../data/dataset_popmusic_v8.csv")
OUTPUT_BASE_DIR = Path("../../llm_as_a_judge/interpretations_prompt_engineering")

# Modelle und ihre Ausgabeordner
MODELS = {
    "deepseek-ai/DeepSeek-V3.2-Exp": OUTPUT_BASE_DIR / "deepseek",
    "openai/gpt-oss-120b": OUTPUT_BASE_DIR / "openai",
    "meta-llama/Llama-3.3-70B-Instruct": OUTPUT_BASE_DIR / "llama"
}

# Anzahl der Wiederholungen und parallele Workers
NUM_REPETITIONS = 10
MAX_WORKERS = 5

def create_output_directories():
    """Erstelle Ausgabeordner für alle Modelle"""
    for output_dir in MODELS.values():
        output_dir.mkdir(parents=True, exist_ok=True)

def load_metadata():
    """Lade Metadaten aus CSV"""
    df = pd.read_csv(CSV_PATH, dtype={'Index': str})
    return df.set_index('Index')

def create_prompt(titel, künstler, jahr, songtext):
    """Erstelle den Interpretations-Prompt"""
    prompt = f"""Context: Du bist ein Kulturwissenschaftler mit Expertise in Textanalyse und Songinterpretationen.
Objective: Interpretiere den Song "{titel}" von {künstler} aus dem Jahr {jahr} aus kulturwissenschaftlicher Perspektive.
Style: Klar verständlich
Tone: Objektiv
Response: auf Deutsch, maximal 300 Wörter, strukturiert in drei Absätze, keine Zwischenüberschriften
SONGTEXT: {songtext}
ANALYSERAHMEN:

Zentrale Botschaft und thematische Komplexität
Literarische Stilmittel (Metaphern, Symbole, Bilder, Narrative)
Emotionale Wirkung und affektive Dimension
Kultureller, historischer und gesellschaftlicher Kontext
Tiefere Bedeutungsebenen und symbolische Ordnungen

ANFORDERUNGEN:

Belege jede Behauptung mit konkreten Textzeilen
Identifiziere Ambiguitäten, Spannungen und Widersprüche statt sie aufzulösen
Vermeide oberflächliche Aussagen"""
    return prompt

def get_interpretation(model, prompt):
    """Rufe LLM API auf und hole Interpretation mit Timeout-Handling"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=30.0  # Zusätzlicher expliziter Timeout
        )
        return response.choices[0].message.content
    except httpx.TimeoutException:
        raise TimeoutError(f"API-Anfrage für Modell {model} hat 30 Sekunden überschritten")
    except Exception as e:
        raise e

def process_single_interpretation(index, model_name, output_dir, prompt, repetition):
    """Verarbeite eine einzelne Interpretation"""
    try:
        # Prüfe ob Interpretation bereits existiert
        output_filename = f"{index}_interpretation_{repetition}.txt"
        output_path = output_dir / output_filename
        
        if output_path.exists():
            return True, None  # Bereits vorhanden, überspringe
        
        # Hole Interpretation
        interpretation = get_interpretation(model_name, prompt)
        
        # Speichere Interpretation
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(interpretation)
        
        return True, None
    except TimeoutError as e:
        return False, f"TIMEOUT: {index}, Modell {model_name}, Rep. {repetition} - übersprungen"
    except Exception as e:
        return False, f"Fehler bei {index}, Modell {model_name}, Rep. {repetition}: {e}"

def process_all_samples():
    """Verarbeite alle Samples mit allen Modellen"""
    # Vorbereitung
    create_output_directories()
    metadata_df = load_metadata()
    
    # Hole alle Textdateien
    sample_files = sorted(SAMPLES_DIR.glob("*.txt"))
    
    # Erstelle alle Tasks
    tasks = []
    for sample_file in sample_files:
        # Extrahiere Index aus Dateinamen als String
        index = sample_file.stem
        
        # Hole Metadaten
        if index not in metadata_df.index:
            continue
        
        row = metadata_df.loc[index]
        titel = row['Titel']
        künstler = row['Künstler']
        jahr = row['Jahr']
        
        # Lese Songtext
        with open(sample_file, 'r', encoding='utf-8') as f:
            songtext = f.read()
        
        # Erstelle Prompt
        prompt = create_prompt(titel, künstler, jahr, songtext)
        
        # Erstelle Tasks für alle Modelle und Wiederholungen
        for model_name, output_dir in MODELS.items():
            for repetition in range(1, NUM_REPETITIONS + 1):
                tasks.append((index, model_name, output_dir, prompt, repetition))
    
    # Verarbeite Tasks parallel mit Progressbar
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_interpretation, *task): task 
            for task in tasks
        }
        
        with tqdm(total=len(tasks), desc="Erstelle Interpretationen") as pbar:
            for future in as_completed(futures):
                success, error_msg = future.result()
                if not success and error_msg:
                    print(f"\n{error_msg}")
                pbar.update(1)

if __name__ == "__main__":
    process_all_samples()
    print("\nAlle Interpretationen wurden erstellt!")
