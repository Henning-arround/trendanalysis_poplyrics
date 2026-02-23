import os
import json
import pandas as pd
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# API Setup
my_api_key = "##############"
client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=my_api_key)

# Pfade
SAMPLES_DIR = Path("../../llm_as_a_judge/samples")
CSV_PATH = Path("../../data/dataset_popmusic_v8.csv")
INTERPRETATIONS_BASE_DIR = Path("../../llm_as_a_judge/interpretations_prompt_engineering")
EVALUATIONS_BASE_DIR = Path("../../llm_as_a_judge/evaluations_prompt_engineering_two_dimensions")

# Modelle
JUDGE_MODELS = [
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "openai/gpt-oss-120b",
    "meta-llama/Llama-3.3-70B-Instruct"
]

INTERPRETATION_SOURCES = {
    "deepseek": INTERPRETATIONS_BASE_DIR / "deepseek",
    "openai": INTERPRETATIONS_BASE_DIR / "openai",
    "llama": INTERPRETATIONS_BASE_DIR / "llama"
}

NUM_REPETITIONS = 5
MAX_WORKERS = 7

def create_evaluation_directories():
    """Erstelle Evaluationsordner für alle Judge-Modelle"""
    for model in JUDGE_MODELS:
        model_dir_name = model.split('/')[-1].lower().replace('-', '_')
        eval_dir = EVALUATIONS_BASE_DIR / model_dir_name
        eval_dir.mkdir(parents=True, exist_ok=True)

def load_metadata():
    """Lade Metadaten aus CSV"""
    df = pd.read_csv(CSV_PATH, dtype={'Index': str})
    return df.set_index('Index')

def load_songtext(index):
    """Lade den Original-Songtext"""
    songtext_path = SAMPLES_DIR / f"{index}.txt"
    if songtext_path.exists():
        with open(songtext_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def load_interpretations(index, source_model_dir):
    """Lade alle 5 Interpretationen für einen Song von einem Modell"""
    interpretations = []
    for rep in range(1, NUM_REPETITIONS + 1):
        interp_path = source_model_dir / f"{index}_interpretation_{rep}.txt"
        if interp_path.exists():
            with open(interp_path, 'r', encoding='utf-8') as f:
                interpretations.append(f.read())
        else:
            interpretations.append(None)
    return interpretations

def create_judge_prompt(song_title, artist, year, original_lyrics, model_name, interpretations, judge_model_name):
    """Erstelle den Judge-Prompt"""
    # Formatiere die Interpretationen
    interp_text = "\n\n".join([
        f"=== INTERPRETATION {i+1} ===\n{interp}" 
        for i, interp in enumerate(interpretations) if interp
    ])
    
    prompt = f"""Du bist ein Judge für Songtext-Interpretationen. Du bewertest 5 Interpretationen desselben Songs nach zwei kritischen Dimensionen. Die Interpretationen stammen selber aus LLMs.

EINGABE:
- Songtext: {original_lyrics}
- Künstler: {artist}
- Titel: {song_title}
- Jahr: {year}
- Interpretationen: {interp_text}

BEWERTUNGSAUFGABE:

Analysiere jede der 5 Interpretationen nach zwei Dimensionen:

## DIMENSION 1: HALLUZINATIONS-SCORE (1-5)
1 = Stark halluziniert (>50% unbelegte Aussagen)
2 = Viele Erfindungen (30-50% unbelegte Aussagen)
3 = Moderate Erfindungen (15-30% unbelegte Aussagen)
4 = Wenige Erfindungen (5-15% unbelegte Aussagen)
5 = Faktentreu (<5% unbelegte Aussagen)

Als HALLUZINATION/ERFINDUNG zählt:
- Biografische Details des Künstlers, die nicht im Songtext erwähnt werden
- Genre-Zuordnungen ohne textliche Belege (z.B. "typisch für Folk" ohne Folk-Elemente im Text)
- Historische Kontextualisierungen ohne Grundlage (z.B. "Post-9/11 Stimmung" für Songs nach 2001 ohne konkrete Bezüge)
- Kulturgeschichtliche Einordnungen ohne Textbezug
- Spezifische Emotionen/Intentionen, die dem Künstler unterstellt werden
- Konkrete Interpretationen, die NICHT aus dem Text ableitbar sind
- Behauptungen über Musikstil, Sound oder Produktion (diese Information liegt nicht vor)

## DIMENSION 2: TIEFE-SCORE (1-5)
1 = Extrem oberflächlich (nur Gemeinplätze, ausschließlich Wiederholungen von Aussagen)
2 = Oberflächlich (hauptsächlich vage Aussagen, viele Wiederholungen von Aussagen)
3 = Mittlere Tiefe (Mix aus allgemein und spezifischen Aussagen)
4 = Substantiell (überwiegend textbezogen und konkret aber gleichzeitig offen für Ambivalenzen in der Interpretation)
5 = Sehr tiefgehend (durchgehend spezifisch und textanalytisch aber gleichzeitig offen für Ambivalenzen in der Interpretation)

Als OBERFLÄCHLICH gilt:
- Phrasen wie "kulturelles Artefakt seiner Zeit" ohne Erläuterung
- "Es geht um Liebe/Leben/Tod" ohne spezifische Textanalyse
- "Gesellschaftskritisch" ohne Benennung der kritisierten Aspekte
- "Philosophisch" oder "existenziell" ohne konkrete Bezüge
- Allgemeinplätze wie "zeitlos", "universell", "menschliche Erfahrung"
- Vage Aussagen über "Emotionen" oder "Gefühle" ohne Spezifizierung

BEWERTUNGSPROZESS:

Für jede Interpretation:
1. ZÄHLE alle substantiellen Aussagen
2. MARKIERE jede Aussage als:
   - [BELEGT]: Direkt aus dem Songtext ableitbar
   - [ERFUNDEN]: Nicht im Text, spekulativ oder biografisch
   - [OBERFLÄCHLICH]: Vage Allgemeinaussage ohne Substanz
   
3. BERECHNE:
   - Halluzinations-Rate = (Anzahl ERFUNDEN / Gesamtaussagen) × 100
   - Oberflächlichkeits-Rate = (Anzahl OBERFLÄCHLICH / Gesamtaussagen) × 100

OUTPUT-FORMAT:
```json
{{
  "song": "{song_title} - {artist} ({year})",
  "judge_modell": "{judge_model_name}",
  "interpretations_modell": "{model_name}",
  "interpretationen": [
    {{
      "nummer": 1,
      "halluzination_score": [1-5],
      "tiefe_score": [1-5],
      "gesamt_score": [Durchschnitt],
      "begründung": {{
        "erfundene_aussagen": ["Beispiel 1", "Beispiel 2"],
        "oberflächliche_aussagen": ["Beispiel 1", "Beispiel 2"],
        "positive_aspekte": ["Was gut war"]
      }}
    }}
    // ... für alle 5 Interpretationen
  ],
  "zusammenfassung": {{
    "beste_interpretation": [Nummer],
    "durchschnitt_halluzination": [Wert],
    "durchschnitt_tiefe": [Wert],
    "konsistenz_über_wiederholungen": "Bewertung der Varianz",
    "empfehlung": "Kurze Einschätzung welcher Ansatz besser funktioniert"
  }}
}}

WICHTIGE REGELN:
- Längere Interpretationen sind NICHT automatisch besser
- Beginne nicht immer mit der ersten Interpretation sondern mache die Reihenfolge zufällig
- Kreativität ohne Textbezug ist NEGATIV zu bewerten
- Die beste Interpretation balanciert Textreue mit analytischer Tiefe
- Ambivalenzen und Mehrdeutigkeiten sind positiv, wenn textbasiert
- Wiederholungen des gleichen Punkts in verschiedenen Worten zählen als oberflächlich

Gib das vollständige JSON für alle 5 Interpretationen aus."""
    
    return prompt

def get_evaluation(judge_model, prompt):
    """Rufe Judge-LLM auf und hole Evaluation"""
    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def extract_json_from_response(response_text):
    """Extrahiere JSON aus der Antwort"""
    try:
        # Versuche direktes JSON-Parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Suche nach JSON-Block in der Antwort
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            try:
                return json.loads(response_text[start:end])
            except json.JSONDecodeError:
                pass
    return None

def process_single_evaluation(index, song_title, artist, year, original_lyrics, 
                              source_model_name, source_model_dir, 
                              judge_model, judge_output_dir):
    """Verarbeite eine einzelne Evaluation"""
    try:
        # Lade Interpretationen
        interpretations = load_interpretations(index, source_model_dir)
        
        # Prüfe ob alle Interpretationen vorhanden sind
        if None in interpretations:
            return False, f"Fehlende Interpretationen für {index} von {source_model_name}"
        
        # Erstelle Prompt
        judge_model_name = judge_model.split('/')[-1]
        prompt = create_judge_prompt(song_title, artist, year, original_lyrics, 
                                    source_model_name, interpretations, judge_model_name)
        
        # Hole Evaluation
        evaluation_response = get_evaluation(judge_model, prompt)
        
        # Extrahiere JSON
        evaluation_json = extract_json_from_response(evaluation_response)
        
        if evaluation_json is None:
            # Wenn JSON-Extraktion fehlschlägt, speichere rohe Antwort
            evaluation_json = {
                "raw_response": evaluation_response,
                "error": "JSON parsing failed"
            }
        
        # Speichere Evaluation
        output_filename = f"{index}_{source_model_name}_judged_by_{judge_model_name}.json"
        output_path = judge_output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_json, f, ensure_ascii=False, indent=2)
        
        return True, None
    except Exception as e:
        return False, f"Fehler bei {index}, Source: {source_model_name}, Judge: {judge_model}: {e}"

def evaluate_all_interpretations():
    """Evaluiere alle Interpretationen mit allen Judge-Modellen"""
    # Vorbereitung
    create_evaluation_directories()
    metadata_df = load_metadata()
    
    # Hole alle Samples
    sample_files = sorted(SAMPLES_DIR.glob("*.txt"))
    
    # Erstelle alle Tasks
    tasks = []
    for sample_file in sample_files:
        index = sample_file.stem
        
        # Hole Metadaten
        if index not in metadata_df.index:
            continue
        
        row = metadata_df.loc[index]
        song_title = row['Titel']
        artist = row['Künstler']
        year = row['Jahr']
        
        # Lade Songtext
        original_lyrics = load_songtext(index)
        if original_lyrics is None:
            continue
        
        # Erstelle Tasks für alle Kombinationen
        for source_name, source_dir in INTERPRETATION_SOURCES.items():
            for judge_model in JUDGE_MODELS:
                judge_model_dir_name = judge_model.split('/')[-1].lower().replace('-', '_')
                judge_output_dir = EVALUATIONS_BASE_DIR / judge_model_dir_name
                
                tasks.append((
                    index, song_title, artist, year, original_lyrics,
                    source_name, source_dir, judge_model, judge_output_dir
                ))
    
    # Verarbeite Tasks parallel mit Progressbar
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_evaluation, *task): task 
            for task in tasks
        }
        
        with tqdm(total=len(tasks), desc="Evaluiere Interpretationen") as pbar:
            for future in as_completed(futures):
                success, error_msg = future.result()
                if not success and error_msg:
                    print(f"\n{error_msg}")
                pbar.update(1)

if __name__ == "__main__":
    evaluate_all_interpretations()
    print("\nAlle Evaluationen wurden erstellt!")
