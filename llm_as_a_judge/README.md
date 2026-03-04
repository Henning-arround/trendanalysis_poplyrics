# LLM-as-a-Judge Evaluation

Dieser Ordner beinhaltet die Daten, Prompts und Ergebnisse zur Evaluation der LLM-generierten Songtext-Interpretationen mittels LLM-as-a-Judge.

## Verzeichnisstruktur im Detail

### `evaluations_two_dimensions/`
Hier liegen die Ergebnisse der "standard" Methode (ohne spezielles Prompt Engineering) anhand von zwei Bewertungsdimensionen (Halluzinationsrate, Interpretationstiefe). 
Die Ergebnisse sind in Unterordnern für die jeweiligen "Judge"-Modelle strukturiert:

- `deepseek_v3.2_exp/`
- `gpt_oss_120b/`
- `llama_3.3_70b_instruct/`

### `evaluations_prompt_engineering_two_dimensions/`
Analog zu *evaluations_two_dimensions*, jedoch enthalten diese Ordner die Bewertungen für Interpretationen mit der Methode "prompt_engineering"

### `interpretations/`
Enthält die zu evaluierenden Interpretationen, die mit der "standard" Methode erzeugt wurden. Sie sind nach dem generierenden Modell gruppiert:
- `deepseek/`
- `llama/`
- `openai/`

### `interpretations_prompt_engineering/`
Enthält die Interpretationen der gleichen Modelle (DeepSeek, LLaMa, OpenAI), jedoch mit der "prompt_engineering" Methode.

### `prompts/`
Dieses Verzeichnis archiviert die exakten Prompts, die in den LLM-Aufrufen genutzt wurden:
- **`standard_prompt.txt`**: Die Basis-Anweisung zur Interpretation von Songtexten.
- **`prompt_engineering.txt`**: Die überarbeitete und optimierte Anweisung zur Interpretation.
- **`judge_prompt.txt`**: Der System-Prompt für die LLM-Judges, in dem die Bewertungskriterien und Antwortformate spezifiziert sind.

### `results/`
Zentrale Sammelstelle für evaluierte Metriken und Vergleiche der Interpretationen:
- **`bertscore_similarity/`** & **`cosine_similarity/`** & **`similarity_baai/`**: Ergebnisse von Ähnlichkeitsberechnungen (Semantische Nähe) zwischen den verschiedenen Texten.
- **`llm_judge/`**: Aggregierte oder spezifische Ergebnisse aus den LLM-as-a-Judge Durchläufen.

### `samples/`
Beinhaltet eine repräsentative Stichprobe (100 Songtexte) aus dem Gesamtdatensatz, auf der die Evaluation (Validierung) durchgeführt wurde.
- **`.txt`-Dateien** (z.B. `10058.txt`): Die rohen Songtexte.
- **`sample_metadata.csv`**: Metadaten zu der gezogenen Stichprobe.
