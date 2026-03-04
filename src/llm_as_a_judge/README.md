# Quellcode der LLM-as-a-Judge Evaluation

Dieser Ordner enthält die Python-Skripte, welche die Methodik der "LLM-as-a-Judge"-Evaluation abbilden. Dazu gehören die Ziehung der Stichprobe, das Generieren der Songtext-Interpretationen, der Evaluierungsprozess durch die Judge-LLMs, sowie die Berechnung verschiedener Ähnlichkeitsmetriken.

## Enthaltene Skripte

### Datenvorbereitung und Generierung
* **`sample.py`**: Skript zur zufälligen Ziehung einer strukturierten Stichprobe (100 Songtexte) aus dem Gesamtdatensatz, auf der die gesamte Evaluation basiert.
* **`create_interpretations.py`**: Generiert die ersten Interpretationen für die gezogenen Songtexte mittels der Methode "standard" für verschiedene LLMs.
* **`create_interpretations_prompt_engineering.py`**: Generiert Interpretationen der gleichen Songtexte, wendet hierfür jedoch die Methode "prompt_engineering" an.

### LLM Judge Ausführung
* **`judge.py`**: Führt den Kernprozess "LLM-as-a-Judge" auf den Interpretationen der "standard" Methode aus.
* **`jude_prompt_engineering.py`**: Führt den "LLM-as-a-Judge"-Prozess identisch für jene Interpretationen durch, die mittels Prompt Engineering erzielt wurden.
* **`evaluation.py`**: Dient als Skript um Evaluationsmetriken gesammelt zu verarbeiten und auszugeben.

### Ähnlichkeitsmetriken (Similarity)
Diese Skripte dienen zur Messung der semantischen Konsistenz und Ähnlichkeit zwischen den verschiedenen generierten Evaluierungen.
* **`similarity_api_embeddings.py`**: Berechnet Textähnlichkeiten basierend auf Embeddings, die durch die externen APIs generiert wurden.
* **`similarity_baai.py`**: Verwendet explizit Modelle/Embeddings von BAAI, um die semantische Nähe der Texte zu bewerten.
* **`similarity_bertscore.py`**: Berechnet den BERTScore für den paarweisen Vergleich von Interpretationen.
* **`similiarity_cosine.py`**: Dient zur Berechnung der Cosine Similarity auf verschiedenen Embeddings.

### Auswertung und Statistik
* **`statistics_judge.py`**: Verarbeitet die Rohergebnisse der LLM-Judges (JSON/CSV) und berechnet darauf basierend Aggregationen, Verteilungen und statistische Kennzahlen, die dann z. B. für Visualisierungen im `visualisations/` Ordner genutzt werden.
