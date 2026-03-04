# Projektübersicht

Dieses Repository enthält den Code und die Ergebnisse für die Masterarbeit zur Analyse von Songtexten.

## Ordnerstruktur

Hier ist eine kurze Übersicht über die wichtigsten Ordner in diesem Projekt (ausgenommen Datensätze):

### `bertopic/`
Dieser Ordner enthält Ergebnisse des Topic Models mittels BERTopic.
- Enthält den Unterordner `dtm` (Dynamic Topic Modeling), bei dem sich auf den Clusteralgorithmus HDBSCAN konzentriert wurde und einen archivierten Ordner der Experimente mit der Methode K-Means beinhaltetn. 

### `llm_as_a_judge/`
Hier befinden sich Dateien für die Evaluation mittels "LLM-as-a-Judge".
- **evaluations_...**: Ergebnisse der Bewertungen durch das LLM (z.B. `evaluations_two_dimensions`, `evaluations_prompt_engineering_two_dimensions`).
- **interpretations**: Vom LLM generierte Interpretationen der Songtexte.
- **prompts**: Die verwendeten Prompts für das LLM.
- **results/samples**: Weitere Ergebnisse und Stichproben.

### `src/`
Der Quellcode des Projekts, unterteilt in verschiedene Module:

#### Datenakquise und Aufbereitung
- **detect_language/**: Spracherkennung der Songtexte.
- **llm_api/**: Zugriff auf LLM-APIs für die Interpretationen er 255k Songs.
- **preprocessing/**: Skripte zur Bereinigung des Korpus.
- **scrape_scripts/**: Web-Scraper zum Sammeln von den Daten (z.B. offiziellecharts, chartsurfer, genius, discogs, etc.).
- **table_manipulations/**: Wieder Skripte zum Erstellen des Korpuses (aufräumen), sowie Pre- und Postprocessing des Korpus.

#### Methoden und Analyse
- **llm_as_a_judge/**: Logik für den LLM-Evaluierungsprozess.
- **topic_modelling/**: Implementierungen und Skripte spezifisch für BERTopic und Dynamic Topic Modeling (DTM). Zudem ein archivierte Ordner der Experimente zu K-Means enthält.
- **visualisation_scripts/**: Code zur Erstellung der Visualisierungen.

### `visualisations/`
Dieser Ordner speichert die generierten Grafiken und Plots.
- **chart_analysis/**: Visualisierungen der Charts und die Ergebnisse des exakten Test nach Fisher.
- **dataset/**: Deskriptive Darstelunngen des Datensatzes.
- **llm_judge/**: Visualisierung der Ergebnisse der LLM-Evaluation.
- **topic_model/**: Visualisierungen zum Embeddingspace, Outliern, spezifischen Topic Entwicklungen, Hierarchische Struktur sowie Netzwerkvisualisierungen.
- **keywords/** Visualisierung für die Keyword Tabelle in Latex.
