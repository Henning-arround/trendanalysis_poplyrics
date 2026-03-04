# BERTopic - Dynamic Topic Modeling 

Dieser Ordner enthält alle Zwischenergebnisse, Modelle, Evaluationen und Visualisierungen, die im Rahmen des Topic Modelings mittels **BERTopic** und dem Cluster-Algorithmus **HDBSCAN** entstanden sind. 

## Verzeichnisstruktur im Detail

### `embedding/`
Speichert die vorberechneten Text-Embeddings der Songtextinterpretationen (z. B. `embeddings.npy`). Durch das lokale Zwischenspeichern der Embeddings müssen diese nicht bei jedem Modellaufruf neu berechnet werden, was den Trainings- und Evaluierungsprozess erheblich beschleunigt.

### `evaluation/`
Beinhaltet die Ergebnisse und Metriken der Modell-Evaluationen.
- **Log-Dateien** (`evaluation_log_*.txt`): Protokolle der verschiedenen Trainings- oder Evaluierungsdurchläufe.
- **CSV-Dateien** (`results_intermediate_*.csv`, `evaluation.csv`): Ergebnisse der verschiedenen Modell-Iterationen und Evaluierungsmetriken (wie Coherence Scores, Outlier-Ratios), um das beste Modell zu identifizieren.

### `evaluation_parameters/`
Enthält spezifische Auswertungen und Parameter-Tests für das Clustering.
Dateien wie `experimental_leaf_eom.csv` deuten auf die Optimierung und Gegenüberstellung verschiedener HDBSCAN-Clusterextraktionsmethoden hin (z. B. "Excess of Mass" vs. "Leaf"). 

### `models/`
Speichert die finalen trainierten Modelle und abgeleitete Datenstrukturen.
- **`bertopic_model_dtm`**: Das gespeicherte, trainierte BERTopic-Modell.
- **`topic_info.csv` & `topic_info_llm.csv`**: Tabellarische Übersicht der gefundenen Topics, deren Größe und repräsentativen Wörtern (ggf. durch ein LLM gelabelt).
- **`topics_over_time.csv`**: Rohdaten zur zeitlichen Entwicklung der Topics, essenziell für das Dynamic Topic Modeling.
- **`probabilities.npy`**: Wahrscheinlichkeitswerte der Dokument-zu-Topic-Zuweisungen.

### `outlier_analysis/`
Hier finden sich Visualisierungen zu Outlieruntersuchungen. 

### `stopwords/`
- **`stopwords.txt`**: Die Top 100 Wörter des Korpus, die dem Modell übergeben wurde, um bedeutungslose Wörter aus der Topic-Repräsentation (c-TF-IDF) herauszufiltern (Stoppwörter).

### `visualizations/`
Dieser Ordner archiviert html-basierte und tabellarische Visualisierungen, die direkt mit den BERTopic-Ergebnissen oder deren Parametrisierungen zusammenhängen:
- **`chart_analysis/`**: Ergebnisse statistischer Auswertungen bezugnehmend auf die Topics (z. B. Chi-Quadrat-Tests).
- **`hierarchical/`**: HTML-Visualisierung der hierarchischen Topic-Struktur (`hierarchy.html`).
- **`model_visualisations/`**: Verschiedene von BERTopic standardmäßig generierte interaktive Visualisierungen (`barchart.html`, `documents.html`, `intertopic_distance_map.html`, `all_topics_datacloud.html`).
- **`over_time/`**: Grafische Aufbereitungen der zeitlichen Entwicklung der Topics (`topics_over_time.html`).
