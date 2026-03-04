# README: Topic Modelling mit HDBSCAN (Ordner "src/topic_modelling/hdbscan")

Dieser Ordner enthält die zentralen Skripte für die Implementierung des Topic Modeling im Rahmen der Masterarbeit. Hier wird BERTopic mit seinem Standard-Clusteralgorithmus HDBSCAN genutzt, um ein Dynamic Topic Model (DTM) für die Songtexte zu trainieren, zu evaluieren und visuell aufzubereiten.

## Enthaltene Skripte

### Modellierung und Evaluierung
* **`create_dtm.py`**
  Das Hauptskript für das Training des BERTopic-Modells. Es nutzt HDBSCAN zur Clusterbildung und berechnet die thematische Entwicklung über die Zeit (Dynamic Topic Modeling), inklusive der c-TF-IDF basierten Repräsentationen.
* **`extract_dtm.py`**
  Extrahiert die Ergebnisse aus dem fertig trainierten Modell (z.B. Topic-Listen, Dokumenten-Zuweisungen und zeitliche Verläufe) und speichert diese in leicht lesbaren Formaten (z.B. CSV-Dateien im Ordner `bertopic/dtm/models/`).
* **`dtm_evaluation.py`**
  Führt die quantitative Bewertung von verschiedenen Modellen durch. Hierbei werden verschiedene Metriken und Hyperparameter (z.B. Coherence-Scores, Verteilung der Clustergrößen) berechnet und in Log-Dateien protokolliert.
* **`analyse_outliers.py`**
  Da der HDBSCAN-Algorithmus Dokumente, die in kein klares Muster passen, als Rauschen/Ausreißer ("Outlier", meist Topic -1) kategorisiert, dient dieses Skript der spezifischen Analyse dieser Dokumente, um die Güte des Modells sicherzustellen.

### Post-Processing und Erweiterungen
* **`create_stopwords.py`**
  Hilfsskript zur Erstellung und Formatierung der angepassten Stoppwortliste, um bedeutungslose Füllwörter aus den Topics herauszufiltern.
* **`create_llm_labels.py`**
  Verwendet ein LLM, um basierend auf den repräsentativen Top-Wörtern und den zugehörigen Songtexten ausdrucksstarke, menschlich verständliche Labels für die einzelnen Topics zu generieren.
* **`chart_analysis.py`**
  Führt statistische Auswertungen durch, indem es die Topic-Zuweisungen der Songs mit deren Chart-Daten verknüpft (z.B. durch Chi-Quadrat-/Fisher-Tests).

### Visualisierungen
Diese Skripte greifen auf das trainierte BERTopic-Modell zu und generieren interaktive HTML-Dateien oder statische Grafiken für den Ordner `visualisations/`:
* **`visualize_tot.py`**
  Generiert Liniendiagramme für die "Topics over Time" (ToT), um aufzuzeigen, wie sich die Popularität bestimmter Themen über die Jahrzehnte entwickelt hat.
* **`visualize_hierarchical.py`**
  Erzeugt eine Darstellung des hierarchischen Clusterbaums, der zeigt, wie einzelne, sehr granulare Topics in größere Überthemen zusammengefasst werden können.
* **`visualize_datacloud.py`**
  Erstellt visuelle Repräsentationen der Topics in Form von DataClouds basierend auf den Embeddings.
* **`visualize_network.py`**
  Erzeugt eine Netzwerk-Visualisierung basierend auf den Inter-Topic Distanzen, um die semantische Nähe und Zusammenhänge zwischen verschiedenen Topics abzubilden.
