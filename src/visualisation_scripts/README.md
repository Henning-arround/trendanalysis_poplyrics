# Visualisierungsskripte

Dieser Ordner enthält alle Skripte, die zur Erstellung der Abbildungen, Plots und Diagramme für die schriftliche Ausarbeitung der Masterarbeit verwendet wurden. Zur besseren Übersicht sind die Skripte thematisch in drei Unterordner gegliedert.

## Verzeichnisstruktur und Skripte

### 1. `chart_analysis/`
Enthält Skripte zur grafischen Aufbereitung statistischer Analysen bezüglich der Chart-Auswertungen.
* **`viz_odds_ratio.py`**: Visualisiert die errechneten Assoziationsmaße (Odds Ratios) sowie Signifikanzen (z.B. aus exakten Tests nach Fisher), um zu verdeutlichen, welche Topics eine statistisch signifikante Beziehung zu kommerziellem bzw. Chart-Erfolg aufweisen.

### 2. `dataset/`
Diese Skripte dienen der explorativen Datenanalyse und erzeugen deskriptive Visualisierungen, die den bereinigten Korpus beschreiben, bevor das Topic Model angewandt wird.
* **`viz_artists.py`**: Erstellt Darstellungen zur Verteilung und Dominanz einzelner Künstler im Datensatz (z. B. Top-Interpreten).
* **`viz_genres.py`**: Visualisiert die Zusammensetzung der zugeordneten Musikgenres der gescrapten Songs.
* **`viz_language.py`**: Untersucht und plottet die Verteilung der gesprochenen Sprachen in den Songtexten.
* **`viz_rap_bias.py`**: Analysiert visuell das Aufkommen und potenziell dominierende Volumen von Rap-Texten im zeitlichen Verlauf der Charts.
* **`viz_time_bins.py`**: Generiert Histogramme und Zeitreihen, die aufzeigen, wie sich die Anzahl der quantifizierbaren Songtexte über die historischen Zeitabschnitte (Jahre, Dekaden) der Charts verteilt.

### 3. `topic_model/`
Diese Skripte befassen sich exklusiv mit den Ergebnissen und der Struktur des trainierten BERTopic-Modells.
* **`viz_embeddings.py`** & **`viz_embeddings_dual_view.py`**: Skripte zur Reduktion (via UMAP/PCA) und grafischen Darstellung des Embedding Raums.
* **`viz_outlier_analysis.py`**: Visualisiert den Anteil der vom HDBSCAN-Algorithmus als Rauschen ("Outlier" / Topic -1) deklarierten Dokumente im Zeit- oder Clustervergleich.
* **`viz_topic_development.py`** & **`viz_trends.py`**: Erstellen generelle Übersichtsgrafiken zur inhaltlichen Entwicklung der Topics über die analysierten Datumsstempel.
* **`viz_specific_trends_combined.py`**: Kombiniert spezifisch ausgewählte zeitliche Trends ("Topics over Time") in zusammengefassten Grafiken, um inhaltliche Gegenüberstellungen oder Verschiebungen zu verdeutlichen (z. B. Liebe)
