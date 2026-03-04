# Tabellenmanipulation

Dieses Verzeichnis enthält verschiedene Skripte zur Verarbeitung, Reinigung und Transformation der gesammelten Datensätze (CSVs und JSONs). Die Skripte sind oft Teil einer sukzessiven Daten-Pipeline.

## Übersicht der Skripte

| Skript | Funktion | Eingabe | Ausgabe |
|---|---|---|---|
| `add_index.py` | Fügt dem initialen Charts-Datensatz eine eindeutige Index-Spalte hinzu. | `charts.csv` | `charts_indexed.csv` |
| `add_language_full_dataset.py` | Liest die Lyrics-Textdateien ein und fügt dem Datensatz die erkannte Sprache als Spalte hinzu. | `dataset_popmusic_v5.csv`, `.txt` Dateien | `dataset_popmusic_v6.csv` |
| `add_missing_lines_chartsurfer.py` | Ergänzt fehlende Information (Genius Quelle, Release Date, Genre) aus Log-Daten für den Chartsurfer-Datensatz. | `charts_chartsurfer_1954_1977_v3.csv` | `charts_chartsurfer_1954_1977_v4.csv` |
| `add_source.py` | Prüft die lokal vorhandenen Songtext-Dateien und aktualisiert den Datensatz mit Quellen-Informationen. | `charts_indexed.csv`, `.txt` Dateien | `charts_indexed.csv` (aktualisiert) |
| `add_source_genius.py` | Führt neu gescrapte Genius-Daten und den Chart-Datensatz zusammen. | `charts_with_language_updated_v2.csv`, `missing_source_lyrics.csv` | `charts_with_language_updated_v3.csv` |
| `add_year_chartsurfer.py` | Extrahiert das Veröffentlichungsjahr im Chartsurfer-Datensatz in eine separate Spalte. | `charts_chartsurfer_1954_1977.csv` | `charts_chartsurfer_1954_1977_with_year.csv` |
| `clean_duplicates_chartsurfer.py` | Entfernt Duplikate aus der zweiten Version des Chartsurfer-Datensatzes. | `charts_chartsurfer_1954_1977_v2.csv` | `charts_chartsurfer_1954_1977_v2_without_duplicates.csv` |
| `clear_duplicats_full_dataset.py` | Bereinigt Duplikate im großen, zusammengeführten Popmusik-Datensatz (basierend auf URLs). | `dataset_popmusic.csv` | `dataset_popmusic_v2.csv`, `url_analyse_ergebnisse.txt` |
| `complete_dataset.py` | Führt die "Offizielle Charts"- und "Chartsurfer"-Daten zu einem finalen Gesamtdatensatz zusammen. | diverse `charts_*.csv` und `charts_chartsurfer_*.csv` Dateien | `dataset_popmusic.csv`, Umbenennung von `.txt` Dateien |
| `create_DNB_dataset.py` | Erzeugt einen speziellen JSON-Datensatz mit integrierten Lyrics (vermutlich für die Deutsche Nationalbibliothek oder spezielle Analysen). | `dataset_popmusic_v8.csv`, `.txt` Dateien | `dataset_popmusic_DNB.json` |
| `csv_to_json.py` | Konvertiert einen der Charts-Datensätze von CSV nach JSON. | `charts_with_language_updated_v4.csv` | `charts_with_language_updated_v4.json` |
| `extract_topic_examples.py` | Extrahiert Songtexte und deren Interpretationen für spezifische Themen (aus BERTopic). | `dataset_popmusic_v8.json`, `dataset_popmusic_DNB.json`, `topic_info_llm.csv` | Einzelne `.txt` Dateien (Lyrics & Interpretation je Beispiel) |
| `lyrics_to_txt.py` | Extrahiert Lyricsstrukturen aus der anfänglichen CSV und speichert jeden Song in einer eigenen Textdatei ab. | `lyrics.csv` | Eigene `.txt` Dateien (Name: `<Index>.txt`) |
| `prepare_data_bertopic.py` | Bereitet die Datensätze (Texte & Interpretationen) als JSON formatiert für die Textanalyse mit BERTopic vor. | `dataset_popmusic_v8.csv`, Interpretations/Text-Dateien | `dataset_popmusic_v8.json`, `dataset_popmusic_v8_lyrics.json` |
| `process_results_eom_leaf.py` | Säubert Evaluations-CSVs aus BERTopic (z.B. Entfernung nicht benötigter Spalten). | `results_eom_leaf_1.csv` | `results_eom_leaf_1_cleaned.csv` |
| `topic_keywords_latex.py` | Liest Themen-Keywords ein und generiert LaTeX-Code für die Arbeit/Thesis. | `topic_info_llm.csv` | LaTeX Code (Terminal / Output) |
| `unify_year_full_dataset.py` | Vereinheitlicht das Datums- und Jahresformat im großen Dataset. | `dataset_popmusic_v2.csv` | `dataset_popmusic_v3.csv`, `jahr_vereinheitlichung_log.txt` |

