# Scrape Scripte

In diesem Verzeichnis befinden sich verschiedene Skripte zum Scraping von Songtexten, Chartdaten und Zusatzinformationen. Im Folgenden wird kurz erläutert, was die Skripte machen und welche Ein- und Ausgabedateien es gibt. Nicht alle Skripte sind noch relevant für den finalen Datensatz, da zwischendurch ein neuer Durchlauf gestartet wurde. Aufgrund der Prozessdokumentation wurden sie jedoch nicht entfernt.

## Übersicht der Skripte

| Skript | Funktion | Eingabe | Ausgabe |
|---|---|---|---|
| `scrape_chart_urls.py` | Ruft Chart-URLs ab | - | `lyrics_ovh_not_found.csv` (Fehlerprotokoll) |
| `scrape_chartsurfer.py` | Scraping der Chartsurfer-Website (1954-1977) um erste Songdaten (Titel, Künstler, Chart-Details) zu erhalten. | - | `charts_chartsurfer_1954_1977.csv` |
| `scrape_offizielle_charts.py` | Scraping der "Offizielle Charts" Webseite um Songs zu extrahieren. | `chart_urls_unique.txt` | `charts.csv`, `progress.txt` |
| `add_additional_songs_to_chartsurfer_dataset.py` | Fügt zusätzliche Songs zum Chartsurfer Datensatz hinzu bzw. bereitet die CSV weiter auf. | `charts_chartsurfer_1954_1977.csv` | `charts_chartsurfer_1954_1977_v2.csv` |
| `scrape_genius_chartsurfer.py` | Sucht und lädt Lyrics für den Chartsurfer-Datensatz von Genius herunter und fügt die Download-URLs in die CSV ein. | `charts_chartsurfer_1954_1977_v2_without_duplicates.csv` | `charts_chartsurfer_1954_1977_v3.csv`, `chartsurfer_download_progress.log` |
| `scrape_missing_date_chartsurfer.py` | Sucht fehlende Veröffentlichungsdaten für den Chartsurfer-Datensatz über MusicBrainz TheAudioDB. | `charts_chartsurfer_1954_1977_v4.csv` | `charts_chartsurfer_1954_1977_v5.csv`, `chartsurfer_missing_dates.txt` |
| `scrape_missing_date_chartsurfer_discogs.py` | Sucht fehlende Veröffentlichungsdaten für den Chartsurfer-Datensatz über Discogs. | `charts_chartsurfer_1954_1977_v4.csv` | `charts_chartsurfer_1954_1977_v5.csv`, `chartsurfer_missing_dates_discogs.txt` |
| `add_missing_dates_full_dataset.py` | Sucht global nach fehlenden Daten für den zusammengesetzten Musikdatensatz. | `dataset_popmusic_v3.csv` | `dataset_popmusic_v4.csv`, `popmusic_missing_dates.txt` |
| `scrape_genre_offizielle_charts.py` | Ermittelt über MusicBrainz fehlende Genre-Tags für Künstler im Offiziellen Chart Datensatz. | `charts_with_language_updated_v4.csv` | `charts_with_language_updated_v5.csv` |
| `scrape_second_run.py` | Zweiter Scrape-Durchlauf zum Herunterladen noch fehlender Lyrics aus dem "Offiziellen Charts" Datensatz bei Genius. | `charts_with_language_updated_v5.csv` | `charts_offizielle_charts_minimal.csv`, Texte im Ordner `lyrics_offiziellecharts_revisited/` |
| `scrape_genius.py` | Lädt Lyrics für die Offiziellen Charts (indiziert) von Genius herunter und verzeichnet die Quelle. | `charts_indexed.csv` | `source_genius.csv`, Texte im Ordner `lyrics/` |
| `scrape_missing_lyrics.py` | Lädt noch fehlende Lyrics von Genius herunter für den ersten großen "Offizielle Charts" Datensatz. | `charts_with_language_updated.csv` | `missing_source_lyrics.csv`, Texte im Ordner `lyrics/` |
| `scrape_lyricsovh.py` | Versuchtes Ersatz-Scraping von Lyrics via lyrics.ovh API falls Genius fehlschlägt. | `charts_with_language.csv` | `lyrics.csv`, `failed_lyrics.csv` |
| `unique_urls.py` | Entfernt Duplikate aus der Liste der Chart-URLs, um Redundanzen beim Scraping zu vermeiden. | `chart_urls.txt` | `chart_urls_unique.txt` |

