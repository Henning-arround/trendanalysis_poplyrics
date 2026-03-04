# Datenbereinigung

Dieser Ordner enthält alle Skripte, die für das Preprocessing der gesammelten Rohdaten verwendet wurden. 

## Enthaltene Skripte

* **`check_year.py`**
  Überprüft und vereinheitlicht die Jahreszahlen der Songs. Dieser Schritt ist besonders wichtig, um sicherzustellen, dass für das Dynamic Topic Modeling verlässliche und einheitliche zeitliche Metadaten vorliegen.

* **`clean_all_offizielle_charts.py`**
  Ein spezifisches Skript zur Bereinigung der Metadaten und Songtexte, welche aus der Quelle der "Offiziellen Deutschen Charts" stammen, um spezifische Eigenheiten oder Fehler dieser Quelle auszubessern.

* **`clean_false_language_full_dataset.py`**
  Dient der Qualitätskontrolle der Sprachdetektion: Identifiziert und entfernt Einträge aus dem Gesamtdatensatz, bei denen die Sprache der Songtexte potenziell falsch gekennzeichnet wurde oder die sich als Instrumentals entpuppt haben.

* **`clean_full_dataset.py`**
  Das zentrale Bereinigungsskript für den zusammengeführten Hauptdatensatz. Hier passieren klassische Bereinigungsschritte wie das Deduplizieren von Einträgen, das Auflösen von Inkonsistenzen und das allgemeine Aufräumen der zusammengeführten tabellarischen Formate.

* **`clean_lyrics.py`**
  Führt die Textbereinigung direkt auf den rohen Songtexten ("Lyrics") aus. Dies umfasst typischerweise das Entfernen von Meta-Tags in eckigen Klammern (wie `[Chorus]`, `[Verse 1]`), überflüssigen Leerzeichen, ungewollten Sonderzeichen oder HTML-Resten.

* **`csv_to_json.py`**
  Ein Konvertierungsskript, das die finalisierten Datensätze aus dem CSV-Format in das JSON-Format überführt (z. B. zur Generierung von `dataset_popmusic_v8.json`).

* **`extract_german_lyrics.py`**
  Filtert und extrahiert gezielt isolierte deutsche Songtexte aus dem größeren gemischtsprachigen Korpus. Dies war vor allem für die DNB interessant.