import os
import csv
import json
from datetime import datetime

def prepare_bertopic_json():
    # Pfade relativ zu diesem Skript
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.normpath(os.path.join(base_dir, "../../data/dataset_popmusic_v8.csv"))
    interp_dir = os.path.normpath(os.path.join(base_dir, "../../data/dataset_popmusic_interpretations_revisited"))
    out_dir = os.path.normpath(os.path.join(base_dir, "../../data"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset_popmusic_v8.json")

    # Lese alle Interpretationsdateien und baue eine Zuordnung index -> text
    interp_map = {}
    if os.path.isdir(interp_dir):
        for fn in os.listdir(interp_dir):
            if not fn.lower().endswith(".txt"):
                continue
            full = os.path.join(interp_dir, fn)
            # Erwartetes Muster: <index>_interpretation.txt (Index kann Unterstrich enthalten)
            name = fn[:-4]  # ohne .txt
            if name.endswith("_interpretation"):
                key = name[: -len("_interpretation")]
            else:
                key = name
            try:
                with open(full, "r", encoding="utf-8") as f:
                    interp_map[key] = f.read().strip()
            except Exception:
                interp_map[key] = None

    # Hilfsfunktionen zum Finden von Spaltennamen unabhängig von Groß/Kleinschreibung
    def find_key_case_insensitive(keys, candidates):
        lower_map = {k.lower(): k for k in keys}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    total = 0
    found_interp = 0
    data = []  # Liste für alle Objekte
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames or []
        # finde Jahr-Spalte und Index-Spalte (fallunabhängig)
        year_key = find_key_case_insensitive(headers, ["Jahr", "jahr", "Year", "year"])
        index_key = find_key_case_insensitive(headers, ["Index", "index", "ID", "Id"])
        for row in reader:
            total += 1
            obj = {}
            # kopiere alle Spalten außer Jahr (wird durch timestamp ersetzt)
            for k, v in row.items():
                if year_key and k == year_key:
                    continue
                obj[k] = v
            # timestamp erzeugen aus Jahr
            year_val = row.get(year_key) if year_key else None
            ts = None
            if year_val and year_val.strip():
                try:
                    yint = int(year_val.strip())
                    # Normiertes ISO-Format: YYYY-01-01T00:00:00Z
                    ts = f"{yint:04d}-01-01T00:00:00Z"
                except Exception:
                    ts = None
            obj["timestamp"] = ts
            # interpretation hinzufügen
            idx_val = row.get(index_key) if index_key else None
            interp_text = None
            if idx_val:
                interp_text = interp_map.get(idx_val)
                if interp_text is None:
                    # letzter Versuch: manchmal index ohne suffix oder mit anderen kleinschreibungen
                    # Suche keys case-insensitive
                    for k in interp_map.keys():
                        if k.lower() == idx_val.lower():
                            interp_text = interp_map[k]
                            break
            if interp_text:
                found_interp += 1
            obj["text"] = interp_text
            data.append(obj)  # Objekt zur Liste hinzufügen
    # Schreibe die gesamte Liste als JSON
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2)

    print(f"Gesamtzeilen: {total}, gefundene Interpretationen: {found_interp}, Ausgabe: {out_path}")

def prepare_bertopic_lyrics_json():
    # Pfade relativ zu diesem Skript
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.normpath(os.path.join(base_dir, "../../data/dataset_popmusic_v8.csv"))
    lyrics_dir = os.path.normpath(os.path.join(base_dir, "../../data/dataset_popmusic_lyrics"))
    out_dir = os.path.normpath(os.path.join(base_dir, "../../data"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset_popmusic_v8_lyrics.json")

    # Lese alle Lyricsdateien und baue eine Zuordnung index -> text
    lyrics_map = {}
    if os.path.isdir(lyrics_dir):
        for fn in os.listdir(lyrics_dir):
            if not fn.lower().endswith(".txt"):
                continue
            full = os.path.join(lyrics_dir, fn)
            # Erwartetes Muster: "2.txt" oder "34905_chartsurfer.txt"
            name = fn[:-4]  # ohne .txt
            
            # Versuche Index zu extrahieren (alles vor dem ersten _ oder .)
            if "_" in name:
                key = name.split("_")[0]
            else:
                key = name
            
            try:
                with open(full, "r", encoding="utf-8") as f:
                    lyrics_map[key] = f.read().strip()
            except Exception:
                lyrics_map[key] = None

    # Hilfsfunktionen zum Finden von Spaltennamen unabhängig von Groß/Kleinschreibung
    def find_key_case_insensitive(keys, candidates):
        lower_map = {k.lower(): k for k in keys}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    total = 0
    found_lyrics = 0
    data = []  # Liste für alle Objekte
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames or []
        # finde Jahr-Spalte und Index-Spalte (fallunabhängig)
        year_key = find_key_case_insensitive(headers, ["Jahr", "jahr", "Year", "year"])
        index_key = find_key_case_insensitive(headers, ["Index", "index", "ID", "Id"])
        for row in reader:
            total += 1
            obj = {}
            # kopiere alle Spalten außer Jahr (wird durch timestamp ersetzt)
            for k, v in row.items():
                if year_key and k == year_key:
                    continue
                obj[k] = v
            # timestamp erzeugen aus Jahr
            year_val = row.get(year_key) if year_key else None
            ts = None
            if year_val and year_val.strip():
                try:
                    yint = int(year_val.strip())
                    # Normiertes ISO-Format: YYYY-01-01T00:00:00Z
                    ts = f"{yint:04d}-01-01T00:00:00Z"
                except Exception:
                    ts = None
            obj["timestamp"] = ts
            
            # Lyrics hinzufügen
            idx_val = row.get(index_key) if index_key else None
            lyrics_text = None
            if idx_val:
                lyrics_text = lyrics_map.get(idx_val)
                if lyrics_text is None:
                    # Fallback case-insensitive
                    for k in lyrics_map.keys():
                        if k.lower() == idx_val.lower():
                            lyrics_text = lyrics_map[k]
                            break
            
            if lyrics_text:
                found_lyrics += 1
            
            obj["Songtext"] = lyrics_text
            data.append(obj)

    # Schreibe die gesamte Liste als JSON
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2)

    print(f"Gesamtzeilen: {total}, gefundene Lyrics: {found_lyrics}, Ausgabe: {out_path}")

if __name__ == "__main__":
    prepare_bertopic_json()
    prepare_bertopic_lyrics_json()