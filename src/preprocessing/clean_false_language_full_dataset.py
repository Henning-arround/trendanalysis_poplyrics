import pandas as pd
import os
from pathlib import Path

# CSV laden
df = pd.read_csv("../../data/dataset_popmusic_v6.csv")

# Zeilen identifizieren, die gelöscht werden sollen
to_delete = df[~df["Sprache Lyrics"].isin(["en", "de"])]

# Textdateien löschen für die zu entfernenden Zeilen
lyrics_dir = Path("../../data/dataset_popmusic_lyrics")
deleted_count = 0

for index in to_delete["Index"]:
    lyrics_file = lyrics_dir / f"{index}.txt"
    if lyrics_file.exists():
        lyrics_file.unlink()
        deleted_count += 1
        print(f"Gelöscht: {lyrics_file}")

print(f"\nAnzahl gelöschter Textdateien: {deleted_count}")

# Nur Zeilen mit "en" oder "de" behalten
df_filtered = df[df["Sprache Lyrics"].isin(["en", "de"])]

# Als v7 speichern
df_filtered.to_csv("../../data/dataset_popmusic_v7.csv", index=False, quoting=1)

print(f"\nOriginal Zeilen: {len(df)}")
print(f"Gefilterte Zeilen: {len(df_filtered)}")
print(f"Entfernte Zeilen: {len(df) - len(df_filtered)}")
print(f"\nGespeichert als: dataset_popmusic_v7.csv")
