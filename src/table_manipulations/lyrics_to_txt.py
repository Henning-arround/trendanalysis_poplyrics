import pandas as pd
import os
from pathlib import Path

# CSV-Datei einlesen
df = pd.read_csv("../data/lyrics.csv")

# Zielordner erstellen
output_dir = Path("../data/lyrics/")
output_dir.mkdir(parents=True, exist_ok=True)

# Für jede Zeile eine Textdatei erstellen
for _, row in df.iterrows():
    # Dateiname aus Index-Spalte
    filename = f"{row['Index']}.txt"
    filepath = output_dir / filename
    
    # Songtext in Datei schreiben
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(row['Songtext'])

print(f"Erfolgreich {len(df)} Dateien erstellt in {output_dir}")