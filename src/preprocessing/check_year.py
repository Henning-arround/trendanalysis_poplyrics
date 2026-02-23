import pandas as pd
import csv

# CSV-Datei einlesen
df = pd.read_csv("../../data/dataset_popmusic_v7.csv")

# Zeilen filtern: nur Jahre >= 1950 und <= 2024 behalten
df_filtered = df[(df["Jahr"] >= 1950) & (df["Jahr"] <= 2024)]

# Gefilterte Daten als v8 speichern mit quoting=1 (QUOTE_ALL)
df_filtered.to_csv("../../data/dataset_popmusic_v8.csv", index=False, quoting=csv.QUOTE_ALL)

print(f"Original: {len(df)} Zeilen")
print(f"Nach Filterung: {len(df_filtered)} Zeilen")
print(f"Entfernt: {len(df) - len(df_filtered)} Zeilen")
