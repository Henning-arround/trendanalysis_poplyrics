import pandas as pd
from collections import Counter
import re
import csv

# CSV-Datei einlesen
df = pd.read_csv("../../data/dataset_popmusic.csv")

print("=" * 80)
print("ANALYSE DER GENIUS API URLS")
print("=" * 80)
print()

# Gruppiere nach Künstler
grouped = df.groupby('Künstler')

results = []
rows_to_keep = []

for artist, group in grouped:
    print(f"\n{'=' * 80}")
    print(f"Künstler: {artist}")
    print(f"Anzahl Songs: {len(group)}")
    print("-" * 80)
    
    # Extrahiere URL-Künstlernamen
    url_artists = []
    
    for idx, row in group.iterrows():
        url = row['Quelle']
        if pd.notna(url) and isinstance(url, str):
            # Entferne https://genius.com/ und extrahiere den ersten Teil
            match = re.search(r'genius\.com/([^-/]+)', url)
            if match:
                url_artist = match.group(1)
                url_artists.append((url_artist, row['Titel'], url, idx))
            else:
                # URL hat nicht das erwartete Format, behalte sie trotzdem
                rows_to_keep.append(idx)
        else:
            # Keine URL vorhanden, behalte die Zeile
            rows_to_keep.append(idx)
    
    if url_artists:
        # Zähle die verschiedenen URL-Künstlernamen
        artist_counts = Counter([ua[0] for ua in url_artists])
        
        print(f"\nGefundene URL-Künstlernamen:")
        for url_artist, count in artist_counts.most_common():
            percentage = (count / len(url_artists)) * 100
            print(f"  - '{url_artist}': {count} mal ({percentage:.1f}%)")
        
        most_common = artist_counts.most_common(1)[0][0]
        most_common_count = artist_counts.most_common(1)[0][1]
        most_common_percentage = (most_common_count / len(url_artists)) * 100
        
        # Prüfe Filterkriterien
        should_filter = False
        if len(artist_counts) > 1:
            if len(group) < 10:
                # Unter 10 Songs: häufigster muss mindestens 51% haben
                if most_common_percentage >= 51:
                    should_filter = True
                    print(f"\n✓ Filterung aktiv: Künstler hat < 10 Songs und häufigster URL-Name hat {most_common_percentage:.1f}% (≥ 51%)")
                else:
                    print(f"\n✗ Keine Filterung: Künstler hat < 10 Songs aber häufigster URL-Name hat nur {most_common_percentage:.1f}% (< 51%)")
            else:
                # 10 oder mehr Songs: immer filtern
                should_filter = True
                print(f"\n✓ Filterung aktiv: Künstler hat ≥ 10 Songs")
        
        if should_filter:
            print(f"  Behalte nur URLs mit: '{most_common}'")
            removed_count = 0
            
            for url_artist, title, url, idx in url_artists:
                if url_artist == most_common:
                    rows_to_keep.append(idx)
                else:
                    removed_count += 1
                    print(f"    ✗ Entfernt: '{title}' (URL-Künstler: '{url_artist}')")
            
            print(f"  → {removed_count} Zeilen entfernt, {most_common_count} behalten")
            
            results.append({
                'Künstler': artist,
                'Anzahl_Songs': len(group),
                'Verschiedene_URL_Künstler': len(artist_counts),
                'Häufigster_URL_Künstler': most_common,
                'Prozent_Häufigster': most_common_percentage,
                'Entfernt': removed_count,
                'Alle_URL_Künstler': dict(artist_counts)
            })
        else:
            # Keine Filterung, behalte alle
            for url_artist, title, url, idx in url_artists:
                rows_to_keep.append(idx)
            print(f"  → Alle {len(url_artists)} Zeilen behalten")

# Erstelle gefilterten DataFrame
df_filtered = df.loc[rows_to_keep].copy()

print("\n" + "=" * 80)
print("ZUSAMMENFASSUNG FILTERUNG")
print("=" * 80)
print(f"\nUrsprüngliche Anzahl Zeilen: {len(df)}")
print(f"Nach URL-Filterung: {len(df_filtered)}")
print(f"Entfernte Zeilen: {len(df) - len(df_filtered)}")

# Entferne Duplikate in der Quellen-Spalte (behalte erstes Vorkommen)
print("\n" + "=" * 80)
print("DUPLIKATE IN QUELLEN-SPALTE")
print("=" * 80)

duplicates_before = df_filtered['Quelle'].duplicated().sum()
print(f"\nAnzahl Duplikate vor Entfernung: {duplicates_before}")

if duplicates_before > 0:
    # Zeige einige Beispiele
    duplicate_sources = df_filtered[df_filtered['Quelle'].duplicated(keep=False)].sort_values('Quelle')
    print(f"\nBeispiele für Duplikate:")
    for source in duplicate_sources['Quelle'].unique()[:5]:
        dup_rows = df_filtered[df_filtered['Quelle'] == source]
        print(f"\n  Quelle: {source}")
        print(f"  Vorkommnisse: {len(dup_rows)}")
        for idx, row in dup_rows.iterrows():
            print(f"    - Künstler: '{row['Künstler']}', Titel: '{row['Titel']}'")

df_filtered = df_filtered.drop_duplicates(subset='Quelle', keep='first')

print(f"\nNach Duplikate-Entfernung: {len(df_filtered)}")
print(f"Entfernte Duplikate: {duplicates_before}")

# Speichere neue CSV
output_path = "../../data/dataset_popmusic_v2.csv"
df_filtered.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

print("\n" + "=" * 80)
print("FINALE STATISTIK")
print("=" * 80)
print(f"\nUrsprüngliche Datei: {len(df)} Zeilen")
print(f"Neue Datei: {len(df_filtered)} Zeilen")
print(f"Gesamt entfernt: {len(df) - len(df_filtered)} Zeilen ({((len(df) - len(df_filtered)) / len(df) * 100):.1f}%)")
print(f"\n✅ Gefilterte Daten gespeichert in: {output_path}")

# Detaillierte Ergebnisse in Textdatei schreiben
with open("../../data/url_analyse_ergebnisse.txt", "w", encoding="utf-8") as f:
    f.write("ANALYSE DER GENIUS API URLS UND FILTERUNG\n")
    f.write("=" * 80 + "\n\n")
    
    if results:
        f.write("KÜNSTLER MIT FILTERUNG:\n\n")
        for r in results:
            f.write(f"Künstler: {r['Künstler']}\n")
            f.write(f"  Anzahl Songs: {r['Anzahl_Songs']}\n")
            f.write(f"  Verschiedene URL-Künstlernamen: {r['Verschiedene_URL_Künstler']}\n")
            f.write(f"  Häufigster: '{r['Häufigster_URL_Künstler']}' ({r['Prozent_Häufigster']:.1f}%)\n")
            f.write(f"  Entfernte Zeilen: {r['Entfernt']}\n")
            f.write(f"  Verteilung: {r['Alle_URL_Künstler']}\n")
            f.write("\n")
    
    f.write("=" * 80 + "\n")
    f.write("STATISTIK:\n\n")
    f.write(f"Künstler mit mehreren URL-Namen: {len(results)}\n")
    f.write(f"Ursprüngliche Zeilen: {len(df)}\n")
    f.write(f"Nach URL-Filterung: {len(df.loc[rows_to_keep])}\n")
    f.write(f"Nach Duplikate-Entfernung: {len(df_filtered)}\n")
    f.write(f"Gesamt entfernt: {len(df) - len(df_filtered)}\n")

print(f"\n✅ Detaillierte Ergebnisse in '../../data/url_analyse_ergebnisse.txt' gespeichert")
