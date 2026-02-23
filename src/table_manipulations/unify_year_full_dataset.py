import pandas as pd
import re
import csv

# CSV-Datei einlesen
df = pd.read_csv("../../data/dataset_popmusic_v2.csv")

print("=" * 80)
print("VEREINHEITLICHUNG DER JAHRESANGABEN")
print("=" * 80)
print(f"\nAnzahl Zeilen: {len(df)}")

def extract_year(text):
    """Extrahiert ein 4-stelliges Jahr aus verschiedenen Datumsformaten."""
    if pd.isna(text):
        return None
    
    text_str = str(text).strip()
    
    # Suche nach 4-stelligem Jahr (YYYY)
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text_str)
    if year_match:
        return year_match.group(1)
    
    return None

def needs_year_update(jahr_value):
    """Prüft ob das Jahr aktualisiert werden muss."""
    if pd.isna(jahr_value):
        return True
    
    jahr_str = str(jahr_value).strip().lower()
    
    # Prüfe auf "k.a.", "ka" oder fehlende YYYY
    if jahr_str in ['k.a.', 'ka', 'k.a', '']:
        return True
    
    # Prüfe ob ein valides Jahr vorhanden ist
    if not re.search(r'\b(19\d{2}|20\d{2})\b', jahr_str):
        return True
    
    return False

# Statistiken
stats = {
    'non_chartsurfer_updated': 0,
    'chartsurfer_updated': 0,
    'chartsurfer_from_external': 0,
    'no_year_found': 0
}

print("\nVerarbeite Zeilen...")

for idx, row in df.iterrows():
    index_value = str(row['Index'])
    jahr_current = row['Jahr']
    
    # Prüfe ob Jahr aktualisiert werden muss
    if not needs_year_update(jahr_current):
        continue
    
    # Fall 1: Index endet NICHT mit "_chartsurfer"
    if not index_value.endswith('_chartsurfer'):
        # Nimm Jahr aus "Erscheinungsjahr Genius"
        erscheinungsjahr = row.get('Erscheinungsjahr Genius', None)
        new_year = extract_year(erscheinungsjahr)
        
        if new_year:
            df.at[idx, 'Jahr'] = new_year
            stats['non_chartsurfer_updated'] += 1
            if stats['non_chartsurfer_updated'] <= 5:  # Zeige erste 5 Beispiele
                print(f"\n  Non-Chartsurfer: {row['Künstler']} - {row['Titel']}")
                print(f"    Alt: '{jahr_current}' → Neu: '{new_year}'")
                print(f"    Quelle: Erscheinungsjahr Genius = '{erscheinungsjahr}'")
        else:
            stats['no_year_found'] += 1
    
    # Fall 2: Index endet mit "_chartsurfer"
    else:
        # Zuerst: "Release Date Genius"
        release_date_genius = row.get('Release Date Genius', None)
        
        # Prüfe ob es "1 viewer" oder ähnliches ist
        if pd.notna(release_date_genius) and '1 viewer' not in str(release_date_genius).lower():
            new_year = extract_year(release_date_genius)
            
            if new_year:
                df.at[idx, 'Jahr'] = new_year
                stats['chartsurfer_updated'] += 1
                if stats['chartsurfer_updated'] <= 5:  # Zeige erste 5 Beispiele
                    print(f"\n  Chartsurfer: {row['Künstler']} - {row['Titel']}")
                    print(f"    Alt: '{jahr_current}' → Neu: '{new_year}'")
                    print(f"    Quelle: Release Date Genius = '{release_date_genius}'")
                continue
        
        # Falls nicht gefunden, nimm "Release Date external"
        release_date_external = row.get('Release Date external', None)
        new_year = extract_year(release_date_external)
        
        if new_year:
            df.at[idx, 'Jahr'] = new_year
            stats['chartsurfer_updated'] += 1
            stats['chartsurfer_from_external'] += 1
            if stats['chartsurfer_from_external'] <= 5:  # Zeige erste 5 Beispiele
                print(f"\n  Chartsurfer (external): {row['Künstler']} - {row['Titel']}")
                print(f"    Alt: '{jahr_current}' → Neu: '{new_year}'")
                print(f"    Quelle: Release Date external = '{release_date_external}'")
        else:
            stats['no_year_found'] += 1

# Speichere neue CSV
output_path = "../../data/dataset_popmusic_v3.csv"
df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

print("\n" + "=" * 80)
print("STATISTIK")
print("=" * 80)
print(f"\nNon-Chartsurfer Einträge aktualisiert: {stats['non_chartsurfer_updated']}")
print(f"Chartsurfer Einträge aktualisiert: {stats['chartsurfer_updated']}")
print(f"  Davon aus 'Release Date external': {stats['chartsurfer_from_external']}")
print(f"Keine Jahresangabe gefunden: {stats['no_year_found']}")
print(f"\nGesamt aktualisiert: {stats['non_chartsurfer_updated'] + stats['chartsurfer_updated']}")

print(f"\n✅ Neue Datei gespeichert: {output_path}")

# Detaillierte Analyse in Textdatei
with open("../../data/jahr_vereinheitlichung_log.txt", "w", encoding="utf-8") as f:
    f.write("VEREINHEITLICHUNG DER JAHRESANGABEN\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Non-Chartsurfer Einträge aktualisiert: {stats['non_chartsurfer_updated']}\n")
    f.write(f"Chartsurfer Einträge aktualisiert: {stats['chartsurfer_updated']}\n")
    f.write(f"  Davon aus 'Release Date external': {stats['chartsurfer_from_external']}\n")
    f.write(f"Keine Jahresangabe gefunden: {stats['no_year_found']}\n")
    f.write(f"\nGesamt aktualisiert: {stats['non_chartsurfer_updated'] + stats['chartsurfer_updated']}\n")

print(f"\n✅ Log-Datei gespeichert: ../../data/jahr_vereinheitlichung_log.txt")
