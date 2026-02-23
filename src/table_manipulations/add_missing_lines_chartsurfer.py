import pandas as pd
import re
from pathlib import Path

def parse_log_file(log_path):
    """
    Parst die Log-Datei und extrahiert SUCCESS-Einträge
    """
    success_entries = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern für SUCCESS-Zeilen
    pattern = r'SUCCESS - Index: (\d+), Artist: (.+?), Title: (.+?), URL: (https://genius\.com/.+?), Release: (.+?), Genres: (.+?)(?=\n|$)'
    
    matches = re.finditer(pattern, content)
    
    for match in matches:
        index = int(match.group(1))
        artist = match.group(2)
        title = match.group(3)
        url = match.group(4)
        release = match.group(5)
        genres = match.group(6)
        
        success_entries.append({
            'Index': index,
            'Artist': artist,
            'Title': title,
            'Quelle Genius': url,
            'Release Date Genius': release,
            'Genre Genius': genres
        })
    
    return pd.DataFrame(success_entries)

def main():
    # Pfade definieren
    base_path = Path(__file__).parent.parent.parent
    log_path = base_path / "data" / "chartsurfer_download_progress.log"
    csv_input_path = base_path / "data" / "charts_chartsurfer_1954_1977_v3.csv"
    csv_output_path = base_path / "data" / "charts_chartsurfer_1954_1977_v4.csv"
    
    print("Lade bestehende CSV...")
    df_existing = pd.read_csv(csv_input_path)
    
    print("Parse Log-Datei...")
    df_log = parse_log_file(log_path)
    
    print(f"Gefundene SUCCESS-Einträge in Log: {len(df_log)}")
    
    # Merge Log mit CSV auf Index
    df_merged = df_existing.merge(df_log, on='Index', how='left', suffixes=('', '_log'))
    
    # Identifiziere Zeilen, wo SUCCESS in Log ist, aber Genius-Daten fehlen
    # Fehlend = Quelle Genius ist leer/NaN UND es gibt Log-Daten
    missing_mask = (
        (df_merged['Quelle Genius'].isna() | (df_merged['Quelle Genius'] == '')) &
        (df_merged['Quelle Genius_log'].notna())
    )
    
    indices_to_update = df_merged[missing_mask]['Index'].tolist()
    print(f"Einträge mit fehlenden Genius-Daten: {len(indices_to_update)}")
    
    if len(indices_to_update) == 0:
        print("Keine fehlenden Einträge gefunden!")
        return
    
    # Aktualisiere die Genius-Spalten für diese Indizes
    for idx in indices_to_update:
        log_data = df_log[df_log['Index'] == idx].iloc[0]
        csv_idx = df_existing[df_existing['Index'] == idx].index[0]
        
        df_existing.at[csv_idx, 'Quelle Genius'] = log_data['Quelle Genius']
        df_existing.at[csv_idx, 'Release Date Genius'] = log_data['Release Date Genius']
        df_existing.at[csv_idx, 'Genre Genius'] = log_data['Genre Genius']
    
    # Speichern mit quoting=1 (QUOTE_ALL)
    print(f"Speichere aktualisierte CSV...")
    df_existing.to_csv(csv_output_path, index=False, quoting=1)
    
    print(f"Fertig! Neue Datei gespeichert: {csv_output_path}")
    print(f"Aktualisierte Zeilen: {len(indices_to_update)}")
    
    # Statistik ausgeben
    print("\nStatistik der aktualisierten Einträge:")
    updated_data = df_log[df_log['Index'].isin(indices_to_update)]
    print(updated_data[['Index', 'Artist', 'Title']].to_string(index=False))

if __name__ == "__main__":
    main()
