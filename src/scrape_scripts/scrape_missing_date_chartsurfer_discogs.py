import pandas as pd
import requests
import musicbrainzngs
import time
import json
import os
from tqdm import tqdm
from datetime import datetime

# Discogs API Base URL
DISCOGS_API_BASE = "https://api.discogs.com"
DISCOGS_USER_AGENT = "ChartsurferDataset/1.0 +http://example.com"

# Discogs Personal Access Token
DISCOGS_TOKEN = "#####################"

# MusicBrainz konfigurieren (Fallback)
musicbrainzngs.set_useragent(
    "ChartsurferDataset",
    "1.0",
    "Henning101@gmx.de"
)
musicbrainzngs.auth("Kaboom1998", "J8s%;]cMc:Rce*F")

# Progress- und Log-Dateien
PROGRESS_FILE = "../../data/chartsurfer_missing_dates_progress.json"
LOG_FILE = "../../data/chartsurfer_missing_dates_discogs.txt"

def log_message(message, print_also=True):
    """Schreibt eine Nachricht ins Log-File und optional auf die Konsole."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"
    
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"✗ Fehler beim Schreiben ins Log: {e}")
    
    if print_also:
        print(message)

def load_progress():
    """Lädt den gespeicherten Fortschritt."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
                log_message(f"✓ Fortschritt geladen: Letzter Index {progress.get('last_index', -1)}")
                return progress
        except Exception as e:
            log_message(f"✗ Fehler beim Laden des Fortschritts: {e}")
    return {'last_index': -1, 'processed_indices': [], 'results': {}}

def save_progress(last_index, processed_indices, results):
    """Speichert den aktuellen Fortschritt."""
    progress = {
        'last_index': last_index,
        'processed_indices': processed_indices,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        log_message(f"✗ Fehler beim Speichern des Fortschritts: {e}")

def get_earliest_date_discogs(artist_name, song_title):
    """Sucht bei Discogs nach dem frühesten Release-Datum."""
    try:
        # Suche nach Releases
        search_url = f"{DISCOGS_API_BASE}/database/search"
        params = {
            'q': song_title,
            'artist': artist_name,
            'type': 'master',
            'per_page': 100,
            'token': DISCOGS_TOKEN
        }
        headers = {
            'User-Agent': DISCOGS_USER_AGENT
        }
        
        response = requests.get(search_url, params=params, headers=headers)
        
        if response.status_code != 200:
            msg = f"  → Discogs API Error: {response.status_code}"
            tqdm.write(msg)
            log_message(msg, print_also=False)
            return None
        
        data = response.json()
        results = data.get('results', [])
        
        earliest_year = None
        earliest_master_id = None
        
        # Durchsuche alle Master-Releases
        for result in results:
            year = result.get('year')
            if year:
                try:
                    year = int(year)
                    if earliest_year is None or year < earliest_year:
                        earliest_year = year
                        earliest_master_id = result.get('id')
                except (ValueError, TypeError):
                    continue
        
        # Wenn ein Master gefunden wurde, hole Details für genaueres Datum
        if earliest_master_id:
            try:
                master_url = f"{DISCOGS_API_BASE}/masters/{earliest_master_id}"
                master_response = requests.get(master_url, params={'token': DISCOGS_TOKEN}, headers=headers)
                time.sleep(1.0)
                
                if master_response.status_code == 200:
                    master_data = master_response.json()
                    # Master-Releases haben oft ein 'year' Feld mit dem Original-Release
                    if 'year' in master_data:
                        master_year = master_data['year']
                        if master_year and int(master_year) < earliest_year:
                            earliest_year = int(master_year)
            except Exception as e:
                msg = f"  → Fehler beim Abrufen von Master-Details: {e}"
                tqdm.write(msg)
                log_message(msg, print_also=False)
        
        time.sleep(1.0)
        
        if earliest_year:
            date_str = str(earliest_year)
            msg = f"  → {artist_name} - {song_title}: Discogs gefunden: {date_str}"
            tqdm.write(msg)
            log_message(msg, print_also=False)
            return date_str
        
        # Fallback: Suche nach normalen Releases (nicht Master)
        params['type'] = 'release'
        params['per_page'] = 100
        response = requests.get(search_url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            earliest_year = None
            
            # Durchsuche ALLE Release-Ergebnisse für das früheste Jahr
            for result in results:
                year = result.get('year')
                if year:
                    try:
                        year = int(year)
                        if earliest_year is None or year < earliest_year:
                            earliest_year = year
                    except (ValueError, TypeError):
                        continue
            
            time.sleep(1.0)
            
            if earliest_year:
                date_str = str(earliest_year)
                msg = f"  → {artist_name} - {song_title}: Discogs gefunden (Release): {date_str}"
                tqdm.write(msg)
                log_message(msg, print_also=False)
                return date_str
        
        return None
        
    except Exception as e:
        msg = f"  → Discogs Fehler bei {artist_name} - {song_title}: {e}"
        tqdm.write(msg)
        log_message(msg, print_also=False)
        return None

def get_earliest_date_musicbrainz(artist_name, song_title):
    """Fallback: Sucht bei MusicBrainz nach dem frühesten Datum."""
    try:
        result = musicbrainzngs.search_recordings(
            query=f'artist:"{artist_name}" AND recording:"{song_title}"',
            limit=20
        )
        
        time.sleep(1.0)  # Rate limit: 1 req/sec
        
        if not result.get('recording-list'):
            return None
        
        relationship_dates = []
        release_dates = []
        
        for i, recording in enumerate(result['recording-list'][:10]):
            recording_id = recording.get('id')
            
            try:
                detailed = musicbrainzngs.get_recording_by_id(
                    recording_id,
                    includes=['work-rels', 'artist-rels', 'releases']
                )
                
                if 'recording' in detailed:
                    rec = detailed['recording']
                    
                    # Suche nach "recording of" Relationship mit Datum
                    if 'work-relation-list' in rec:
                        for relation in rec['work-relation-list']:
                            if relation.get('type') == 'performance':
                                if 'begin' in relation:
                                    date_str = relation['begin']
                                    if date_str:
                                        year_str = date_str.split('-')[0]
                                        if year_str.isdigit():
                                            relationship_dates.append((int(year_str), date_str))
                    
                    # Sammle auch Release-Daten
                    if 'release-list' in rec:
                        for release in rec['release-list']:
                            if 'date' in release:
                                date_str = release['date']
                                year_str = date_str.split('-')[0]
                                if year_str.isdigit():
                                    release_dates.append((int(year_str), date_str))
                
                time.sleep(1.0)
                
            except Exception as e:
                time.sleep(1.0)
                continue
        
        if relationship_dates:
            relationship_dates.sort()
            earliest_year, earliest_date = relationship_dates[0]
            msg = f"  → {artist_name} - {song_title}: MusicBrainz gefunden (Relationships): {earliest_date}"
            tqdm.write(msg)
            log_message(msg, print_also=False)
            return earliest_date
        elif release_dates:
            release_dates.sort()
            earliest_year, earliest_date = release_dates[0]
            msg = f"  → {artist_name} - {song_title}: MusicBrainz gefunden (Releases): {earliest_date}"
            tqdm.write(msg)
            log_message(msg, print_also=False)
            return earliest_date
        
        return None
        
    except Exception as e:
        msg = f"  → MusicBrainz Fehler bei {artist_name} - {song_title}: {e}"
        tqdm.write(msg)
        log_message(msg, print_also=False)
        time.sleep(1.0)
        return None

def get_earliest_release_date(artist_name, song_title):
    """Hauptfunktion: Versucht zuerst Discogs, dann MusicBrainz als Fallback."""
    
    # 1. Versuche Discogs (schneller, bessere Rate Limits)
    discogs_date = get_earliest_date_discogs(artist_name, song_title)
    
    if discogs_date:
        return discogs_date, "Discogs"
    
    # 2. Fallback auf MusicBrainz
    msg = f"  → Discogs keine Daten, versuche MusicBrainz..."
    tqdm.write(msg)
    log_message(msg, print_also=False)
    
    mb_date = get_earliest_date_musicbrainz(artist_name, song_title)
    
    if mb_date:
        return mb_date, "MusicBrainz"
    
    return None, None

def main():
    log_message("="*60)
    log_message("SCRAPE MISSING DATES (DISCOGS + MUSICBRAINZ) - START")
    log_message("="*60 + "\n")
    
    # Lade CSV
    input_path = "../../data/charts_chartsurfer_1954_1977_v4.csv"
    log_message(f"Lade CSV: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
        log_message(f"✓ CSV geladen: {len(df)} Zeilen\n")
    except Exception as e:
        log_message(f"✗ Fehler beim Laden der CSV: {e}")
        return
    
    # Lade Fortschritt
    progress = load_progress()
    last_index = progress.get('last_index', -1)
    processed_indices = set(progress.get('processed_indices', []))
    saved_results = progress.get('results', {})
    
    # Erstelle neue Spalten falls nicht vorhanden
    if 'MusicBrainz Release Date' not in df.columns:
        df['MusicBrainz Release Date'] = None
    
    if 'Data Source' not in df.columns:
        df['Data Source'] = None
    
    # Lade bereits gespeicherte Ergebnisse
    for idx_str, data in saved_results.items():
        idx = int(idx_str)
        if idx in df.index and isinstance(data, dict):
            df.at[idx, 'MusicBrainz Release Date'] = data.get('date')
            df.at[idx, 'Data Source'] = data.get('source')
    
    mask = df['Release Date Genius'] == '1 viewer'
    rows_to_process = df[mask]
    rows_to_process = rows_to_process[~rows_to_process.index.isin(processed_indices)]
    
    if last_index >= 0:
        log_message(f"✓ Fortsetzen ab Index {last_index + 1}")
        log_message(f"  Bereits verarbeitet: {len(processed_indices)} Songs")
        log_message(f"  Verbleibend: {len(rows_to_process)} Songs\n")
    
    log_message(f"Gefunden: {len(rows_to_process)} Zeilen zu verarbeiten\n")
    log_message("="*60)
    log_message("Starte Suche (Discogs → MusicBrainz Fallback)...")
    log_message("="*60 + "\n")
    
    success_count = len([v for v in saved_results.values() if isinstance(v, dict) and v.get('date')])
    fail_count = len([v for v in saved_results.values() if isinstance(v, dict) and not v.get('date')])
    discogs_count = len([v for v in saved_results.values() if isinstance(v, dict) and v.get('source') == 'Discogs'])
    mb_count = len([v for v in saved_results.values() if isinstance(v, dict) and v.get('source') == 'MusicBrainz'])
    
    idx = last_index  # Initialize idx for exception handling
    
    try:
        for idx, row in tqdm(rows_to_process.iterrows(), 
                             total=len(rows_to_process), 
                             desc="Verarbeite Songs", 
                             unit="song",
                             initial=0):
            
            artist = row['Künstler']
            title = row['Titel']
            
            release_date, source = get_earliest_release_date(artist, title)
            
            if release_date:
                df.at[idx, 'MusicBrainz Release Date'] = release_date
                df.at[idx, 'Data Source'] = source
                success_count += 1
                if source == "Discogs":
                    discogs_count += 1
                else:
                    mb_count += 1
                msg = f"✓ {artist} - {title}: {release_date} [{source}]"
                tqdm.write(msg)
                log_message(msg, print_also=False)
            else:
                fail_count += 1
                msg = f"✗ {artist} - {title}: Nicht gefunden"
                tqdm.write(msg)
                log_message(msg, print_also=False)
            
            processed_indices.add(int(idx))
            saved_results[str(idx)] = {'date': release_date, 'source': source}
            
            if len(processed_indices) % 10 == 0:
                save_progress(int(idx), list(processed_indices), saved_results)
                temp_output = "../../data/charts_chartsurfer_1954_1977_v5_temp.csv"
                df.to_csv(temp_output, index=False, quoting=1)
    
    except KeyboardInterrupt:
        log_message("\n\n⚠ Unterbrochen durch Benutzer!")
        log_message("Speichere Fortschritt...")
        save_progress(int(idx), list(processed_indices), saved_results)
        df.to_csv("../../data/charts_chartsurfer_1954_1977_v5_temp.csv", index=False, quoting=1)
        log_message("✓ Fortschritt gespeichert. Beim nächsten Start wird hier fortgesetzt.")
        return
    
    except Exception as e:
        log_message(f"\n\n✗ Fehler aufgetreten: {e}")
        log_message("Speichere Fortschritt...")
        save_progress(int(idx), list(processed_indices), saved_results)
        df.to_csv("../../data/charts_chartsurfer_1954_1977_v5_temp.csv", index=False, quoting=1)
        log_message("✓ Fortschritt gespeichert.")
        raise
    
    output_path = "../../data/charts_chartsurfer_1954_1977_v5.csv"
    df.to_csv(output_path, index=False, quoting=1)
    
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        log_message("✓ Progress-Datei gelöscht (Verarbeitung abgeschlossen)")
    
    temp_file = "../../data/charts_chartsurfer_1954_1977_v5_temp.csv"
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    log_message("\n" + "="*60)
    log_message("ZUSAMMENFASSUNG")
    log_message("="*60)
    log_message(f"✓ Neue CSV erstellt: {output_path}")
    log_message(f"  Zeilen gesamt: {len(df)}")
    log_message(f"  Zeilen mit '1 viewer': {len(df[mask])}")
    log_message(f"  Erfolgreich gefunden: {success_count}")
    log_message(f"    - Via Discogs: {discogs_count}")
    log_message(f"    - Via MusicBrainz: {mb_count}")
    log_message(f"  Nicht gefunden: {fail_count}")
    log_message(f"  Erfolgsrate: {(success_count / (success_count + fail_count) * 100):.1f}%" if (success_count + fail_count) > 0 else "N/A")
    log_message("="*60)
    log_message("SCRAPE MISSING DATES - ABGESCHLOSSEN")
    log_message("="*60 + "\n")

if __name__ == "__main__":
    main()