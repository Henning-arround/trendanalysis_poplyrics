import pandas as pd
import requests
import musicbrainzngs
import time
import json
import os
from tqdm import tqdm
from datetime import datetime

# Discogs API Configuration
DISCOGS_API_BASE = "https://api.discogs.com"
DISCOGS_USER_AGENT = "PopMusicDataset/1.0 +http://example.com"
DISCOGS_TOKEN = "######################'"

# MusicBrainz Configuration (Fallback)
musicbrainzngs.set_useragent("PopMusicDataset", "1.0", "############")
musicbrainzngs.auth("##########", "###########")

# Progress and Log Files
PROGRESS_FILE = "../../data/popmusic_missing_dates_progress.json"
LOG_FILE = "../../data/popmusic_missing_dates.txt"

def log_message(message, print_also=True):
    """Writes message to log file and optionally to console."""
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
    """Loads saved progress."""
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
    """Saves current progress."""
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

def get_earliest_year_discogs(artist_name, song_title):
    """Searches Discogs for earliest release year."""
    try:
        search_url = f"{DISCOGS_API_BASE}/database/search"
        params = {
            'q': song_title,
            'artist': artist_name,
            'type': 'master',
            'per_page': 100,
            'token': DISCOGS_TOKEN
        }
        headers = {'User-Agent': DISCOGS_USER_AGENT}
        
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
        
        if earliest_master_id:
            try:
                master_url = f"{DISCOGS_API_BASE}/masters/{earliest_master_id}"
                master_response = requests.get(master_url, params={'token': DISCOGS_TOKEN}, headers=headers)
                time.sleep(1.0)
                
                if master_response.status_code == 200:
                    master_data = master_response.json()
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
            msg = f"  → {artist_name} - {song_title}: Discogs gefunden: {earliest_year}"
            tqdm.write(msg)
            log_message(msg, print_also=False)
            return str(earliest_year)
        
        # Fallback: Search normal releases
        params['type'] = 'release'
        response = requests.get(search_url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            earliest_year = None
            
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
                msg = f"  → {artist_name} - {song_title}: Discogs gefunden (Release): {earliest_year}"
                tqdm.write(msg)
                log_message(msg, print_also=False)
                return str(earliest_year)
        
        return None
        
    except Exception as e:
        msg = f"  → Discogs Fehler bei {artist_name} - {song_title}: {e}"
        tqdm.write(msg)
        log_message(msg, print_also=False)
        return None

def get_earliest_year_musicbrainz(artist_name, song_title):
    """Fallback: Searches MusicBrainz for earliest year."""
    try:
        result = musicbrainzngs.search_recordings(
            query=f'artist:"{artist_name}" AND recording:"{song_title}"',
            limit=20
        )
        
        time.sleep(1.0)
        
        if not result.get('recording-list'):
            return None
        
        all_years = []
        
        for recording in result['recording-list'][:10]:
            recording_id = recording.get('id')
            
            try:
                detailed = musicbrainzngs.get_recording_by_id(
                    recording_id,
                    includes=['work-rels', 'releases']
                )
                
                if 'recording' in detailed:
                    rec = detailed['recording']
                    
                    # Check work relationships for date
                    if 'work-relation-list' in rec:
                        for relation in rec['work-relation-list']:
                            if relation.get('type') == 'performance' and 'begin' in relation:
                                date_str = relation['begin']
                                if date_str:
                                    year_str = date_str.split('-')[0]
                                    if year_str.isdigit():
                                        all_years.append(int(year_str))
                    
                    # Check release dates
                    if 'release-list' in rec:
                        for release in rec['release-list']:
                            if 'date' in release:
                                date_str = release['date']
                                year_str = date_str.split('-')[0]
                                if year_str.isdigit():
                                    all_years.append(int(year_str))
                
                time.sleep(1.0)
                
            except Exception:
                time.sleep(1.0)
                continue
        
        if all_years:
            earliest_year = min(all_years)
            msg = f"  → {artist_name} - {song_title}: MusicBrainz gefunden: {earliest_year}"
            tqdm.write(msg)
            log_message(msg, print_also=False)
            return str(earliest_year)
        
        return None
        
    except Exception as e:
        msg = f"  → MusicBrainz Fehler bei {artist_name} - {song_title}: {e}"
        tqdm.write(msg)
        log_message(msg, print_also=False)
        time.sleep(1.0)
        return None

def get_earliest_year(artist_name, song_title):
    """Main function: Try Discogs first, then MusicBrainz fallback."""
    discogs_year = get_earliest_year_discogs(artist_name, song_title)
    
    if discogs_year:
        return discogs_year
    
    msg = f"  → Discogs keine Daten, versuche MusicBrainz..."
    tqdm.write(msg)
    log_message(msg, print_also=False)
    
    mb_year = get_earliest_year_musicbrainz(artist_name, song_title)
    
    return mb_year

def main():
    log_message("="*60)
    log_message("ADD MISSING DATES - FULL DATASET - START")
    log_message("="*60 + "\n")
    
    input_path = "../../data/dataset_popmusic_v3.csv"
    log_message(f"Lade CSV: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
        log_message(f"✓ CSV geladen: {len(df)} Zeilen\n")
    except Exception as e:
        log_message(f"✗ Fehler beim Laden der CSV: {e}")
        return
    
    progress = load_progress()
    last_index = progress.get('last_index', -1)
    processed_indices = set(progress.get('processed_indices', []))
    saved_results = progress.get('results', {})
    
    # Load saved results
    for idx_str, year in saved_results.items():
        idx = int(idx_str)
        if idx in df.index and year:
            df.at[idx, 'Jahr'] = year
    
    mask = df['Jahr'] == 'k.A.'
    rows_to_process = df[mask]
    rows_to_process = rows_to_process[~rows_to_process.index.isin(processed_indices)]
    
    if last_index >= 0:
        log_message(f"✓ Fortsetzen ab Index {last_index + 1}")
        log_message(f"  Bereits verarbeitet: {len(processed_indices)} Songs")
        log_message(f"  Verbleibend: {len(rows_to_process)} Songs\n")
    
    log_message(f"Gefunden: {len(rows_to_process)} Zeilen mit 'k.A.' zu verarbeiten\n")
    log_message("="*60)
    log_message("Starte Suche (Discogs → MusicBrainz Fallback)...")
    log_message("="*60 + "\n")
    
    success_count = len([v for v in saved_results.values() if v])
    fail_count = len([v for v in saved_results.values() if not v])
    
    idx = last_index
    
    try:
        for idx, row in tqdm(rows_to_process.iterrows(), 
                             total=len(rows_to_process), 
                             desc="Verarbeite Songs", 
                             unit="song"):
            
            artist = row['Künstler']
            title = row['Titel']
            
            year = get_earliest_year(artist, title)
            
            if year:
                df.at[idx, 'Jahr'] = year
                success_count += 1
                msg = f"✓ {artist} - {title}: {year}"
                tqdm.write(msg)
                log_message(msg, print_also=False)
            else:
                fail_count += 1
                msg = f"✗ {artist} - {title}: Nicht gefunden"
                tqdm.write(msg)
                log_message(msg, print_also=False)
            
            processed_indices.add(int(idx))
            saved_results[str(idx)] = year
            
            if len(processed_indices) % 10 == 0:
                save_progress(int(idx), list(processed_indices), saved_results)
                temp_output = "../../data/dataset_popmusic_v4_temp.csv"
                df.to_csv(temp_output, index=False, quoting=1)
    
    except KeyboardInterrupt:
        log_message("\n\n⚠ Unterbrochen durch Benutzer!")
        log_message("Speichere Fortschritt...")
        save_progress(int(idx), list(processed_indices), saved_results)
        df.to_csv("../../data/dataset_popmusic_v4_temp.csv", index=False, quoting=1)
        log_message("✓ Fortschritt gespeichert. Beim nächsten Start wird hier fortgesetzt.")
        return
    
    except Exception as e:
        log_message(f"\n\n✗ Fehler aufgetreten: {e}")
        log_message("Speichere Fortschritt...")
        save_progress(int(idx), list(processed_indices), saved_results)
        df.to_csv("../../data/dataset_popmusic_v4_temp.csv", index=False, quoting=1)
        log_message("✓ Fortschritt gespeichert.")
        raise
    
    # Remove rows with empty Jahr
    initial_count = len(df)
    df = df[df['Jahr'] != 'k.A.']
    df = df[df['Jahr'].notna()]
    df = df[df['Jahr'] != '']
    removed_count = initial_count - len(df)
    log_message(f"\n✓ {removed_count} Zeilen mit leerem Jahr entfernt")
    
    # Remove specified columns
    columns_to_remove = ['Erscheinungsjahr Genius', 'Release Date external', 'Release Date Genius']
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)
        log_message(f"✓ Spalten entfernt: {', '.join(existing_columns)}")
    
    output_path = "../../data/dataset_popmusic_v4.csv"
    df.to_csv(output_path, index=False, quoting=1)
    
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        log_message("✓ Progress-Datei gelöscht (Verarbeitung abgeschlossen)")
    
    temp_file = "../../data/dataset_popmusic_v4_temp.csv"
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    log_message("\n" + "="*60)
    log_message("ZUSAMMENFASSUNG")
    log_message("="*60)
    log_message(f"✓ Neue CSV erstellt: {output_path}")
    log_message(f"  Zeilen im Output: {len(df)}")
    log_message(f"  Erfolgreich gefunden: {success_count}")
    log_message(f"  Nicht gefunden: {fail_count}")
    log_message(f"  Zeilen entfernt: {removed_count}")
    if (success_count + fail_count) > 0:
        log_message(f"  Erfolgsrate: {(success_count / (success_count + fail_count) * 100):.1f}%")
    log_message("="*60)
    log_message("ADD MISSING DATES - ABGESCHLOSSEN")
    log_message("="*60 + "\n")

if __name__ == "__main__":
    main()
