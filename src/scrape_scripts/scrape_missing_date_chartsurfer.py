import pandas as pd
import musicbrainzngs
import time
import json
import os
from tqdm import tqdm
from datetime import datetime

# MusicBrainz konfigurieren
musicbrainzngs.set_useragent(
    "ChartsurferDataset",
    "1.0",
    "#############"
)

# MusicBrainz Account Authentifizierung
musicbrainzngs.auth("##########", "###########")
# Progress- und Log-Dateien
PROGRESS_FILE = "../../data/chartsurfer_missing_dates_progress.json"
LOG_FILE = "../../data/chartsurfer_missing_dates.txt"

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

def get_earliest_release_date(artist_name, song_title):
    """Sucht nach dem Recording und findet das Aufnahmedatum aus den Relationships."""
    try:
        # Suche nach Recordings
        result = musicbrainzngs.search_recordings(
            query=f'artist:"{artist_name}" AND recording:"{song_title}"',
            limit=20
        )
        
        if not result.get('recording-list'):
            return None
        
        # Sammle erst alle Daten aus ALLEN Recordings
        relationship_dates = []
        release_dates = []
        
        # Durchsuche die ersten 20 relevantesten Recordings
        for i, recording in enumerate(result['recording-list'][:20]):
            recording_id = recording.get('id')
            
            # Hole detaillierte Informationen MIT Relationships
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
                    
                    # Sammle auch Release-Daten als Fallback
                    if 'release-list' in rec:
                        for release in rec['release-list']:
                            if 'date' in release:
                                date_str = release['date']
                                year_str = date_str.split('-')[0]
                                if year_str.isdigit():
                                    release_dates.append((int(year_str), date_str))
                
                # Kürzeres Rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                msg = f"  → Fehler bei Recording {recording_id}: {e}"
                tqdm.write(msg)
                log_message(msg, print_also=False)
                # Fallback: Versuche Releases aus Suchergebnis
                if 'release-list' in recording:
                    for release in recording['release-list']:
                        if 'date' in release:
                            date_str = release['date']
                            year_str = date_str.split('-')[0]
                            if year_str.isdigit():
                                release_dates.append((int(year_str), date_str))
        
        # Jetzt wähle das früheste Datum aus
        # Priorisiere Relationship-Daten
        if relationship_dates:
            relationship_dates.sort()
            earliest_year, earliest_date = relationship_dates[0]
            msg = f"  → {artist_name} - {song_title}: Gefunden in Relationships: {earliest_date}"
            tqdm.write(msg)
            log_message(msg, print_also=False)
        elif release_dates:
            release_dates.sort()
            earliest_year, earliest_date = release_dates[0]
            msg = f"  → {artist_name} - {song_title}: Gefunden in Releases (Fallback): {earliest_date}"
            tqdm.write(msg)
            log_message(msg, print_also=False)
        else:
            earliest_date = None
        
        return earliest_date
    
    except Exception as e:
        msg = f"✗ Fehler bei {artist_name} - {song_title}: {e}"
        tqdm.write(msg)
        log_message(msg, print_also=False)
        return None

def main():
    # Initialisiere Log-Datei
    log_message("="*60)
    log_message("SCRAPE MISSING DATES - START")
    log_message("="*60)
    
    # CSV einlesen
    csv_path = "../../data/charts_chartsurfer_1954_1977_v4.csv"
    df = pd.read_csv(csv_path)
    
    log_message(f"CSV eingelesen: {len(df)} Zeilen")
    log_message(f"Spalten: {df.columns.tolist()}\n")
    
    # Lade Fortschritt
    progress = load_progress()
    last_index = progress.get('last_index', -1)
    processed_indices = set(progress.get('processed_indices', []))
    saved_results = progress.get('results', {})
    
    # Neue Spalte für MusicBrainz Release Date
    if 'MusicBrainz Release Date' not in df.columns:
        df['MusicBrainz Release Date'] = None
    
    # Lade bereits gespeicherte Ergebnisse
    for idx_str, date in saved_results.items():
        idx = int(idx_str)
        if idx in df.index:
            df.at[idx, 'MusicBrainz Release Date'] = date
    
    # Filtere Zeilen mit "1 viewer" in "Release Date Genius"
    mask = df['Release Date Genius'] == '1 viewer'
    rows_to_process = df[mask]
    
    # Filtere bereits verarbeitete Zeilen aus
    rows_to_process = rows_to_process[~rows_to_process.index.isin(processed_indices)]
    
    if last_index >= 0:
        log_message(f"✓ Fortsetzen ab Index {last_index + 1}")
        log_message(f"  Bereits verarbeitet: {len(processed_indices)} Songs")
        log_message(f"  Verbleibend: {len(rows_to_process)} Songs\n")
    
    log_message(f"Gefunden: {len(rows_to_process)} Zeilen zu verarbeiten\n")
    log_message("="*60)
    log_message("Starte MusicBrainz Suche...")
    log_message("="*60 + "\n")
    
    success_count = len([v for v in saved_results.values() if v is not None])
    fail_count = len([v for v in saved_results.values() if v is None])
    
    # Verarbeite jede Zeile
    try:
        for idx, row in tqdm(rows_to_process.iterrows(), 
                             total=len(rows_to_process), 
                             desc="Verarbeite Songs", 
                             unit="song",
                             initial=0):
            
            artist = row['Künstler']
            title = row['Titel']
            
            # Suche nach Release-Datum
            release_date = get_earliest_release_date(artist, title)
            
            if release_date:
                df.at[idx, 'MusicBrainz Release Date'] = release_date
                success_count += 1
                msg = f"✓ {artist} - {title}: {release_date}"
                tqdm.write(msg)
                log_message(msg, print_also=False)
            else:
                fail_count += 1
                msg = f"✗ {artist} - {title}: Nicht gefunden"
                tqdm.write(msg)
                log_message(msg, print_also=False)
            
            # Aktualisiere Fortschritt
            processed_indices.add(int(idx))
            saved_results[str(idx)] = release_date
            
            # Speichere Fortschritt alle 10 Songs
            if len(processed_indices) % 10 == 0:
                save_progress(int(idx), list(processed_indices), saved_results)
                # Speichere auch Zwischen-CSV
                temp_output = "../../data/charts_chartsurfer_1954_1977_v5_temp.csv"
                df.to_csv(temp_output, index=False, quoting=1)
            
            # Reduziertes Rate limiting
            time.sleep(0.5)
    
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
        save_progress(int(idx) if 'idx' in locals() else last_index, list(processed_indices), saved_results)
        df.to_csv("../../data/charts_chartsurfer_1954_1977_v5_temp.csv", index=False, quoting=1)
        log_message("✓ Fortschritt gespeichert.")
        raise
    
    # Speichere finale CSV
    output_path = "../../data/charts_chartsurfer_1954_1977_v5.csv"
    df.to_csv(output_path, index=False, quoting=1)
    
    # Lösche Progress-Datei nach erfolgreichem Abschluss
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        log_message("✓ Progress-Datei gelöscht (Verarbeitung abgeschlossen)")
    
    # Lösche temporäre CSV
    temp_file = "../../data/charts_chartsurfer_1954_1977_v5_temp.csv"
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # Statistik
    log_message("\n" + "="*60)
    log_message("ZUSAMMENFASSUNG")
    log_message("="*60)
    log_message(f"✓ Neue CSV erstellt: {output_path}")
    log_message(f"  Zeilen gesamt: {len(df)}")
    log_message(f"  Zeilen mit '1 viewer': {len(df[mask])}")
    log_message(f"  Erfolgreich gefunden: {success_count}")
    log_message(f"  Nicht gefunden: {fail_count}")
    log_message(f"  Erfolgsrate: {(success_count / (success_count + fail_count) * 100):.1f}%" if (success_count + fail_count) > 0 else "N/A")
    log_message("="*60)
    log_message("SCRAPE MISSING DATES - ABGESCHLOSSEN")
    log_message("="*60 + "\n")

if __name__ == "__main__":
    main()
