import pandas as pd
import lyricsgenius
import os
import time
import logging
import requests
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re

# ================================
# KONFIGURATION
# ================================
GENIUS_API_TOKEN = "##########################"

# Rate Limiting
SLEEP_BETWEEN_REQUESTS = 0.3
MAX_RETRIES = 10
TIMEOUT = 3600
RATE_LIMIT_WAIT_TIME = 61 * 60  # 61 Minuten in Sekunden

# Pfade
INPUT_CSV = "../../data/charts_with_language_updated_v5.csv"
OUTPUT_CSV = "../../data/charts_offizielle_charts_minimal.csv"
LYRICS_DIR = "../../data/lyrics_offiziellecharts_revisited/"
PROGRESS_LOG_PATH = "../../data/second_run_download_progress.log"

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Progress Logger Setup
progress_logger = logging.getLogger('progress')
progress_logger.setLevel(logging.INFO)
progress_handler = logging.FileHandler(PROGRESS_LOG_PATH, encoding='utf-8')
progress_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
progress_logger.addHandler(progress_handler)
progress_logger.propagate = False

# ================================
# FUNKTIONEN
# ================================
def get_last_processed_index():
    """Liest den zuletzt verarbeiteten Index aus dem Progress Log"""
    progress_log = Path(PROGRESS_LOG_PATH)
    
    if not progress_log.exists():
        print("📝 Kein Progress Log gefunden - starte von vorne")
        return None
    
    try:
        with open(progress_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Suche rückwärts nach dem letzten "Processed Index: XXX" Eintrag
        for line in reversed(lines):
            match = re.search(r'Processed Index: (\d+)', line)
            if match:
                last_index = int(match.group(1))
                print(f"🔄 Letzter verarbeiteter Index gefunden: {last_index}")
                return last_index
        
        print("📝 Kein verarbeiteter Index im Log gefunden - starte von vorne")
        return None
        
    except Exception as e:
        logger.warning(f"Fehler beim Lesen des Progress Logs: {e}")
        return None

def wait_for_rate_limit_reset():
    """Wartet 61 Minuten mit Countdown-Anzeige"""
    wait_seconds = RATE_LIMIT_WAIT_TIME
    end_time = datetime.now() + timedelta(seconds=wait_seconds)
    
    print(f"\n\n{'='*60}")
    print(f"🛑 RATE LIMIT ERREICHT!")
    print(f"{'='*60}")
    print(f"⏰ Wartezeit: {wait_seconds // 60} Minuten ({wait_seconds} Sekunden)")
    print(f"🕐 Aktuell: {datetime.now().strftime('%H:%M:%S')}")
    print(f"🕐 Fortsetzung um: {end_time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")
    
    progress_logger.info(f"Rate Limit erreicht. Warte {wait_seconds // 60} Minuten bis {end_time.strftime('%H:%M:%S')}")
    
    # Countdown mit tqdm
    with tqdm(total=wait_seconds, desc="⏳ Warte auf Rate Limit Reset", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]',
              unit='s', leave=False) as pbar:
        for _ in range(wait_seconds):
            time.sleep(1)
            pbar.update(1)
    
    print(f"\n✅ Wartezeit abgelaufen! Setze Download fort...\n")
    progress_logger.info("Wartezeit abgelaufen. Setze Download fort.")

def setup_genius_client():
    """Initialisiert den Genius Client"""
    genius = lyricsgenius.Genius(
        GENIUS_API_TOKEN,
        sleep_time=SLEEP_BETWEEN_REQUESTS,
        timeout=TIMEOUT,
        retries=MAX_RETRIES,
        remove_section_headers=True,
        skip_non_songs=True,
        verbose=False
    )
    return genius

def load_input_data():
    """Lädt die Input-CSV und filtert nach nicht-leeren 'Quelle Lyrics'"""
    try:
        df = pd.read_csv(INPUT_CSV, sep=',', quotechar='"', escapechar='\\')
        print(f"✅ CSV erfolgreich geladen: {len(df)} Einträge")
        
        # Filtere nur Zeilen mit nicht-leerer "Quelle Lyrics"
        df_filtered = df[df['Quelle Lyrics'].notna() & (df['Quelle Lyrics'].str.strip() != '')]
        print(f"📊 Einträge mit Quelle Lyrics: {len(df_filtered)}")
        
        return df_filtered
    except Exception as e:
        logger.error(f"Fehler beim Laden der CSV: {e}")
        return None

def load_or_create_output_csv():
    """Lädt vorhandene Output CSV oder erstellt eine neue"""
    output_path = Path(OUTPUT_CSV)
    
    if output_path.exists():
        try:
            df = pd.read_csv(OUTPUT_CSV)
            print(f"✅ Output CSV geladen: {len(df)} Einträge")
            return df
        except Exception as e:
            logger.warning(f"Fehler beim Laden der Output CSV: {e}")
    
    # Erstelle neue leere DataFrame
    df = pd.DataFrame(columns=['Index', 'Künstler', 'Titel', 'Jahr', 'Quelle', 'Erscheinungsjahr Genius', 'Genre Genius'])
    print("📝 Neue Output CSV erstellt")
    return df

def save_output_csv(df):
    """Speichert die Output CSV"""
    try:
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False, sep=',', quotechar='"', quoting=1)
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Output CSV: {e}")

def scrape_genius_metadata(url):
    """Scraped Release-Jahr und Genre-Tags von der Genius-Seite"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Release-Jahr extrahieren
        release_date = ''
        release_span = soup.find('span', class_='LabelWithIcon__Label-sc-a1922d73-1 hFYGNw')
        if release_span:
            release_date = release_span.get_text(strip=True)
        
        # Genre-Tags extrahieren
        genres = []
        tags_container = soup.find('div', class_='SongTags__Container-sc-ca7b71cb-1 jSHrVV')
        if tags_container:
            tag_links = tags_container.find_all('a')
            genres = [tag.get_text(strip=True) for tag in tag_links]
        
        genres_string = ';'.join(genres) if genres else ''
        
        return release_date, genres_string
            
    except Exception as e:
        logger.warning(f"Fehler beim Scrapen von {url}: {e}")
        return '', ''

class RateLimitExceeded(Exception):
    """Custom Exception für Rate Limit Überschreitung"""
    pass

def search_and_download_lyrics(genius, artist, title, index):
    """Sucht und lädt Songtext herunter und gibt Metadaten zurück"""
    try:
        song = genius.search_song(title, artist)
        
        if song:
            lyrics = song.lyrics
            url = song.url
            
            # Lyrics speichern
            lyrics_file = Path(LYRICS_DIR) / f"{index}.txt"
            lyrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(lyrics_file, 'w', encoding='utf-8') as f:
                f.write(lyrics)
            
            # Metadaten von Genius-Seite scrapen
            release_year, genres = scrape_genius_metadata(url)
            
            return True, url, release_year, genres
        else:
            return False, '', '', ''
            
    except requests.exceptions.HTTPError as e:
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
            raise RateLimitExceeded(f"Rate Limit erreicht!")
        logger.error(f"HTTP Fehler bei {artist} - {title}: {e}")
        return False, '', '', ''
    except Exception as e:
        error_msg = str(e).lower()
        if any(term in error_msg for term in ['rate limit', 'too many requests', '429']):
            raise RateLimitExceeded(f"Rate Limit erreicht! Fehler: {e}")
        logger.error(f"Fehler bei {artist} - {title}: {e}")
        return False, '', '', ''

def main():
    """Hauptfunktion"""
    print("🎵 Second Run Lyrics Downloader gestartet...")
    
    if GENIUS_API_TOKEN == "YOUR_GENIUS_API_TOKEN_HERE":
        print("❌ FEHLER: Bitte setze deinen Genius API Token!")
        return
    
    # Lese letzten verarbeiteten Index
    last_processed_index = get_last_processed_index()
    
    genius = setup_genius_client()
    
    # Lade Input-Daten (nur mit Quelle Lyrics)
    df_input = load_input_data()
    if df_input is None or df_input.empty:
        print("❌ Keine Daten zum Verarbeiten gefunden!")
        return
    
    # Lade oder erstelle Output CSV
    df_output = load_or_create_output_csv()
    
    # Bereits verarbeitete Indices
    processed_indices = set(df_output['Index'].tolist()) if not df_output.empty else set()
    
    # Filtere bereits verarbeitete aus
    to_process = df_input[~df_input['Index'].isin(processed_indices)]
    
    # Wenn ein letzter Index existiert, filtere auch danach
    if last_processed_index is not None:
        # Überspringe alle Indices bis einschließlich last_processed_index
        to_process = to_process[to_process['Index'] > last_processed_index]
        print(f"⏭️  Überspringe bereits verarbeitete Indices bis {last_processed_index}")
    
    if to_process.empty:
        print("✅ Alle Einträge bereits verarbeitet!")
        return
    
    print(f"📊 Zu verarbeiten: {len(to_process)} Einträge")
    print(f"⏱️  Geschätzte Zeit: ~{len(to_process) * SLEEP_BETWEEN_REQUESTS / 60:.1f} Minuten")
    
    # Logge Start
    progress_logger.info(f"=== Download-Session gestartet ===")
    progress_logger.info(f"Zu verarbeiten: {len(to_process)} Einträge")
    if last_processed_index:
        progress_logger.info(f"Resume ab Index: {last_processed_index + 1}")
    
    success_count = 0
    failed_count = 0
    total_count = len(to_process)
    rate_limit_hits = 0
    
    with tqdm(total=total_count, desc="📥 Downloading Lyrics", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] ✅{postfix[0]} ❌{postfix[1]}',
              postfix=[0, 0], leave=True) as pbar:
        
        rows_list = list(to_process.iterrows())
        i = 0
        
        while i < len(rows_list):
            _, row = rows_list[i]
            index = row['Index']
            artist = row['Künstler']
            title = row['Titel']
            jahr = row['Jahr']
            
            try:
                success, url, release_year, genres = search_and_download_lyrics(genius, artist, title, index)
                
                if success:
                    success_count += 1
                    # Füge zur Output CSV hinzu
                    new_row = pd.DataFrame({
                        'Index': [index],
                        'Künstler': [artist],
                        'Titel': [title],
                        'Jahr': [jahr],
                        'Quelle': [url],
                        'Erscheinungsjahr Genius': [release_year],
                        'Genre Genius': [genres]
                    })
                    df_output = pd.concat([df_output, new_row], ignore_index=True)
                    save_output_csv(df_output)
                    
                    # Logge erfolgreichen Download
                    progress_logger.info(f"Processed Index: {index} | Status: SUCCESS | Artist: {artist} | Title: {title}")
                else:
                    failed_count += 1
                    # Logge fehlgeschlagenen Download
                    progress_logger.info(f"Processed Index: {index} | Status: FAILED | Artist: {artist} | Title: {title}")
                
                pbar.postfix[0] = success_count
                pbar.postfix[1] = failed_count
                pbar.update(1)
                
                # Gehe zum nächsten Song
                i += 1
                
                # Normale Wartezeit zwischen Requests
                if i < len(rows_list):
                    time.sleep(SLEEP_BETWEEN_REQUESTS)
                
            except RateLimitExceeded as e:
                rate_limit_hits += 1
                print(f"\n⚠️  Rate Limit Hit #{rate_limit_hits}")
                progress_logger.warning(f"Rate Limit Hit #{rate_limit_hits} bei Index {index} ({artist} - {title})")
                
                # Warte 61 Minuten
                wait_for_rate_limit_reset()
                
                # Versuche den gleichen Song nochmal (i wird nicht erhöht)
                # Genius Client neu initialisieren
                genius = setup_genius_client()
                continue
            
            except KeyboardInterrupt:
                print(f"\n\n⚠️  ABBRUCH DURCH BENUTZER!")
                print(f"⏰ Verarbeitet: {success_count + failed_count}/{total_count}")
                print(f"✅ Erfolgreich: {success_count}")
                print(f"❌ Fehlgeschlagen: {failed_count}")
                print(f"🔄 Rate Limit Hits: {rate_limit_hits}")
                progress_logger.info(f"=== Abbruch durch Benutzer bei Index {index} ===")
                return
    
    # Logge erfolgreichen Abschluss
    progress_logger.info(f"=== Download-Session abgeschlossen ===")
    progress_logger.info(f"Erfolgreich: {success_count} | Fehlgeschlagen: {failed_count} | Rate Limits: {rate_limit_hits}")
    
    print(f"\n{'='*50}")
    print("📈 ZUSAMMENFASSUNG")
    print(f"{'='*50}")
    print(f"✅ Erfolgreich heruntergeladen: {success_count}")
    print(f"❌ Fehlgeschlagen: {failed_count}")
    print(f"🔄 Rate Limit Hits: {rate_limit_hits}")
    print(f"📁 Lyrics Ordner: {os.path.abspath(LYRICS_DIR)}")
    print(f"📊 Output CSV: {os.path.abspath(OUTPUT_CSV)}")
    print("🎵 Download abgeschlossen!")

if __name__ == "__main__":
    main()