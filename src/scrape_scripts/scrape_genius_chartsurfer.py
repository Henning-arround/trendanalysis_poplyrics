#!/usr/bin/env python3
"""
Skript zum Herunterladen von Songtexten für Chartsurfer Daten (1954-1977)
Fügt Index-Spalte hinzu und speichert Genius URLs in "Quelle Genius" Spalte
"""

import pandas as pd
import lyricsgenius
import os
import time
import logging
import requests
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from bs4 import BeautifulSoup
import threading
from queue import Queue

# ================================
# KONFIGURATION
# ================================

# Genius API Token
GENIUS_API_TOKEN = "############################"

# Rate Limiting Konfiguration
SLEEP_BETWEEN_REQUESTS = 1
MAX_RETRIES = 10
TIMEOUT = 3600
RATE_LIMIT_WAIT_TIME = 61 * 60  # 61 Minuten in Sekunden

# Pfade
CSV_INPUT_PATH = "../../data/charts_chartsurfer_1954_1977_v2_without_duplicates.csv"
CSV_OUTPUT_PATH = "../../data/charts_chartsurfer_1954_1977_v3.csv"
LYRICS_DIR = "../../data/lyrics_chartsurfer/"
PROGRESS_LOG_PATH = "../../data/chartsurfer_download_progress.log"

# Logging Setup
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
# HAUPTFUNKTIONEN
# ================================

def setup_genius_client():
    """Initialisiert den Genius Client mit Rate Limiting"""
    genius = lyricsgenius.Genius(
        GENIUS_API_TOKEN,
        sleep_time=SLEEP_BETWEEN_REQUESTS,
        timeout=TIMEOUT,
        retries=MAX_RETRIES,
        remove_section_headers=True,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        verbose=False
    )
    return genius

def load_and_prepare_csv():
    """Lädt CSV und fügt Index sowie Quelle Genius Spalte hinzu"""
    try:
        df = pd.read_csv(CSV_INPUT_PATH)
        print(f"✅ CSV erfolgreich geladen: {len(df)} Einträge")
        
        # Füge Index-Spalte hinzu falls nicht vorhanden
        if 'Index' not in df.columns:
            df.insert(0, 'Index', range(1, len(df) + 1))
            print("➕ Index-Spalte hinzugefügt")
        
        # Füge "Quelle Genius" Spalte hinzu falls nicht vorhanden
        if 'Quelle Genius' not in df.columns:
            df['Quelle Genius'] = ''
            print("➕ 'Quelle Genius' Spalte hinzugefügt")
        
        # Füge "Release Date Genius" Spalte hinzu falls nicht vorhanden
        if 'Release Date Genius' not in df.columns:
            df['Release Date Genius'] = ''
            print("➕ 'Release Date Genius' Spalte hinzugefügt")
        
        # Füge "Genre Genius" Spalte hinzu falls nicht vorhanden
        if 'Genre Genius' not in df.columns:
            df['Genre Genius'] = ''
            print("➕ 'Genre Genius' Spalte hinzugefügt")
        
        return df
    except Exception as e:
        logger.error(f"Fehler beim Laden der CSV: {e}")
        return None

def save_csv(df):
    """Speichert die CSV mit Updates"""
    try:
        df.to_csv(CSV_OUTPUT_PATH, index=False, sep=',', quotechar='"', quoting=1)
    except Exception as e:
        logger.error(f"Fehler beim Speichern der CSV: {e}")

def get_existing_lyrics():
    """Ermittelt welche Lyrics bereits vorhanden sind"""
    lyrics_dir = Path(LYRICS_DIR)
    if not lyrics_dir.exists():
        lyrics_dir.mkdir(parents=True, exist_ok=True)
        return set()
    
    existing_files = set()
    for file in lyrics_dir.glob("*.txt"):
        try:
            index = int(file.stem)
            existing_files.add(index)
        except ValueError:
            continue
    
    print(f"📂 Bereits vorhandene Lyrics: {len(existing_files)} Dateien")
    return existing_files

class RateLimitExceeded(Exception):
    """Custom Exception für Rate Limit Überschreitung"""
    pass

def scrape_release_date_from_url(url):
    """Scraped das Release-Datum und Genre-Tags direkt von der Genius-Seite"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Suche nach dem Release-Datum im spezifischen span-Element
        release_date = ''
        release_span = soup.find('span', class_='LabelWithIcon__Label-sc-a1922d73-1 hFYGNw')
        if release_span:
            release_date = release_span.get_text(strip=True)
        
        # Suche nach Genre-Tags im div mit der Klasse SongTags__Container
        genres = []
        tags_container = soup.find('div', class_='SongTags__Container-sc-ca7b71cb-1 jSHrVV')
        if tags_container:
            # Finde alle <a> Elemente innerhalb des Containers
            tag_links = tags_container.find_all('a')
            genres = [tag.get_text(strip=True) for tag in tag_links]
        
        # Genres mit Semikolon verbinden
        genres_string = ';'.join(genres) if genres else ''
        
        return release_date, genres_string
            
    except Exception as e:
        logger.warning(f"Fehler beim Scrapen von {url}: {e}")
        return '', ''

def search_and_download_lyrics(genius, artist, title, index):
    """Sucht und lädt Songtext herunter, gibt URL, Release Date und Genres zurück"""
    try:
        # Suche Song
        song = genius.search_song(title, artist)
        
        if song:
            lyrics = song.lyrics
            url = song.url
            
            # Lyrics in Datei speichern
            lyrics_file = Path(LYRICS_DIR) / f"{index}.txt"
            with open(lyrics_file, 'w', encoding='utf-8') as f:
                f.write(lyrics)
            
            # Scrape Release-Datum und Genres von der Genius-Seite
            release_date, genres = scrape_release_date_from_url(url)
            
            # Log Erfolg
            log_progress("SUCCESS", index, artist, title, f"URL: {url}, Release: {release_date}, Genres: {genres}")
            return url, release_date, genres
        else:
            # Log Fehlschlag (nicht gefunden)
            log_progress("NOT_FOUND", index, artist, title, "Song nicht auf Genius gefunden")
            return None, None, None
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            log_progress("RATE_LIMIT", index, artist, title, "Rate Limit erreicht")
            raise RateLimitExceeded(f"Rate Limit erreicht! API-Anfragen zu häufig.")
        else:
            log_progress("ERROR", index, artist, title, f"HTTP Error: {e}")
            logger.error(f"HTTP Fehler bei {artist} - {title}: {e}")
            return None, None, None
    except Exception as e:
        error_msg = str(e).lower()
        if any(term in error_msg for term in ['rate limit', 'too many requests', '429', 'quota exceeded']):
            log_progress("RATE_LIMIT", index, artist, title, f"Rate Limit: {e}")
            raise RateLimitExceeded(f"Rate Limit erreicht! Fehler: {e}")
        else:
            log_progress("ERROR", index, artist, title, f"Exception: {e}")
            logger.error(f"Fehler bei {artist} - {title}: {e}")
            return None, None, None

def get_artist_title_columns(df):
    """Ermittelt die Spalten für Künstler und Titel"""
    # Mögliche Spaltennamen
    artist_candidates = ['Künstler', 'Artist', 'Interpret', 'künstler', 'artist']
    title_candidates = ['Titel', 'Title', 'Song', 'titel', 'title', 'song']
    
    artist_col = None
    title_col = None
    
    for col in df.columns:
        if col in artist_candidates:
            artist_col = col
        if col in title_candidates:
            title_col = col
    
    if not artist_col or not title_col:
        print(f"📋 Verfügbare Spalten: {list(df.columns)}")
        raise ValueError(f"Künstler oder Titel Spalte nicht gefunden! Gefunden: Artist={artist_col}, Title={title_col}")
    
    print(f"📋 Verwende Spalten: Künstler='{artist_col}', Titel='{title_col}'")
    return artist_col, title_col

def get_last_processed_index_from_log():
    """Liest den letzten erfolgreich verarbeiteten Index aus der Log-Datei"""
    log_path = Path(PROGRESS_LOG_PATH)
    
    if not log_path.exists():
        print("📝 Keine Progress-Log gefunden - Start von Beginn")
        return 0
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        last_success_index = 0
        for line in reversed(lines):
            if 'SUCCESS' in line and 'Index:' in line:
                # Extrahiere Index aus Log-Zeile
                try:
                    index_str = line.split('Index:')[1].split(',')[0].strip()
                    last_success_index = int(index_str)
                    print(f"🔄 Letzter erfolgreicher Download aus Log: Index {last_success_index}")
                    print(f"   Fortsetzung wird automatisch ab Index {last_success_index + 1} gestartet")
                    break
                except (IndexError, ValueError):
                    continue
        
        return last_success_index
    except Exception as e:
        logger.warning(f"Fehler beim Lesen der Progress-Log: {e}")
        return 0

def log_progress(status, index, artist, title, message=""):
    """Schreibt Fortschritt in die Log-Datei"""
    progress_logger.info(f"{status} - Index: {index}, Artist: {artist}, Title: {title}, {message}")

def log_summary(total, success, failed, start_index):
    """Schreibt Zusammenfassung in die Log-Datei"""
    progress_logger.info(f"{'='*80}")
    progress_logger.info(f"ZUSAMMENFASSUNG - Gestartet bei Index: {start_index + 1}")
    progress_logger.info(f"Total verarbeitet: {total}, Erfolgreich: {success}, Fehlgeschlagen: {failed}")
    progress_logger.info(f"{'='*80}")

def wait_for_rate_limit(wait_time=RATE_LIMIT_WAIT_TIME):
    """Wartet mit Countdown-Anzeige bis Rate Limit aufgehoben ist"""
    print(f"\n⏳ Warte {wait_time // 60} Minuten bis Rate Limit aufgehoben ist...")
    progress_logger.info(f"WARTE {wait_time // 60} MINUTEN - Rate Limit Wartezeit")
    
    end_time = time.time() + wait_time
    
    with tqdm(total=wait_time, desc="⏰ Wartezeit", unit="s", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]') as pbar:
        
        last_update = 0
        while time.time() < end_time:
            elapsed = int(time.time() - (end_time - wait_time))
            if elapsed > last_update:
                pbar.update(elapsed - last_update)
                last_update = elapsed
            time.sleep(1)
        
        # Finale Aktualisierung
        if last_update < wait_time:
            pbar.update(wait_time - last_update)
    
    print("✅ Wartezeit beendet - Download wird fortgesetzt!\n")
    progress_logger.info("Wartezeit beendet - Fortsetzung des Downloads")

def main():
    """Hauptfunktion"""
    print("🎵 Chartsurfer Lyrics Downloader gestartet...")
    
    # Log Start
    progress_logger.info(f"{'='*80}")
    progress_logger.info(f"NEUER DOWNLOAD-LAUF GESTARTET")
    progress_logger.info(f"{'='*80}")
    
    # Prüfe ob API Token gesetzt ist
    if GENIUS_API_TOKEN == "YOUR_GENIUS_API_TOKEN_HERE":
        print("❌ FEHLER: Bitte setze deinen Genius API Token!")
        return
    
    # Initialisiere Genius Client
    genius = setup_genius_client()
    
    # Lade und bereite CSV vor
    df = load_and_prepare_csv()
    if df is None:
        return
    
    # Ermittle Spalten für Künstler und Titel
    try:
        artist_col, title_col = get_artist_title_columns(df)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Ermittle bereits vorhandene Lyrics
    existing_lyrics = get_existing_lyrics()
    
    # Ermittle letzten verarbeiteten Index aus Log (automatisch)
    last_processed_index = get_last_processed_index_from_log()
    
    # Filtere für fehlende Lyrics UND Indices oberhalb des letzten verarbeiteten Index
    missing_lyrics = df[
        (~df['Index'].isin(existing_lyrics)) & 
        (df['Index'] > last_processed_index)
    ].copy()
    
    if missing_lyrics.empty:
        print("✅ Alle Songtexte sind bereits vorhanden!")
        # Speichere CSV auch wenn nichts zu tun ist (damit v3 existiert)
        save_csv(df)
        progress_logger.info("Alle Songtexte bereits vorhanden - nichts zu tun")
        return
    
    print(f"📊 Zu downloaden: {len(missing_lyrics)} von {len(df)} Songtexten")
    if last_processed_index > 0:
        print(f"🚀 Automatische Fortsetzung ab Index: {last_processed_index + 1}")
        progress_logger.info(f"Automatische Fortsetzung ab Index: {last_processed_index + 1}")
    else:
        print(f"🆕 Start von Beginn (Index 1)")
        progress_logger.info(f"Start von Beginn")
    print(f"⏱️  Geschätzte Zeit: ~{len(missing_lyrics) * SLEEP_BETWEEN_REQUESTS / 60:.1f} Minuten")
    print(f"📋 Progress-Log: {os.path.abspath(PROGRESS_LOG_PATH)}")
    print(f"⚠️  Bei Rate Limit: Automatische 61 Minuten Wartezeit\n")
    
    # Download starten
    success_count = 0
    failed_count = 0
    total_count = len(missing_lyrics)
    
    # Progress Bar Setup
    pbar = tqdm(total=total_count, desc="📥 Downloading Lyrics", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] ✅{postfix[0]} ❌{postfix[1]}',
                postfix=[0, 0], leave=True)
    
    try:
        # Konvertiere zu Liste für einfacheres Iterieren mit Index
        rows_list = list(missing_lyrics.iterrows())
        i = 0
        
        while i < len(rows_list):
            idx, row = rows_list[i]
            index = row['Index']
            artist = row[artist_col]
            title = row[title_col]
            
            try:
                genius_url, release_date, genres = search_and_download_lyrics(genius, artist, title, index)
                
                if genius_url:
                    success_count += 1
                    # Speichere Genius URL, Release Date und Genres in DataFrame
                    df.loc[df['Index'] == index, 'Quelle Genius'] = genius_url
                    df.loc[df['Index'] == index, 'Release Date Genius'] = release_date
                    df.loc[df['Index'] == index, 'Genre Genius'] = genres
                    # SOFORT speichern
                    save_csv(df)
                else:
                    failed_count += 1
                
                # Update Progress Bar
                pbar.postfix[0] = success_count
                pbar.postfix[1] = failed_count
                pbar.update(1)
                
                # Rate Limiting
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                
                # Nächsten Song
                i += 1
                
            except RateLimitExceeded as e:
                # Log, aber nicht abbrechen
                progress_logger.warning(f"RATE LIMIT ERREICHT - Index: {index} - Warte 61 Minuten")
                
                # Schließe aktuellen Progress Bar
                pbar.close()
                
                # Warte 61 Minuten
                wait_for_rate_limit()
                
                # Erstelle neuen Progress Bar
                pbar = tqdm(total=total_count, initial=success_count + failed_count,
                           desc="📥 Downloading Lyrics", 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] ✅{postfix[0]} ❌{postfix[1]}',
                           postfix=[success_count, failed_count], leave=True)
                
                # NICHT i erhöhen - versuche denselben Song nochmal
                continue
                
    except KeyboardInterrupt:
        pbar.close()
        log_summary(success_count + failed_count, success_count, failed_count, last_processed_index)
        progress_logger.warning(f"MANUELLER ABBRUCH - Letzter Index: {index}")
        
        print(f"\n\n⚠️ MANUELL ABGEBROCHEN!")
        print(f"⏰ Bisher verarbeitet: {success_count + failed_count}/{total_count}")
        print(f"✅ Erfolgreich: {success_count}")
        print(f"❌ Fehlgeschlagen: {failed_count}")
        print(f"\n📊 Progress in CSV gespeichert: {os.path.abspath(CSV_OUTPUT_PATH)}")
        print(f"📋 Progress-Log: {os.path.abspath(PROGRESS_LOG_PATH)}")
        print(f"🔄 Beim nächsten Start wird automatisch fortgesetzt")
        raise SystemExit(0)
    
    finally:
        pbar.close()
    
    # Log finale Zusammenfassung
    log_summary(success_count + failed_count, success_count, failed_count, last_processed_index)
    progress_logger.info("DOWNLOAD-LAUF ERFOLGREICH ABGESCHLOSSEN")
    
    # Zusammenfassung
    print(f"\n{'='*50}")
    print("📈 ZUSAMMENFASSUNG")
    print(f"{'='*50}")
    print(f"✅ Erfolgreich heruntergeladen: {success_count}")
    print(f"❌ Fehlgeschlagen: {failed_count}")
    print(f"📁 Lyrics Ordner: {os.path.abspath(LYRICS_DIR)}")
    print(f"📊 CSV aktualisiert: {os.path.abspath(CSV_OUTPUT_PATH)}")
    print(f"📋 Progress-Log: {os.path.abspath(PROGRESS_LOG_PATH)}")
    print("🎵 Download abgeschlossen!")

if __name__ == "__main__":
    main()