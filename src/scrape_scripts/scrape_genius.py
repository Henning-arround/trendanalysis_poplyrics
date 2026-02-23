#!/usr/bin/env python3
"""
Skript zum Herunterladen von Songtexten mit lyricsgenius
Lädt nur fehlende Songtexte herunter und speichert sie mit Index als Dateiname
Erstellt zusätzlich eine CSV mit den Quelleninformationen
"""

import pandas as pd
import lyricsgenius
import os
import time
import logging
import requests
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup

# ================================
# KONFIGURATION
# ================================

# Genius API Token - ersetze mit deinem eigenen Token
GENIUS_API_TOKEN = "##################"

# Rate Limiting Konfiguration
SLEEP_BETWEEN_REQUESTS = 0.5  # Sekunden zwischen Anfragen
MAX_RETRIES = 10              # Maximale Wiederholungsversuche
TIMEOUT = 3600                 # Timeout für Anfragen in Sekunden

# Pfade
CSV_PATH = "../data/charts_indexed.csv"
LYRICS_DIR = "../data/lyrics/"
SOURCE_CSV_PATH = "../data/source_genius.csv"

# Logging Setup - nur Errors und Warnings
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        remove_section_headers=True,  # Entfernt [Verse], [Chorus] etc.
        skip_non_songs=True,         # Überspringt Nicht-Songs
        excluded_terms=["(Remix)", "(Live)"],  # Schließt Remixes/Live-Versionen aus
        verbose=False
    )
    return genius

def load_csv_data():
    """Lädt die CSV-Datei und gibt DataFrame zurück"""
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✅ CSV erfolgreich geladen: {len(df)} Einträge")
        return df
    except Exception as e:
        logger.error(f"Fehler beim Laden der CSV: {e}")
        return None

def get_last_processed_index():
    """Ermittelt den höchsten Index aus der Source CSV"""
    source_path = Path(SOURCE_CSV_PATH)
    
    if source_path.exists():
        try:
            df = pd.read_csv(SOURCE_CSV_PATH)
            if not df.empty and 'Index' in df.columns:
                last_index = df['Index'].max()
                print(f"🔄 Letzter verarbeiteter Index aus Source CSV: {last_index}")
                return last_index
            else:
                print("📝 Source CSV ist leer oder hat keine Index-Spalte")
                return 0
        except Exception as e:
            logger.warning(f"Fehler beim Lesen der Source CSV: {e}")
            return 0
    else:
        print("📝 Keine Source CSV gefunden - Start von Beginn")
        return 0

def load_or_create_source_csv():
    """Lädt vorhandene Source CSV oder erstellt eine neue"""
    source_path = Path(SOURCE_CSV_PATH)
    
    if source_path.exists():
        try:
            df = pd.read_csv(SOURCE_CSV_PATH)
            print(f"✅ Source CSV geladen: {len(df)} Einträge")
            # Füge Release Date Spalte hinzu falls nicht vorhanden
            if 'Release Date' not in df.columns:
                df['Release Date'] = ''
            # Füge Genre Spalte hinzu falls nicht vorhanden
            if 'Genre' not in df.columns:
                df['Genre'] = ''
            return df
        except Exception as e:
            logger.warning(f"Fehler beim Laden der Source CSV: {e}")
    
    # Erstelle neue leere DataFrame
    df = pd.DataFrame(columns=['Index', 'Quelle Lyrics', 'Release Date', 'Genre'])
    print("📝 Neue Source CSV erstellt")
    return df

def save_source_csv(df):
    """Speichert die Source CSV"""
    try:
        # Stelle sicher, dass der Ordner existiert
        os.makedirs(os.path.dirname(SOURCE_CSV_PATH), exist_ok=True)
        df.to_csv(SOURCE_CSV_PATH, index=False, sep=',', quotechar='"', quoting=1)
        # Keine print-Ausgabe mehr, um Progress Bar nicht zu stören
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Source CSV: {e}")

def add_source_entry(source_df, index, release_date='', genre=''):
    """Fügt einen Eintrag zur Source DataFrame hinzu"""
    new_row = pd.DataFrame({
        'Index': [index],
        'Quelle Lyrics': ['https://genius.com/'],
        'Release Date': [release_date],
        'Genre': [genre]
    })
    return pd.concat([source_df, new_row], ignore_index=True)

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
    """Sucht und lädt Songtext für einen einzelnen Song herunter"""
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
            
            return True, release_date, genres
        else:
            return False, '', ''
            
    except requests.exceptions.HTTPError as e:
        # Prüfe auf Rate Limit Fehler (HTTP 429)
        if e.response.status_code == 429:
            raise RateLimitExceeded(f"Rate Limit erreicht! API-Anfragen zu häufig. Warte und versuche es später erneut.")
        else:
            logger.error(f"HTTP Fehler bei {artist} - {title}: {e}")
            return False, '', ''
    except Exception as e:
        # Prüfe auf Rate Limit in der Fehlermeldung
        error_msg = str(e).lower()
        if any(term in error_msg for term in ['rate limit', 'too many requests', '429', 'quota exceeded']):
            raise RateLimitExceeded(f"Rate Limit erreicht! Fehler: {e}")
        else:
            logger.error(f"Fehler bei {artist} - {title}: {e}")
            return False, '', ''

def main():
    """Hauptfunktion"""
    print("🎵 Lyrics Downloader gestartet...")
    
    # Prüfe ob API Token gesetzt ist
    if GENIUS_API_TOKEN == "YOUR_GENIUS_API_TOKEN_HERE":
        print("❌ FEHLER: Bitte setze deinen Genius API Token in der Konfiguration!")
        print("Erstelle einen kostenlosen Account auf https://genius.com/api-clients")
        return
    
    # Initialisiere Genius Client
    genius = setup_genius_client()
    
    # Lade CSV Daten
    df = load_csv_data()
    if df is None:
        return
    
    # Lade oder erstelle Source CSV
    source_df = load_or_create_source_csv()
    
    # Ermittle den letzten verarbeiteten Index
    last_processed_index = get_last_processed_index()
    
    # Ermittle bereits vorhandene Lyrics
    existing_lyrics = get_existing_lyrics()
    
    # Filtere DataFrame für fehlende Lyrics UND Indices oberhalb des letzten verarbeiteten Index
    missing_lyrics = df[
        (~df['Index'].isin(existing_lyrics)) & 
        (df['Index'] > last_processed_index)
    ].sort_values('Index')  # Sortiere nach Index für sequenzielle Verarbeitung
    
    if missing_lyrics.empty:
        print("✅ Alle Songtexte sind bereits vorhanden oder verarbeitet!")
        return
    
    print(f"📊 Zu downloaden: {len(missing_lyrics)} von {len(df)} Songtexten")
    if last_processed_index > 0:
        print(f"🚀 Fortsetzung ab Index: {last_processed_index + 1}")
    print(f"⏱️  Geschätzte Zeit: ~{len(missing_lyrics) * SLEEP_BETWEEN_REQUESTS / 60:.1f} Minuten")
    
    # Download starten
    success_count = 0
    failed_count = 0
    total_count = len(missing_lyrics)
    
    # Progress Bar Setup
    with tqdm(total=total_count, desc="📥 Downloading Lyrics", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] ✅{postfix[0]} ❌{postfix[1]}',
              postfix=[0, 0], leave=True) as pbar:
        
        try:
            for _, row in missing_lyrics.iterrows():
                index = row['Index']
                artist = row['Künstler']
                title = row['Titel']
                
                success, release_date, genres = search_and_download_lyrics(genius, artist, title, index)
                
                if success:
                    success_count += 1
                    # Füge Eintrag zur Source CSV hinzu mit Release Date und Genres
                    source_df = add_source_entry(source_df, index, release_date, genres)
                    # SOFORT speichern damit nichts verloren geht
                    save_source_csv(source_df)
                else:
                    failed_count += 1
                
                # Update Progress Bar mit aktuellen Zählern
                pbar.postfix[0] = success_count  # Erfolgreich
                pbar.postfix[1] = failed_count   # Fehlgeschlagen
                pbar.update(1)
                
                # Rate Limiting
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                
        except RateLimitExceeded as e:
            print(f"\n\n🛑 RATE LIMIT ERREICHT!")
            print(f"📛 {e}")
            print(f"⏰ Bisher verarbeitet: {success_count + failed_count}/{total_count}")
            print(f"✅ Erfolgreich: {success_count}")
            print(f"❌ Fehlgeschlagen: {failed_count}")
            print(f"\n💡 Empfehlung:")
            print(f"   - Warte mindestens 1 Stunde vor dem nächsten Versuch")
            print(f"   - Erhöhe SLEEP_BETWEEN_REQUESTS auf 2.0 oder höher")
            print(f"   - Bereits heruntergeladene Songs bleiben erhalten")
            print(f"\n📊 Progress in Source CSV gespeichert: {os.path.abspath(SOURCE_CSV_PATH)}")
            
            # Beende das Programm mit Fehlercode
            raise SystemExit(1)
    
    # Zusammenfassung
    print(f"\n{'='*50}")
    print("📈 ZUSAMMENFASSUNG")
    print(f"{'='*50}")
    print(f"✅ Erfolgreich heruntergeladen: {success_count}")
    print(f"❌ Fehlgeschlagen: {failed_count}")
    print(f"📁 Lyrics Ordner: {os.path.abspath(LYRICS_DIR)}")
    print(f"📊 Source CSV: {os.path.abspath(SOURCE_CSV_PATH)}")
    print("🎵 Download abgeschlossen!")

if __name__ == "__main__":
    main()