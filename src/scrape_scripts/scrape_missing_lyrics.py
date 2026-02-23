import pandas as pd
import lyricsgenius
import os
import time
import logging
import requests
from pathlib import Path
from tqdm import tqdm
import re

# ================================
# KONFIGURATION
# ================================

# Genius API Token - ersetze mit deinem eigenen Token
GENIUS_API_TOKEN = "#######################"

# Rate Limiting Konfiguration
SLEEP_BETWEEN_REQUESTS = 1.0  # Sekunden zwischen Anfragen
MAX_RETRIES = 10              # Maximale Wiederholungsversuche
TIMEOUT = 3600                 # Timeout für Anfragen in Sekunden

# Pfade
CSV_PATH = "../data/charts_with_language_updated.csv"
LYRICS_DIR = "../data/lyrics/"
SOURCE_CSV_PATH = "../data/missing_source_lyrics.csv"

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
        remove_section_headers=True,
        skip_non_songs=True,
        excluded_terms=[],
        verbose=False
    )
    return genius

def load_csv_data():
    """Lädt die CSV-Datei und gibt DataFrame zurück"""
    try:
        df = pd.read_csv(CSV_PATH, sep=',', quotechar='"', escapechar='\\')
        print(f"✅ CSV erfolgreich geladen: {len(df)} Einträge")
        return df
    except Exception as e:
        logger.error(f"Fehler beim Laden der CSV: {e}")
        return None

def has_special_pattern(text):
    """Prüft ob ein Text spezielle Pattern enthält"""
    if pd.isna(text):
        return False
    text = str(text).lower()  # Konvertiere zu lowercase für case-insensitive matching
    patterns = ['feat', 'ft.', ' and ', '&', '/']
    return any(pattern in text for pattern in patterns)

def filter_missing_lyrics_with_patterns(df):
    """Filtert Einträge ohne Lyrics die spezielle Pattern enthalten"""
    # Fehlende Lyrics
    missing_lyrics = df[df['Quelle Lyrics'].isna() | (df['Quelle Lyrics'] == '') | (df['Quelle Lyrics'].str.strip() == '')]
    
    # Nur Einträge mit speziellen Pattern
    pattern_entries = missing_lyrics[
        missing_lyrics['Titel'].apply(has_special_pattern) | 
        missing_lyrics['Künstler'].apply(has_special_pattern)
    ]
    
    return pattern_entries

def generate_search_variations(artist, title):
    """Generiert verschiedene Suchvariationen basierend auf Pattern"""
    variations = []
    
    # Original
    variations.append((artist, title))
    
    # Behandle feat/ft Pattern
    for text_type, text in [('artist', artist), ('title', title)]:
        if pd.isna(text):
            continue
        text = str(text)
        
        # Entferne feat/ft und alles danach (case-insensitive)
        feat_pattern = re.compile(r'\s+(feat|ft\.?)\s+.*', re.IGNORECASE)
        if feat_pattern.search(text):
            clean_text = feat_pattern.sub('', text).strip()
            if text_type == 'artist':
                variations.append((clean_text, title))
            else:
                variations.append((artist, clean_text))
    
    # Behandle Trennzeichen (case-insensitive für "and")
    separators = [' and ', ' And ', '&', '/']
    
    for text_type, text in [('artist', artist), ('title', title)]:
        if pd.isna(text):
            continue
        text = str(text)
        
        for sep in separators:
            if sep in text:
                # Ersetze Trennzeichen durch andere
                for replacement in separators:
                    if replacement != sep:
                        new_text = text.replace(sep, replacement)
                        if text_type == 'artist':
                            variations.append((new_text, title))
                        else:
                            variations.append((artist, new_text))
                
                # Splitte bei Trennzeichen und verwende einzelne Teile
                parts = [part.strip() for part in text.split(sep) if part.strip()]
                if len(parts) > 1:
                    for part in parts:
                        if text_type == 'artist':
                            variations.append((part, title))
                        else:
                            variations.append((artist, part))
    
    # Entferne Duplikate
    seen = set()
    unique_variations = []
    for var in variations:
        if var not in seen:
            seen.add(var)
            unique_variations.append(var)
    
    return unique_variations

def load_or_create_source_csv():
    """Lädt vorhandene Source CSV oder erstellt eine neue"""
    source_path = Path(SOURCE_CSV_PATH)
    
    if source_path.exists():
        try:
            df = pd.read_csv(SOURCE_CSV_PATH)
            print(f"✅ Source CSV geladen: {len(df)} Einträge")
            return df
        except Exception as e:
            logger.warning(f"Fehler beim Laden der Source CSV: {e}")
    
    # Erstelle neue leere DataFrame
    df = pd.DataFrame(columns=['Index', 'Quelle Lyrics'])
    print("📝 Neue Source CSV erstellt")
    return df

def save_source_csv(df):
    """Speichert die Source CSV"""
    try:
        os.makedirs(os.path.dirname(SOURCE_CSV_PATH), exist_ok=True)
        df.to_csv(SOURCE_CSV_PATH, index=False)
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Source CSV: {e}")

def add_source_entry(source_df, index):
    """Fügt einen Eintrag zur Source DataFrame hinzu"""
    new_row = pd.DataFrame({
        'Index': [index],
        'Quelle Lyrics': ['https://genius.com/']
    })
    return pd.concat([source_df, new_row], ignore_index=True)

class RateLimitExceeded(Exception):
    """Custom Exception für Rate Limit Überschreitung"""
    pass

def search_and_download_lyrics(genius, artist, title, index):
    """Sucht und lädt Songtext für einen einzelnen Song mit Variationen herunter"""
    try:
        # Generiere Suchvariationen
        variations = generate_search_variations(artist, title)
        
        for i, (var_artist, var_title) in enumerate(variations):
            if pd.isna(var_artist) or pd.isna(var_title):
                continue
                
            try:
                # Suche Song
                song = genius.search_song(var_title, var_artist)
                
                if song:
                    lyrics = song.lyrics
                    
                    # Lyrics in Datei speichern
                    lyrics_file = Path(LYRICS_DIR) / f"{index}.txt"
                    lyrics_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(lyrics_file, 'w', encoding='utf-8') as f:
                        f.write(lyrics)
                    
                    return True
                    
            except requests.exceptions.HTTPError as e:
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                    raise RateLimitExceeded(f"Rate Limit erreicht!")
                continue
            except requests.exceptions.RequestException as e:
                # Behandle andere Request-Fehler
                error_msg = str(e).lower()
                if any(term in error_msg for term in ['rate limit', 'too many requests', '429']):
                    raise RateLimitExceeded(f"Rate Limit erreicht! Fehler: {e}")
                continue
            except Exception as e:
                error_msg = str(e).lower()
                if any(term in error_msg for term in ['rate limit', 'too many requests', '429']):
                    raise RateLimitExceeded(f"Rate Limit erreicht! Fehler: {e}")
                continue
            
            # Pause zwischen Variationen um Rate Limiting zu vermeiden
            # Nur wenn es nicht die letzte Variation ist
            if i < len(variations) - 1:
                time.sleep(SLEEP_BETWEEN_REQUESTS)
        
        return False
            
    except RateLimitExceeded:
        raise
    except Exception as e:
        logger.error(f"Fehler bei {artist} - {title}: {e}")
        return False

def main():
    """Hauptfunktion"""
    print("🎵 Missing Lyrics Downloader (Pattern-basiert) gestartet...")
    
    # Prüfe API Token
    if GENIUS_API_TOKEN == "YOUR_GENIUS_API_TOKEN_HERE":
        print("❌ FEHLER: Bitte setze deinen Genius API Token!")
        return
    
    # Initialisiere Genius Client
    genius = setup_genius_client()
    
    # Lade CSV Daten
    df = load_csv_data()
    if df is None:
        return
    
    # Filtere fehlende Lyrics mit speziellen Pattern
    missing_lyrics = filter_missing_lyrics_with_patterns(df)
    
    if missing_lyrics.empty:
        print("✅ Keine fehlenden Lyrics mit speziellen Pattern gefunden!")
        return
    
    print(f"📊 Gefundene Einträge mit speziellen Pattern: {len(missing_lyrics)}")
    
    # Lade oder erstelle Source CSV
    source_df = load_or_create_source_csv()
    
    # Bereits verarbeitete Indices
    processed_indices = set(source_df['Index'].tolist()) if not source_df.empty else set()
    
    # Filtere bereits verarbeitete aus
    to_process = missing_lyrics[~missing_lyrics['Index'].isin(processed_indices)]
    
    if to_process.empty:
        print("✅ Alle Pattern-Einträge bereits verarbeitet!")
        return
    
    print(f"📊 Zu verarbeiten: {len(to_process)} Einträge")
    print(f"⏱️  Geschätzte Zeit: ~{len(to_process) * SLEEP_BETWEEN_REQUESTS * 5 / 60:.1f} Minuten")
    
    # Download starten
    success_count = 0
    failed_count = 0
    total_count = len(to_process)
    
    # Progress Bar Setup
    with tqdm(total=total_count, desc="📥 Downloading Missing Lyrics", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] ✅{postfix[0]} ❌{postfix[1]}',
              postfix=[0, 0], leave=True) as pbar:
        
        try:
            for _, row in to_process.iterrows():
                index = row['Index']
                artist = row['Künstler']
                title = row['Titel']
                
                if search_and_download_lyrics(genius, artist, title, index):
                    success_count += 1
                    source_df = add_source_entry(source_df, index)
                    save_source_csv(source_df)
                else:
                    failed_count += 1
                
                pbar.postfix[0] = success_count
                pbar.postfix[1] = failed_count
                pbar.update(1)
                
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                
        except RateLimitExceeded as e:
            print(f"\n\n🛑 RATE LIMIT ERREICHT!")
            print(f"📛 {e}")
            print(f"⏰ Bisher verarbeitet: {success_count + failed_count}/{total_count}")
            print(f"✅ Erfolgreich: {success_count}")
            print(f"❌ Fehlgeschlagen: {failed_count}")
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
