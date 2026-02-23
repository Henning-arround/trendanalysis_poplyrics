import re
import csv
import requests
from bs4 import BeautifulSoup
import time
import logging
import os
from urllib.parse import urlparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

# KONFIGURATION - Zentrale Stelle für alle Dateipfade und Einstellungen
CONFIG = {
    'urls_file': '../data/chart_urls_unique.txt',
    'output_csv': '../data/charts.csv',
    'log_file': '../data/scraper.log',
    'progress_file': '../data/progress.txt',  # Datei für Progress-Tracking
    'test_limit': None,  # Setze auf None für alle URLs
    'request_delay': 0,  # Sekunden zwischen Requests
    'timeout': 30,  # Timeout für HTTP-Requests
    'deep_scraping': True,  # Soll auch andere Titel des Künstlers gescraped werden?
    'album_matching': True,  # Soll Album-Matching durchgeführt werden?
    'max_related_songs': 10000,  # Maximum Anzahl verwandter Songs pro Künstler
    'save_interval': 100,  # CSV alle X Einträge speichern
    'max_albums_per_song': 30,  # Maximum Anzahl Alben pro Song für CSV-Spalten
    'max_workers': 3,  # Anzahl paralleler Threads
    'max_retries': 3,  # Maximum Anzahl Wiederholungsversuche
    'retry_delay': 180,  # Wartezeit bei Retry in Sekunden (3 Minuten)
    'blocked_retry_delay': 300  # Wartezeit bei IP-Block in Sekunden (5 Minuten)
}

# Globale Thread-Locks für sichere Datenoperationen
records_lock = threading.Lock()
save_lock = threading.Lock()

def save_progress(processed_count):
    """Speichert den aktuellen Fortschritt"""
    try:
        with open(CONFIG['progress_file'], 'w') as f:
            f.write(str(processed_count))
    except Exception as e:
        print(f"Fehler beim Speichern des Fortschritts: {e}")

def load_progress():
    """Lädt den gespeicherten Fortschritt"""
    try:
        if os.path.exists(CONFIG['progress_file']):
            with open(CONFIG['progress_file'], 'r') as f:
                return int(f.read().strip())
    except Exception as e:
        print(f"Fehler beim Laden des Fortschritts: {e}")
    return 0

def setup_logging():
    """
    Richtet das Logging-System ein
    """
    # Log-Verzeichnis erstellen falls es nicht existiert
    log_dir = os.path.dirname(CONFIG['log_file'])
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Logging konfigurieren
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(CONFIG['log_file'], encoding='utf-8'),
            # Keine Console-Handler, da wir nur tqdm im Terminal wollen
        ]
    )
    
    return logging.getLogger(__name__)

def is_duplicate_song(artist, title, existing_records):
    """
    Prüft ob die Kombination aus Künstler und Titel bereits in den Records vorhanden ist
    """
    if not artist and not title:
        return False
    
    # Normalisiere die Strings für besseren Vergleich (lowercase, stripped)
    artist_norm = artist.lower().strip() if artist else ""
    title_norm = title.lower().strip() if title else ""
    
    for record in existing_records:
        existing_artist = record.get("Künstler", "").lower().strip()
        existing_title = record.get("Titel", "").lower().strip()
        
        if artist_norm == existing_artist and title_norm == existing_title:
            return True
    
    return False

def extract_basic_info_from_html(html_content):
    """
    Extrahiert nur Künstler und Titel aus HTML für Duplikat-Check (schneller als full extract)
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    artist = ""
    title = ""
    
    # Künstler extrahieren
    artist_element = soup.find('h1', class_='big-head')
    if artist_element:
        artist = artist_element.get_text(strip=True)
    
    # Titel extrahieren
    title_element = soup.find('h2', class_='sub-head')
    if title_element:
        title_text = title_element.get_text(strip=True)
        # Entferne "Single" am Ende falls vorhanden
        title = re.sub(r'\s*Single\s*$', '', title_text)
    
    return artist, title

def extract_related_song_urls(html_content):
    """
    Extrahiert URLs zu weiteren Titeln des Künstlers aus der "TOP Titel" und "NICHT TOP Titel" Sektion
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    related_urls = []
    
    # Suche nach div mit class="panel-body" die "TOP Titel" oder "NICHT TOP Titel" enthalten
    panel_bodies = soup.find_all('div', class_='panel-body')
    
    for panel_body in panel_bodies:
        # Prüfe ob dieses panel-body "TOP Titel" oder "NICHT TOP Titel" enthält
        h4_elements = panel_body.find_all('h4')
        relevant_panel = False
        
        for h4 in h4_elements:
            h4_text = h4.get_text(strip=True)
            if "TOPTitel" in h4_text or "NICHTTOPTitel" in h4_text:
                relevant_panel = True
                break
        
        if relevant_panel:
            # Alle Links zu Titeln in diesem panel-body finden
            title_links = panel_body.find_all('a', href=True)
            
            for link in title_links:
                href = link.get('href')
                # Nur Links zu Titel-Details nehmen
                if href and '/details/titel-details-' in href:
                    # Relative URLs zu absoluten URLs konvertieren
                    if href.startswith('/'):
                        full_url = 'https://www.offiziellecharts.de' + href
                    else:
                        full_url = href
                    
                    related_urls.append(full_url)
            
            # Nur das erste relevante panel-body verwenden
            break
    
    return related_urls

def extract_album_urls(html_content):
    """
    Extrahiert URLs zu Alben des Künstlers aus der "TOP Alben" und "NICHT TOP Alben" Sektion
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    album_urls = []
    
    # Suche nach div mit class="panel-body" die "TOP Alben" oder "NICHT TOP Alben" enthalten
    panel_bodies = soup.find_all('div', class_='panel-body')
    
    for panel_body in panel_bodies:
        # Prüfe ob dieses panel-body "TOP Alben" oder "NICHT TOP Alben" enthält
        h4_elements = panel_body.find_all('h4')
        relevant_panel = False
        
        for h4 in h4_elements:
            h4_text = h4.get_text(strip=True)
            if "TOPAlben" in h4_text or "NICHTTOPAlben" in h4_text:
                relevant_panel = True
                break
        
        if relevant_panel:
            # Alle Links zu Alben in diesem panel-body finden
            album_links = panel_body.find_all('a', href=True)
            
            for link in album_links:
                href = link.get('href')
                # Nur Links zu Album-Details nehmen
                if href and '/details/album-details-' in href:
                    # Relative URLs zu absoluten URLs konvertieren
                    if href.startswith('/'):
                        full_url = 'https://www.offiziellecharts.de' + href
                    else:
                        full_url = href
                    
                    album_urls.append(full_url)
            
            # Nur das erste relevante panel-body verwenden
            break
    
    return album_urls

def extract_album_tracks(html_content):
    """
    Extrahiert Album-Name, Erscheinungsjahr und Tracklist aus einer Album-Detail-Seite
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Album-Name extrahieren
    album_name = ""
    album_element = soup.find('h2', class_='sub-head')
    if album_element:
        album_name = album_element.get_text(strip=True)
        # Entferne "Album" am Ende falls vorhanden
        album_name = re.sub(r'\s*Album\s*$', '', album_name)
    
    # Album-Jahr extrahieren
    album_year = ""
    table_rows = soup.find_all('tr')
    for row in table_rows:
        label_cell = row.find('td', class_='detail-info-label')
        if label_cell and label_cell.get_text(strip=True) == "Jahr:":
            value_cell = label_cell.find_next_sibling('td')
            if value_cell:
                album_year = value_cell.get_text(strip=True)
                break
    
    # Tracklist extrahieren
    # Dictionary um Songs nach (artist, title) zu gruppieren und EANs zu sammeln
    songs_dict = {}
    
    track_table = soup.find('table', class_='table table-striped track-table')
    
    if track_table:
        # Alle <tr> Elemente finden
        track_trs = track_table.find_all('tr')
        
        for tr in track_trs:
            # EAN Identifier aus h4 und span extrahieren
            ean_identifier = ""
            h4_element = tr.find('h4')

            if h4_element:
                # span ist INNERHALB des h4 Elements
                span_element = h4_element.find('span')
                
                # Text vor dem span (direkt im h4)
                h4_text = h4_element.get_text(strip=True)
                if span_element:
                    # Text des span Elements
                    span_text = span_element.get_text(strip=True)
                    # Text des h4 ohne den span Inhalt
                    h4_only_text = h4_element.get_text(strip=True).replace(span_text, '').strip()
                    
                    # Mit " --- " verbinden
                    if h4_only_text and span_text:
                        ean_identifier = f"{h4_only_text} --- {span_text}"
                    elif h4_only_text:
                        ean_identifier = h4_only_text
                    elif span_text:
                        ean_identifier = span_text
                else:
                    # Kein span gefunden, nur h4 Text
                    ean_identifier = h4_text
            
            # Alle <a> Elemente in diesem TR finden
            track_links = tr.find_all('a', href=True)
            
            for link in track_links:
                # Nur Links zu Titel-Details verarbeiten
                if '/titel-details-' in link.get('href', ''):
                    title = link.get_text(strip=True)
                    
                    # Sibling <b> Element für Künstler suchen
                    artist = ""
                    
                    # Versuche verschiedene Wege den Künstler zu finden
                    parent_li = link.find_parent('li')
                    if parent_li:
                        b_element = parent_li.find('b')
                        if b_element:
                            artist = b_element.get_text(strip=True)
                    
                    if title and artist:
                        # Song-Key erstellen (artist, title)
                        song_key = (artist, title)
                        
                        # Song in Dictionary hinzufügen oder EAN hinzufügen
                        if song_key not in songs_dict:
                            songs_dict[song_key] = {
                                'artist': artist,
                                'title': title,
                                'ean_identifiers': []
                            }
                        
                        # EAN hinzufügen falls vorhanden und noch nicht in der Liste
                        if ean_identifier and ean_identifier not in songs_dict[song_key]['ean_identifiers']:
                            songs_dict[song_key]['ean_identifiers'].append(ean_identifier)
    
    # Dictionary in Liste umwandeln und EANs mit ";" verbinden
    tracks = []
    for song_data in songs_dict.values():
        track_entry = {
            'artist': song_data['artist'],
            'title': song_data['title'],
            'ean_identifier': '; '.join(song_data['ean_identifiers'])
        }
        tracks.append(track_entry)
    
    return album_name, album_year, tracks

def extract_chart_data(html_content, source_url=""):
    """
    Extrahiert Chart-Daten aus der HTML-Datei
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    record = {
        "Künstler": "",
        "Titel": "",
        "Jahr": "",
        "Musik/Text": "",
        "Produzent": "",
        "Chartentry": "",
        "Letzte Chartposition": "",
        "Chartverlauf": "",
        "Anzahl Wochen": "",
        "URL": source_url,  # URL von der die Daten stammen
        "albums": []  # Liste für alle gefundenen Alben
    }
    
    # Künstler extrahieren
    artist_element = soup.find('h1', class_='big-head')
    if artist_element:
        record["Künstler"] = artist_element.get_text(strip=True)
    
    # Titel extrahieren (ohne "Single")
    title_element = soup.find('h2', class_='sub-head')
    if title_element:
        title_text = title_element.get_text(strip=True)
        # Entferne "Single" am Ende falls vorhanden
        title_text = re.sub(r'\s*Single\s*$', '', title_text)
        record["Titel"] = title_text
    
    # Chart-Informationen aus der Tabelle extrahieren
    table_rows = soup.find_all('tr')
    
    for row in table_rows:
        label_cell = row.find('td', class_='detail-info-label')
        if label_cell:
            label_text = label_cell.get_text(strip=True)
            value_cell = row.find('td', class_='detail-info-label').find_next_sibling('td')
            
            if label_text == "Jahr:" and value_cell:
                record["Jahr"] = value_cell.get_text(strip=True)
            
            elif label_text == "Musik/Text:" and value_cell:
                # Alle Links in der Zelle finden
                links = value_cell.find_all('a')
                if links:
                    # Falls Links vorhanden sind, Text aus Links extrahieren
                    composers = [link.get_text(strip=True) for link in links]
                else:
                    # Falls keine Links, direkten Text aus der Zelle nehmen
                    # Text in Listen-Items suchen
                    list_items = value_cell.find_all('li')
                    if list_items:
                        composers = [li.get_text(strip=True) for li in list_items if li.get_text(strip=True)]
                    else:
                        # Falls auch keine Listen-Items, gesamten Zelltext nehmen
                        cell_text = value_cell.get_text(strip=True)
                        composers = [cell_text] if cell_text else []
                record["Musik/Text"] = "; ".join(composers)
            
            elif label_text == "Produzent:" and value_cell:
                # Alle Links in der Zelle finden
                links = value_cell.find_all('a')
                if links:
                    # Falls Links vorhanden sind, Text aus Links extrahieren
                    producers = [link.get_text(strip=True) for link in links]
                else:
                    # Falls keine Links, direkten Text aus der Zelle nehmen
                    # Text in Listen-Items suchen
                    list_items = value_cell.find_all('li')
                    if list_items:
                        producers = [li.get_text(strip=True) for li in list_items if li.get_text(strip=True)]
                    else:
                        # Falls auch keine Listen-Items, gesamten Zelltext nehmen
                        cell_text = value_cell.get_text(strip=True)
                        producers = [cell_text] if cell_text else []
                record["Produzent"] = "; ".join(producers)
            
            elif label_text == "Chartentry:" and value_cell:
                record["Chartentry"] = value_cell.get_text(strip=True)
            
            elif label_text == "Letzte Chartposition:" and value_cell:
                record["Letzte Chartposition"] = value_cell.get_text(strip=True)
            
            elif label_text == "Anzahl Wochen:" and value_cell:
                record["Anzahl Wochen"] = value_cell.get_text(strip=True)
    
        # Chartverlauf extrahieren
        chart_element = soup.find('div', id='chartgraph')
        if chart_element and chart_element.get('data-chart'):
            chart_data = chart_element.get('data-chart')
            # Extrahiere den kompletten Wert inklusive nulls
            record["Chartverlauf"] = chart_data     
    
    return record

def fetch_url_content(url, session, logger=None):
    """
    Ruft eine URL auf und gibt den HTML-Inhalt zurück mit Retry-Logik
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(CONFIG['max_retries']):
        try:
            # Kleine zufällige Verzögerung um Server nicht zu überlasten
            if CONFIG['request_delay'] > 0:
                time.sleep(CONFIG['request_delay'])
            
            response = session.get(url, headers=headers, timeout=CONFIG['timeout'])
            
            # Prüfe auf verschiedene Fehlertypen
            if response.status_code == 403:
                logger.warning(f"403 Forbidden für {url} - möglicherweise IP geblockt (Versuch {attempt + 1}/{CONFIG['max_retries']})")
                if attempt < CONFIG['max_retries'] - 1:
                    logger.info(f"Warte {CONFIG['blocked_retry_delay']} Sekunden vor erneutem Versuch...")
                    time.sleep(CONFIG['blocked_retry_delay'])
                    continue
                    
            elif response.status_code == 429:
                logger.warning(f"429 Too Many Requests für {url} (Versuch {attempt + 1}/{CONFIG['max_retries']})")
                if attempt < CONFIG['max_retries'] - 1:
                    logger.info(f"Warte {CONFIG['retry_delay']} Sekunden vor erneutem Versuch...")
                    time.sleep(CONFIG['retry_delay'])
                    continue
                    
            elif response.status_code >= 500:
                logger.warning(f"Server-Fehler {response.status_code} für {url} (Versuch {attempt + 1}/{CONFIG['max_retries']})")
                if attempt < CONFIG['max_retries'] - 1:
                    logger.info(f"Warte {CONFIG['retry_delay']} Sekunden vor erneutem Versuch...")
                    time.sleep(CONFIG['retry_delay'])
                    continue
            
            # Bei Erfolg Response zurückgeben
            response.raise_for_status()
            return response.text
            
        except requests.exceptions.ConnectTimeout:
            logger.warning(f"Verbindungs-Timeout für {url} (Versuch {attempt + 1}/{CONFIG['max_retries']})")
            if attempt < CONFIG['max_retries'] - 1:
                time.sleep(CONFIG['retry_delay'])
                continue
                
        except requests.exceptions.ReadTimeout:
            logger.warning(f"Lese-Timeout für {url} (Versuch {attempt + 1}/{CONFIG['max_retries']})")
            if attempt < CONFIG['max_retries'] - 1:
                time.sleep(CONFIG['retry_delay'])
                continue
                
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Verbindungsfehler für {url}: {e} (Versuch {attempt + 1}/{CONFIG['max_retries']})")
            if attempt < CONFIG['max_retries'] - 1:
                time.sleep(CONFIG['retry_delay'])
                continue
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request-Fehler für {url}: {e} (Versuch {attempt + 1}/{CONFIG['max_retries']})")
            if attempt < CONFIG['max_retries'] - 1:
                time.sleep(CONFIG['retry_delay'])
                continue
    
    # Alle Versuche fehlgeschlagen
    logger.error(f"Alle {CONFIG['max_retries']} Versuche für {url} fehlgeschlagen")
    return None

def read_urls_from_file(file_path):
    """
    Liest URLs aus einer Textdatei
    """
    urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                url = line.strip()
                if url:  # Nur nicht-leere Zeilen
                    urls.append(url)
        return urls
    except FileNotFoundError:
        print(f"Datei {file_path} nicht gefunden")
        return []
    except Exception as e:
        print(f"Fehler beim Lesen der Datei {file_path}: {e}")
        return []

def prepare_csv_record(record):
    """
    Bereitet einen Record für die CSV-Ausgabe vor und erstellt separate Album-Spalten
    """
    csv_record = {
        "Künstler": record.get("Künstler", ""),
        "Titel": record.get("Titel", ""),
        "Jahr": record.get("Jahr", ""),
        "Musik/Text": record.get("Musik/Text", ""),
        "Produzent": record.get("Produzent", ""),
        "Chartentry": record.get("Chartentry", ""),
        "Letzte Chartposition": record.get("Letzte Chartposition", ""),
        "Chartverlauf": record.get("Chartverlauf", ""),
        "Anzahl Wochen": record.get("Anzahl Wochen", ""),
        "URL": record.get("URL", "")
    }
    
    # Album-Informationen in separate Spalten aufteilen
    albums = record.get("albums", [])
    for i in range(CONFIG['max_albums_per_song']):
        album_key = f"Album {i + 1}"
        year_key = f"Jahr Album {i + 1}"
        ean_key = f"EAN Identifier {i + 1}"
        
        if i < len(albums):
            csv_record[album_key] = albums[i].get("name", "")
            csv_record[year_key] = albums[i].get("year", "")
            csv_record[ean_key] = albums[i].get("ean", "")
        else:
            csv_record[album_key] = ""
            csv_record[year_key] = ""
            csv_record[ean_key] = ""
    
    return csv_record

def save_to_csv(records, filename=None, mode='w'):
    """
    Speichert die extrahierten Daten in eine CSV-Datei
    """
    if not records:
        return
    
    # Verwende Standard-Dateiname aus Config falls nicht angegeben
    if filename is None:
        filename = CONFIG['output_csv']
    
    # Verzeichnis erstellen falls es nicht existiert
    csv_dir = os.path.dirname(filename)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    # Fieldnames für CSV definieren
    fieldnames = [
        "Künstler", "Titel", "Jahr", "Musik/Text", "Produzent",
        "Chartentry", "Letzte Chartposition", "Chartverlauf", 
        "Anzahl Wochen", "URL"
    ]
    
    # Album-Spalten hinzufügen
    for i in range(CONFIG['max_albums_per_song']):
        fieldnames.append(f"Album {i + 1}")
        fieldnames.append(f"Jahr Album {i + 1}")
        fieldnames.append(f"EAN Identifier {i + 1}")
    
    file_exists = os.path.exists(filename)
    
    with open(filename, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        
        # Header nur schreiben wenn neue Datei oder 'w' Modus
        if not file_exists or mode == 'w':
            writer.writeheader()
            
        for record in records:
            csv_record = prepare_csv_record(record)
            writer.writerow(csv_record)

def save_records_incrementally(all_records, last_saved_count, logger):
    """
    Speichert neue Records inkrementell in die CSV (Thread-safe)
    """
    with save_lock:
        new_records = all_records[last_saved_count:]
        if new_records:
            mode = 'a' if last_saved_count > 0 else 'w'
            save_to_csv(new_records, mode=mode)
            logger.info(f"CSV aktualisiert: {len(new_records)} neue Einträge hinzugefügt (Gesamt: {len(all_records)})")
        return len(all_records)

def add_album_to_record(record, album_name, album_year="", ean_identifier=""):
    """
    Fügt ein Album zu einem Record hinzu, falls es noch nicht vorhanden ist
    """
    if not album_name:
        return False
    
    # Prüfen ob Album bereits vorhanden ist
    for album in record.get("albums", []):
        if album.get("name", "").lower().strip() == album_name.lower().strip():
            # Album bereits vorhanden, ggf. Jahr oder EAN hinzufügen
            if album_year and not album.get("year"):
                album["year"] = album_year
            if ean_identifier and not album.get("ean"):
                album["ean"] = ean_identifier
            return False
    
    # Album hinzufügen
    if "albums" not in record:
        record["albums"] = []
    
    record["albums"].append({
        "name": album_name,
        "year": album_year,
        "ean": ean_identifier
    })
    
    return True

def process_album_matching(url, session, all_records, logger):
    """
    Verarbeitet Album-URLs und matcht Tracks zu bereits gescrapten Songs
    """
    # HTML der Hauptseite abrufen (falls noch nicht geschehen)
    html_content = fetch_url_content(url, session, logger)
    if not html_content:
        return 0
    
    # Album-URLs extrahieren
    album_urls = extract_album_urls(html_content)
    if not album_urls:
        logger.info("📀 Keine Album-URLs gefunden")
        return 0
    
    logger.info(f"📀 Gefunden: {len(album_urls)} Album-URLs")
    
    # Song-Lookup Dictionary erstellen (Thread-safe)
    with records_lock:
        song_lookup = {}
        for i, record in enumerate(all_records):
            artist = record.get("Künstler", "").lower().strip()
            title = record.get("Titel", "").lower().strip()
            if artist and title:
                key = f"{artist}|{title}"
                if key not in song_lookup:
                    song_lookup[key] = []
                song_lookup[key].append(i)
    
    matches_found = 0
    
    # Jede Album-URL verarbeiten
    for i, album_url in enumerate(album_urls, 1):
        logger.info(f"💿 {i}/{len(album_urls)}: {album_url}")
        
        album_html = fetch_url_content(album_url, session, logger)
        if album_html:
            try:
                album_name, album_year, tracks = extract_album_tracks(album_html)
                
                if album_name and tracks:
                    logger.info(f"📀 Album: {album_name} ({album_year}) - {len(tracks)} Tracks")
                    
                    # Jeden Track im Album prüfen
                    for track in tracks:
                        # Normalisierte Suche
                        artist_norm = track['artist'].lower().strip()
                        title_norm = track['title'].lower().strip()
                        key = f"{artist_norm}|{title_norm}"
                        
                        # Prüfen ob dieser Track in unseren gescrapten Songs vorhanden ist
                        if key in song_lookup:
                            # Album-Information zu allen passenden Records hinzufügen (Thread-safe)
                            with records_lock:
                                for record_index in song_lookup[key]:
                                    if record_index < len(all_records):
                                        record = all_records[record_index]
                                        
                                        # Album hinzufügen
                                        if add_album_to_record(record, album_name, album_year, track.get('ean_identifier', '')):
                                            matches_found += 1
                                            logger.info(f"🎯 Match: {track['artist']} - {track['title']} → {album_name} ({album_year})")
                else:
                    logger.warning(f"⚠ Album-Daten nicht extrahierbar: {album_url}")
                    
            except Exception as e:
                logger.error(f"✗ Fehler beim Verarbeiten des Albums: {album_url} - {e}")
        else:
            logger.error(f"✗ Fehler beim Abrufen des Albums: {album_url}")
    
    logger.info(f"📊 Album-Matching abgeschlossen: {matches_found} neue Album-Zuordnungen")
    return matches_found

def process_url_with_deep_scraping(url, all_records, logger):
    """
    Verarbeitet eine URL und optional auch alle verwandten Song-URLs (für Threading optimiert)
    """
    # Neue Session für jeden Thread
    session = requests.Session()
    
    logger.info(f"Hauptseite: {url}")
    
    # Hauptseite abrufen
    html_content = fetch_url_content(url, session, logger)
    if not html_content:
        logger.error(f"Fehler beim Abrufen der Hauptseite: {url}")
        return False
    
    # Schneller Duplikat-Check vor vollständiger Extraktion (Thread-safe)
    artist, title = extract_basic_info_from_html(html_content)
    with records_lock:
        if is_duplicate_song(artist, title, all_records):
            logger.info(f"⏭ Duplikat übersprungen (Hauptseite): {artist} - {title}")
            return True  # Als erfolgreich zählen, da bereits vorhanden
    
    try:
        # Daten der Hauptseite extrahieren
        main_record = extract_chart_data(html_content, url)
        
        if main_record["Künstler"] or main_record["Titel"]:
            # Record thread-safe hinzufügen
            with records_lock:
                all_records.append(main_record)
            
            logger.info(f"✓ Hauptseite: {main_record['Künstler']} - {main_record['Titel']}")
            
            # Deep Scraping aktiviert?
            if CONFIG['deep_scraping']:
                related_urls = extract_related_song_urls(html_content)
                
                if related_urls:
                    logger.info(f"📂 Gefunden: {len(related_urls)} verwandte Titel")
                    
                    # Begrenzen auf max_related_songs
                    if len(related_urls) > CONFIG['max_related_songs']:
                        related_urls = related_urls[:CONFIG['max_related_songs']]
                        logger.info(f"⚠ Begrenzt auf {CONFIG['max_related_songs']} Titel")
                    
                    processed_related = 0
                    skipped_duplicates = 0
                    
                    # Jede verwandte URL verarbeiten
                    for i, related_url in enumerate(related_urls, 1):
                        logger.info(f"🎵 {i}/{len(related_urls)}: {related_url}")
                        
                        related_html = fetch_url_content(related_url, session, logger)
                        if related_html:
                            # Duplikat-Check für verwandte URL (Thread-safe)
                            related_artist, related_title = extract_basic_info_from_html(related_html)
                            
                            with records_lock:
                                if is_duplicate_song(related_artist, related_title, all_records):
                                    logger.info(f"⏭ Duplikat übersprungen: {related_artist} - {related_title}")
                                    skipped_duplicates += 1
                                    continue
                            
                            try:
                                related_record = extract_chart_data(related_html, related_url)
                                
                                if related_record["Künstler"] or related_record["Titel"]:
                                    # Record thread-safe hinzufügen
                                    with records_lock:
                                        all_records.append(related_record)
                                    
                                    logger.info(f"✓ {related_record['Künstler']} - {related_record['Titel']}")
                                    processed_related += 1
                                else:
                                    logger.warning(f"⚠ Keine Daten gefunden: {related_url}")
                                    
                            except Exception as e:
                                logger.error(f"✗ Fehler beim Extrahieren: {related_url} - {e}")
                        else:
                            logger.error(f"✗ Fehler beim Abrufen: {related_url}")
                    
                    logger.info(f"📊 Verwandte Titel - Verarbeitet: {processed_related}, Duplikate übersprungen: {skipped_duplicates}")
                else:
                    logger.info("📂 Keine verwandten Titel gefunden")
            
            # Album-Matching durchführen (nach dem Deep Scraping)
            if CONFIG['album_matching']:
                album_matches = process_album_matching(url, session, all_records, logger)
            
            return True
        else:
            logger.warning(f"⚠ Keine relevanten Daten auf Hauptseite gefunden: {url}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Fehler beim Extrahieren der Hauptseite: {url} - {e}")
        return False

def main():
    # Logging einrichten
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("SCRAPER GESTARTET")
    logger.info("=" * 60)
    
    # URLs aus Textdatei lesen (Pfad aus Config)
    urls = read_urls_from_file(CONFIG['urls_file'])
    
    # Progress laden und ggf. fortsetzen
    start_index = load_progress()
    original_url_count = len(urls)
    
    if start_index > 0:
        print(f"📍 Setze bei URL {start_index + 1} fort (Zeile {start_index + 1} in der Datei)")
        logger.info(f"Resume: Starte bei URL-Index {start_index}")
        urls = urls[start_index:]  # Überspringe bereits verarbeitete URLs
    
    # Für Testzwecke begrenzen falls test_limit gesetzt ist
    if CONFIG['test_limit'] is not None:
        urls = urls[:CONFIG['test_limit']]
        logger.info(f"TEST-MODUS: Begrenzt auf {CONFIG['test_limit']} URLs")
    
    if not urls:
        logger.error("Keine URLs gefunden oder Datei konnte nicht gelesen werden")
        print("❌ Keine URLs gefunden!")
        return
    
    logger.info(f"Gefunden: {len(urls)} URLs (Start-Index: {start_index}, Original: {original_url_count})")
    logger.info(f"Input-Datei: {CONFIG['urls_file']}")
    logger.info(f"Output-Datei: {CONFIG['output_csv']}")
    logger.info(f"Log-Datei: {CONFIG['log_file']}")
    logger.info(f"Deep Scraping: {'Aktiviert' if CONFIG['deep_scraping'] else 'Deaktiviert'}")
    logger.info(f"Album Matching: {'Aktiviert' if CONFIG['album_matching'] else 'Deaktiviert'}")
    logger.info(f"CSV Speicher-Intervall: alle {CONFIG['save_interval']} Einträge")
    logger.info(f"Max. Alben pro Song: {CONFIG['max_albums_per_song']}")
    logger.info(f"Max. Workers: {CONFIG['max_workers']}")
    logger.info(f"Max. Retries: {CONFIG['max_retries']}")
    
    all_records = []
    successful = 0
    failed = 0
    last_saved_count = 0
    
    # Progress bar initialisieren
    with tqdm(total=len(urls), desc="URLs verarbeitet", unit="URL") as pbar:
        
        # ThreadPoolExecutor für parallele Verarbeitung
        with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
            # Alle URLs als Futures submitten
            future_to_url = {
                executor.submit(process_url_with_deep_scraping, url, all_records, logger): url 
                for url in urls
            }
            
            # Futures verarbeiten, sobald sie fertig sind
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                
                try:
                    success = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        
                except Exception as exc:
                    logger.error(f"✗ Exception bei URL {url}: {exc}")
                    failed += 1
                
                # Progress bar aktualisieren
                pbar.update(1)
                pbar.set_description(f"URLs: {successful + failed}/{len(urls)} | Songs: {len(all_records)} | ✓{successful} ✗{failed}")
                
                # Progress speichern (alle 10 URLs)
                if (successful + failed) % 10 == 0:
                    save_progress(start_index + successful + failed)
                
                # CSV inkrementell speichern alle X Einträge
                if len(all_records) - last_saved_count >= CONFIG['save_interval']:
                    last_saved_count = save_records_incrementally(all_records, last_saved_count, logger)
    
    # Finale CSV-Speicherung (falls noch nicht gespeicherte Records vorhanden)
    if len(all_records) > last_saved_count:
        save_records_incrementally(all_records, last_saved_count, logger)
    
    # Progress-Datei löschen wenn erfolgreich abgeschlossen
    if os.path.exists(CONFIG['progress_file']):
        os.remove(CONFIG['progress_file'])
        logger.info("Progress-Datei gelöscht - Scraping vollständig abgeschlossen")
    
    # Zusammenfassung
    logger.info("=" * 60)
    logger.info("ZUSAMMENFASSUNG:")
    logger.info(f"Erfolgreich verarbeitet: {successful}")
    logger.info(f"Fehlgeschlagen: {failed}")
    logger.info(f"Gesamt URLs verarbeitet: {len(urls)}")
    logger.info(f"Start-Index: {start_index}")
    logger.info(f"Original URL-Anzahl: {original_url_count}")
    logger.info(f"Extrahierte Songs: {len(all_records)}")
    logger.info(f"CSV-Datei: {CONFIG['output_csv']}")
    logger.info("SCRAPER BEENDET")
    logger.info("=" * 60)
    
    # Kurze Zusammenfassung im Terminal
    print(f"\n✅ Scraping abgeschlossen!")
    if start_index > 0:
        print(f"📍 Fortgesetzt von URL {start_index + 1}")
    print(f"📊 {len(all_records)} Songs extrahiert aus {successful}/{len(urls)} URLs")
    print(f"💾 Daten gespeichert in: {CONFIG['output_csv']}")
    print(f"📝 Logs verfügbar in: {CONFIG['log_file']}")
    
    if failed > 0:
        print(f"⚠️  {failed} URLs fehlgeschlagen (siehe Log für Details)")

if __name__ == "__main__":
    main()