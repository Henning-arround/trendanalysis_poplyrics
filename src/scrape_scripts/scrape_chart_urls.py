import pandas as pd
import requests
import time
import csv
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

# KONFIGURATION - NUR LYRICS.OVH
class Config:
    MAX_WORKERS = 5             # Mehr Worker da lyrics.ovh keine Rate Limits hat
    REQUEST_TIMEOUT = 5          # Kurze Timeouts, schnell aufgeben
    RATE_LIMIT_DELAY = 0.1      # Minimal delay
    MIN_LYRICS_LENGTH = 30       # Reduziert für lyrics.ovh
    
    # Retry-Mechanismus für Timeouts
    MAX_RETRIES = 2             # 2 Versuche pro Song
    RETRY_DELAY = 1             # 1s zwischen Retries
    
    # Batch Processing
    BATCH_SIZE = 5000           # Große Batches da API schnell ist
    SAVE_PROGRESS = True
    
    # API URL
    LYRICS_OVH_URL = "https://api.lyrics.ovh/v1/{artist}/{title}"
    
    # Request Session für bessere Performance
    USE_SESSION = True

class LyricsOVHScraper:
    def __init__(self, log_file="lyrics_ovh_not_found.csv"):
        self.log_file = log_file
        self.setup_logging()
        
        # Session für bessere Performance
        if Config.USE_SESSION:
            self.session = requests.Session()
            # Session optimieren
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=Config.MAX_WORKERS,
                pool_maxsize=Config.MAX_WORKERS * 2,
                max_retries=0  # Wir machen eigene Retries
            )
            self.session.mount('https://', adapter)
            self.session.mount('http://', adapter)
        else:
            self.session = requests
        
        self.stats = {
            'processed': 0,
            'found': 0,
            'not_found': 0,
            'timeouts': 0,
            'errors': 0,
            'batch_count': 0
        }
    
    def setup_logging(self):
        """Setup für Logging"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Zeitstempel', 'Index', 'Künstler', 'Titel', 'Grund'])
    
    def log_not_found(self, index, artist, title, reason):
        """Loggt nicht gefundene Songs"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, index, artist, title, reason])
    
    def clean_artist_title(self, text):
        """Bereinigt Artist/Title für bessere API-Kompatibilität"""
        if not text:
            return ""
        
        # Entferne problematische Zeichen
        text = str(text).strip()
        
        # Entferne common features/remix indicators
        removals = [
            " feat.", " ft.", " featuring", 
            " (feat.", " (ft.", " (featuring",
            " - feat.", " - ft.",
            " [feat.", " [ft.",
            " remix", " (remix)", " [remix]",
            " live", " (live)", " [live]",
            " acoustic", " (acoustic)", " [acoustic]"
        ]
        
        text_lower = text.lower()
        for removal in removals:
            if removal in text_lower:
                # Find position case-insensitive
                pos = text_lower.find(removal)
                if pos != -1:
                    text = text[:pos]
                    break
        
        return text.strip()
    
    def fetch_lyrics_ovh(self, index, artist, title):
        """Lyrics von lyrics.ovh API mit verbessertem Error Handling"""
        
        # Artist/Title bereinigen
        clean_artist = self.clean_artist_title(artist)
        clean_title = self.clean_artist_title(title)
        
        if not clean_artist or not clean_title:
            self.log_not_found(index, artist, title, "Leerer Artist/Title nach Bereinigung")
            return {
                'success': False,
                'index': index,
                'artist': artist,
                'title': title,
                'lyrics': "",
                'source': 'invalid_input'
            }
        
        # URL erstellen
        try:
            url = Config.LYRICS_OVH_URL.format(
                artist=requests.utils.quote(clean_artist, safe=''),
                title=requests.utils.quote(clean_title, safe='')
            )
        except Exception as e:
            self.log_not_found(index, artist, title, f"URL-Erstellung fehlgeschlagen: {str(e)}")
            return {
                'success': False,
                'index': index,
                'artist': artist,
                'title': title,
                'lyrics': "",
                'source': 'url_error'
            }
        
        # Versuche API-Aufruf mit Retries
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        lyrics = data.get("lyrics", "").strip()
                        
                        if lyrics and len(lyrics) > Config.MIN_LYRICS_LENGTH:
                            return {
                                'success': True,
                                'index': index,
                                'artist': artist,
                                'title': title,
                                'lyrics': lyrics,
                                'source': 'lyrics.ovh'
                            }
                        else:
                            # Leere Antwort oder zu kurz
                            break
                            
                    except ValueError as e:
                        # JSON decode error
                        self.log_not_found(index, artist, title, f"JSON-Parse-Fehler: {str(e)}")
                        return {
                            'success': False,
                            'index': index,
                            'artist': artist,
                            'title': title,
                            'lyrics': "",
                            'source': 'json_error'
                        }
                
                elif response.status_code == 404:
                    # Song nicht gefunden - kein Retry nötig
                    break
                    
                elif response.status_code == 429:
                    # Rate limit - warten und retry
                    if attempt < Config.MAX_RETRIES - 1:
                        time.sleep(Config.RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        break
                        
                else:
                    # Andere HTTP-Fehler
                    if attempt < Config.MAX_RETRIES - 1:
                        time.sleep(Config.RETRY_DELAY)
                        continue
                    else:
                        self.log_not_found(index, artist, title, f"HTTP {response.status_code}")
                        return {
                            'success': False,
                            'index': index,
                            'artist': artist,
                            'title': title,
                            'lyrics': "",
                            'source': 'http_error'
                        }
            
            except requests.exceptions.Timeout:
                # Timeout - retry
                if attempt < Config.MAX_RETRIES - 1:
                    continue
                else:
                    self.log_not_found(index, artist, title, "Timeout nach allen Versuchen")
                    self.stats['timeouts'] += 1
                    return {
                        'success': False,
                        'index': index,
                        'artist': artist,
                        'title': title,
                        'lyrics': "",
                        'source': 'timeout'
                    }
            
            except requests.exceptions.RequestException as e:
                # Andere Request-Fehler
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)
                    continue
                else:
                    self.log_not_found(index, artist, title, f"Request-Fehler: {str(e)}")
                    self.stats['errors'] += 1
                    return {
                        'success': False,
                        'index': index,
                        'artist': artist,
                        'title': title,
                        'lyrics': "",
                        'source': 'request_error'
                    }
            
            except Exception as e:
                # Unerwartete Fehler
                self.log_not_found(index, artist, title, f"Unerwarteter Fehler: {str(e)}")
                self.stats['errors'] += 1
                return {
                    'success': False,
                    'index': index,
                    'artist': artist,
                    'title': title,
                    'lyrics': "",
                    'source': 'unexpected_error'
                }
        
        # Nicht gefunden nach allen Versuchen
        self.log_not_found(index, artist, title, "Nicht gefunden (alle Versuche)")
        return {
            'success': False,
            'index': index,
            'artist': artist,
            'title': title,
            'lyrics': "",
            'source': 'not_found'
        }
    
    def get_lyrics_for_song(self, song_data):
        """Wrapper für Threading"""
        index, artist, title = song_data
        
        result = self.fetch_lyrics_ovh(index, artist, title)
        
        # Minimal delay um Server nicht zu überlasten
        time.sleep(Config.RATE_LIMIT_DELAY)
        
        return result
    
    def load_existing_results(self, output_csv):
        """Lädt bereits verarbeitete Ergebnisse"""
        if os.path.exists(output_csv):
            try:
                existing_df = pd.read_csv(output_csv)
                processed_indices = set(existing_df['Index'].tolist())
                print(f"📄 {len(processed_indices)} bereits verarbeitete Songs gefunden")
                return processed_indices, existing_df
            except Exception as e:
                print(f"⚠️  Warnung beim Laden bestehender Ergebnisse: {e}")
        return set(), pd.DataFrame()
    
    def process_csv_lyrics_ovh(self, input_csv: str, output_csv: str):
        """Verarbeitet CSV nur mit lyrics.ovh API"""
        print("🎯 LYRICS.OVH ONLY SCRAPER")
        print("=" * 50)
        print(f"📁 Input: {input_csv}")
        print(f"💾 Output: {output_csv}")
        print(f"📝 Log: {self.log_file}")
        print(f"⚙️  Workers: {Config.MAX_WORKERS}")
        print(f"📦 Batch Size: {Config.BATCH_SIZE}")
        print(f"⏱️  Timeout: {Config.REQUEST_TIMEOUT}s")
        print(f"🔄 Max Retries: {Config.MAX_RETRIES}")
        print("-" * 50)
        
        # CSV einlesen
        try:
            df = pd.read_csv(input_csv, low_memory=False)
            print(f"📊 {len(df)} Songs in Input-Datei gefunden")
        except Exception as e:
            print(f"❌ Fehler beim Lesen der CSV: {e}")
            return
        
        # Index hinzufügen falls nötig
        if 'Index' not in df.columns:
            df.insert(0, 'Index', range(1, len(df) + 1))
            print("📋 Index-Spalte hinzugefügt")
        
        # Bereits verarbeitete Songs laden
        processed_indices, existing_df = self.load_existing_results(output_csv)
        
        # Nicht verarbeitete Songs filtern
        if processed_indices:
            df = df[~df['Index'].isin(processed_indices)]
            print(f"⏭️  {len(df)} Songs verbleiben zu verarbeiten")
        
        if len(df) == 0:
            print("✅ Alle Songs bereits verarbeitet!")
            return
        
        # Batch-Verarbeitung
        all_new_results = []
        total_batches = (len(df) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
        
        print(f"🚀 Starte Verarbeitung in {total_batches} Batches")
        
        for start_idx in range(0, len(df), Config.BATCH_SIZE):
            end_idx = min(start_idx + Config.BATCH_SIZE, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            self.stats['batch_count'] += 1
            
            print(f"\n🔄 Batch {self.stats['batch_count']}/{total_batches}")
            print(f"📈 Songs {start_idx + 1}-{end_idx} von {len(df)}")
            
            # Batch verarbeiten
            batch_results = self.process_batch(batch_df)
            all_new_results.extend(batch_results)
            
            # Zwischenspeichern
            if Config.SAVE_PROGRESS:
                self.save_results(output_csv, existing_df, all_new_results)
                print(f"💾 Batch {self.stats['batch_count']} gespeichert")
            
            # Batch-Statistiken
            batch_found = sum(1 for r in batch_results if r.get('Lyrics'))
            print(f"✅ {batch_found}/{len(batch_results)} Songs in diesem Batch gefunden")
            
            # Performance-Info
            if self.stats['timeouts'] > 0:
                print(f"⏱️  Timeouts in diesem Lauf: {self.stats['timeouts']}")
        
        # Finale Speicherung
        self.save_results(output_csv, existing_df, all_new_results)
        self.print_final_stats(output_csv)
    
    def process_batch(self, batch_df):
        """Verarbeitet einen Batch mit Threading"""
        song_data_list = [
            (row['Index'], row['Künstler'], row['Titel']) 
            for _, row in batch_df.iterrows()
        ]
        
        results = []
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            # Alle Tasks einreichen
            future_to_song = {
                executor.submit(self.get_lyrics_for_song, song_data): song_data 
                for song_data in song_data_list
            }
            
            # Progress Bar für aktuellen Batch
            for future in tqdm(
                as_completed(future_to_song), 
                total=len(song_data_list), 
                desc="🎵 lyrics.ovh",
                leave=False,
                unit="song"
            ):
                result = future.result()
                self.stats['processed'] += 1
                
                if result['success']:
                    results.append({
                        'Index': result['index'],
                        'Artist': result['artist'],
                        'Title': result['title'],
                        'Lyrics': result['lyrics'],
                        'Source': 'lyrics.ovh'
                    })
                    self.stats['found'] += 1
                else:
                    self.stats['not_found'] += 1
        
        return results
    
    def save_results(self, output_csv, existing_df, new_results):
        """Speichert Ergebnisse (bestehende + neue)"""
        try:
            # Neue Ergebnisse zu DataFrame
            if new_results:
                new_df = pd.DataFrame(new_results)
                
                # Kombiniere mit bestehenden Ergebnissen
                if not existing_df.empty:
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = new_df
                
                # Nach Index sortieren
                combined_df = combined_df.sort_values('Index').reset_index(drop=True)
                
                # Speichern
                combined_df.to_csv(output_csv, index=False, encoding='utf-8')
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern: {e}")
    
    def print_final_stats(self, output_file):
        """Finale Statistiken"""
        print("\n" + "=" * 60)
        print("🎯 LYRICS.OVH FINAL STATS")
        print("=" * 60)
        print(f"Verarbeitete Songs: {self.stats['processed']}")
        print(f"Gefundene Lyrics: {self.stats['found']}")
        print(f"Nicht gefunden: {self.stats['not_found']}")
        print(f"Timeouts: {self.stats['timeouts']}")
        print(f"Andere Fehler: {self.stats['errors']}")
        print(f"Verarbeitete Batches: {self.stats['batch_count']}")
        
        if self.stats['processed'] > 0:
            success_rate = (self.stats['found'] / self.stats['processed']) * 100
            timeout_rate = (self.stats['timeouts'] / self.stats['processed']) * 100
            print(f"Erfolgsrate: {success_rate:.1f}%")
            print(f"Timeout-Rate: {timeout_rate:.1f}%")
        
        print(f"\n💾 Ergebnisse: {output_file}")
        print(f"📝 Log: {self.log_file}")
        
        # Zeige Log-Zusammenfassung
        self.show_log_summary()
    
    def show_log_summary(self):
        """Zeigt Zusammenfassung der nicht gefundenen Songs"""
        if not os.path.exists(self.log_file):
            return
        
        try:
            log_df = pd.read_csv(self.log_file)
            if len(log_df) > 0:
                print(f"\n📋 Nicht gefundene Songs: {len(log_df)}")
                
                # Gründe analysieren
                if 'Grund' in log_df.columns:
                    reasons = log_df['Grund'].value_counts()
                    print("Top 5 Gründe:")
                    for reason, count in reasons.head(5).items():
                        print(f"  {reason}: {count}")
                
                print(f"Vollständige Liste: {self.log_file}")
        except Exception as e:
            print(f"⚠️  Fehler beim Lesen des Logs: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'session') and hasattr(self.session, 'close'):
            self.session.close()

# Analyse-Funktionen
def analyze_results(results_csv, log_csv="lyrics_ovh_not_found.csv"):
    """Analysiert die Ergebnisse"""
    print("\n📊 ERGEBNIS-ANALYSE")
    print("=" * 40)
    
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        print(f"✅ Erfolgreich: {len(df)} Songs mit Lyrics")
        
        if len(df) > 0:
            avg_length = df['Lyrics'].str.len().mean()
            print(f"📏 Durchschnittliche Lyrics-Länge: {avg_length:.0f} Zeichen")
    
    if os.path.exists(log_csv):
        log_df = pd.read_csv(log_csv)
        print(f"❌ Nicht gefunden: {len(log_df)} Songs")

def estimate_completion_time(results_csv, total_songs=386888):
    """Schätzt verbleibende Zeit"""
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        completed = len(df)
        remaining = total_songs - completed
        
        print(f"\n⏱️  FORTSCHRITT:")
        print(f"Abgeschlossen: {completed:,} / {total_songs:,} ({completed/total_songs*100:.1f}%)")
        print(f"Verbleibend: {remaining:,} Songs")

# Verwendung
if __name__ == "__main__":
    print("🎯 LYRICS.OVH ONLY SCRAPER")
    print("Optimiert für große Datenmengen mit robustem Error-Handling")
    print()
    
    scraper = LyricsOVHScraper(log_file="lyrics_ovh_not_found.csv")
    
    try:
        # Nur lyrics.ovh verwenden
        scraper.process_csv_lyrics_ovh(
            '../data/charts_with_language.csv',
            '../data/lyrics_ovh_only.csv'
        )
        
        # Ergebnisse analysieren
        analyze_results('../data/lyrics_ovh_only.csv', 'lyrics_ovh_not_found.csv')
        estimate_completion_time('../data/lyrics_ovh_only.csv')
        
    except KeyboardInterrupt:
        print("\n⏹️  Script durch Benutzer gestoppt")
        print("💾 Fortschritt wurde gespeichert - kann später fortgesetzt werden")
        
    finally:
        # Cleanup
        scraper.cleanup()