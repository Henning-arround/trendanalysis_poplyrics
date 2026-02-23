import pandas as pd
import requests
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import urllib.parse
import os
from pathlib import Path

# Logging konfigurieren (nur in Datei, nicht in Terminal)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lyrics_scraper.log')
    ]
)
logger = logging.getLogger(__name__)

class LyricsScraper:
    def __init__(self, max_workers=5, delay=0.1):
        """
        max_workers: Anzahl der parallelen Threads
        delay: Verzögerung zwischen Requests (in Sekunden)
        """
        self.max_workers = max_workers
        self.delay = delay
        self.base_url = "https://api.lyrics.ovh/v1"
        
    def clean_text(self, text):
        """Text für URL-Encoding bereinigen"""
        if pd.isna(text):
            return ""
        # Grundlegende Bereinigung
        text = str(text).strip()
        # Entfernen von feat., ft., etc.
        import re
        text = re.sub(r'\s*\(.*?\)\s*', '', text)  # Entfernt Klammern und Inhalt
        text = re.sub(r'\s*feat\..*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*ft\..*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*featuring.*', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    def get_lyrics(self, artist, title, index):
        """Einzelnen Songtext von lyrics.ovh API abrufen"""
        try:
            # Text bereinigen
            clean_artist = self.clean_text(artist)
            clean_title = self.clean_text(title)
            
            if not clean_artist or not clean_title:
                return index, None, f"Leerer Künstler oder Titel"
            
            # URL erstellen
            url = f"{self.base_url}/{urllib.parse.quote(clean_artist)}/{urllib.parse.quote(clean_title)}"
            
            # Request senden
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                lyrics = data.get('lyrics', '')
                if lyrics.strip():
                    logger.info(f"✓ Erfolgreich: {artist} - {title}")
                    return index, lyrics.strip(), None
                else:
                    error_msg = "Leerer Songtext"
                    logger.info(f"✗ {artist} - {title}: {error_msg}")
                    return index, None, error_msg
            else:
                error_msg = f"HTTP {response.status_code}"
                logger.info(f"✗ {artist} - {title}: {error_msg}")
                return index, None, error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "Timeout"
            logger.info(f"✗ {artist} - {title}: {error_msg}")
            return index, None, error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Request Error: {str(e)}"
            logger.info(f"✗ {artist} - {title}: {error_msg}")
            return index, None, error_msg
        except Exception as e:
            error_msg = f"Unbekannter Fehler: {str(e)}"
            logger.info(f"✗ {artist} - {title}: {error_msg}")
            return index, None, error_msg
    
    def _save_intermediate_results(self, results, output_file, processed_count):
        """Zwischenergebnisse speichern"""
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('Index')  # Nach Index sortieren
            results_df.to_csv(output_file, index=False)
            logger.info(f"Zwischenspeicherung nach {processed_count} Songs: {len(results)} Lyrics gespeichert")
    
    def scrape_lyrics(self, input_file="data/charts_with_language.csv", output_file="lyrics.csv"):
        """Hauptfunktion zum Scrapen aller Lyrics"""
        
        # CSV einlesen
        try:
            df = pd.read_csv(input_file)
            logger.info(f"CSV erfolgreich geladen: {len(df)} Zeilen")
        except Exception as e:
            print(f"FEHLER: CSV konnte nicht geladen werden: {e}")
            logger.info(f"Fehler beim Laden der CSV: {e}")
            return
        
        # Prüfen ob erforderliche Spalten vorhanden sind
        required_columns = ['Künstler', 'Titel']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"FEHLER: Fehlende Spalten: {missing_columns}")
            logger.info(f"Fehlende Spalten: {missing_columns}")
            return
        
        # Index hinzufügen falls nicht vorhanden
        if 'Index' not in df.columns:
            df.reset_index(drop=True, inplace=True)
            df.insert(0, 'Index', range(len(df)))
            # Aktualisierte CSV speichern
            df.to_csv(input_file, index=False)
            logger.info("Index-Spalte hinzugefügt und CSV aktualisiert")
        
        # Leere Zeilen entfernen
        initial_count = len(df)
        df = df.dropna(subset=['Künstler', 'Titel'])
        final_count = len(df)
        if initial_count != final_count:
            logger.info(f"{initial_count - final_count} Zeilen mit leeren Werten entfernt")
        
        # Ergebnisse sammeln
        results = []
        failed_results = []
        processed_count = 0
        save_interval = 100  # Alle 100 Songs speichern
        
        # Parallelisierte Ausführung
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Tasks erstellen
            future_to_row = {
                executor.submit(self.get_lyrics, row['Künstler'], row['Titel'], row['Index']): row 
                for _, row in df.iterrows()
            }
            
            # Progress Bar
            with tqdm(total=len(future_to_row), desc="Scraping Lyrics") as pbar:
                for future in as_completed(future_to_row):
                    row = future_to_row[future]
                    try:
                        index, lyrics, error = future.result()
                        
                        if lyrics:
                            results.append({'Index': index, 'Songtext': lyrics})
                        else:
                            failed_results.append({
                                'Index': index,
                                'Künstler': row['Künstler'],
                                'Titel': row['Titel'],
                                'Fehler': error
                            })
                        
                        processed_count += 1
                        pbar.update(1)
                        
                        # Alle 100 Songs zwischenspeichern
                        if processed_count % save_interval == 0 and results:
                            self._save_intermediate_results(results, output_file, processed_count)
                        
                        # Kleine Verzögerung um API nicht zu überlasten
                        time.sleep(self.delay)
                        
                    except Exception as exc:
                        logger.info(f"Task Fehler für {row['Künstler']} - {row['Titel']}: {exc}")
                        failed_results.append({
                            'Index': row['Index'],
                            'Künstler': row['Künstler'],
                            'Titel': row['Titel'],
                            'Fehler': f"Task Fehler: {exc}"
                        })
                        processed_count += 1
                        pbar.update(1)
                        
                        # Auch bei Fehlern zwischenspeichern falls nötig
                        if processed_count % save_interval == 0 and results:
                            self._save_intermediate_results(results, output_file, processed_count)
        
        # Finale Ergebnisse speichern (falls noch nicht alle durch Zwischenspeicherung erfasst)
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('Index')  # Nach Index sortieren
            results_df.to_csv(output_file, index=False)
            logger.info(f"✓ Finale Speicherung: {len(results)} Lyrics gespeichert in {output_file}")
        else:
            logger.info("Keine Lyrics gefunden!")
        
        # Fehler-Report speichern
        if failed_results:
            failed_df = pd.DataFrame(failed_results)
            failed_df.to_csv("failed_lyrics.csv", index=False)
            logger.info(f"✗ {len(failed_results)} fehlgeschlagene Requests gespeichert in failed_lyrics.csv")
        
        # Zusammenfassung
        total = len(df)
        success = len(results)
        failed = len(failed_results)
        
        print(f"\n=== ZUSAMMENFASSUNG ===")
        print(f"Gesamt verarbeitet: {total}")
        print(f"Erfolgreich: {success} ({success/total*100:.1f}%)")
        print(f"Fehlgeschlagen: {failed} ({failed/total*100:.1f}%)")
        print(f"Output: {output_file}")
        if failed_results:
            print(f"Fehler-Log: failed_lyrics.csv")

def main():
    """Hauptfunktion"""
    # Datenordner erstellen falls nicht vorhanden
    Path("data").mkdir(exist_ok=True)
    
    # Scraper initialisieren
    scraper = LyricsScraper(max_workers=15, delay=0.1)  # Konservative Einstellungen
    
    # Scraping starten
    scraper.scrape_lyrics(
        input_file="../data/charts_with_language.csv",
        output_file="../data/lyrics.csv"
    )

if __name__ == "__main__":
    main()