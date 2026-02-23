import pandas as pd
import asyncio
from googletrans import Translator
from collections import Counter
import time

# Versuche tqdm zu importieren
try:
    from tqdm.asyncio import tqdm as async_tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("⚠️ tqdm nicht verfügbar - installiere mit: pip install tqdm")
    TQDM_AVAILABLE = False
    # Fallback: eigene Progress-Anzeige
    class FakeAsyncTqdm:
        @staticmethod
        async def gather(*tasks, desc="", unit="", colour="", ncols=80, bar_format=""):
            return await asyncio.gather(*tasks)
    async_tqdm = FakeAsyncTqdm

class LanguageDetector:
    def __init__(self, max_concurrent=10, delay=0.05):
        """
        max_concurrent: Maximale Anzahl gleichzeitiger Requests
        delay: Verzögerung zwischen Requests in Sekunden
        """
        self.max_concurrent = max_concurrent
        self.delay = delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def detect_single_language(self, translator, title, index):
        """Erkennt die Sprache für einen einzelnen Titel"""
        async with self.semaphore:
            try:
                # Kurze Pause um Rate Limiting zu vermeiden
                await asyncio.sleep(self.delay)
                
                # Sprache erkennen
                result = await translator.detect(str(title))
                return index, result.lang
                
            except Exception as e:
                # Bei Fehlern 'unknown' zurückgeben
                return index, 'unknown'
    
    async def detect_languages_parallel(self, titles):
        """Erkennt Sprachen parallel für alle Titel"""
        async with Translator() as translator:
            # Tasks für alle Titel erstellen
            tasks = [
                self.detect_single_language(translator, title, i) 
                for i, title in enumerate(titles)
            ]
            
            total_tasks = len(tasks)
            
            if TQDM_AVAILABLE:
                # Verwende tqdm Progress Bar
                results = await async_tqdm.gather(
                    *tasks, 
                    desc="🎵 Spracherkennung", 
                    unit="songs",
                    colour="green",
                    ncols=80,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                )
            else:
                # Fallback: eigene Progress-Anzeige
                print(f"🎵 Starte Spracherkennung für {total_tasks} Titel...")
                
                # Manuelle Progress-Anzeige alle 5%
                async def run_with_progress():
                    completed = 0
                    results = []
                    
                    # Tasks in kleineren Batches verarbeiten für bessere Progress-Anzeige
                    batch_size = max(1, total_tasks // 20)  # 20 Updates
                    
                    for i in range(0, total_tasks, batch_size):
                        batch = tasks[i:i + batch_size]
                        batch_results = await asyncio.gather(*batch)
                        results.extend(batch_results)
                        
                        completed += len(batch)
                        progress = (completed / total_tasks) * 100
                        
                        # Progress Bar zeichnen
                        bar_length = 40
                        filled_length = int(progress * bar_length // 100)
                        bar = "█" * filled_length + "░" * (bar_length - filled_length)
                        
                        print(f"\r🎵 [{bar}] {completed}/{total_tasks} ({progress:.1f}%)", end="", flush=True)
                    
                    print()  # Neue Zeile nach Abschluss
                    return results
                
                results = await run_with_progress()
            
            # Ergebnisse sortieren nach Index
            results.sort(key=lambda x: x[0])
            detected_languages = [lang for _, lang in results]
            
            return detected_languages

async def main():
    print("🎵 Spracherkennung für Chart-Titel")
    print("=" * 50)
    
    # CSV-Datei einlesen mit korrekten Parametern
    print("📁 Lade CSV-Datei...")
    try:
        df = pd.read_csv("../data/charts_indexed.csv",
                         delimiter=',',
                         quotechar='"',
                         quoting=1,
                         skipinitialspace=True)
    except FileNotFoundError:
        print("❌ Fehler: charts_indexed.csv nicht gefunden!")
        return
    
    print(f"📊 Anzahl Songs: {len(df)}")
    print(f"📋 Spalten: {list(df.columns)}")
    
    # Prüfen ob 'Titel' Spalte existiert
    if 'Titel' not in df.columns:
        print("❌ Fehler: Spalte 'Titel' nicht gefunden!")
        print("Verfügbare Spalten:", list(df.columns))
        return
    
    # Leere oder NaN Titel behandeln
    df['Titel'] = df['Titel'].fillna('Unknown Title')
    
    # Language Detector initialisieren
    detector = LanguageDetector(max_concurrent=15, delay=0.03)
    
    print(f"\n🚀 Starte parallele Spracherkennung für {len(df)} Titel...")
    print(f"⚡ Maximale gleichzeitige Requests: {detector.max_concurrent}")
    
    start_time = time.time()
    
    # Sprachen parallel erkennen
    detected_languages = await detector.detect_languages_parallel(df['Titel'].tolist())
    
    # Neue Spalte hinzufügen (nur Sprachcodes, z.B. 'en', 'de', 'es')
    df['Spracherkennung Titel'] = detected_languages
    
    end_time = time.time()
    processing_time = end_time - start_time
    songs_per_second = len(df) / processing_time
    
    print(f"\n✅ Spracherkennung abgeschlossen!")
    print(f"⏱️  Zeit: {processing_time:.2f} Sekunden")
    print(f"🏎️  Geschwindigkeit: {songs_per_second:.1f} Songs/Sekunde")
    
    # Statistik erstellen
    print("\n" + "="*60)
    print("📈 SPRACHSTATISTIK")
    print("="*60)
    
    language_counts = Counter(detected_languages)
    total_songs = len(detected_languages)
    
    # Erweiterte Sprach-Codes zu Namen mapping
    language_names = {
        'en': 'Englisch 🇺🇸',
        'de': 'Deutsch 🇩🇪', 
        'es': 'Spanisch 🇪🇸',
        'fr': 'Französisch 🇫🇷',
        'it': 'Italienisch 🇮🇹',
        'nl': 'Niederländisch 🇳🇱',
        'pt': 'Portugiesisch 🇵🇹',
        'ko': 'Koreanisch 🇰🇷',
        'ja': 'Japanisch 🇯🇵',
        'zh': 'Chinesisch 🇨🇳',
        'ru': 'Russisch 🇷🇺',
        'sv': 'Schwedisch 🇸🇪',
        'no': 'Norwegisch 🇳🇴',
        'da': 'Dänisch 🇩🇰',
        'fi': 'Finnisch 🇫🇮',
        'pl': 'Polnisch 🇵🇱',
        'cs': 'Tschechisch 🇨🇿',
        'hu': 'Ungarisch 🇭🇺',
        'tr': 'Türkisch 🇹🇷',
        'ar': 'Arabisch 🇸🇦',
        'hi': 'Hindi 🇮🇳',
        'th': 'Thailändisch 🇹🇭',
        'vi': 'Vietnamesisch 🇻🇳',
        'id': 'Indonesisch 🇮🇩',
        'ms': 'Malaiisch 🇲🇾',
        'unknown': 'Unbekannt ❓'
    }
    
    # Top 15 Sprachen anzeigen
    top_languages = language_counts.most_common(15)
    
    print(f"{'Sprache':<25} {'Code':<4} {'Anzahl':<8} {'Prozent':<8} {'Balken'}")
    print("-" * 60)
    
    max_count = top_languages[0][1] if top_languages else 1
    
    for lang_code, count in top_languages:
        lang_name = language_names.get(lang_code, f"Unbekannt ({lang_code})")
        percentage = (count / total_songs) * 100
        bar_length = int((count / max_count) * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        
        print(f"{lang_name:<25} {lang_code:<4} {count:<8} {percentage:<7.1f}% {bar}")
    
    print("-" * 60)
    print(f"{'🎵 GESAMT':<25} {'ALL':<4} {total_songs:<8} {'100.0%':<8}")
    
    # Weitere Statistiken
    unique_languages = len(language_counts)
    print(f"\n📊 Zusätzliche Statistiken:")
    print(f"   🌍 Verschiedene Sprachen erkannt: {unique_languages}")
    print(f"   🏆 Häufigste Sprache: {language_names.get(top_languages[0][0], top_languages[0][0])} ({top_languages[0][1]} Songs)")
    print(f"   ❓ Unbekannte Sprachen: {language_counts.get('unknown', 0)} Songs")
    
    # Erweiterte CSV speichern mit korrekten Parametern
    output_file = "../data/charts_with_language.csv"
    df.to_csv(output_file, 
              index=False,
              quoting=1)
    print(f"\n💾 Erweiterte CSV-Datei gespeichert als: {output_file}")
    print(f"   📋 Spalte 'Spracherkennung Titel' enthält nur Sprachcodes (z.B. 'en', 'de', 'es')")
    
    # Beispiele anzeigen
    print(f"\n📝 Beispiele (erste 10 Songs):")
    print("-" * 70)
    sample_df = df[['Index', 'Titel', 'Spracherkennung Titel']].head(10)
    for _, row in sample_df.iterrows():
        lang_name = language_names.get(row['Spracherkennung Titel'], row['Spracherkennung Titel'])
        title_short = row['Titel'][:45] + "..." if len(row['Titel']) > 45 else row['Titel']
        print(f"{row['Index']:3}: {title_short:<48} → {lang_name}")

# Programm ausführen
if __name__ == "__main__":
    print("🔍 Überprüfe Abhängigkeiten...")
    
    # Abhängigkeiten prüfen
    missing_packages = []
    
    try:
        import googletrans
        print("✅ googletrans verfügbar")
    except ImportError:
        missing_packages.append('googletrans')
    
    try:
        import pandas
        print("✅ pandas verfügbar")
    except ImportError:
        missing_packages.append('pandas')
    
    try:
        import tqdm
        print("✅ tqdm verfügbar")
    except ImportError:
        print("⚠️ tqdm nicht verfügbar - Fallback-Progress-Bar wird verwendet")
        print("💡 Für bessere Progress-Bar installiere: pip install tqdm")
    
    if missing_packages:
        print(f"❌ Fehlende Pakete: {', '.join(missing_packages)}")
        print(f"📦 Installiere mit: pip install {' '.join(missing_packages)}")
        exit()
    
    print("✅ Alle kritischen Abhängigkeiten verfügbar!")
    
    # Async Funktion ausführen
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Programm durch Benutzer abgebrochen!")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")