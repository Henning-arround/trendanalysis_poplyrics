import pandas as pd
import musicbrainzngs
import time
from pathlib import Path
from tqdm import tqdm

# MusicBrainz API konfigurieren
musicbrainzngs.set_useragent("Masterarbeit", "1.0", "your-email@example.com")

def get_earliest_release_year(artist, title):
    """
    Fragt das früheste Erscheinungsjahr für einen Song bei MusicBrainz ab.
    """
    try:
        # Suche nach Recording
        result = musicbrainzngs.search_recordings(
            artist=artist,
            recording=title,
            limit=5
        )
        
        if 'recording-list' not in result or len(result['recording-list']) == 0:
            return None
        
        earliest_year = None
        
        # Durchsuche alle gefundenen Recordings
        for recording in result['recording-list']:
            if 'release-list' in recording:
                for release in recording['release-list']:
                    if 'date' in release:
                        date = release['date']
                        # Überprüfe ob Datum nicht leer ist
                        if date and date.strip():
                            # Extrahiere Jahr (Format: YYYY-MM-DD oder YYYY)
                            year_str = date.split('-')[0]
                            if year_str:
                                year = int(year_str)
                                if earliest_year is None or year < earliest_year:
                                    earliest_year = year
        
        return earliest_year
    
    except Exception as e:
        return None

# CSV einlesen
input_path = Path("../../data/charts_chartsurfer_1954_1977.csv")
df = pd.read_csv(input_path)

# Neue Spalte für Erscheinungsjahr erstellen
df['Erscheinungsjahr'] = None

# Für jede Zeile das Erscheinungsjahr abfragen
for index, row in tqdm(df.iterrows(), total=len(df), desc="Verarbeite Songs"):
    artist = row['Künstler']
    title = row['Titel']
    
    year = get_earliest_release_year(artist, title)
    df.at[index, 'Erscheinungsjahr'] = year
    
    # Rate limiting: 1 Anfrage pro Sekunde (MusicBrainz Richtlinien)
    time.sleep(1)

# Neue CSV speichern
output_path = input_path.parent / "charts_chartsurfer_1954_1977_with_year.csv"
df.to_csv(output_path, index=False)

print(f"\nFertig! Gespeichert unter: {output_path}")
