import pandas as pd
import musicbrainzngs
import time
from collections import defaultdict
from tqdm import tqdm

# MusicBrainz konfigurieren
musicbrainzngs.set_useragent(
    "ChartsurferDataset",
    "1.0",
    "############"
)

# MusicBrainz Account Authentifizierung
musicbrainzngs.auth("##########", "###########")

def find_artist_mbid(artist_name, reference_song):
    """Findet die MusicBrainz Artist ID über einen bekannten Song."""
    try:
        # Suche nach dem Song zusammen mit dem Künstler
        result = musicbrainzngs.search_recordings(
            query=f'artist:"{artist_name}" AND recording:"{reference_song}"',
            limit=5
        )
        
        if result['recording-list']:
            # Nehme das erste Ergebnis mit dem höchsten Score
            for recording in result['recording-list']:
                if 'artist-credit' in recording:
                    artist_credit = recording['artist-credit'][0]
                    if 'artist' in artist_credit:
                        mbid = artist_credit['artist']['id']
                        name = artist_credit['artist']['name']
                        print(f"✓ Gefunden: {artist_name} -> {name} (MBID: {mbid})")
                        return mbid
        
        print(f"✗ Nicht gefunden: {artist_name} - {reference_song}")
        return None
    except Exception as e:
        print(f"✗ Fehler bei {artist_name}: {e}")
        return None

def get_release_year_for_song(artist_name, song_title):
    """Holt das Erscheinungsjahr für einen bestimmten Song."""
    try:
        result = musicbrainzngs.search_recordings(
            query=f'artist:"{artist_name}" AND recording:"{song_title}"',
            limit=5
        )
        
        if result['recording-list']:
            for recording in result['recording-list']:
                # Versuche das früheste Release-Datum zu finden
                if 'release-list' in recording:
                    years = []
                    for release in recording['release-list']:
                        if 'date' in release:
                            date_str = release['date']
                            # Extrahiere Jahr (Format: YYYY oder YYYY-MM-DD)
                            year = date_str.split('-')[0]
                            if year.isdigit():
                                years.append(int(year))
                    
                    if years:
                        return min(years)  # Frühestes Jahr
        
        return None
    except Exception as e:
        print(f"  ✗ Fehler beim Abrufen des Jahres für {song_title}: {e}")
        return None

def get_all_artist_recordings(artist_mbid, artist_name):
    """Holt alle Recordings eines Künstlers mit Erscheinungsjahr."""
    all_recordings = {}  # Dict: {title: year}
    
    try:
        # Schritt 1: Hole alle Releases des Künstlers mit Tracklists
        print(f"  → Hole Releases für {artist_name}...")
        release_offset = 0
        release_limit = 100
        recording_years = {}  # Dict: {recording_id: earliest_year}
        
        while True:
            releases_result = musicbrainzngs.browse_releases(
                artist=artist_mbid,
                limit=release_limit,
                offset=release_offset,
                includes=['recordings']
            )
            
            releases = releases_result.get('release-list', [])
            if not releases:
                break
            
            for release in releases:
                release_date = release.get('date', '')
                year = None
                if release_date:
                    year_str = release_date.split('-')[0]
                    if year_str.isdigit():
                        year = int(year_str)
                
                # Extrahiere Recordings aus diesem Release
                if 'medium-list' in release:
                    for medium in release['medium-list']:
                        if 'track-list' in medium:
                            for track in medium['track-list']:
                                if 'recording' in track:
                                    rec = track['recording']
                                    rec_id = rec.get('id')
                                    rec_title = rec.get('title')
                                    
                                    if rec_id and rec_title and year:
                                        # Speichere frühestes Jahr für dieses Recording
                                        if rec_id not in recording_years or year < recording_years[rec_id]:
                                            recording_years[rec_id] = year
                                        
                                        # Speichere auch im finalen Dict
                                        if rec_title not in all_recordings or year < (all_recordings[rec_title] or float('inf')):
                                            all_recordings[rec_title] = year
            
            release_count = releases_result.get('release-count', 0)
            release_offset += release_limit
            
            if release_offset >= release_count:
                break
            
            time.sleep(1)
        
        # Schritt 2: Hole alle Recordings (für die ohne Release-Datum)
        print(f"  → Hole Recordings ohne Release-Datum...")
        recording_offset = 0
        recording_limit = 100
        
        while True:
            result = musicbrainzngs.browse_recordings(
                artist=artist_mbid,
                limit=recording_limit,
                offset=recording_offset
            )
            
            recordings = result.get('recording-list', [])
            if not recordings:
                break
            
            for recording in recordings:
                title = recording.get('title', '')
                rec_id = recording.get('id', '')
                
                if title and title not in all_recordings:
                    # Prüfe ob wir ein Jahr aus den Releases haben
                    year = recording_years.get(rec_id)
                    all_recordings[title] = year
            
            recording_count = result.get('recording-count', 0)
            recording_offset += recording_limit
            
            if recording_offset >= recording_count:
                break
            
            time.sleep(1)
        
        print(f"  → {len(all_recordings)} Songs gefunden für {artist_name}")
        return all_recordings
    
    except Exception as e:
        print(f"✗ Fehler beim Abrufen der Recordings für {artist_name}: {e}")
        return {}

def main():
    # CSV einlesen
    csv_path = "../../data/charts_chartsurfer_1954_1977.csv"
    df = pd.read_csv(csv_path)
    
    print(f"CSV eingelesen: {len(df)} Zeilen")
    print(f"Spalten: {df.columns.tolist()}\n")
    
    # Annahme: Spalten heißen 'Künstler' und 'Song' oder ähnlich
    # Passe die Spaltennamen an falls nötig
    artist_col = 'Künstler' if 'Künstler' in df.columns else df.columns[0]
    song_col = 'Song' if 'Song' in df.columns else df.columns[1]
    
    # Gruppiere nach Künstler mit einem Beispiel-Song
    artist_songs = df.groupby(artist_col)[song_col].first().to_dict()
    
    print(f"Gefunden: {len(artist_songs)} eindeutige Künstler\n")
    print("="*60)
    
    # Sammle alle Songs pro Künstler mit Jahr
    all_artist_recordings = {}
    
    # Hole alle Songs der Künstler (inklusive Chart-Songs)
    print("\n[SCHRITT 1] Hole alle Songs der Künstler...")
    print("="*60)
    
    for artist_name, reference_song in tqdm(artist_songs.items(), desc="Verarbeite Künstler", unit="artist"):
        tqdm.write(f"\nVerarbeite: {artist_name}")
        
        # Finde Artist MBID
        mbid = find_artist_mbid(artist_name, reference_song)
        
        if mbid:
            # Hole alle Recordings mit Jahr
            recordings = get_all_artist_recordings(mbid, artist_name)
            all_artist_recordings[artist_name] = recordings
        
        # Rate limiting zwischen Künstlern
        time.sleep(1)
    
    # Schritt 2: Füge Erscheinungsjahr zu bestehenden Chart-Songs hinzu
    print("\n\n[SCHRITT 2] Füge Erscheinungsjahr zu Chart-Songs hinzu...")
    print("="*60)
    
    df['Erscheinungsjahr'] = None
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Füge Erscheinungsjahr hinzu", unit="song"):
        artist = row[artist_col]
        song = row[song_col]
        
        # Suche den Song in den bereits extrahierten Recordings
        if artist in all_artist_recordings:
            # Versuche exakten Match zu finden
            if song in all_artist_recordings[artist]:
                year = all_artist_recordings[artist][song]
                df.at[idx, 'Erscheinungsjahr'] = year
            else:
                # Versuche case-insensitive Match
                song_lower = song.lower()
                for recording_title, year in all_artist_recordings[artist].items():
                    if recording_title.lower() == song_lower:
                        df.at[idx, 'Erscheinungsjahr'] = year
                        break
    
    # Ausgabe der Ergebnisse
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)
    
    total_songs = 0
    artists_found = 0
    artists_not_found = len(artist_songs) - len(all_artist_recordings)
    
    for artist, songs in sorted(all_artist_recordings.items()):
        artists_found += 1
        total_songs += len(songs)
        print(f"\n{artist}: {len(songs)} Songs")
        # Zeige erste 10 Songs als Beispiel
        for song_title, year in sorted(list(songs.items()))[:10]:
            year_str = f"({year})" if year else "(Jahr unbekannt)"
            print(f"  - {song_title} {year_str}")
        if len(songs) > 10:
            print(f"  ... und {len(songs) - 10} weitere")
    
    # Erstelle neue CSV
    print("\n" + "="*60)
    print("ERSTELLE NEUE CSV")
    print("="*60)
    
    # Neue Zeilen für zusätzliche Songs
    new_rows = []
    
    for artist, songs in all_artist_recordings.items():
        for song_title, year in songs.items():
            # Prüfe ob dieser Song bereits in den Charts ist (case-insensitive)
            existing = df[(df[artist_col] == artist) & (df[song_col].str.lower() == song_title.lower())]
            
            if len(existing) == 0:
                # Neuer Song, füge hinzu
                new_row = {col: None for col in df.columns}
                new_row[artist_col] = artist
                new_row[song_col] = song_title
                new_row['Erscheinungsjahr'] = year
                new_rows.append(new_row)
    
    # Kombiniere alte und neue Daten
    new_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    
    # Speichere neue CSV
    output_path = "../../data/charts_chartsurfer_1954_1977_v2.csv"
    new_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Neue CSV erstellt: {output_path}")
    print(f"  Zeilen gesamt: {len(new_df)}")
    print(f"  Chart-Songs (original): {len(df)}")
    print(f"  Zusätzliche Songs: {len(new_rows)}")
    
    # Statistik für Erscheinungsjahre bei Chart-Songs
    chart_songs_with_year = df['Erscheinungsjahr'].notna().sum()
    print(f"  Chart-Songs mit Erscheinungsjahr: {chart_songs_with_year}/{len(df)} ({chart_songs_with_year/len(df)*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print("STATISTIK")
    print(f"{'='*60}")
    print(f"Künstler in CSV gesamt: {len(artist_songs)}")
    print(f"Künstler erfolgreich gefunden: {artists_found}")
    print(f"Künstler nicht gefunden: {artists_not_found}")
    print(f"Erfolgsrate: {(artists_found / len(artist_songs) * 100):.1f}%")
    print(f"\nGesamt eindeutige Songs (neu gescraped): {total_songs}")
    print(f"Durchschnitt Songs pro Künstler: {(total_songs / artists_found):.1f}" if artists_found > 0 else "N/A")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
