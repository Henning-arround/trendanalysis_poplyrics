import pandas as pd
import musicbrainzngs
import time
from pathlib import Path
from tqdm import tqdm

# Configure MusicBrainz API
musicbrainzngs.set_useragent(
    "ChartGenreScraper",
    "1.0",
    "################"  # Replace with your email
)

def get_artist_tags(artist):
    """Get tags for an artist from MusicBrainz"""
    try:
        # Search for artist
        result = musicbrainzngs.search_artists(artist=artist, limit=5)
        
        if result.get('artist-list'):
            for artist_result in result['artist-list']:
                # Get detailed artist info with tags
                artist_id = artist_result['id']
                detailed = musicbrainzngs.get_artist_by_id(
                    artist_id,
                    includes=['tags']
                )
                
                if 'artist' in detailed and 'tag-list' in detailed['artist']:
                    tags = [tag['name'] for tag in detailed['artist']['tag-list']]
                    if tags:
                        return ';'.join(tags)
        
    except Exception as e:
        pass
    
    return None

def main():
    # Read CSV file with explicit separator and quoting
    input_path = Path("../../data/charts_with_language_updated_v4.csv")
    df = pd.read_csv(input_path, sep=',', quotechar='"', quoting=1)
    
    # Add new column if not exists
    if 'Genre Künstler' not in df.columns:
        df['Genre Künstler'] = ''
    
    # Get unique artists that have lyrics but no genre yet
    artists_to_process = df[
        (pd.notna(df.get('Quelle Lyrics'))) & 
        (df.get('Quelle Lyrics') != '') &
        ((pd.isna(df.get('Genre Künstler'))) | (df.get('Genre Künstler') == ''))
    ]['Künstler'].unique()
    
    total_artists = len(artists_to_process)
    
    print(f"Total rows in CSV: {len(df)}")
    print(f"Unique artists to process: {total_artists}")
    print()
    
    # Create a mapping of artist -> tags
    artist_tags_map = {}
    
    # Process each unique artist with progress bar
    with tqdm(total=total_artists, desc="Processing artists") as pbar:
        for artist in artists_to_process:
            if not artist or pd.isna(artist):
                pbar.update(1)
                continue
            
            # Get artist tags
            artist_tags = get_artist_tags(artist)
            if artist_tags:
                artist_tags_map[artist] = artist_tags
            
            # Rate limiting (MusicBrainz allows 1 request per second)
            #time.sleep(1.1)
            pbar.update(1)
    
    # Apply tags to all rows with matching artists
    print("\nApplying tags to rows...")
    for artist, tags in tqdm(artist_tags_map.items(), desc="Updating rows"):
        mask = (df['Künstler'] == artist) & (pd.notna(df.get('Quelle Lyrics'))) & (df.get('Quelle Lyrics') != '')
        df.loc[mask, 'Genre Künstler'] = tags
    
    # Save to new file with same format
    output_path = Path("../../data/charts_with_language_updated_v5.csv")
    df.to_csv(output_path, index=False, sep=',', quotechar='"', quoting=1)
    print(f"\n✓ Saved to {output_path}")
    print(f"Found tags for {len(artist_tags_map)}/{total_artists} artists")

if __name__ == "__main__":
    main()