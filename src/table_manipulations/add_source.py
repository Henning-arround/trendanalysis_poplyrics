import pandas as pd
import os

def add_lyrics_source_column():
    # CSV-Datei einlesen
    csv_path = "../data/charts_indexed.csv"
    lyrics_folder = "../data/lyrics"
    
    try:
        # CSV laden mit den gleichen Parametern wie vorher
        df = pd.read_csv(csv_path, 
                         delimiter=',',
                         quotechar='"',
                         quoting=1,
                         skipinitialspace=True)
        
        print(f"CSV geladen. Anzahl Zeilen: {len(df)}")
        
        # Prüfen ob Index-Spalte existiert
        if 'Index' not in df.columns:
            print("Fehler: Spalte 'Index' nicht gefunden!")
            return
        
        # Neue Spalte 'Quelle Lyrics' hinzufügen (zunächst leer)
        df['Quelle Lyrics'] = ''
        
        # Zähler für gefundene Dateien
        files_found = 0
        
        # Durch jede Zeile iterieren
        for index, row in df.iterrows():
            index_value = row['Index']
            
            # Dateiname mit Index erstellen (als .txt Datei)
            filename = f"{index_value}.txt"
            file_path = os.path.join(lyrics_folder, filename)
            
            # Prüfen ob Datei existiert
            if os.path.exists(file_path):
                df.at[index, 'Quelle Lyrics'] = 'https://lyrics.ovh/'
                files_found += 1
                print(f"✓ Datei gefunden für Index {index_value}")
        
        # Aktualisierte CSV speichern mit konsistenten Parametern
        df.to_csv(csv_path, 
                  index=False,
                  quoting=1)
        
        print(f"\nFertig! {files_found} Zeilen wurden mit der Lyrics-Quelle aktualisiert.")
        print(f"Die aktualisierte CSV wurde gespeichert: {csv_path}")
        
        # Kurze Übersicht anzeigen
        filled_rows = df[df['Quelle Lyrics'] != ''].shape[0]
        print(f"Gesamtanzahl Zeilen mit Lyrics-Quelle: {filled_rows}")
        
    except FileNotFoundError as e:
        print(f"Fehler: Datei nicht gefunden - {e}")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

# Funktion ausführen
if __name__ == "__main__":
    add_lyrics_source_column()