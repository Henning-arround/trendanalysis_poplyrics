import pandas as pd
import json

def csv_to_json():
    # Pfade definieren
    csv_path = "../../data/charts_with_language_updated_v4.csv"
    json_path = "../../data/charts_with_language_updated_v4.json"
    
    try:
        # CSV laden mit den spezifizierten Parametern
        df = pd.read_csv(csv_path, 
                         delimiter=',',
                         quotechar='"',
                         quoting=1,
                         skipinitialspace=True)
        
        print(f"CSV geladen. Anzahl Zeilen: {len(df)}")
        print(f"Spalten: {list(df.columns)}")
        
        # DataFrame zu JSON konvertieren
        json_data = df.to_json(orient='records', force_ascii=False, indent=2)
        
        # JSON in Datei speichern
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        print(f"JSON erfolgreich gespeichert: {json_path}")
        
    except FileNotFoundError as e:
        print(f"Fehler: CSV-Datei nicht gefunden - {e}")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    csv_to_json()
