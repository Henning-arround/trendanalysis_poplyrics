import pandas as pd

# CSV mit expliziter Quote-Behandlung einlesen
df = pd.read_csv("../data/charts.csv", 
                 delimiter=',',
                 quotechar='"',
                 quoting=1,  # QUOTE_ALL
                 skipinitialspace=True)

# Datentypen überprüfen BEVOR Index hinzugefügt wird
print("Datentypen vor Index-Hinzufügung:")
print(df.dtypes)
print("\nErste 3 Zeilen roh:")
print(df.head(3))

# Numeric columns automatisch konvertieren
df = df.apply(pd.to_numeric, errors='ignore')

print("\nDatentypen nach Konvertierung:")
print(df.dtypes)

# Jetzt Index hinzufügen
df.insert(0, 'Index', range(len(df)))

# Speichern ohne Quotes um alles (nur wo nötig)
df.to_csv("../data/charts_indexed.csv", 
          index=False,
          quoting=1)  # Nur bei Bedarf quoten

print("\nErgebnis:")
print(df.head())