def main():
    input_file = "../data/chart_urls.txt"
    output_file = "../data/chart_urls_unique.txt"
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            # Einlesen aller Zeilen und entfernen von überflüssigen Leerzeichen
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Die Datei '{input_file}' wurde nicht gefunden.")
        return
    
    # Entferne Zeilenumbrüche und leere Zeilen, dann Duplikate mit einem Set entfernen
    unique_urls = set(line.strip() for line in lines if line.strip())
    
    # Optional: Sortieren der eindeutigen URLs
    sorted_urls = sorted(unique_urls)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for url in sorted_urls:
            f.write(url + "\n")
    
    print(f"Es wurden {len(sorted_urls)} eindeutige URLs in '{output_file}' gespeichert.")

if __name__ == "__main__":
    main()