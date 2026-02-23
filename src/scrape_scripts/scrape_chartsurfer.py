import csv
import requests
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

BASE_URL = "https://www.chartsurfer.de/archiv/single-charts-deutschland/"
DETAIL_BASE_URL = "https://www.chartsurfer.de"

def scrape_year(year: int, session: requests.Session) -> List[Dict[str, str]]:
    """Scrape all pages for a given year."""
    collected: List[Dict[str, str]] = []
    page = 1
    while True:
        suffix = f"-{page}" if page > 1 else ""
        url = f"{BASE_URL}{year}{suffix}"
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 404:
                break
            response.raise_for_status()
            entries = parse_entries(response.text, year, session)
            if not entries:
                break
            collected.extend(entries)
            page += 1
        except Exception as e:
            print(f"\nError scraping {url}: {e}")
            break
    return collected

def parse_entries(html: str, year: int, session: requests.Session) -> List[Dict[str, str]]:
    """Parse chart entries from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", id="dataList")
    if not container:
        return []
    
    rows: List[Dict[str, str]] = []
    song_links = container.select("a.style-1.font-weight-bold")
    
    for anchor in tqdm(song_links, desc=f"Processing entries", leave=False):
        title = anchor.get_text(strip=True)
        
        # Get the text that comes after the anchor element
        next_sibling = anchor.next_sibling
        artist = ""
        if next_sibling and isinstance(next_sibling, str):
            artist = next_sibling.strip()
        
        if artist.lower().startswith("von "):
            artist = artist[4:].strip()
        
        # Get detail page URL
        detail_url = None
        if anchor.get("href"):
            detail_url = DETAIL_BASE_URL + anchor.get("href")
        
        # Scrape detail page for additional information
        detail_info = scrape_detail_page(detail_url, session) if detail_url else {}
        
        if artist and title:
            row = {
                "Künstler": artist,
                "Titel": title,
                "Jahr der Chartplatzierung": str(year),
                "Wochen Gesamt": detail_info.get("wochen_gesamt", ""),
                "Top-10 Wochen": detail_info.get("top10_wochen", ""),
                "Nr. 1 Wochen": detail_info.get("nr1_wochen", ""),
                "Erste Notierung": detail_info.get("erste_notierung", ""),
                "Letzte Notierung": detail_info.get("letzte_notierung", ""),
                "Höchstposition": detail_info.get("hoechstposition", ""),
            }
            rows.append(row)
        
        # Be polite - wait a bit between requests (increased from 0.5 to 1 second)
        time.sleep(0.2)
    
    return rows

def scrape_detail_page(url: str, session: requests.Session) -> Dict[str, str]:
    """Scrape additional information from detail page."""
    info = {}
    
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find the row-item div containing Deutschland
        row_items = soup.find_all("div", class_="row item")
        deutschland_div = None
        
        for item in row_items:
            img = item.find("img")
            if img and "Deutschland" in item.get_text():
                deutschland_div = item
                break
        
        if deutschland_div:
            # Extract "Wochen Gesamt"
            canvas1 = deutschland_div.find("canvas", id="chartwgt1")
            if canvas1:
                parent_div = canvas1.find_parent("div")
                if parent_div and parent_div.next_sibling:
                    text = parent_div.next_sibling
                    if isinstance(text, str):
                        info["wochen_gesamt"] = text.strip()
            
            # Extract "Top-10 Wochen"
            canvas2 = deutschland_div.find("canvas", id="chartwgt2")
            if canvas2:
                parent_div = canvas2.find_parent("div")
                if parent_div and parent_div.next_sibling:
                    text = parent_div.next_sibling
                    if isinstance(text, str):
                        info["top10_wochen"] = text.strip()
            
            # Extract "Nr. 1 Wochen"
            canvas3 = deutschland_div.find("canvas", id="chartwgt3")
            if canvas3:
                parent_div = canvas3.find_parent("div")
                if parent_div and parent_div.next_sibling:
                    text = parent_div.next_sibling
                    if isinstance(text, str):
                        info["nr1_wochen"] = text.strip()
            
            # Extract "Erste Notierung"
            erste_span = deutschland_div.find("span", string=lambda s: s and "Erste Notierung:" in s)
            if erste_span and erste_span.next_sibling:
                text = erste_span.next_sibling
                if isinstance(text, str):
                    info["erste_notierung"] = text.strip()
            
            # Extract "Letzte Notierung"
            letzte_span = deutschland_div.find("span", string=lambda s: s and "Letzte Notierung:" in s)
            if letzte_span and letzte_span.next_sibling:
                text = letzte_span.next_sibling
                if isinstance(text, str):
                    info["letzte_notierung"] = text.strip()
            
            # Extract "Höchstposition" (Note: website has typo "Höchstpostion")
            hoechst_span = deutschland_div.find("span", string=lambda s: s and "Höchstpostion:" in s)
            if hoechst_span and hoechst_span.next_sibling:
                text = hoechst_span.next_sibling
                if isinstance(text, str):
                    info["hoechstposition"] = text.strip()
        
    except Exception as e:
        print(f"\nError scraping detail page {url}: {e}")
    
    return info

def write_csv(rows: List[Dict[str, str]]) -> None:
    """Write all collected data to CSV file."""
    target = (Path(__file__).resolve().parent / "../../data/charts_chartsurfer_1954_1977.csv").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "Künstler", 
            "Titel", 
            "Jahr der Chartplatzierung",
            "Wochen Gesamt",
            "Top-10 Wochen",
            "Nr. 1 Wochen",
            "Erste Notierung",
            "Letzte Notierung",
            "Höchstposition"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} entries to {target}")

def main() -> None:
    """Main scraping function."""
    all_rows: List[Dict[str, str]] = []
    
    # Create a session for better performance
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    try:
        for year in tqdm(range(1954, 1978), desc="Scraping years"):
            year_rows = scrape_year(year, session)
            all_rows.extend(year_rows)
            tqdm.write(f"Year {year}: {len(year_rows)} entries")
        
        write_csv(all_rows)
        print(f"Total entries scraped: {len(all_rows)}")
    
    finally:
        session.close()

if __name__ == "__main__":
    main()