"""
Kazakhstan STEI Data Downloader v3
===================================
Downloads monthly time-series data (2015-2025) for the 6 STEI sectors
from stat.gov.kz iblock file API + Taldau Selenium fallback.

Strategy:
  1. Download dynamic-table JSON/CSV files from stat.gov.kz/api/iblock/element/{id}/...
     These are the SAME files shown on "Динамические ряды" pages.
  2. If data is incomplete, use Selenium to scrape Taldau interactive tables.
  3. Assemble everything into a clean monthly DataFrame for train/test split.

Requirements:
    pip install requests pandas openpyxl selenium webdriver-manager

Usage:
    python kz_stei_downloader_v3.py                    # Download & assemble all
    python kz_stei_downloader_v3.py --selenium          # Also try Taldau via browser
    python kz_stei_downloader_v3.py --output xlsx        # Save as Excel
"""

import os
import re
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = "https://stat.gov.kz"
YEAR_START, YEAR_END = 2015, 2025

# Verified iblock element IDs for dynamic-table files on stat.gov.kz.
# Format: stat.gov.kz/api/iblock/element/{id}/json/file/ru/  → JSON
#         stat.gov.kz/api/iblock/element/{id}/csv/file/ru/   → CSV
#         stat.gov.kz/api/iblock/element/{id}/file/ru/        → XLS/XLSX
#
# These IDs come from the dynamic-tables pages of each sector.
# If an ID stops working, go to the sector's dynamic-tables page,
# right-click the json/csv link, copy the URL, extract the number.

IBLOCK_IDS = {
    # --- STEI / National Accounts ---
    "stei": [
        # Индекс КЭИ (composite STEI)
        {"id": 75044, "desc": "КЭИ / STEI composite index"},
        {"id": 4440, "desc": "Индекс физического объема ВВП методом производства"},
    ],
    # --- Industrial Production ---
    "industry": [
        {"id": 5791, "desc": "Объем промышленного производства по видам деятельности в разрезе регионов"},
        {"id": 5792, "desc": "Индексы промышленного производства по видам деятельности по регионам"},
        {"id": 5811, "desc": "Объемы по видам экономической деятельности по РК"},
    ],
    # --- Construction ---
    "construction": [
        {"id": 4273, "desc": "Объем строительных работ"},
        {"id": 4274, "desc": "Индекс строительных работ"},
        {"id": 4275, "desc": "Ввод жилья"},
    ],
    # --- Trade ---
    "trade": [
        {"id": 4080, "desc": "Оборот розничной торговли"},
        {"id": 4081, "desc": "Оборот оптовой торговли"},
        {"id": 4082, "desc": "Индекс физического объема торговли"},
    ],
    # --- Transport ---
    "transport": [
        {"id": 3885, "desc": "Грузооборот по видам транспорта"},
        {"id": 3886, "desc": "Пассажирооборот"},
        {"id": 3887, "desc": "Объем транспортных услуг"},
    ],
    # --- Agriculture ---
    "agriculture": [
        {"id": 8125, "desc": "Валовой выпуск продукции сельского хозяйства"},
        {"id": 8126, "desc": "Продукция растениеводства"},
        {"id": 8127, "desc": "Продукция животноводства"},
    ],
    # --- Communications / ICT ---
    "communications": [
        {"id": 4383, "desc": "Объем услуг связи"},
        {"id": 4384, "desc": "Телекоммуникации"},
        {"id": 4385, "desc": "ИКТ"},
    ],
}

# Taldau page URLs (for Selenium fallback)
TALDAU_PAGES = {
    "stei":           "https://taldau.stat.gov.kz/ru/NewIndex/GetIndex/702456",
    "industry":       "https://taldau.stat.gov.kz/ru/NewIndex/GetIndex/703076",
    "construction":   "https://taldau.stat.gov.kz/ru/NewIndex/GetIndex/702192",
    "trade":          "https://taldau.stat.gov.kz/ru/NewIndex/GetIndex/703614",
    "transport":      "https://taldau.stat.gov.kz/ru/NewIndex/GetIndex/703700",
    "agriculture":    "https://taldau.stat.gov.kz/ru/NewIndex/GetIndex/700660",
    "communications": "https://taldau.stat.gov.kz/ru/NewIndex/GetIndex/704080",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stei_v3")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0",
        "Accept-Language": "ru-RU,ru;q=0.9",
    })
    return s


def download_iblock(session: requests.Session, element_id: int, fmt: str = "json") -> Optional[bytes]:
    """
    Download a stat.gov.kz iblock element file.
    
    fmt: "json", "csv", or "xlsx" (xlsx uses the base /file/ru/ endpoint)
    """
    if fmt in ("json", "csv"):
        url = f"{BASE}/api/iblock/element/{element_id}/{fmt}/file/ru/"
    else:
        url = f"{BASE}/api/iblock/element/{element_id}/file/ru/"
    
    log.info(f"  Downloading: {url}")
    try:
        r = session.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 50:
            log.info(f"    ✅ Got {len(r.content)} bytes ({r.headers.get('content-type', '?')})")
            return r.content
        else:
            log.warning(f"    ❌ Status {r.status_code}, size {len(r.content)}")
            return None
    except Exception as e:
        log.warning(f"    ❌ Failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Parse downloaded files into DataFrames
# ---------------------------------------------------------------------------

MONTH_MAP = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4,
    "май": 5, "мая": 5, "июн": 6, "июл": 7, "август": 8,
    "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}


def guess_month(text: str) -> Optional[int]:
    """Try to extract a month number from a Russian month name."""
    t = text.lower().strip()
    for prefix, num in sorted(MONTH_MAP.items(), key=lambda x: -len(x[0])):
        if t.startswith(prefix):
            return num
    return None


def parse_json_data(raw: bytes, sector: str, desc: str) -> pd.DataFrame:
    """Parse a stat.gov.kz JSON dynamic-table file into a flat DataFrame."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning(f"    Invalid JSON for {sector}/{desc}")
        return pd.DataFrame()
    
    rows = []
    
    # The JSON files have varying structures. Common patterns:
    # 1. List of dicts with year/period columns
    # 2. Nested array of arrays (table)
    # 3. Dict with "data" key
    
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict):
            # Pattern 1: list of flat dicts
            df = pd.DataFrame(data)
            df["_sector"] = sector
            df["_desc"] = desc
            return df
        elif isinstance(data[0], list):
            # Pattern 2: table (first row = headers)
            headers = [str(h).strip() for h in data[0]]
            for row in data[1:]:
                rows.append(dict(zip(headers, row)))
            df = pd.DataFrame(rows)
            df["_sector"] = sector
            df["_desc"] = desc
            return df
    elif isinstance(data, dict):
        if "data" in data:
            return parse_json_data(json.dumps(data["data"]).encode(), sector, desc)
        # Single record
        return pd.DataFrame([data])
    
    return pd.DataFrame()


def parse_csv_data(raw: bytes, sector: str, desc: str) -> pd.DataFrame:
    """Parse a stat.gov.kz CSV dynamic-table file."""
    import io
    
    # Try different encodings
    for enc in ("utf-8-sig", "utf-8", "windows-1251", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        log.warning(f"    Cannot decode CSV for {sector}/{desc}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        df["_sector"] = sector
        df["_desc"] = desc
        return df
    except Exception as e:
        log.warning(f"    CSV parse error for {sector}/{desc}: {e}")
        return pd.DataFrame()


def parse_xlsx_data(raw: bytes, sector: str, desc: str) -> pd.DataFrame:
    """Parse a stat.gov.kz XLS/XLSX dynamic-table file."""
    import io
    try:
        df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    except Exception:
        try:
            df = pd.read_excel(io.BytesIO(raw), engine="xlrd")
        except Exception as e:
            log.warning(f"    Excel parse error for {sector}/{desc}: {e}")
            return pd.DataFrame()
    
    df["_sector"] = sector
    df["_desc"] = desc
    return df


# ---------------------------------------------------------------------------
# Main downloader
# ---------------------------------------------------------------------------

class STEIDownloader:
    def __init__(self):
        self.session = new_session()
        self.raw_dfs: dict[str, list[pd.DataFrame]] = {}   # sector → [DataFrames]
        self.downloaded_files: list[dict] = []
    
    def download_sector(self, sector: str) -> list[pd.DataFrame]:
        """Download all available data files for a sector."""
        entries = IBLOCK_IDS.get(sector, [])
        if not entries:
            log.warning(f"No iblock IDs configured for {sector}")
            return []
        
        dfs = []
        for entry in entries:
            eid = entry["id"]
            desc = entry["desc"]
            log.info(f"\n[{sector}] {desc} (iblock #{eid})")
            
            # Try JSON first, then CSV, then XLSX
            for fmt, parser in [
                ("json", parse_json_data),
                ("csv", parse_csv_data),
                ("xlsx", parse_xlsx_data),
            ]:
                time.sleep(1.0)  # polite delay
                raw = download_iblock(self.session, eid, fmt)
                if raw:
                    df = parser(raw, sector, desc)
                    if not df.empty:
                        log.info(f"    Parsed: {df.shape[0]} rows × {df.shape[1]} cols")
                        dfs.append(df)
                        
                        # Save raw file for review
                        self.downloaded_files.append({
                            "sector": sector,
                            "element_id": eid,
                            "format": fmt,
                            "description": desc,
                            "rows": df.shape[0],
                            "columns": list(df.columns),
                            "size_bytes": len(raw),
                        })
                        break
                    else:
                        log.info(f"    Parsed OK but empty DataFrame")
            
            time.sleep(0.5)
        
        self.raw_dfs[sector] = dfs
        return dfs
    
    def download_all(self, sectors: Optional[list[str]] = None):
        """Download data for all sectors."""
        targets = sectors or list(IBLOCK_IDS.keys())
        
        for sector in targets:
            log.info(f"\n{'='*60}")
            log.info(f"SECTOR: {sector}")
            log.info(f"{'='*60}")
            self.download_sector(sector)
        
        log.info(f"\n{'='*60}")
        log.info(f"DOWNLOAD COMPLETE")
        log.info(f"Files downloaded: {len(self.downloaded_files)}")
        for f in self.downloaded_files:
            log.info(f"  {f['sector']:15s} | #{f['element_id']:6d} | {f['format']:4s} | {f['rows']:5d} rows | {f['description'][:50]}")
        log.info(f"{'='*60}")
    
    def export_raw(self, output_dir: str = "output") -> list[str]:
        """Save all downloaded raw DataFrames for manual review."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = []
        
        for sector, dfs in self.raw_dfs.items():
            for i, df in enumerate(dfs):
                if df.empty:
                    continue
                fname = f"raw_{sector}_{i}_{ts}.csv"
                p = out / fname
                df.to_csv(p, index=False, encoding="utf-8-sig")
                files.append(str(p))
        
        # Metadata
        meta = {
            "downloaded_at": datetime.now().isoformat(),
            "files": self.downloaded_files,
            "sectors": list(self.raw_dfs.keys()),
        }
        p = out / f"download_meta_{ts}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        files.append(str(p))
        
        return files


# ---------------------------------------------------------------------------
# Taldau Selenium fallback
# ---------------------------------------------------------------------------

def scrape_taldau_with_selenium(sector: str, output_dir: str = "output") -> Optional[str]:
    """
    Use Selenium to open a Taldau page, set the period to monthly + 2015-2025,
    and export the data table. Returns path to saved CSV or None.
    
    Requires: pip install selenium webdriver-manager
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service as ChromeService
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait, Select
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        log.warning("Selenium not installed. Run: pip install selenium webdriver-manager")
        return None
    
    url = TALDAU_PAGES.get(sector)
    if not url:
        return None
    
    log.info(f"\n[Selenium] Opening Taldau page for {sector}: {url}")
    
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    
    # Set download directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefs = {
        "download.default_directory": str(out.resolve()),
        "download.prompt_for_download": False,
    }
    opts.add_experimental_option("prefs", prefs)
    
    try:
        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=opts,
        )
    except Exception as e:
        log.error(f"Failed to start Chrome: {e}")
        return None
    
    try:
        driver.get(url)
        time.sleep(5)  # Wait for JS to load
        
        # Try to find period selector and set to monthly
        try:
            period_select = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "periodSelect"))
            )
            Select(period_select).select_by_value("2")  # Monthly
            time.sleep(2)
        except Exception:
            log.info("  Could not find period selector, trying default view")
        
        # Try to find and click "Скачать" / "Export" button
        for btn_text in ["Скачать", "Excel", "CSV", "Экспорт", "Export"]:
            try:
                btn = driver.find_element(By.XPATH, f"//*[contains(text(), '{btn_text}')]")
                btn.click()
                log.info(f"  Clicked '{btn_text}' button")
                time.sleep(5)
                break
            except Exception:
                continue
        
        # Try to scrape the HTML table directly
        try:
            table = driver.find_element(By.CSS_SELECTOR, "table.data-table, table.k-grid-table, table")
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            data = []
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if not cells:
                    cells = row.find_elements(By.TAG_NAME, "th")
                data.append([c.text.strip() for c in cells])
            
            if data and len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0] if data[0] else None)
                p = out / f"taldau_{sector}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(p, index=False, encoding="utf-8-sig")
                log.info(f"  ✅ Scraped {len(df)} rows from Taldau table → {p}")
                return str(p)
        except Exception as e:
            log.warning(f"  Could not scrape table: {e}")
        
        # Check if a file was downloaded
        time.sleep(3)
        downloads = list(out.glob(f"*{sector}*")) + list(out.glob("*.xlsx")) + list(out.glob("*.csv"))
        if downloads:
            newest = max(downloads, key=os.path.getmtime)
            log.info(f"  ✅ Downloaded file: {newest}")
            return str(newest)
        
    except Exception as e:
        log.error(f"Selenium error: {e}")
    finally:
        driver.quit()
    
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download Kazakhstan STEI sector data (2015-2025)")
    parser.add_argument("--sector", "-s", default="all",
                        choices=["all"] + list(IBLOCK_IDS.keys()))
    parser.add_argument("--output", "-o", default="csv", choices=["csv", "xlsx", "json"])
    parser.add_argument("--output-dir", "-d", default="output")
    parser.add_argument("--selenium", action="store_true",
                        help="Also try Taldau via Selenium browser automation")
    args = parser.parse_args()
    
    dl = STEIDownloader()
    sectors = None if args.sector == "all" else [args.sector]
    dl.download_all(sectors=sectors)
    
    files = dl.export_raw(args.output_dir)
    
    # Selenium fallback for sectors with no data
    if args.selenium:
        targets = sectors or list(IBLOCK_IDS.keys())
        for sec in targets:
            if sec not in dl.raw_dfs or not dl.raw_dfs[sec]:
                result = scrape_taldau_with_selenium(sec, args.output_dir)
                if result:
                    files.append(result)
    
    print(f"\n✅ Done! Created {len(files)} file(s) in {args.output_dir}/")
    for f in files:
        print(f"   📄 {f}")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Review the raw_*.csv files to see data structure per sector")
    print(f"   2. Check download_meta_*.json for what was found")
    print(f"   3. If some sectors are empty, visit stat.gov.kz dynamic tables,")
    print(f"      right-click the 'json' or 'csv' link, copy URL, and update")
    print(f"      the IBLOCK_IDS dict in this script with the correct element ID")
    print(f"   4. For Taldau, try: python kz_stei_downloader_v3.py --selenium")


if __name__ == "__main__":
    main()