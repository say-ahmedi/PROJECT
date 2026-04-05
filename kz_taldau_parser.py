"""
Kazakhstan Taldau (taldau.stat.gov.kz) — STEI Sector Data Parser
==================================================================

Pulls MONTHLY data for 2015–2025 from the Taldau analytical system's
JSON API for the 6 STEI component sectors:

  1. Industrial Production   (Промышленность)
  2. Construction             (Строительство)
  3. Trade                    (Торговля)
  4. Transport & Warehousing  (Транспорт и складирование)
  5. Agriculture              (Сельское хозяйство)
  6. Communications / ICT     (Связь / ИКТ)

Plus the composite STEI (КЭИ) itself.

Data source:
    Taldau IAS – https://taldau.stat.gov.kz
    JSON API  – https://taldau.stat.gov.kz/ru/Api/GetIndexData/{id}?period={p}&dics={d}
    Web view  – https://taldau.stat.gov.kz/ru/NewIndex/GetIndex/{id}

API conventions (reverse-engineered from open-data-kazakhstan repos):
    period = 2  → monthly
    period = 4  → quarterly
    period = 7  → annual
    dics   = 67 → by region
    dics   = 0  → nationwide (no breakdown)

Requirements:
    pip install requests pandas openpyxl

Usage:
    python kz_taldau_parser.py                       # Fetch all sectors, save CSV
    python kz_taldau_parser.py --sector industry      # Single sector
    python kz_taldau_parser.py --output xlsx           # Save as Excel
    python kz_taldau_parser.py --discover              # Auto-discover valid index IDs
"""

import re
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TALDAU_BASE = "https://taldau.stat.gov.kz"

# Period codes
PERIOD_MONTHLY = 2
PERIOD_QUARTERLY = 4
PERIOD_ANNUAL = 7

# Year range for your train/test split
YEAR_START = 2015
YEAR_END = 2025

# --------------------------------------------------------------------------
# Known / candidate Taldau index IDs per sector.
#
# IMPORTANT: Taldau index IDs can change. The script includes a discovery
# mode (--discover) that probes candidate IDs and reports which ones return
# data. If an ID stops working, run discovery to find the replacement.
#
# How IDs were sourced:
#   - open-data-kazakhstan GitHub repos (confirmed working)
#   - BNS publication cross-references to Taldau
#   - Taldau search page results
#   - Web crawling of /NewIndex/GetIndex/{id} pages
#
# Each entry lists MULTIPLE candidate IDs — the script tries them in order
# and uses the first one that returns valid data.
# --------------------------------------------------------------------------

SECTOR_CONFIG = {
    "stei": {
        "name_ru": "Краткосрочный экономический индикатор (КЭИ)",
        "name_en": "Short-Term Economic Indicator (STEI)",
        "unit": "% к соотв. периоду прошлого года",
        "candidate_ids": [
            702456,  # КЭИ composite
            2709380, # GDP per capita (fallback proxy)
        ],
        "preferred_period": PERIOD_MONTHLY,
    },
    "industry": {
        "name_ru": "Объем промышленного производства",
        "name_en": "Volume of Industrial Production",
        "unit": "млн. тенге / index %",
        "candidate_ids": [
            703076,  # Объем промышленной продукции
            703115,  # Цены на нефтепродукты (related)
            703885,  # Промышленное производство
            701764,  # ИПП – Индекс промышленного производства
        ],
        "preferred_period": PERIOD_MONTHLY,
    },
    "construction": {
        "name_ru": "Объем строительных работ",
        "name_en": "Volume of Construction",
        "unit": "млн. тенге / index %",
        "candidate_ids": [
            701830,  # Строительство / инвестиции
            702192,  # Объем строительных работ
            702160,  # Ввод жилья
        ],
        "preferred_period": PERIOD_MONTHLY,
    },
    "trade": {
        "name_ru": "Объем торговли",
        "name_en": "Volume of Trade",
        "unit": "млн. тенге / index %",
        "candidate_ids": [
            703614,  # Оборот розничной торговли
            703610,  # Оборот оптовой торговли
            703580,  # Товарооборот
            702710,  # Торговля
        ],
        "preferred_period": PERIOD_MONTHLY,
    },
    "transport": {
        "name_ru": "Объем транспорта и складирования",
        "name_en": "Volume of Transport and Warehousing",
        "unit": "млн. тенге / index %",
        "candidate_ids": [
            703700,  # Грузооборот
            703710,  # Пассажирооборот
            703680,  # Объем транспортных услуг
            702850,  # Транспорт
        ],
        "preferred_period": PERIOD_MONTHLY,
    },
    "agriculture": {
        "name_ru": "Объем валового выпуска сельского хозяйства",
        "name_en": "Volume of Agriculture",
        "unit": "млн. тенге / index %",
        "candidate_ids": [
            700660,  # Валовой выпуск продукции с/х
            700670,  # Продукция растениеводства
            700680,  # Продукция животноводства
            2972846, # Пчелосемьи (known valid page, for API testing)
        ],
        "preferred_period": PERIOD_MONTHLY,
    },
    "communications": {
        "name_ru": "Объем услуг связи",
        "name_en": "Volume of Communications / ICT",
        "unit": "млн. тенге / index %",
        "candidate_ids": [
            704080,  # Услуги связи
            704100,  # Телекоммуникации
            704000,  # ИКТ
        ],
        "preferred_period": PERIOD_MONTHLY,
    },
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kz_taldau")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaldauRecord:
    """One data point from the Taldau API."""
    sector: str
    index_id: int
    indicator_name: str
    period_label: str        # "Январь 2023", "1 квартал 2024", "2023 год"
    year: Optional[int] = None
    month: Optional[int] = None
    quarter: Optional[int] = None
    value: Optional[float] = None
    unit: str = ""
    region: str = "Республика Казахстан"
    source_url: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class TaldauClient:
    """HTTP client for taldau.stat.gov.kz API."""

    def __init__(self, delay: float = 1.0, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "ru-RU,ru;q=0.9",
            "Referer": "https://taldau.stat.gov.kz/ru/",
        })
        self.delay = delay
        self.timeout = timeout
        self._last_req = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_req
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def get_json(self, url: str) -> Optional[dict | list]:
        """Fetch JSON from Taldau API."""
        self._throttle()
        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                self._last_req = time.time()
                if resp.status_code == 404:
                    logger.debug(f"  404 for {url}")
                    return None
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as e:
                logger.warning(f"  Attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
        return None

    def get_html(self, url: str) -> Optional[str]:
        """Fetch HTML page."""
        self._throttle()
        try:
            resp = self.session.get(url, timeout=self.timeout)
            self._last_req = time.time()
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"  HTML fetch failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Period / date helpers
# ---------------------------------------------------------------------------

MONTH_NAMES_RU = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
}

MONTH_LOOKUP = {}
for _num, _name in MONTH_NAMES_RU.items():
    for prefix_len in range(3, len(_name) + 1):
        MONTH_LOOKUP[_name[:prefix_len].lower()] = _num


def parse_period_label(label: str) -> dict:
    """
    Parse Taldau period labels into structured date info.
    
    Examples:
        "Январь 2023"         → {year: 2023, month: 1}
        "1 квартал 2024"      → {year: 2024, quarter: 1}
        "2023 год"            → {year: 2023}
        "январь-декабрь 2024" → {year: 2024, month: 1, month_end: 12}
    """
    result = {"year": None, "month": None, "quarter": None}

    # Year
    year_match = re.search(r"(\d{4})", label)
    if year_match:
        result["year"] = int(year_match.group(1))

    # Quarter
    q_match = re.search(r"(\d)\s*кварт", label, re.I)
    if q_match:
        result["quarter"] = int(q_match.group(1))
        return result

    # Month — check for single month name
    label_lower = label.lower().strip()
    for prefix in sorted(MONTH_LOOKUP.keys(), key=len, reverse=True):
        if label_lower.startswith(prefix):
            result["month"] = MONTH_LOOKUP[prefix]
            break

    return result


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class TaldauParser:
    """
    Fetches time-series data from Taldau for the 6 STEI sectors.
    
    Strategy:
    1. For each sector, try candidate index IDs via the JSON API
    2. Request monthly data (period=2) nationwide (no region filter)
    3. Filter to 2015-2025
    4. If monthly fails, fall back to quarterly (period=4) then annual (period=7)
    5. Export a unified DataFrame ready for train/test split
    """

    def __init__(self, client: Optional[TaldauClient] = None):
        self.client = client or TaldauClient()
        self.records: list[TaldauRecord] = []
        self.api_results_raw: dict[str, list] = {}   # sector → raw JSON
        self.working_ids: dict[str, int] = {}         # sector → confirmed index_id

    # -----------------------------------------------------------------
    # Core API call
    # -----------------------------------------------------------------
    def fetch_index_data(
        self,
        index_id: int,
        period: int = PERIOD_MONTHLY,
        dics: int = 0,
    ) -> Optional[list]:
        """
        Call Taldau API:
          GET /ru/Api/GetIndexData/{index_id}?period={period}&dics={dics}

        Returns the raw JSON list/dict or None on failure.
        """
        url = (
            f"{TALDAU_BASE}/ru/Api/GetIndexData/{index_id}"
            f"?period={period}&dics={dics}"
        )
        logger.info(f"  API call: {url}")
        data = self.client.get_json(url)

        if data is None:
            return None

        # The API returns different shapes:
        # - A list of dicts with period/value pairs
        # - A dict with nested structure
        # Normalize to a list
        if isinstance(data, dict):
            # Sometimes wrapped in {"data": [...]}
            if "data" in data:
                return data["data"]
            return [data]
        elif isinstance(data, list):
            return data
        return None

    # -----------------------------------------------------------------
    # Discover working index IDs
    # -----------------------------------------------------------------
    def discover_ids(self) -> dict[str, dict]:
        """
        Probe all candidate IDs across all sectors and report which
        ones return valid data. Useful when IDs change.
        """
        results = {}
        for sector, config in SECTOR_CONFIG.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Discovering IDs for: {sector} ({config['name_en']})")
            sector_results = {}

            for idx_id in config["candidate_ids"]:
                for period_code, period_name in [
                    (PERIOD_MONTHLY, "monthly"),
                    (PERIOD_QUARTERLY, "quarterly"),
                    (PERIOD_ANNUAL, "annual"),
                ]:
                    data = self.fetch_index_data(idx_id, period=period_code)
                    if data and len(data) > 0:
                        # Check if data has any records in 2015-2025
                        count = 0
                        sample = None
                        for item in data[:5]:
                            if isinstance(item, dict):
                                sample = item
                                count += 1
                        sector_results[idx_id] = {
                            "period": period_name,
                            "record_count": len(data),
                            "sample": sample,
                            "status": "OK",
                        }
                        logger.info(
                            f"  ✅ ID {idx_id} ({period_name}): "
                            f"{len(data)} records"
                        )
                        break  # found working period for this ID
                    else:
                        logger.info(f"  ❌ ID {idx_id} ({period_name}): no data")

            results[sector] = sector_results
        return results

    # -----------------------------------------------------------------
    # Parse one sector
    # -----------------------------------------------------------------
    def parse_sector(self, sector: str) -> list[TaldauRecord]:
        """
        Fetch and parse data for a single sector.
        Tries candidate IDs and period types until data is found.
        """
        config = SECTOR_CONFIG.get(sector)
        if not config:
            logger.error(f"Unknown sector: {sector}")
            return []

        logger.info(f"\n{'='*60}")
        logger.info(f"Parsing: {sector} — {config['name_en']}")
        logger.info(f"{'='*60}")

        records = []
        found = False

        for idx_id in config["candidate_ids"]:
            # Try monthly first, then quarterly, then annual
            for period_code, period_name in [
                (config["preferred_period"], "preferred"),
                (PERIOD_MONTHLY, "monthly"),
                (PERIOD_QUARTERLY, "quarterly"),
                (PERIOD_ANNUAL, "annual"),
            ]:
                data = self.fetch_index_data(idx_id, period=period_code)
                if not data or len(data) == 0:
                    continue

                logger.info(
                    f"  ✅ Using ID {idx_id} ({period_name}): "
                    f"{len(data)} raw records"
                )

                # Store raw for inspection
                self.api_results_raw[sector] = data
                self.working_ids[sector] = idx_id

                # Parse records
                records = self._parse_api_response(
                    data, sector, idx_id, config
                )

                if records:
                    found = True
                    break

            if found:
                break

        if not found:
            logger.warning(
                f"  ⚠️  No data found for {sector}. "
                f"Run --discover to check available IDs."
            )

        # Filter to 2015-2025
        records = [
            r for r in records
            if r.year and YEAR_START <= r.year <= YEAR_END
        ]

        logger.info(f"  Records in {YEAR_START}-{YEAR_END}: {len(records)}")
        self.records.extend(records)
        return records

    def _parse_api_response(
        self,
        data: list,
        sector: str,
        index_id: int,
        config: dict,
    ) -> list[TaldauRecord]:
        """
        Parse the raw API JSON into TaldauRecord objects.

        The Taldau API has several response formats depending on the
        indicator. Common patterns:

        Pattern A (flat list of period-value pairs):
        [
            {"period": "Январь 2023", "value": 12345.6, ...},
            ...
        ]

        Pattern B (nested with dimensions):
        {
            "dimensions": [...],
            "data": [
                {"period": "...", "value": ..., "dimValues": [...]}
            ]
        }

        Pattern C (tabular with headers):
        [
            ["", "2015", "2016", ...],
            ["Январь", 100.1, 100.2, ...],
            ...
        ]
        """
        records = []

        if not data:
            return records

        first = data[0] if data else None

        # --- Pattern A: list of dicts with "period" key ---
        if isinstance(first, dict) and any(
            k in first for k in ("period", "Period", "periodName", "date")
        ):
            for item in data:
                period_label = (
                    item.get("period") or
                    item.get("Period") or
                    item.get("periodName") or
                    item.get("date", "")
                )
                value = (
                    item.get("value") or
                    item.get("Value") or
                    item.get("indexValue")
                )
                if value is not None:
                    try:
                        value = float(str(value).replace(",", ".").replace(" ", ""))
                    except (ValueError, TypeError):
                        value = None

                period_info = parse_period_label(str(period_label))
                region = (
                    item.get("region") or
                    item.get("Region") or
                    item.get("areaName") or
                    "Республика Казахстан"
                )

                # Only keep national-level data
                region_str = str(region)
                if any(kw in region_str.lower() for kw in [
                    "республика", "казахстан", "всего", "итого"
                ]) or region_str == "Республика Казахстан":
                    records.append(TaldauRecord(
                        sector=sector,
                        index_id=index_id,
                        indicator_name=config["name_ru"],
                        period_label=str(period_label),
                        year=period_info["year"],
                        month=period_info["month"],
                        quarter=period_info["quarter"],
                        value=value,
                        unit=config["unit"],
                        region=region_str,
                        source_url=f"{TALDAU_BASE}/ru/NewIndex/GetIndex/{index_id}",
                    ))

        # --- Pattern B: nested dict ---
        elif isinstance(first, dict) and "data" in first:
            return self._parse_api_response(
                first["data"], sector, index_id, config
            )

        # --- Pattern C: tabular (list of lists) ---
        elif isinstance(first, list):
            headers = data[0]
            for row in data[1:]:
                if not row or not isinstance(row, list):
                    continue
                month_label = str(row[0]) if row else ""
                for col_idx, year_str in enumerate(headers[1:], start=1):
                    if col_idx >= len(row):
                        break
                    try:
                        year = int(str(year_str).strip())
                    except ValueError:
                        continue

                    value = row[col_idx]
                    if value is not None:
                        try:
                            value = float(str(value).replace(",", ".").replace(" ", ""))
                        except (ValueError, TypeError):
                            value = None

                    period_info = parse_period_label(f"{month_label} {year}")
                    records.append(TaldauRecord(
                        sector=sector,
                        index_id=index_id,
                        indicator_name=config["name_ru"],
                        period_label=f"{month_label} {year}",
                        year=year,
                        month=period_info["month"],
                        quarter=period_info["quarter"],
                        value=value,
                        unit=config["unit"],
                        source_url=f"{TALDAU_BASE}/ru/NewIndex/GetIndex/{index_id}",
                    ))

        # --- Fallback: try to extract any dict-like data ---
        else:
            logger.warning(f"  Unknown API response format. First element type: {type(first)}")
            if isinstance(first, dict):
                logger.info(f"  Keys: {list(first.keys())[:10]}")

        return records

    # -----------------------------------------------------------------
    # Also try scraping the HTML page (fallback)
    # -----------------------------------------------------------------
    def scrape_html_table(self, sector: str) -> list[TaldauRecord]:
        """
        Fallback: scrape the Taldau HTML page for tabular data.
        This works when the JSON API returns an empty result but
        the interactive page shows data.
        """
        config = SECTOR_CONFIG.get(sector)
        if not config:
            return []

        records = []
        for idx_id in config["candidate_ids"]:
            url = f"{TALDAU_BASE}/ru/NewIndex/GetIndex/{idx_id}"
            logger.info(f"  Scraping HTML: {url}")

            html = self.client.get_html(url)
            if not html:
                continue

            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
            except ImportError:
                logger.warning("  beautifulsoup4 not installed, skipping HTML scrape")
                return []

            # Look for data tables
            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                if len(rows) < 2:
                    continue

                headers = [
                    td.get_text(strip=True) for td in rows[0].find_all(["th", "td"])
                ]

                for row in rows[1:]:
                    cells = [td.get_text(strip=True) for td in row.find_all("td")]
                    if len(cells) < 2:
                        continue

                    for i, header in enumerate(headers[1:], start=1):
                        if i >= len(cells):
                            break
                        period_label = f"{cells[0]} {header}".strip()
                        value_str = cells[i].replace(",", ".").replace(" ", "")
                        try:
                            value = float(value_str)
                        except ValueError:
                            continue

                        period_info = parse_period_label(period_label)
                        if (period_info["year"] and
                                YEAR_START <= period_info["year"] <= YEAR_END):
                            records.append(TaldauRecord(
                                sector=sector,
                                index_id=idx_id,
                                indicator_name=config["name_ru"],
                                period_label=period_label,
                                year=period_info["year"],
                                month=period_info["month"],
                                quarter=period_info["quarter"],
                                value=value,
                                unit=config["unit"],
                                source_url=url,
                            ))

            if records:
                logger.info(f"  Scraped {len(records)} records from HTML")
                break

        self.records.extend(records)
        return records

    # -----------------------------------------------------------------
    # Run all sectors
    # -----------------------------------------------------------------
    def run_all(self, sectors: Optional[list[str]] = None) -> dict:
        """Parse all (or selected) sectors."""
        targets = sectors or list(SECTOR_CONFIG.keys())

        logger.info(f"Starting Taldau parse for: {targets}")
        logger.info(f"Date range: {YEAR_START}-01 to {YEAR_END}-12")
        logger.info("=" * 60)

        for sector in targets:
            records = self.parse_sector(sector)

            # If JSON API returned nothing, try HTML scrape
            if not records:
                logger.info(f"  Falling back to HTML scrape for {sector}...")
                self.scrape_html_table(sector)

        # Summary
        summary = {
            "sectors_parsed": targets,
            "total_records": len(self.records),
            "working_ids": self.working_ids,
            "records_by_sector": {},
            "date_range": f"{YEAR_START}-{YEAR_END}",
            "parsed_at": datetime.now().isoformat(),
        }
        for r in self.records:
            summary["records_by_sector"].setdefault(r.sector, 0)
            summary["records_by_sector"][r.sector] += 1

        logger.info(f"\n{'='*60}")
        logger.info("PARSING COMPLETE")
        for sec, count in summary["records_by_sector"].items():
            logger.info(f"  {sec:20s}: {count:6d} records")
        logger.info(f"  {'TOTAL':20s}: {summary['total_records']:6d} records")
        logger.info(f"{'='*60}")

        return summary

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all records to a pandas DataFrame."""
        if not self.records:
            return pd.DataFrame()

        df = pd.DataFrame([asdict(r) for r in self.records])

        # Create a proper date column for sorting / time series
        def make_date(row):
            y = row.get("year")
            m = row.get("month")
            q = row.get("quarter")
            if y and m:
                return pd.Timestamp(year=y, month=m, day=1)
            elif y and q:
                return pd.Timestamp(year=y, month=(q - 1) * 3 + 1, day=1)
            elif y:
                return pd.Timestamp(year=y, month=1, day=1)
            return pd.NaT

        df["date"] = df.apply(make_date, axis=1)
        df = df.sort_values(["sector", "date"]).reset_index(drop=True)
        return df

    def to_pivot(self) -> pd.DataFrame:
        """
        Create a wide-format DataFrame with columns = sectors,
        rows = monthly dates. Ready for train/test split.
        """
        df = self.to_dataframe()
        if df.empty:
            return df

        # Keep only monthly national data with non-null values
        monthly = df[df["month"].notna() & df["value"].notna()].copy()

        if monthly.empty:
            # Fall back to whatever granularity we have
            monthly = df[df["value"].notna()].copy()

        pivot = monthly.pivot_table(
            index="date",
            columns="sector",
            values="value",
            aggfunc="first",
        )
        pivot = pivot.sort_index()
        return pivot

    def export(self, output_dir: str = "output", fmt: str = "csv") -> list[str]:
        """Save results to files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = []

        # 1. Long-format (all records)
        df = self.to_dataframe()
        if not df.empty:
            fname = f"kz_taldau_stei_long_{ts}"
            if fmt == "csv":
                p = out / f"{fname}.csv"
                df.to_csv(p, index=False, encoding="utf-8-sig")
            elif fmt == "json":
                p = out / f"{fname}.json"
                df.to_json(p, orient="records", force_ascii=False, indent=2)
            elif fmt == "xlsx":
                p = out / f"{fname}.xlsx"
                df.to_excel(p, index=False, engine="openpyxl")
            files.append(str(p))
            logger.info(f"Saved long-format → {p}")

        # 2. Wide / pivot format (date × sector)
        pivot = self.to_pivot()
        if not pivot.empty:
            fname = f"kz_taldau_stei_wide_{ts}"
            if fmt == "csv":
                p = out / f"{fname}.csv"
                pivot.to_csv(p, encoding="utf-8-sig")
            elif fmt == "json":
                p = out / f"{fname}.json"
                pivot.to_json(p, force_ascii=False, indent=2)
            elif fmt == "xlsx":
                p = out / f"{fname}.xlsx"
                pivot.to_excel(p, engine="openpyxl")
            files.append(str(p))
            logger.info(f"Saved wide-format → {p}")

        # 3. Raw API responses (for debugging)
        if self.api_results_raw:
            p = out / f"kz_taldau_raw_api_{ts}.json"
            with open(p, "w", encoding="utf-8") as f:
                # Only save first 10 records per sector for brevity
                trimmed = {
                    k: v[:10] for k, v in self.api_results_raw.items()
                }
                json.dump(trimmed, f, ensure_ascii=False, indent=2, default=str)
            files.append(str(p))
            logger.info(f"Saved raw API    → {p}")

        # 4. Summary / metadata
        meta = {
            "date_range": f"{YEAR_START}-01 to {YEAR_END}-12",
            "sectors": list(SECTOR_CONFIG.keys()),
            "working_ids": self.working_ids,
            "total_records": len(self.records),
            "parsed_at": datetime.now().isoformat(),
            "coverage": {},
        }
        for sec in SECTOR_CONFIG:
            sec_records = [r for r in self.records if r.sector == sec]
            if sec_records:
                years = sorted(set(r.year for r in sec_records if r.year))
                months = sorted(set(
                    (r.year, r.month) for r in sec_records
                    if r.year and r.month
                ))
                meta["coverage"][sec] = {
                    "records": len(sec_records),
                    "year_range": f"{min(years)}-{max(years)}" if years else "N/A",
                    "monthly_points": len(months),
                    "expected_monthly": (YEAR_END - YEAR_START + 1) * 12,
                    "completeness_pct": round(
                        len(months) / ((YEAR_END - YEAR_START + 1) * 12) * 100, 1
                    ) if months else 0,
                }

        p = out / f"kz_taldau_meta_{ts}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        files.append(str(p))
        logger.info(f"Saved metadata   → {p}")

        return files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse Kazakhstan STEI sector data from Taldau "
            "(taldau.stat.gov.kz) for 2015-2025"
        ),
    )
    parser.add_argument(
        "--sector", "-s",
        choices=["all"] + list(SECTOR_CONFIG.keys()),
        default="all",
        help="Sector to parse (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        choices=["csv", "json", "xlsx"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--output-dir", "-d",
        default="output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run discovery mode to find valid index IDs",
    )
    args = parser.parse_args()

    client = TaldauClient(delay=args.delay)
    scraper = TaldauParser(client=client)

    if args.discover:
        print("\n🔍 Running index ID discovery...")
        results = scraper.discover_ids()

        print("\n" + "=" * 60)
        print("DISCOVERY RESULTS")
        print("=" * 60)
        for sector, ids in results.items():
            print(f"\n{sector} ({SECTOR_CONFIG[sector]['name_en']}):")
            if ids:
                for idx_id, info in ids.items():
                    print(
                        f"  ✅ ID {idx_id}: {info['record_count']} records "
                        f"({info['period']})"
                    )
            else:
                print("  ❌ No working IDs found")
        return

    sectors = None if args.sector == "all" else [args.sector]
    scraper.run_all(sectors=sectors)
    files = scraper.export(output_dir=args.output_dir, fmt=args.output)

    print(f"\n✅ Done! Created {len(files)} file(s):")
    for f in files:
        print(f"   📄 {f}")

    # Quick data quality check
    pivot = scraper.to_pivot()
    if not pivot.empty:
        print(f"\n📊 Data shape: {pivot.shape[0]} time periods × {pivot.shape[1]} sectors")
        print(f"   Date range: {pivot.index.min()} → {pivot.index.max()}")
        print(f"   Missing values per sector:")
        for col in pivot.columns:
            missing = pivot[col].isna().sum()
            total = len(pivot)
            print(f"     {col:20s}: {missing}/{total} missing ({missing/total*100:.1f}%)")


if __name__ == "__main__":
    main()