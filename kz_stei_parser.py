"""
Kazakhstan BNS (stat.gov.kz) — Short-Term Economic Indicator (STEI / КЭИ) Parser
==================================================================================

Parses 6 core sector indicators from the Bureau of National Statistics:
  1. Industrial Production      (Промышленность)
  2. Construction               (Строительство)
  3. Trade                      (Торговля)
  4. Transport & Warehousing    (Транспорт и складирование)
  5. Agriculture                (Сельское хозяйство)
  6. Communications / ICT       (Связь / ИКТ)

Data sources:
  - STEI publication pages on stat.gov.kz (HTML + PDF links)
  - Sector-specific pages with downloadable xlsx/json/csv tables
  - Homepage headline indicators

Usage:
    python kz_stei_parser.py                    # Run all parsers, save to CSV + JSON
    python kz_stei_parser.py --sector industry   # Parse only industrial production
    python kz_stei_parser.py --output xlsx       # Save as Excel instead

Requirements:
    pip install requests beautifulsoup4 lxml openpyxl pandas
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
from bs4 import BeautifulSoup
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://stat.gov.kz"
LANG = "ru"

# Sector page URLs on stat.gov.kz
SECTOR_URLS = {
    "stei": {
        "name_ru": "Краткосрочный экономический индикатор (КЭИ)",
        "name_en": "Short-Term Economic Indicator (STEI)",
        "publications": f"{BASE_URL}/{LANG}/industries/economy/national-accounts/publications/",
        "dynamic_tables": f"{BASE_URL}/{LANG}/industries/economy/national-accounts/dynamic-tables/",
    },
    "industry": {
        "name_ru": "Объем промышленного производства",
        "name_en": "Volume of Industrial Production",
        "main": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-industrial-production/",
        "publications": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-industrial-production/publications/",
        "spreadsheets": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-industrial-production/spreadsheets/",
        "dynamic_tables": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-industrial-production/dynamic-tables/",
    },
    "construction": {
        "name_ru": "Объем строительных работ",
        "name_en": "Volume of Construction",
        "main": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-inno-build/",
        "publications": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-inno-build/publications/",
        "spreadsheets": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-inno-build/spreadsheets/",
        "dynamic_tables": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-inno-build/dynamic-tables/",
    },
    "trade": {
        "name_ru": "Объем торговли",
        "name_en": "Volume of Trade",
        "main": f"{BASE_URL}/{LANG}/industries/economy/local-market/",
        "publications": f"{BASE_URL}/{LANG}/industries/economy/local-market/publications/",
        "dynamic_tables": f"{BASE_URL}/{LANG}/industries/economy/local-market/dynamic-tables/",
    },
    "transport": {
        "name_ru": "Объем транспорта и складирования",
        "name_en": "Volume of Transport and Warehousing",
        "main": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-transport/",
        "publications": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-transport/publications/",
        "dynamic_tables": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-transport/dynamic-tables/",
    },
    "agriculture": {
        "name_ru": "Объем валового выпуска сельского хозяйства",
        "name_en": "Volume of Agriculture",
        "main": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-forrest-village-hunt-fish/",
        "publications": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-forrest-village-hunt-fish/publications/",
        "dynamic_tables": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-forrest-village-hunt-fish/dynamic-tables/",
    },
    "communications": {
        "name_ru": "Объем услуг связи",
        "name_en": "Volume of Communications / ICT",
        "main": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-it/",
        "publications": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-it/publications/",
        "dynamic_tables": f"{BASE_URL}/{LANG}/industries/business-statistics/stat-it/dynamic-tables/",
    },
}

# Keywords to identify relevant publications per sector
SECTOR_KEYWORDS = {
    "stei": [
        "краткосрочный экономический индикатор",
        "КЭИ",
    ],
    "industry": [
        "промышленного производства",
        "промышленности",
        "индекс промышленного",
    ],
    "construction": [
        "строительных работ",
        "строительство",
        "ввод в эксплуатацию",
    ],
    "trade": [
        "торговля",
        "товарооборот",
        "розничная торговля",
        "оптовая торговля",
    ],
    "transport": [
        "транспорт",
        "складирование",
        "грузоперевозк",
        "пассажирооборот",
    ],
    "agriculture": [
        "сельского хозяйства",
        "сельскохозяйствен",
        "валовой выпуск продукции",
        "растениеводств",
        "животноводств",
    ],
    "communications": [
        "связь",
        "телекоммуникац",
        "ИКТ",
        "информационно-коммуникац",
    ],
}

# Month name → number mapping (Russian)
MONTH_MAP = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4,
    "ма": 5, "июн": 6, "июл": 7, "август": 8,
    "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kz_stei_parser")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Publication:
    """A single BNS publication entry."""
    sector: str
    title: str
    url: str
    period: str = ""                  # e.g. "январь-декабрь 2024"
    year: Optional[int] = None
    month_start: Optional[int] = None
    month_end: Optional[int] = None
    file_url: str = ""                # direct download link (pdf/xlsx)
    file_format: str = ""             # pdf, xlsx, json, csv
    file_size_kb: Optional[float] = None
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class IndicatorValue:
    """A parsed numeric indicator value."""
    sector: str
    indicator_name: str
    period: str
    value: Optional[float] = None
    unit: str = "%"
    year: Optional[int] = None
    month_start: Optional[int] = None
    month_end: Optional[int] = None
    source_url: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DynamicTableMeta:
    """Metadata for a dynamic table (time series) available on BNS."""
    sector: str
    title: str
    page_url: str
    download_links: dict = field(default_factory=dict)  # {format: url}
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class BNSClient:
    """HTTP client with polite rate-limiting for stat.gov.kz."""

    def __init__(self, delay: float = 1.5, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        })
        self.delay = delay
        self.timeout = timeout
        self._last_request_time = 0.0

    def get(self, url: str, **kwargs) -> requests.Response:
        """GET with rate-limit delay and retries."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=self.timeout, **kwargs)
                self._last_request_time = time.time()
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt+1}/3 failed for {url}: {e}")
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    raise

    def get_soup(self, url: str) -> BeautifulSoup:
        """Fetch a page and return parsed BeautifulSoup."""
        resp = self.get(url)
        return BeautifulSoup(resp.text, "lxml")


# ---------------------------------------------------------------------------
# Period parsing helpers
# ---------------------------------------------------------------------------

def parse_period(text: str) -> dict:
    """
    Extract period info from strings like:
      "январь-декабрь 2024г."   → {year: 2024, month_start: 1, month_end: 12}
      "январь 2026г."           → {year: 2026, month_start: 1, month_end: 1}
      "2024г."                  → {year: 2024}
    """
    result = {"year": None, "month_start": None, "month_end": None, "period": text.strip()}

    # Extract year
    year_match = re.search(r"(\d{4})\s*г", text)
    if year_match:
        result["year"] = int(year_match.group(1))

    # Extract month(s)
    text_lower = text.lower()
    months_found = []
    for prefix, num in sorted(MONTH_MAP.items(), key=lambda x: -len(x[0])):
        # find all occurrences
        for m in re.finditer(re.escape(prefix), text_lower):
            months_found.append((m.start(), num))

    months_found.sort(key=lambda x: x[0])
    unique_months = list(dict.fromkeys(m[1] for m in months_found))

    if len(unique_months) >= 2:
        result["month_start"] = unique_months[0]
        result["month_end"] = unique_months[-1]
    elif len(unique_months) == 1:
        result["month_start"] = unique_months[0]
        result["month_end"] = unique_months[0]

    return result


def extract_percentage(text: str) -> Optional[float]:
    """Extract a percentage value like '109,1%' → 109.1"""
    match = re.search(r"(\d+[,.]?\d*)\s*%", text)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

class STEIParser:
    """
    Main parser that orchestrates data collection from stat.gov.kz
    for the 6 STEI sectors + the composite STEI indicator.
    """

    def __init__(self, client: Optional[BNSClient] = None):
        self.client = client or BNSClient()
        self.publications: list[Publication] = []
        self.indicators: list[IndicatorValue] = []
        self.dynamic_tables: list[DynamicTableMeta] = []

    # ------------------------------------------------------------------
    # 1. Parse the homepage headline indicators
    # ------------------------------------------------------------------
    def parse_homepage(self) -> list[IndicatorValue]:
        """
        The stat.gov.kz homepage shows headline numbers for STEI,
        industrial production index, trade turnover, etc.
        """
        logger.info("Parsing homepage headline indicators...")
        url = f"{BASE_URL}/{LANG}/"
        try:
            soup = self.client.get_soup(url)
        except Exception as e:
            logger.error(f"Failed to fetch homepage: {e}")
            return []

        results = []

        # Look for indicator cards/blocks on the homepage
        # The homepage uses div blocks with indicator values
        indicator_blocks = soup.find_all("div", class_=re.compile(r"indicator|stat-block|widget", re.I))
        if not indicator_blocks:
            # Fallback: search all text for known patterns
            indicator_blocks = [soup]

        page_text = soup.get_text(separator="\n")

        # Pattern: "Краткосрочный экономический индикатор ... 109,1%"
        stei_pattern = re.compile(
            r"Краткосрочный\s+экономический\s+индикатор.*?"
            r"(\d+[,.]?\d*)\s*%.*?"
            r"(январ[а-я]*[\s\-]*(?:декабр[а-я]*|ноябр[а-я]*|октябр[а-я]*|"
            r"сентябр[а-я]*|август[а-я]*|июл[а-я]*|июн[а-я]*|ма[а-я]*|"
            r"апрел[а-я]*|март[а-я]*|феврал[а-я]*|январ[а-я]*)?\s*\d{4})",
            re.IGNORECASE | re.DOTALL,
        )
        for m in stei_pattern.finditer(page_text):
            val = float(m.group(1).replace(",", "."))
            period_text = m.group(2).strip()
            period_info = parse_period(period_text)
            iv = IndicatorValue(
                sector="stei",
                indicator_name="Краткосрочный экономический индикатор (КЭИ)",
                period=period_text,
                value=val,
                unit="%",
                year=period_info["year"],
                month_start=period_info["month_start"],
                month_end=period_info["month_end"],
                source_url=url,
            )
            results.append(iv)
            logger.info(f"  STEI headline: {val}% for {period_text}")

        self.indicators.extend(results)
        return results

    # ------------------------------------------------------------------
    # 2. Parse publication listings for a sector
    # ------------------------------------------------------------------
    def parse_publications(self, sector: str, max_pages: int = 3) -> list[Publication]:
        """
        Scrape the publications listing page for a given sector.
        Collects title, period, download links.
        """
        config = SECTOR_URLS.get(sector)
        if not config or "publications" not in config:
            logger.warning(f"No publications URL for sector '{sector}'")
            return []

        base_pub_url = config["publications"]
        keywords = SECTOR_KEYWORDS.get(sector, [])
        results = []

        for page_num in range(1, max_pages + 1):
            url = base_pub_url if page_num == 1 else f"{base_pub_url}?PAGEN_1={page_num}"
            logger.info(f"Parsing publications for '{sector}' page {page_num}: {url}")

            try:
                soup = self.client.get_soup(url)
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                break

            # Find publication entries — BNS uses various card/list structures
            entries = soup.find_all("a", href=re.compile(r"/publications/\d+"))
            if not entries:
                entries = soup.find_all("a", href=re.compile(r"/publications/"))

            if not entries:
                logger.info(f"  No more publication entries found on page {page_num}")
                break

            seen_hrefs = set()
            for link in entries:
                href = link.get("href", "")
                if href in seen_hrefs:
                    continue
                seen_hrefs.add(href)

                title = link.get_text(strip=True)
                if not title:
                    continue

                # Filter by sector keywords
                title_lower = title.lower()
                if keywords and not any(kw.lower() in title_lower for kw in keywords):
                    continue

                full_url = href if href.startswith("http") else BASE_URL + href
                period_info = parse_period(title)

                pub = Publication(
                    sector=sector,
                    title=title,
                    url=full_url,
                    period=period_info["period"],
                    year=period_info["year"],
                    month_start=period_info["month_start"],
                    month_end=period_info["month_end"],
                )

                # Try to find associated file download links nearby
                parent = link.find_parent(["div", "li", "tr", "article"])
                if parent:
                    file_links = parent.find_all("a", href=re.compile(r"\.(pdf|xlsx|xls|csv|json)"))
                    for fl in file_links:
                        fhref = fl.get("href", "")
                        if not fhref:
                            continue
                        pub.file_url = fhref if fhref.startswith("http") else BASE_URL + fhref
                        ext = fhref.rsplit(".", 1)[-1].lower()
                        pub.file_format = ext
                        # Try to extract size
                        size_text = fl.get_text(strip=True)
                        size_match = re.search(r"([\d,.]+)\s*(Кб|Мб|KB|MB)", size_text, re.I)
                        if size_match:
                            size_val = float(size_match.group(1).replace(",", "."))
                            unit = size_match.group(2).lower()
                            if unit in ("мб", "mb"):
                                size_val *= 1024
                            pub.file_size_kb = size_val
                        break

                results.append(pub)
                logger.info(f"  Found: {title[:80]}...")

        self.publications.extend(results)
        return results

    # ------------------------------------------------------------------
    # 3. Parse a single STEI publication page for sector-level values
    # ------------------------------------------------------------------
    def parse_stei_publication(self, pub_url: str) -> list[IndicatorValue]:
        """
        Parse an individual STEI publication page to extract
        percentage values for each of the 6 sectors.
        """
        logger.info(f"Parsing STEI publication detail: {pub_url}")
        try:
            soup = self.client.get_soup(pub_url)
        except Exception as e:
            logger.error(f"Failed to fetch {pub_url}: {e}")
            return []

        page_text = soup.get_text(separator=" ")
        results = []

        # Extract the main STEI value
        main_match = re.search(
            r"(?:КЭИ|краткосрочный\s+экономический\s+индикатор)\s*"
            r".*?составил\s+(\d+[,.]?\d*)\s*%",
            page_text, re.IGNORECASE,
        )
        period_text = ""
        period_info = {}

        # Find the period
        period_match = re.search(
            r"за\s+(январ[а-яё\-\s]+\d{4})\s*г",
            page_text, re.IGNORECASE,
        )
        if period_match:
            period_text = period_match.group(1).strip()
            period_info = parse_period(period_text)

        if main_match:
            val = float(main_match.group(1).replace(",", "."))
            results.append(IndicatorValue(
                sector="stei",
                indicator_name="КЭИ (композитный)",
                period=period_text,
                value=val,
                year=period_info.get("year"),
                month_start=period_info.get("month_start"),
                month_end=period_info.get("month_end"),
                source_url=pub_url,
            ))

        # Extract per-sector indices
        # Pattern: "отрасль (XX,X%)" or "отрасль – на X,X%"
        sector_patterns = {
            "agriculture": r"сельское\s+хозяйство\s*\(?(\d+[,.]?\d*)\s*%",
            "industry": r"промышленность\s*\(?(\d+[,.]?\d*)\s*%",
            "construction": r"строительство\s*\(?(\d+[,.]?\d*)\s*%",
            "trade": r"торговл[а-я]*\s*\(?(\d+[,.]?\d*)\s*%",
            "transport": r"транспорт[а-я\s]*(?:и\s+складировани[а-я]*)?\s*\(?(\d+[,.]?\d*)\s*%",
            "communications": r"связь?\s*\(?(\d+[,.]?\d*)\s*%",
        }

        for sec, pattern in sector_patterns.items():
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                val = float(match.group(1).replace(",", "."))
                results.append(IndicatorValue(
                    sector=sec,
                    indicator_name=f"Индекс физического объема ({SECTOR_URLS[sec]['name_ru']})",
                    period=period_text,
                    value=val,
                    year=period_info.get("year"),
                    month_start=period_info.get("month_start"),
                    month_end=period_info.get("month_end"),
                    source_url=pub_url,
                ))
                logger.info(f"  {sec}: {val}%")

        self.indicators.extend(results)
        return results

    # ------------------------------------------------------------------
    # 4. Parse dynamic tables listing
    # ------------------------------------------------------------------
    def parse_dynamic_tables(self, sector: str) -> list[DynamicTableMeta]:
        """
        List available dynamic tables (time series) for a sector,
        with download links in xlsx / json / csv formats.
        """
        config = SECTOR_URLS.get(sector)
        if not config or "dynamic_tables" not in config:
            logger.warning(f"No dynamic_tables URL for sector '{sector}'")
            return []

        url = config["dynamic_tables"]
        logger.info(f"Parsing dynamic tables for '{sector}': {url}")

        try:
            soup = self.client.get_soup(url)
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return []

        results = []

        # Dynamic tables are listed as items with links to xlsx | json | csv
        # Look for list items or table rows containing download format links
        items = soup.find_all(["li", "div", "tr"], recursive=True)

        for item in items:
            text = item.get_text(separator=" ", strip=True)
            links = item.find_all("a", href=True)

            # Check if this item has format links
            format_links = {}
            for link in links:
                href = link.get("href", "")
                link_text = link.get_text(strip=True).lower()
                for fmt in ("xlsx", "xls", "json", "csv"):
                    if fmt in link_text or href.endswith(f".{fmt}"):
                        full_href = href if href.startswith("http") else BASE_URL + href
                        format_links[fmt] = full_href

            if format_links:
                # Extract the title (the non-link text or first link text)
                title = ""
                for link in links:
                    t = link.get_text(strip=True)
                    if t and t.lower() not in ("xlsx", "xls", "json", "csv"):
                        title = t
                        page_url = link.get("href", "")
                        page_url = page_url if page_url.startswith("http") else BASE_URL + page_url
                        break

                if not title:
                    # Use the full text minus format labels
                    title = re.sub(r"\b(xlsx|xls|json|csv)\b", "", text, flags=re.I).strip()

                if title:
                    dt = DynamicTableMeta(
                        sector=sector,
                        title=title[:200],
                        page_url=page_url if title else url,
                        download_links=format_links,
                    )
                    results.append(dt)
                    logger.info(f"  Table: {title[:80]}... formats={list(format_links.keys())}")

        self.dynamic_tables.extend(results)
        return results

    # ------------------------------------------------------------------
    # 5. Parse sector main page for headline indicators
    # ------------------------------------------------------------------
    def parse_sector_main(self, sector: str) -> list[IndicatorValue]:
        """
        Parse a sector's main page for headline stats
        (volume in mln tenge, physical volume index, etc.)
        """
        config = SECTOR_URLS.get(sector)
        if not config or "main" not in config:
            return []

        url = config["main"]
        logger.info(f"Parsing sector main page for '{sector}': {url}")

        try:
            soup = self.client.get_soup(url)
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return []

        page_text = soup.get_text(separator=" ")
        results = []

        # Look for patterns like "Объем X ... N,N млн.тенге" or "X ... N,N%"
        # and "за январь-декабрь 2025г."
        vol_patterns = [
            (r"(?:Объем|объем)\s+.*?(\d[\d\s,\.]+)\s*(?:млн|млрд)\.?\s*(?:тенге|тг)",
             "Объем (млн/млрд тенге)"),
            (r"(?:Индекс|индекс)\s+.*?(\d+[,.]?\d*)\s*%",
             "Индекс физического объема (%)"),
        ]

        for pattern, indicator_name in vol_patterns:
            for match in re.finditer(pattern, page_text, re.IGNORECASE):
                val_str = match.group(1).replace(" ", "").replace(",", ".")
                try:
                    val = float(val_str)
                except ValueError:
                    continue

                # Find the nearest period reference
                context_start = max(0, match.start() - 200)
                context = page_text[context_start:match.end() + 100]
                p_match = re.search(r"(январ[а-яё\-\s]+\d{4})\s*г", context, re.I)
                period_text = p_match.group(1).strip() if p_match else ""
                period_info = parse_period(period_text) if period_text else {}

                unit = "%" if "%" in match.group(0) else "млн/млрд тенге"

                results.append(IndicatorValue(
                    sector=sector,
                    indicator_name=f"{config['name_ru']} — {indicator_name}",
                    period=period_text,
                    value=val,
                    unit=unit,
                    year=period_info.get("year"),
                    month_start=period_info.get("month_start"),
                    month_end=period_info.get("month_end"),
                    source_url=url,
                ))

        if results:
            logger.info(f"  Found {len(results)} indicator values on sector main page")

        self.indicators.extend(results)
        return results

    # ------------------------------------------------------------------
    # 6. Run all parsers
    # ------------------------------------------------------------------
    def run_all(self, sectors: Optional[list[str]] = None) -> dict:
        """
        Execute the full parsing pipeline for selected (or all) sectors.

        Returns a summary dict with counts.
        """
        all_sectors = ["stei", "industry", "construction", "trade",
                       "transport", "agriculture", "communications"]
        target_sectors = sectors or all_sectors

        logger.info(f"Starting full parse for sectors: {target_sectors}")
        logger.info("=" * 70)

        # 1. Homepage headlines
        self.parse_homepage()

        for sector in target_sectors:
            logger.info(f"\n{'='*70}")
            logger.info(f"SECTOR: {sector} — {SECTOR_URLS[sector]['name_en']}")
            logger.info(f"{'='*70}")

            # 2. Publications listing
            self.parse_publications(sector, max_pages=2)

            # 3. If STEI, parse the latest publication detail page
            if sector == "stei" and self.publications:
                stei_pubs = [p for p in self.publications if p.sector == "stei"]
                if stei_pubs:
                    # Parse the most recent one
                    latest = sorted(stei_pubs, key=lambda p: (p.year or 0, p.month_end or 0), reverse=True)
                    self.parse_stei_publication(latest[0].url)

            # 4. Sector main page
            if sector != "stei":
                self.parse_sector_main(sector)

            # 5. Dynamic tables catalog
            self.parse_dynamic_tables(sector)

        summary = {
            "sectors_parsed": target_sectors,
            "total_publications": len(self.publications),
            "total_indicators": len(self.indicators),
            "total_dynamic_tables": len(self.dynamic_tables),
            "parsed_at": datetime.now().isoformat(),
        }

        logger.info(f"\n{'='*70}")
        logger.info(f"PARSING COMPLETE")
        logger.info(f"  Publications found:  {summary['total_publications']}")
        logger.info(f"  Indicator values:    {summary['total_indicators']}")
        logger.info(f"  Dynamic tables:      {summary['total_dynamic_tables']}")
        logger.info(f"{'='*70}")

        return summary

    # ------------------------------------------------------------------
    # 7. Export results
    # ------------------------------------------------------------------
    def export(self, output_dir: str = "output", fmt: str = "csv") -> list[str]:
        """
        Save all parsed data to files.

        Args:
            output_dir: Directory to save files
            fmt: "csv", "json", or "xlsx"

        Returns:
            List of created file paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        files_created = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # -- Publications catalog --
        if self.publications:
            df = pd.DataFrame([asdict(p) for p in self.publications])
            fname = f"kz_bns_publications_{ts}"
            if fmt == "csv":
                path = out / f"{fname}.csv"
                df.to_csv(path, index=False, encoding="utf-8-sig")
            elif fmt == "json":
                path = out / f"{fname}.json"
                df.to_json(path, orient="records", force_ascii=False, indent=2)
            elif fmt == "xlsx":
                path = out / f"{fname}.xlsx"
                df.to_excel(path, index=False, engine="openpyxl")
            files_created.append(str(path))
            logger.info(f"Saved publications → {path}")

        # -- Indicator values --
        if self.indicators:
            df = pd.DataFrame([asdict(iv) for iv in self.indicators])
            fname = f"kz_bns_indicators_{ts}"
            if fmt == "csv":
                path = out / f"{fname}.csv"
                df.to_csv(path, index=False, encoding="utf-8-sig")
            elif fmt == "json":
                path = out / f"{fname}.json"
                df.to_json(path, orient="records", force_ascii=False, indent=2)
            elif fmt == "xlsx":
                path = out / f"{fname}.xlsx"
                df.to_excel(path, index=False, engine="openpyxl")
            files_created.append(str(path))
            logger.info(f"Saved indicators  → {path}")

        # -- Dynamic tables catalog --
        if self.dynamic_tables:
            records = []
            for dt in self.dynamic_tables:
                rec = asdict(dt)
                rec["download_links"] = json.dumps(dt.download_links, ensure_ascii=False)
                records.append(rec)
            df = pd.DataFrame(records)
            fname = f"kz_bns_dynamic_tables_{ts}"
            if fmt == "csv":
                path = out / f"{fname}.csv"
                df.to_csv(path, index=False, encoding="utf-8-sig")
            elif fmt == "json":
                path = out / f"{fname}.json"
                df.to_json(path, orient="records", force_ascii=False, indent=2)
            elif fmt == "xlsx":
                path = out / f"{fname}.xlsx"
                df.to_excel(path, index=False, engine="openpyxl")
            files_created.append(str(path))
            logger.info(f"Saved dynamic tables → {path}")

        # -- Summary report --
        summary = {
            "parsed_at": datetime.now().isoformat(),
            "sectors": list(SECTOR_URLS.keys()),
            "publications_count": len(self.publications),
            "indicators_count": len(self.indicators),
            "dynamic_tables_count": len(self.dynamic_tables),
            "indicators_by_sector": {},
        }
        for iv in self.indicators:
            summary["indicators_by_sector"].setdefault(iv.sector, []).append({
                "indicator": iv.indicator_name,
                "value": iv.value,
                "unit": iv.unit,
                "period": iv.period,
            })

        path = out / f"kz_bns_summary_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        files_created.append(str(path))
        logger.info(f"Saved summary     → {path}")

        return files_created


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse Kazakhstan BNS (stat.gov.kz) STEI economic indicators",
    )
    parser.add_argument(
        "--sector", "-s",
        choices=["all", "stei", "industry", "construction", "trade",
                 "transport", "agriculture", "communications"],
        default="all",
        help="Which sector(s) to parse (default: all)",
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
        default=1.5,
        help="Delay between requests in seconds (default: 1.5)",
    )
    args = parser.parse_args()

    client = BNSClient(delay=args.delay)
    scraper = STEIParser(client=client)

    sectors = None if args.sector == "all" else [args.sector]
    scraper.run_all(sectors=sectors)
    files = scraper.export(output_dir=args.output_dir, fmt=args.output)

    print(f"\n✅ Done! Created {len(files)} file(s):")
    for f in files:
        print(f"   📄 {f}")


if __name__ == "__main__":
    main()