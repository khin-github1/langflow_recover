# Recovered Langflow component
# type: MultiWebScraper
# class: MultiWebScraper
# used in 3 flow(s): Baseline- Naive, Evaluation_for_all, Table-aware-scraper
# json path: node.data.node.template.code.value

import re
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin

from lfx.custom.custom_component.component import Component
from lfx.io import MultilineInput, Output
from lfx.schema.data import Data
from typing import List


# ---------------------------------------------------------------------------
# Per-source table registry (matched as a SUBSTRING of the URL).
#   prefix : institution id -> gold_table_id ({prefix}-t000k)
#   n      : how many MAIN data tables to keep (gold order)
#   hints  : ordered keyword groups to pin which table gets t0001, t0002, ...
#            (matched against caption + nearest heading + cell text)
# Must line up with evidence_ground_truth.json:
#   bu -> bu-t0001 (foreign), bu-t0002 (Thai)
#   ait -> ait-t0001, ait-t0002, ait-t0003 (dormitory)
#   iis -> iis-t0001
# ---------------------------------------------------------------------------
SOURCE_CONFIG = [
    {
        "match": "bu.ac.th/en/tuition-fees/master-degree/2026",
        "prefix": "bu",
        "n": 2,
        "hints": [["foreign", "international"], ["thai", "thailand"]],
    },
    {
        "match": "ait.ac.th/tuition-and-fees",
        "prefix": "ait",
        "n": 3,
        "hints": [],
    },
    {
        "match": "iis.ru.ac.th",
        "prefix": "iis",
        "n": 1,
        "hints": [],
    },
]

NUMERIC_RE = re.compile(r"\d")


class MultiWebScraper(Component):
    display_name = "Multi-URL Scraper-v2"
    description = (
        "Emits page TEXT plus each MAIN table as PLAIN TEXT (no markdown). "
        "Drops row-filter/dropdown UI and nested 'inside link' tables, and "
        "stamps gold_table_id per table from the source registry."
    )
    icon = "globe"

    inputs = [
        MultilineInput(
            name="urls",
            display_name="URLs",
            info="Enter URLs (one per line).",
            value="https://www.bu.ac.th/en/tuition-fees/master-degree/2026",
        ),
    ]

    outputs = [
        Output(display_name="Scraped Data", name="scraped_data", method="scrape_urls"),
    ]

    # ------------------------------------------------------------------ utils
    def decode_cf_email(self, hex_string: str) -> str:
        """Decodes Cloudflare's XOR-encoded email strings."""
        try:
            key = int(hex_string[:2], 16)
            return "".join(
                chr(int(hex_string[i:i + 2], 16) ^ key)
                for i in range(2, len(hex_string), 2)
            )
        except Exception:
            return "[Decoding Failed]"

    def _config_for(self, url: str) -> dict:
        for cfg in SOURCE_CONFIG:
            if cfg["match"] in url:
                return cfg
        return {"prefix": "src", "n": 999, "hints": []}

    # ------------------------------------------------------ table -> grid/text
    def _table_to_grid(self, table) -> List[List[str]]:
        """Expand a <table> into a rectangular grid, honouring colspan/rowspan.
        recursive=False on rows/cells keeps nested tables out of this grid."""
        trs = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
        grid: List[dict] = []

        for r, tr in enumerate(trs):
            while len(grid) <= r:
                grid.append({})
            cells = tr.find_all(["td", "th"], recursive=False)
            c = 0
            for cell in cells:
                while c in grid[r]:
                    c += 1
                text = cell.get_text(separator=" ", strip=True)
                try:
                    colspan = int(cell.get("colspan", 1) or 1)
                except ValueError:
                    colspan = 1
                try:
                    rowspan = int(cell.get("rowspan", 1) or 1)
                except ValueError:
                    rowspan = 1
                for dr in range(rowspan):
                    while len(grid) <= r + dr:
                        grid.append({})
                    for dc in range(colspan):
                        grid[r + dr][c + dc] = text
                c += colspan

        max_col = max((max(rowd) + 1 if rowd else 0) for rowd in grid) if grid else 0
        return [[rowd.get(i, "") for i in range(max_col)] for rowd in grid]

    def _grid_to_plaintext(self, grid: List[List[str]]) -> str:
        """Plain text: one row per line, cells separated by a tab. No markdown."""
        lines = []
        for row in grid:
            cells = [(c or "").replace("\n", " ").strip() for c in row]
            lines.append("\t".join(cells).rstrip())
        return "\n".join(lines).strip()

    # --------------------------------------------------- table classification
    def _numeric_count(self, grid: List[List[str]]) -> int:
        return sum(1 for row in grid for cell in row if NUMERIC_RE.search(cell or ""))

    def _context_text(self, table) -> str:
        parts = []
        cap = table.find("caption")
        if cap:
            parts.append(cap.get_text(" ", strip=True))
        for prev in table.find_all_previous(["h1", "h2", "h3", "h4", "h5", "strong", "p"], limit=3):
            parts.append(prev.get_text(" ", strip=True))
        parts.append(table.get_text(" ", strip=True)[:300])
        return " ".join(parts).lower()

    def _extract_main_tables(self, soup, cfg):
        """Return [{table_id, text, tag}, ...] for the main data tables."""
        # Remove 'inside link' tables so their text can't bleed into a parent cell.
        for t in soup.find_all("table"):
            if t.find_parent("table") is not None or t.find_parent("a") is not None:
                t.decompose()

        candidates = []
        for dom_index, table in enumerate(soup.find_all("table")):
            if table.find(["select", "input", "button", "textarea"]):
                continue  # row-filter / dropdown UI
            grid = self._table_to_grid(table)
            rows, cols = len(grid), (len(grid[0]) if grid else 0)
            if rows < 2 or cols < 2:
                continue
            numeric = self._numeric_count(grid)
            if numeric == 0:
                continue
            candidates.append({
                "dom_index": dom_index,
                "grid": grid,
                "tag": table,
                "score": rows * cols + 5 * numeric,
                "context": self._context_text(table),
            })

        keep = sorted(candidates, key=lambda x: x["score"], reverse=True)[: cfg["n"]]
        keep = sorted(keep, key=lambda x: x["dom_index"])

        prefix = cfg["prefix"]
        assigned = [None] * len(keep)
        used = set()
        for slot, hint_group in enumerate(cfg.get("hints") or []):
            if slot >= len(keep):
                break
            for i, cand in enumerate(keep):
                if i in used:
                    continue
                if any(kw in cand["context"] for kw in hint_group):
                    assigned[slot] = cand
                    used.add(i)
                    break
        remaining = [c for i, c in enumerate(keep) if i not in used]
        for slot in range(len(assigned)):
            if assigned[slot] is None and remaining:
                assigned[slot] = remaining.pop(0)

        out = []
        for slot, cand in enumerate(assigned):
            if cand is None:
                continue
            out.append({
                "table_id": f"{prefix}-t{slot + 1:04d}",
                "text": self._grid_to_plaintext(cand["grid"]),
                "tag": cand["tag"],
            })
        return out

    # -------------------------------------------------------------- pipeline
    def _preclean(self, soup):
        for cf_tag in soup.find_all(class_="__cf_email__"):
            enc = cf_tag.get("data-cfemail")
            if enc:
                cf_tag.replace_with(self.decode_cf_email(enc))
        for sel in [".o-breadcrumb", ".cli-bar-container", ".cli-bar-message",
                    ".fusion-breadcrumb", ".fusion-breadcrumbs", ".breadcrumb",
                    ".fusion-page-title-bar", ".fusion-page-title-row"]:
            for t in soup.select(sel):
                t.decompose()
        for node in soup.find_all(string=re.compile(r"Home\s*>\s*", re.I)):
            parent = node.find_parent()
            if parent:
                parent.decompose()
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

    def _capture_links(self, soup, url) -> str:
        important = []
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            if any(x in text for x in ["Apply Now", "brochure", "Click this link"]):
                important.append(f"{text}: {urljoin(url, a['href'])}")
        internal = []
        for link in soup.find_all("a", string=re.compile(r"click(ing)? here", re.I)):
            href = link.get("href")
            if href:
                ctx = link.find_parent().get_text(strip=True)
                internal.append(f"Context: {ctx} -> URL: {urljoin(url, href)}")
        extra = ""
        if important:
            extra += "\n\nImportant Links:\n" + "\n".join(important)
        if internal:
            extra += "\n\nAdditional Resources Found:\n" + "\n".join(internal)
        return extra

    def scrape_urls(self) -> List[Data]:
        urls = self.urls
        url_list = (
            [u.strip() for u in urls.split("\n") if u.strip()]
            if isinstance(urls, str) else list(urls)
        )
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        results: List[Data] = []

        for url in url_list:
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                self._preclean(soup)

                cfg = self._config_for(url)
                link_text = self._capture_links(soup, url)

                # 1) main tables as PLAIN TEXT, each with its gold_table_id
                tables = self._extract_main_tables(soup, cfg)
                for t in tables:
                    results.append(Data(data={
                        "text": t["text"],
                        "source": url,
                        "table_id": t["table_id"],
                        "gold_table_id": t["table_id"],
                        "institution": cfg["prefix"],
                        "kind": "table",
                    }))
                    t["tag"].decompose()  # drop so it isn't duplicated in page text

                # 2) remaining page TEXT (tables removed above, plus link capture)
                body = (soup.find("main") or soup.find("article")
                        or soup.find(id="content") or soup.find("body") or soup)
                page_text = body.get_text(separator=" ", strip=True) + link_text
                if page_text.strip():
                    results.append(Data(data={
                        "text": page_text.strip(),
                        "source": url,
                        "table_id": None,
                        "gold_table_id": None,
                        "institution": cfg["prefix"],
                        "kind": "text",
                    }))

            except Exception as e:
                results.append(Data(data={
                    "text": f"Error: {e}", "source": url,
                    "table_id": None, "gold_table_id": None,
                    "institution": None, "kind": "error",
                }))

        # Show the actual scraped Data in the node output.
        self.status = results
        return results