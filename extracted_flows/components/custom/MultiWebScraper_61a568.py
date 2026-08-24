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
# Per-source table registry.
#
# The key is matched as a SUBSTRING of the URL. For every source we declare:
#   - prefix : institution id used to build gold_table_id ({prefix}-t000k)
#   - n      : how many MAIN data tables to keep (in gold order)
#   - hints  : optional ordered keyword groups used to pin which physical
#              table gets t0001, t0002, ... If a table's context text
#              (caption + nearest heading + cell text) matches a hint group,
#              it takes that slot. Anything unmatched falls back to DOM order.
#
# These ids must line up with evidence_ground_truth.json:
#   bu  -> bu-t0001 (foreign), bu-t0002 (Thai)
#   ait -> ait-t0001, ait-t0002, ait-t0003 (dormitory)
#   iis -> iis-t0001
# ---------------------------------------------------------------------------
SOURCE_CONFIG = [
    {
        "match": "bu.ac.th/en/tuition-fees/master-degree/2026",
        "prefix": "bu",
        "n": 2,
        # foreign/international table first, Thai table second
        "hints": [["foreign", "international"], ["thai", "thailand"]],
    },
    {
        "match": "ait.ac.th/tuition-and-fees",
        "prefix": "ait",
        "n": 3,
        "hints": [],  # keep DOM order: main tuition, SOM, dormitory
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
    display_name = "Multi-URL Scraper-v1"
    description = (
        "Extracts only the MAIN data tables per source (drops row-filter / "
        "dropdown UI and nested 'inside link' tables), flattens each to markdown, "
        "and stamps gold_table_id from the source registry."
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
        # Unknown source: keep every real table, generic prefix.
        return {"prefix": "src", "n": 999, "hints": []}

    # ------------------------------------------------------- table -> grid/md
    def _table_to_grid(self, table) -> List[List[str]]:
        """Expand a <table> into a rectangular grid, honouring colspan/rowspan.

        Uses recursive=False on rows/cells so nested tables inside a cell are
        never pulled into this table's grid.
        """
        trs = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
        grid: List[dict] = []  # list of {col_index: text}

        for r, tr in enumerate(trs):
            while len(grid) <= r:
                grid.append({})
            cells = tr.find_all(["td", "th"], recursive=False)
            c = 0
            for cell in cells:
                # skip columns already filled by a previous rowspan
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

    def _grid_to_markdown(self, grid: List[List[str]]) -> str:
        if not grid:
            return ""
        width = max(len(r) for r in grid)
        header = (grid[0] + [""] * width)[:width]

        def esc(x: str) -> str:
            return (x or "").replace("|", "\\|").replace("\n", " ").strip()

        lines = ["| " + " | ".join(esc(h) for h in header) + " |",
                 "| " + " | ".join(["---"] * width) + " |"]
        for row in grid[1:]:
            row = (row + [""] * width)[:width]
            lines.append("| " + " | ".join(esc(c) for c in row) + " |")
        return "\n".join(lines)

    # --------------------------------------------------- table classification
    def _numeric_count(self, grid: List[List[str]]) -> int:
        return sum(1 for row in grid for cell in row if NUMERIC_RE.search(cell or ""))

    def _context_text(self, table) -> str:
        """Caption + nearest preceding headings + table text, for hint matching."""
        parts = []
        cap = table.find("caption")
        if cap:
            parts.append(cap.get_text(" ", strip=True))
        for prev in table.find_all_previous(["h1", "h2", "h3", "h4", "h5", "strong", "p"], limit=3):
            parts.append(prev.get_text(" ", strip=True))
        parts.append(table.get_text(" ", strip=True)[:300])
        return " ".join(parts).lower()

    def _extract_main_tables(self, soup, cfg):
        """Return [(table_id, markdown, context), ...] for the main data tables."""
        # Drop "inside link" tables entirely so their text cannot bleed into a
        # parent cell: any table nested in another table, or wrapped in <a>.
        for t in soup.find_all("table"):
            if t.find_parent("table") is not None or t.find_parent("a") is not None:
                t.decompose()

        candidates = []
        for dom_index, table in enumerate(soup.find_all("table")):
            # drop nested "inside link" tables and tables wrapped in an anchor
            if table.find_parent("table") is not None:
                continue
            if table.find_parent("a") is not None:
                continue
            # drop row-filter / dropdown / search UI rendered as a table
            if table.find(["select", "input", "button", "textarea"]):
                continue
            grid = self._table_to_grid(table)
            rows, cols = len(grid), (len(grid[0]) if grid else 0)
            if rows < 2 or cols < 2:
                continue
            numeric = self._numeric_count(grid)
            if numeric == 0:  # tuition tables always carry numbers
                continue
            score = rows * cols + 5 * numeric
            candidates.append({
                "dom_index": dom_index,
                "grid": grid,
                "score": score,
                "context": self._context_text(table),
            })

        # keep the top-n by score, then restore DOM order among the kept ones
        keep = sorted(candidates, key=lambda x: x["score"], reverse=True)[: cfg["n"]]
        keep = sorted(keep, key=lambda x: x["dom_index"])

        # assign gold_table_ids: honour hints first, then DOM order for the rest
        prefix = cfg["prefix"]
        assigned = [None] * len(keep)
        used = set()
        hints = cfg.get("hints") or []
        for slot, hint_group in enumerate(hints):
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
            table_id = f"{prefix}-t{slot + 1:04d}"
            out.append((table_id, self._grid_to_markdown(cand["grid"]), cand["context"]))
        return out

    # -------------------------------------------------------------- pipeline
    def _preclean(self, soup):
        """Decode CF emails and strip obvious non-content noise (not tables)."""
        for cf_tag in soup.find_all(class_="__cf_email__"):
            enc = cf_tag.get("data-cfemail")
            if enc:
                cf_tag.replace_with(self.decode_cf_email(enc))
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        for sel in [".o-breadcrumb", ".cli-bar-container", ".cli-bar-message",
                    ".fusion-breadcrumb", ".fusion-breadcrumbs", ".breadcrumb"]:
            for t in soup.select(sel):
                t.decompose()

    def scrape_urls(self) -> List[Data]:
        urls = self.urls
        url_list = (
            [u.strip() for u in urls.split("\n") if u.strip()]
            if isinstance(urls, str) else list(urls)
        )
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        results: List[Data] = []
        preview = []

        for url in url_list:
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                self._preclean(soup)

                cfg = self._config_for(url)
                tables = self._extract_main_tables(soup, cfg)

                if not tables:
                    results.append(Data(data={
                        "text": "Error: no main tables found",
                        "source": url, "table_id": None, "gold_table_id": None,
                        "institution": cfg["prefix"],
                    }))
                    preview.append(f"{url} -> 0 tables (CHECK SELECTORS)")
                    continue

                for table_id, md, _ctx in tables:
                    results.append(Data(data={
                        "text": md,               # naive markdown flatten (M1 representation)
                        "source": url,
                        "table_id": table_id,
                        "gold_table_id": table_id,
                        "institution": cfg["prefix"],
                    }))
                    first_row = md.splitlines()[0] if md else ""
                    preview.append(f"{table_id}: {first_row[:70]}")

            except Exception as e:
                results.append(Data(data={
                    "text": f"Error: {e}", "source": url,
                    "table_id": None, "gold_table_id": None, "institution": None,
                }))
                preview.append(f"{url} -> ERROR {e}")

        # Eyeball preview: verify the right tables were picked and ordered.
        self.status = "\n".join(preview)
        return results