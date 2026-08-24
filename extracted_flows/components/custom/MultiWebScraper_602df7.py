# Recovered Langflow component
# type: MultiWebScraper
# class: MultiWebScraper
# used in 2 flow(s): Method 1-Markdown Serialization, Method2- Summary
# json path: node.data.node.template.code.value

import hashlib
import re
from typing import List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString

from langflow.custom import Component
from langflow.io import BoolInput, IntInput, MultilineInput, Output
from langflow.schema import Data


# ---------------------------------------------------------------------------
# Per-source registry -> gold_table_id ({prefix}-t000k), matched by DOMAIN so
# path variants (e.g. /admissions/) and trailing spaces still resolve.
# Must line up with evidence_ground_truth.json:
#   bu  -> bu-t0001 (foreign), bu-t0002 (Thai)
#   ait -> ait-t0001, ait-t0002, ait-t0003 (dormitory)
#   iis -> iis-t0001
# k is a per-SOURCE 1-based counter (t0001, t0002, ...).
# ---------------------------------------------------------------------------
SOURCE_CONFIG = [
    {"matches": ["bu.ac.th"], "prefix": "bu", "n": 2,
     "hints": [["foreign", "international"], ["thai", "thailand"]]},
    {"matches": ["ait.ac.th"], "prefix": "ait", "n": 3, "hints": []},
    {"matches": ["iis.ru.ac.th"], "prefix": "iis", "n": 1, "hints": []},
]


def config_for(url: str) -> dict:
    u = (url or "").strip().lower()
    for cfg in SOURCE_CONFIG:
        if any(m in u for m in cfg["matches"]):
            return cfg
    return {"prefix": "src", "n": 999, "hints": []}


class MultiWebScraper(Component):
    display_name = "Markdown Web Scraper v3"
    description = "Scrapes to Markdown, consolidates fragment/duplicate tables, and marks each table with its gold_table_id."
    icon = "markdown"

    inputs = [
        MultilineInput(
            name="urls",
            display_name="URLs",
            info="Enter URLs (one per line).",
            value="https://ait.ac.th/admissions/tuition-and-fees/",
        ),
        BoolInput(name="skip_nested", display_name="Skip layout-wrapper tables", value=True,
                  info="Drop a <table> that contains another <table> (layout wrapper); keep the real inner table."),
        BoolInput(name="skip_hidden", display_name="Skip hidden/responsive tables", value=True,
                  info="Ignore tables hidden via display:none / aria-hidden / mobile-only classes (responsive duplicates)."),
        BoolInput(name="dedup_tables", display_name="Drop duplicate tables", value=True,
                  info="Drop a table whose content is identical to one already kept."),
        BoolInput(name="merge_same_header", display_name="Merge consecutive same-header tables", value=True,
                  info="Fold fragment tables that share the exact header row into one logical table."),
        IntInput(name="min_rows", display_name="Min rows to keep", value=2, advanced=True),
        IntInput(name="min_cols", display_name="Min cols to keep", value=2, advanced=True),
    ]

    outputs = [
        Output(display_name="Scraped Data List", name="scraped_data", method="scrape_urls"),
        Output(display_name="Merged Markdown", name="merged_data", method="merge_scraped_data"),
        Output(display_name="Table Report", name="table_report", method="report_tables"),
    ]

    # ------------------------------------------------------------------ utils
    def decode_cf_email(self, hex_string: str) -> str:
        try:
            key = int(hex_string[:2], 16)
            return "".join(chr(int(hex_string[i:i + 2], 16) ^ key) for i in range(2, len(hex_string), 2))
        except Exception:
            return "[Email Protected]"

    def clean_cell(self, cell) -> str:
        text = cell.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text.replace("|", "\\|")

    @staticmethod
    def _is_hidden(tag) -> bool:
        style = (tag.get("style") or "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            return True
        if tag.get("aria-hidden") == "true" or tag.has_attr("hidden"):
            return True
        cls = " ".join(tag.get("class") or []).lower()
        return any(k in cls for k in ["hidden", "mobile-only", "sr-only", "visually-hidden", "d-none"])

    def _extract_rows(self, table) -> list:
        raw = []
        for tr in table.find_all("tr"):
            if tr.find_parent("table") is not table:
                continue
            cells = [self.clean_cell(c) for c in tr.find_all(["td", "th"]) if c.find_parent("table") is table]
            if cells:
                raw.append(cells)
        return raw

    @staticmethod
    def _rows_to_md(raw: list) -> str:
        width = max(len(r) for r in raw)
        lines = []
        for i, cells in enumerate(raw):
            cells = cells + [""] * (width - len(cells))
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * width) + " |")
        return "\n".join(lines)

    @staticmethod
    def _section_index(soup) -> dict:
        """Map id(table) -> section counter that increments at every heading.
        Two tables in different sections must never be merged, even if their
        header rows are identical (e.g. BU foreign vs Thai fee tables)."""
        idx = 0
        m = {}
        for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "table"]):
            if el.name == "table":
                m[id(el)] = idx
            else:
                idx += 1
        return m

    def _context_text(self, table) -> str:
        """Nearest heading(s) + table text, lowercased — used for hint matching."""
        parts = []
        cap = table.find("caption")
        if cap:
            parts.append(cap.get_text(" ", strip=True))
        for prev in table.find_all_previous(["h1", "h2", "h3", "h4", "h5", "strong", "p"], limit=4):
            parts.append(prev.get_text(" ", strip=True))
        parts.append(table.get_text(" ", strip=True)[:300])
        return " ".join(parts).lower()

    @staticmethod
    def _assign_gold_ids(kept: list, prefix: str, hints: list) -> dict:
        """kept: ordered [{'pid','context'}]. Returns {pid: '{prefix}-t000k'}."""
        slots = [None] * len(kept)
        used = set()
        for slot, group in enumerate(hints or []):
            if slot >= len(kept):
                break
            for i, rec in enumerate(kept):
                if i in used:
                    continue
                if any(kw in rec["context"] for kw in group):
                    slots[slot] = rec
                    used.add(i)
                    break
        remaining = [rec for i, rec in enumerate(kept) if i not in used]
        for s in range(len(slots)):
            if slots[s] is None and remaining:
                slots[s] = remaining.pop(0)
        mapping = {}
        for slot, rec in enumerate(slots):
            if rec is not None:
                mapping[rec["pid"]] = f"{prefix}-t{slot + 1:04d}"
        return mapping

    # ----------------------------------------------- table consolidation pass
    def _consolidate_tables(self, soup, url, cfg, report):
        """Replace each kept top-level table with a [[pid]] placeholder.

        Returns (placeholders {pid: markdown}, kept_order [{'pid','context'}]).
        Gold ids are assigned AFTER this pass (so hints can reorder).
        """
        placeholders = {}
        kept_order = []
        seen_hashes = set()
        prev_sig = None
        prev_pid = None
        prev_rows = None
        prev_sec = None
        counter = 0
        secmap = self._section_index(soup)

        for n, table in enumerate(soup.find_all("table")):
            sec = secmap.get(id(table))
            raw = self._extract_rows(table)
            nrows = len(raw)
            ncols = max((len(r) for r in raw), default=0)
            reason = "kept"

            if self.skip_nested and table.find("table") is not None:
                reason = "dropped: layout wrapper (contains a nested table)"
            elif self.skip_hidden and self._is_hidden(table):
                reason = "dropped: hidden/responsive"
            elif nrows < self.min_rows or ncols < self.min_cols:
                reason = f"dropped: degenerate ({nrows}x{ncols})"

            sig = tuple(raw[0]) if raw else ()
            content_hash = hashlib.md5(self._rows_to_md(raw).encode("utf-8")).hexdigest()[:10] if raw else ""

            if reason == "kept" and self.dedup_tables and content_hash in seen_hashes:
                reason = "dropped: duplicate"

            entry = {"source": url, "dom_index": n, "rows": nrows, "cols": ncols,
                     "header": " | ".join(sig)[:80], "decision": reason, "pid": None, "table_id": ""}
            report.append(entry)

            if reason != "kept":
                table.decompose()
                continue

            seen_hashes.add(content_hash)

            if (self.merge_same_header and prev_sig == sig and prev_sec == sec
                    and prev_pid is not None and len(raw) > 1):
                prev_rows.extend(raw[1:])
                placeholders[prev_pid] = self._rows_to_md(prev_rows)
                entry["decision"] = "merged (same header, same section) into previous kept table"
                table.decompose()
                continue

            pid = f"PH_{cfg['prefix']}_{counter:03d}"
            counter += 1
            placeholders[pid] = self._rows_to_md(raw)
            kept_order.append({"pid": pid, "context": self._context_text(table)})
            entry["pid"] = pid
            table.replace_with(NavigableString(f"\n\n[[{pid}]]\n\n"))
            prev_sig, prev_pid, prev_rows, prev_sec = sig, pid, raw, sec

        return placeholders, kept_order

    # ---------------------------------------------------------- html -> md
    def html_to_markdown(self, element, url):
        if isinstance(element, NavigableString):
            return str(element).strip()
        tag = element.name
        if tag == "a" and element.get("href"):
            href = urljoin(url, element["href"])
            return f" [{element.get_text(strip=True)}]({href}) "
        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(tag[1])
            return f"\n\n{'#' * level} {element.get_text(strip=True)}\n"
        if tag == "li":
            return f"\n* {element.get_text(strip=True)}"
        if tag == "p":
            return f"\n\n{element.get_text(strip=True)}\n"
        return "".join(
            self.html_to_markdown(child, url)
            for child in element.children
            if getattr(child, "name", None) not in ["script", "style"]
        )

    # --------------------------------------------------------------- pipeline
    def _scrape(self):
        urls = self.urls
        url_list = [u.strip() for u in urls.split("\n") if u.strip()] if isinstance(urls, str) else urls
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        results, report = [], []

        for url in url_list:
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                for cf_tag in soup.find_all(class_="__cf_email__"):
                    enc = cf_tag.get("data-cfemail")
                    if enc:
                        cf_tag.replace_with(self.decode_cf_email(enc))
                for selector in [".o-breadcrumb", "nav", "footer", "header", "script", "style"]:
                    for tag in soup.select(selector):
                        tag.decompose()

                page_title = soup.title.string if soup.title else url
                main_body = soup.find("main") or soup.find("article") or soup.find("body")

                cfg = config_for(url)
                placeholders, kept_order = self._consolidate_tables(main_body or soup, url, cfg, report)

                # assign gold ids (hints first, then DOM order), then fill the report
                mapping = self._assign_gold_ids(kept_order, cfg["prefix"], cfg["hints"])
                for r in report:
                    if r.get("pid") in mapping:
                        r["table_id"] = mapping[r["pid"]]

                md = f"# PAGE: {page_title}\nURL: {url}\n\n"
                md += self.html_to_markdown(main_body, url) if main_body else "No content found."

                # substitute placeholders with marker-wrapped whole tables carrying the gold id
                for pid, table_md in placeholders.items():
                    gold_id = mapping.get(pid, f"{cfg['prefix']}-tXXXX")
                    block = f"<!-- TABLE id={gold_id} source={url} -->\n{table_md}\n<!-- /TABLE -->"
                    md = md.replace(f"[[{pid}]]", block)

                results.append(Data(data={"text": re.sub(r"\n{3,}", "\n\n", md), "source": url}))
            except Exception as e:
                results.append(Data(data={"text": f"Error: {e}", "source": url}))
                report.append({"source": url, "dom_index": -1, "rows": 0, "cols": 0,
                               "header": "", "decision": f"error: {e}", "pid": None, "table_id": ""})

        return results, report

    # ----------------------------------------------------------------- outputs
    def scrape_urls(self) -> List[Data]:
        results, _ = self._scrape()
        return results

    def merge_scraped_data(self) -> Data:
        results, _ = self._scrape()
        return Data(data={"text": "\n\n---\n\n".join(item.text for item in results)})

    def report_tables(self) -> Data:
        _, report = self._scrape()
        n_kept = sum(1 for r in report if r["decision"] == "kept")
        lines = [f"TABLE INVENTORY — {len(report)} <table> elements found, {n_kept} kept as logical tables", ""]
        lines.append(f"{'gold_table_id':14} {'src':24} {'dom#':>4} {'rows':>4} {'cols':>4}  decision | header")
        lines.append("-" * 110)
        for r in report:
            src = r["source"].split("//")[-1][:24]
            lines.append(f"{r['table_id']:14} {src:24} {r['dom_index']:>4} {r['rows']:>4} {r['cols']:>4}  {r['decision']} | {r['header']}")
        self.status = "\n".join(lines)
        return Data(data={"text": "\n".join(lines)})