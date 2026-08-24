# Recovered Langflow component
# type: MultiWebScraper
# class: MultiWebScraper
# used in 1 flow(s): Method 1-Markdown Serialization
# json path: node.data.node.template.code.value

import hashlib
import re
from typing import Any, List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString

from langflow.custom import Component
from langflow.io import BoolInput, IntInput, MultilineInput, Output
from langflow.schema import Data


class MultiWebScraper(Component):
    display_name = "Markdown Web Scraper v4 (table-aware)"
    description = "Scrapes pages to Markdown, consolidates duplicate/fragment tables, and marks whole tables."
    icon = "markdown"

    inputs = [
        MultilineInput(
            name="urls",
            display_name="URLs",
            info="Enter URLs (one per line).",
            value="https://ait.ac.th/admissions/tuition-and-fees/",
        ),
        BoolInput(
            name="skip_nested",
            display_name="Skip layout-wrapper tables",
            value=True,
            info="Drop a <table> that contains another <table> (layout wrapper); keep the real inner table.",
        ),
        BoolInput(
            name="skip_hidden",
            display_name="Skip hidden/responsive tables",
            value=True,
            info="Ignore tables hidden via display:none / aria-hidden / mobile-only classes (responsive duplicates).",
        ),
        BoolInput(
            name="dedup_tables",
            display_name="Drop duplicate tables",
            value=True,
            info="Drop a table whose content is identical to one already kept.",
        ),
        BoolInput(
            name="merge_same_header",
            display_name="Merge consecutive same-header tables",
            value=True,
            info="Fold fragment tables that share the exact header row into one logical table.",
        ),
        BoolInput(
            name="merge_side_by_side",
            display_name="Merge side-by-side tables (frozen-column split)",
            value=True,
            info="Horizontally join adjacent tables (same parent, same row count) — e.g. a frozen first column split from its data columns.",
        ),
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
            return "".join(chr(int(hex_string[i : i + 2], 16) ^ key) for i in range(2, len(hex_string), 2))
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

    def _extract_rows(self, table) -> list[list[str]]:
        raw = []
        # only this table's OWN rows — exclude rows that belong to a nested table
        for tr in table.find_all("tr"):
            if tr.find_parent("table") is not table:
                continue
            cells = [self.clean_cell(c) for c in tr.find_all(["td", "th"]) if c.find_parent("table") is table]
            if cells:
                raw.append(cells)
        return raw

    @staticmethod
    def _rows_to_md(raw: list[list[str]]) -> str:
        width = max(len(r) for r in raw)
        lines = []
        for i, cells in enumerate(raw):
            cells = cells + [""] * (width - len(cells))
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * width) + " |")
        return "\n".join(lines)

    # ----------------------------------------------- table consolidation pass
    def _consolidate_tables(self, soup, url, start_idx, report):
        """Replace each kept top-level table with a [[TABLE:tid]] placeholder.

        Returns {placeholder_id: (table_id, markdown)} and updates the running index.
        Drops/merges duplicates, hidden, nested, degenerate, and same-header fragments,
        and horizontally joins frozen-column (side-by-side) splits.
        """
        idx = start_idx
        placeholders = {}
        seen_hashes = set()
        prev_sig = None
        prev_pid = None      # placeholder id of the previous KEPT table (for merging)
        prev_rows = None
        prev_parent = None   # DOM parent of the previous KEPT table (for horizontal join)

        for n, table in enumerate(soup.find_all("table")):
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

            # record in report regardless
            report.append(
                {
                    "source": url,
                    "dom_index": n,
                    "rows": nrows,
                    "cols": ncols,
                    "header": " | ".join(sig)[:80],
                    "decision": reason,
                }
            )

            if reason != "kept":
                table.decompose()
                continue

            seen_hashes.add(content_hash)

            # (a) vertical fold: consecutive fragments sharing the header row
            if self.merge_same_header and prev_sig == sig and prev_pid is not None and len(raw) > 1:
                prev_rows.extend(raw[1:])  # append data rows only
                placeholders[prev_pid] = (placeholders[prev_pid][0], self._rows_to_md(prev_rows))
                report[-1]["decision"] = f"v-merged into {placeholders[prev_pid][0]}"
                table.decompose()
                continue

            # (b) horizontal join: frozen-column split — same parent, same row count
            if (
                self.merge_side_by_side
                and prev_rows is not None
                and prev_pid is not None
                and len(raw) == len(prev_rows)
                and table.parent is prev_parent
            ):
                merged = [pr + cr for pr, cr in zip(prev_rows, raw)]
                placeholders[prev_pid] = (placeholders[prev_pid][0], self._rows_to_md(merged))
                report[-1]["decision"] = f"h-merged into {placeholders[prev_pid][0]}"
                prev_rows = merged  # keep chaining for 3+ split groups
                table.decompose()
                continue

            tid = f"t{idx:04d}"
            idx += 1
            pid = f"PH_{tid}"
            placeholders[pid] = (tid, self._rows_to_md(raw))
            parent = table.parent
            table.replace_with(NavigableString(f"\n\n[[{pid}]]\n\n"))
            prev_sig, prev_pid, prev_rows, prev_parent = sig, pid, raw, parent

        return placeholders, idx

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
            self.html_to_markdown(child, url) for child in element.children if getattr(child, "name", None) not in ["script", "style"]
        )

    # --------------------------------------------------------------- pipeline
    def _scrape(self):
        urls = self.urls
        url_list = [u.strip() for u in urls.split("\n") if u.strip()] if isinstance(urls, str) else urls
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        results, report = [], []
        tindex = 0

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

                placeholders, tindex = self._consolidate_tables(main_body or soup, url, tindex, report)

                md = f"# PAGE: {page_title}\nURL: {url}\n\n"
                md += self.html_to_markdown(main_body, url) if main_body else "No content found."

                # substitute placeholders with marker-wrapped whole tables
                for pid, (tid, table_md) in placeholders.items():
                    block = f"<!-- TABLE id={tid} source={url} -->\n{table_md}\n<!-- /TABLE -->"
                    md = md.replace(f"[[{pid}]]", block)

                results.append(Data(data={"text": re.sub(r"\n{3,}", "\n\n", md), "source": url}))
            except Exception as e:
                results.append(Data(data={"text": f"Error: {e}", "source": url}))
                report.append({"source": url, "dom_index": -1, "rows": 0, "cols": 0, "header": "", "decision": f"error: {e}"})

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
        kept = [r for r in report if r["decision"] == "kept" or r["decision"].startswith(("merged", "v-merged", "h-merged"))]
        lines = [f"TABLE INVENTORY — {len(report)} <table> elements found, {sum(1 for r in report if r['decision']=='kept')} kept as logical tables", ""]
        lines.append(f"{'src':28} {'dom#':>4} {'rows':>4} {'cols':>4}  decision | header")
        lines.append("-" * 100)
        for r in report:
            src = r["source"].split("//")[-1][:28]
            lines.append(f"{src:28} {r['dom_index']:>4} {r['rows']:>4} {r['cols']:>4}  {r['decision']} | {r['header']}")
        self.status = "\n".join(lines)
        return Data(data={"text": "\n".join(lines)})