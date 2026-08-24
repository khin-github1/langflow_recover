# Recovered Langflow component
# type: MultiWebScraper
# class: MultiWebScraper
# used in 1 flow(s): Method 1-Markdown Serialization
# json path: node.data.node.template.code.value

import requests
import re
from bs4 import BeautifulSoup, NavigableString
from langflow.custom import Component
from langflow.io import MultilineInput, Output
from langflow.schema import Data
from typing import List, Any
from urllib.parse import urljoin


class MultiWebScraper(Component):
    display_name = "Markdown Web Scraper v1"
    description = "Extracts structured Markdown. Use 'Merged Markdown' for Write File components."
    icon = "markdown"

    inputs = [
        MultilineInput(
            name="urls",
            display_name="URLs",
            info="Enter URLs (one per line).",
            value="https://ait.ac.th/admissions/tuition-and-fees/",
        ),
    ]

    outputs = [
        # Returns a list of Data objects (one per URL)
        Output(display_name="Scraped Data List", name="scraped_data", method="scrape_urls"),
        # Returns ONE merged string — use this for Write File components
        Output(display_name="Merged Markdown", name="merged_data", method="merge_scraped_data"),
    ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def decode_cf_email(self, hex_string: str) -> str:
        try:
            key = int(hex_string[:2], 16)
            return "".join(
                chr(int(hex_string[i:i + 2], 16) ^ key)
                for i in range(2, len(hex_string), 2)
            )
        except Exception:
            return "[Email Protected]"

    def clean_cell(self, cell) -> str:
        """Flatten ONE table cell into a single, structure-safe line.

        Why this exists: a header or data cell can hold <br> tags or stacked
        block elements (e.g. "PhD Program / 7 Semesters / (84 credits) / 42
        Months"). Markdown requires one table row = one physical line, so any
        newline inside a cell shatters the table. We therefore:
          1. join inner text with spaces (separator=' ') instead of letting it
             concatenate,
          2. collapse every whitespace run (newlines, tabs, doubled spaces)
             into a single space,
          3. escape '|' so stray pipes don't create phantom columns.
        We deliberately use get_text here (NOT the recursive converter) so no
        block markup like '#' or '*' can leak into a cell.
        """
        text = cell.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("|", "\\|")
        return text

    def table_to_markdown(self, table) -> str:
        """Convert a <table> into a rectangular Markdown table.

        Rows are padded to the widest row so the grid never goes ragged when a
        cell is merged (colspan) or missing. The divider row is emitted exactly
        once, right after the first row.
        """
        raw_rows = []
        for tr in table.find_all("tr"):
            cells = [self.clean_cell(c) for c in tr.find_all(["td", "th"])]
            if cells:
                raw_rows.append(cells)

        if not raw_rows:
            return ""

        width = max(len(r) for r in raw_rows)
        lines = []
        for i, cells in enumerate(raw_rows):
            cells = cells + [""] * (width - len(cells))  # pad short rows
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * width) + " |")

        return "\n\n" + "\n".join(lines) + "\n\n"

    def html_to_markdown(self, element, url):
        if isinstance(element, NavigableString):
            return element.strip()

        tag = element.name

        if tag == "a" and element.get("href"):
            href = urljoin(url, element["href"])
            return f" [{element.get_text(strip=True)}]({href}) "

        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(tag[1])
            return f"\n\n{'#' * level} {element.get_text(strip=True)}\n"

        if tag == "li":
            return f"\n* {element.get_text(strip=True)}"

        if tag == "table":
            return self.table_to_markdown(element)

        if tag == "p":
            return f"\n\n{element.get_text(strip=True)}\n"

        return "".join(
            self.html_to_markdown(child, url)
            for child in element.children
            if child.name not in ["script", "style"]
        )

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def scrape_urls(self) -> List[Data]:
        urls = self.urls
        url_list = (
            [u.strip() for u in urls.split("\n") if u.strip()]
            if isinstance(urls, str)
            else urls
        )
        results = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        for url in url_list:
            try:
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Decode Cloudflare-obfuscated emails
                for cf_tag in soup.find_all(class_="__cf_email__"):
                    encoded = cf_tag.get("data-cfemail")
                    if encoded:
                        cf_tag.replace_with(self.decode_cf_email(encoded))

                # Strip navigation / chrome / noise
                for selector in [".o-breadcrumb", "nav", "footer", "header", "script", "style"]:
                    for tag in soup.select(selector):
                        tag.decompose()

                page_title = soup.title.string if soup.title else url
                main_body = soup.find("main") or soup.find("article") or soup.find("body")

                md = f"# PAGE: {page_title}\nURL: {url}\n\n"
                md += self.html_to_markdown(main_body, url) if main_body else "No content found."

                results.append(Data(data={"text": re.sub(r"\n{3,}", "\n\n", md), "source": url}))
            except Exception as e:
                results.append(Data(data={"text": f"Error: {str(e)}", "source": url}))

        return results

    def merge_scraped_data(self) -> Data:
        """Combine all scraped pages into one Data object for file writing."""
        scraped_list = self.scrape_urls()
        combined_text = "\n\n---\n\n".join(item.text for item in scraped_list)
        return Data(data={"text": combined_text})