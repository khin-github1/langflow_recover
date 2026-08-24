# Recovered Langflow component
# type: MultiWebScraper
# class: MultiWebScraper
# used in 11 flow(s): 280 Pages, 280pages store in pgvector, Baseline- Naive, Evaluation_for_all, Method 1-Markdown Serialization, Table-aware-scraper, multi_webpage_chunk after summary, problem example ...
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
    display_name = "Markdown Web Scraper (Fixed)"
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
        # This one returns a list (causes the error with Write File)
        Output(display_name="Scraped Data List", name="scraped_data", method="scrape_urls"),
        # This one returns a SINGLE string (works with Write File)
        Output(display_name="Merged Markdown", name="merged_data", method="merge_scraped_data"),
    ]

    def decode_cf_email(self, hex_string: str) -> str:
        try:
            key = int(hex_string[:2], 16)
            return "".join([chr(int(hex_string[i:i+2], 16) ^ key) for i in range(2, len(hex_string), 2)])
        except Exception:
            return "[Email Protected]"

    def html_to_markdown(self, element, url):
        if isinstance(element, NavigableString):
            return element.strip()
        
        tag = element.name
        if tag == 'a' and element.get('href'):
            href = urljoin(url, element['href'])
            return f" [{element.get_text(strip=True)}]({href}) "
        if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = tag[1]
            return f"\n\n{'#' * int(level)} {element.get_text(strip=True)}\n"
        if tag == 'li':
            return f"\n* {element.get_text(strip=True)}"
        if tag == 'table':
            rows = []
            for tr in element.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if not cells: continue
                rows.append("| " + " | ".join(cells) + " |")
                if tr == element.find('tr'):
                    rows.append("| " + " | ".join(['---'] * len(cells)) + " |")
            return "\n\n" + "\n".join(rows) + "\n\n"
        if tag == 'p':
            return f"\n\n{element.get_text(strip=True)}\n"
        
        return "".join(self.html_to_markdown(child, url) for child in element.children if child.name not in ['script', 'style'])

    def scrape_urls(self) -> List[Data]:
        urls = self.urls
        url_list = [u.strip() for u in urls.split('\n') if u.strip()] if isinstance(urls, str) else urls
        results = []
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        for url in url_list:
            try:
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                # Clean Emails & Noise
                for cf_tag in soup.find_all(class_="__cf_email__"):
                    encoded = cf_tag.get("data-cfemail")
                    if encoded: cf_tag.replace_with(self.decode_cf_email(encoded))
                
                for selector in [".o-breadcrumb", "nav", "footer", "header", "script", "style"]:
                    for tag in soup.select(selector): tag.decompose()

                page_title = soup.title.string if soup.title else url
                main_body = soup.find('main') or soup.find('article') or soup.find('body')
                
                md = f"# PAGE: {page_title}\nURL: {url}\n\n"
                md += self.html_to_markdown(main_body, url) if main_body else "No content found."
                
                results.append(Data(data={"text": re.sub(r'\n{3,}', '\n\n', md), "source": url}))
            except Exception as e:
                results.append(Data(data={"text": f"Error: {str(e)}", "source": url}))
        return results

    def merge_scraped_data(self) -> Data:
        """Helper to combine all scraped pages into one Data object for file writing."""
        scraped_list = self.scrape_urls()
        combined_text = "\n\n---\n\n".join([item.text for item in scraped_list])
        return Data(data={"text": combined_text})