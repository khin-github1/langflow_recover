# Recovered Langflow component
# type: MultiWebScraper
# class: MultiWebScraper
# used in 6 flow(s): 280pages store in pgvector, Baseline- Naive, Evaluation_for_all, Method 1-Markdown Serialization, Table-aware-scraper, URL Test
# json path: node.data.node.template.code.value

import requests
import re
from bs4 import BeautifulSoup
from langflow.custom import Component
from langflow.io import MultilineInput, Output
from langflow.schema import Data
from typing import List, Any
from urllib.parse import urljoin

class MultiWebScraper(Component):
    display_name = "Multi-URL Scraper_v3_Final"
    description = "Advanced cleansing: Decodes emails, removes specific AIT breadcrumbs, and captures contextual links."
    icon = "globe"

    inputs = [
        MultilineInput(
            name="urls",
            display_name="URLs",
            info="Enter URLs (one per line).",
            value="https://ait.ac.th/programs/flexible-masters/",
        ),
    ]

    outputs = [
        Output(display_name="Scraped Data", name="scraped_data", method="scrape_urls"),
    ]

    def decode_cf_email(self, hex_string: str) -> str:
        """Decodes Cloudflare's XOR-encoded email strings."""
        try:
            key = int(hex_string[:2], 16)
            email = "".join(
                [chr(int(hex_string[i:i+2], 16) ^ key) 
                 for i in range(2, len(hex_string), 2)]
            )
            return email
        except Exception:
            return "[Decoding Failed]"

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

                # 1. FIX EMAIL PROTECTION: Decode [email protected]
                for cf_tag in soup.find_all(class_="__cf_email__"):
                    encoded = cf_tag.get("data-cfemail")
                    if encoded:
                        decoded_email = self.decode_cf_email(encoded)
                        cf_tag.replace_with(decoded_email)

                # 2. CAPTURE IMPORTANT LINKS (Buttons/Brochures)
                important_links = []
                for a in soup.find_all('a', href=True):
                    link_text = a.get_text(strip=True)
                    if any(x in link_text for x in ["Apply Now", "brochure", "Click this link"]):
                        full_url = urljoin(url, a['href'])
                        important_links.append(f"{link_text}: {full_url}")

                # 3. TARGETED LINK EXTRACTION ("Clicking here" patterns)
                internal_links = []
                for link in soup.find_all('a', string=re.compile(r'click(ing)? here', re.I)):
                    href = link.get('href')
                    if href:
                        full_url = urljoin(url, href)
                        link_context = link.find_parent().get_text(strip=True) 
                        internal_links.append(f"Context: {link_context} -> URL: {full_url}")

                # 4. AGGRESSIVE CLEANING (Breadcrumbs & UI Noise)
                # Specific AIT classes identified from your inspection photos
                noise_selectors = [
                    ".cli-bar-message", ".cli-privacy-content-text", ".cli-bar-container",
                    ".fusion-breadcrumb", ".fusion-breadcrumbs", ".breadcrumb", 
                    ".fusion-page-title-row", ".fusion-page-title-bar", ".fusion-page-title-captions"
                ]
                for selector in noise_selectors:
                    for tag in soup.select(selector):
                        tag.decompose()

                # Pattern-based removal for any remaining "Home >" text strings
                breadcrumb_pattern = re.compile(r'Home\s*>\s*', re.I)
                for bc_tag in soup.find_all(string=breadcrumb_pattern):
                    if bc_tag.parent:
                        bc_tag.parent.decompose()

                # Remove structural junk tags
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                    tag.decompose()

                # 5. EXTRACT TEXT BODY (Prioritize main knowledge area)
                # This explicitly avoids the page title area
                content_body = soup.find('main') or soup.find('article') or soup.find(id='content') or soup.find('body')
                clean_text = content_body.get_text(separator=' ', strip=True)

                # 6. APPEND COLLECTED LINKS
                if important_links:
                    clean_text += "\n\nImportant Links:\n" + "\n".join(important_links)
                
                if internal_links:
                    clean_text += "\n\nAdditional Resources Found:\n" + "\n".join(internal_links)

                results.append(Data(data={"text": clean_text, "source": url}))

            except Exception as e:
                results.append(Data(data={"text": f"Error: {str(e)}", "source": url}))

        self.status = results
        return results