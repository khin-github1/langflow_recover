# Recovered Langflow component
# type: MultiWebScraper
# class: MultiWebScraper
# used in 1 flow(s): URL Test
# json path: node.data.node.template.code.value

import requests
from bs4 import BeautifulSoup
from langflow.custom import Component
from langflow.io import MultilineInput, Output
from langflow.schema import Data
from typing import List, Any

class MultiWebScraper(Component):
    display_name = "Multi-URL Scraper_New"
    description = "Scrapes multiple URLs with aggressive cleaning for AITGPT knowledge base."
    icon = "globe"

    inputs = [
        MultilineInput(
            name="urls",
            display_name="URLs",
            info="Enter URLs (one per line or separated by commas).",
            value="https://ait.ac.th/\nhttps://ait.ac.th/admissions/",
        ),
    ]

    outputs = [
        Output(display_name="Scraped Data", name="scraped_data", method="scrape_urls"),
    ]

    def decode_cloudflare_email(self, encoded_string: str) -> str:
        """Decodes Cloudflare [email protected] hex strings."""
        try:
            # First 2 chars are the XOR key
            key = int(encoded_string[:2], 16)
            email = ""
            # Process remaining hex pairs
            for i in range(2, len(encoded_string), 2):
                char_code = int(encoded_string[i:i+2], 16) ^ key
                email += chr(char_code)
            return email
        except Exception:
            return "[Email Decoding Failed]"

    def scrape_urls(self) -> List[Data]:
        urls = self.urls
        url_list = []
        
        if isinstance(urls, str):
            raw_input = urls.replace(',', '\n')
            url_list = [u.strip() for u in raw_input.split('\n') if u.strip()]
        elif isinstance(urls, list):
            url_list = [str(u) for u in urls]
        
        results = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        for url in url_list:
            if not url.startswith('http'):
                continue
            try:
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 1. FIX [email protected]: Decode Cloudflare emails
                for encrypted_email in soup.find_all("a", class_="__cf_email__"):
                    encoded_data = encrypted_email.get("data-cfemail")
                    if encoded_data:
                        real_email = self.decode_cloudflare_email(encoded_data)
                        encrypted_email.replace_with(real_email)

                # 2. REMOVE SPECIFIC NOISE: Cookie banners and privacy text
                # Targets the classes you identified: .cli-bar-message and .cli-privacy-content-text
                noise_selectors = [
                    ".cli-bar-message", 
                    ".cli-privacy-content-text", 
                    ".cli-bar-container",
                    "#cookie-law-info-bar",
                    ".breadcrumb" # Removes the "Home > About" noise
                ]
                for selector in noise_selectors:
                    for element in soup.select(selector):
                        element.decompose()

                # 3. STRIP STRUCTURAL JUNK: Tags that don't contain knowledge content
                for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                    element.decompose()

                # 4. TARGET MAIN CONTENT (Optional but Recommended)
                # This focuses the scraper on the actual article/body text
                main_content = soup.find('main') or soup.find('article') or soup.find(id='content')
                
                if main_content:
                    clean_text = main_content.get_text(separator=' ', strip=True)
                else:
                    # Fallback to body if specific main tags aren't present
                    clean_text = soup.get_text(separator=' ', strip=True)

                results.append(Data(data={"text": clean_text, "source": url}))
            except Exception as e:
                results.append(Data(data={"text": f"Error scraping {url}: {str(e)}", "source": url}))

        self.status = results
        return results