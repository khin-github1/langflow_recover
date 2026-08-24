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
    display_name = "Multi-URL Scraper_old"
    description = "Scrapes multiple URLs and returns them as individual chunks."
    icon = "globe" # Optional icon

    # Modern input definition
    inputs = [
        MultilineInput(
            name="urls",
            display_name="URLs",
            info="Enter URLs (one per line or separated by commas).",
            value="https://ait.ac.th/\nhttps://ait.ac.th/admissions/",
        ),
    ]

    # Modern output definition
    outputs = [
        Output(display_name="Scraped Data", name="scraped_data", method="scrape_urls"),
    ]

    def scrape_urls(self) -> List[Data]:
        # Access the input directly using the modern attribute pattern
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
                
                # Clean navigation and footer junk
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()

                clean_text = soup.get_text(separator=' ', strip=True)
                results.append(Data(data={"text": clean_text, "source": url}))
            except Exception as e:
                results.append(Data(data={"text": f"Error scraping {url}: {str(e)}", "source": url}))

        self.status = results # For visual feedback in the UI
        return results