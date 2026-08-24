# Recovered Langflow component
# type: MultiWebScraper
# class: MultiWebScraper
# used in 1 flow(s): URL Test
# json path: node.data.node.template.code.value

import requests
from bs4 import BeautifulSoup
from langflow.custom import Component
from langflow.schema import Data
from langflow.io import Output, MultilineInput
from typing import List, Any 

class MultiWebScraper(Component):
    display_name = "Multi-URL Scraper"
    description = "Scrapes multiple URLs and returns them as individual chunks."
    icon = "globe"

    # --- INPUTS DECLARATION (This creates the Pink/Black Dot) ---
    inputs = [
        MultilineInput(
            name="urls",
            display_name="URLs",
            info="Enter URLs (one per line or separated by commas).",
            value="https://ait.ac.th/",
            tool_mode=True, # Makes it compatible with Agent tools
        ),
    ]

    # --- OUTPUTS DECLARATION (This creates the Red Dot) ---
    outputs = [
        Output(
            name="scraped_data", 
            display_name="Scraped Data", 
            method="build", 
            output_type=Data
        ),
    ]

    def build(self, urls: str, **kwargs) -> List[Data]:
        # Handle empty input
        if not urls:
            return []

        # Convert input to a clean list of URLs
        raw_input = urls.replace(',', '\n')
        url_list = [u.strip() for u in raw_input.split('\n') if u.strip()]
        
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
                
                # Cleanup navigation and junk
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()

                clean_text = soup.get_text(separator=' ', strip=True)
                results.append(Data(data={"text": clean_text, "source": url}))
            except Exception as e:
                results.append(Data(data={"text": f"Error scraping {url}: {str(e)}", "source": url}))

        return results