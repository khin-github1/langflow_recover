# Recovered Langflow component
# type: MultiWebScraper
# class: MultiWebScraper
# used in 1 flow(s): URL Test
# json path: node.data.node.template.code.value

import requests
from bs4 import BeautifulSoup
from langflow.custom import CustomComponent
from langflow.schema import Data
from typing import List, Union, Any 

class MultiWebScraper(CustomComponent):
    display_name = "Multi-URL Scraper"
    description = "Scrapes multiple URLs and returns them as individual chunks."

    def build_config(self):
        return {
            "urls": {
                "display_name": "URLs",
                "info": "Enter URLs (one per line or separated by commas).",
                "input_types": ["Text", "Message", "Data"],
                "field_type": "textarea",
                "value": "https://ait.ac.th/\nhttps://ait.ac.th/admissions/"
            },
        }

    def build(self, urls: Any, **kwargs) -> List[Data]:
        url_list = []
        
        # Split input by commas or new lines to get individual URLs
        if isinstance(urls, str):
            raw_input = urls.replace(',', '\n')
            url_list = [u.strip() for u in raw_input.split('\n') if u.strip()]
        elif isinstance(urls, list):
            url_list = [str(u) for u in urls]
        elif hasattr(urls, "text"):
            raw_input = urls.text.replace(',', '\n')
            url_list = [u.strip() for u in raw_input.split('\n') if u.strip()]
        
        results = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        # Scrape each URL and store results in a list
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
                # Append each successful result to the list
                results.append(Data(data={"text": clean_text, "source": url}))
            except Exception as e:
                # Append error message to list to keep the flow moving
                results.append(Data(data={"text": f"Error scraping {url}: {str(e)}", "source": url}))

        # Return the final list of all scraped Data objects
        return results