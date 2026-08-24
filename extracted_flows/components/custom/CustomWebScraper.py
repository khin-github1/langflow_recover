# Recovered Langflow component
# type: CustomWebScraper
# class: CustomWebScraper
# used in 2 flow(s): URL Test, URL Test (Backup)
# json path: node.data.node.template.code.value

import requests
from bs4 import BeautifulSoup
from langflow.custom import CustomComponent
from langflow.schema import Data

class CustomWebScraper(CustomComponent):
    display_name = "Advanced Web Scraper"
    description = "Scrapes a URL using a custom User-Agent to avoid blocks."

    def build_config(self):
        return {
            "url": {"display_name": "URL to Scrape", "value": "https://ait.ac.th/"},
        }

    # Added **kwargs to catch the 'code' argument passed by Langflow
    def build(self, url: str, **kwargs) -> Data:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Using lxml if available, otherwise html.parser
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements from the text
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()

            # Extract clean text
            clean_text = soup.get_text(separator=' ', strip=True)
            
            return Data(data={"text": clean_text, "source": url})
        except Exception as e:
            # Return the error as text so the flow doesn't break entirely
            return Data(data={"text": f"Error scraping {url}: {str(e)}", "source": url})