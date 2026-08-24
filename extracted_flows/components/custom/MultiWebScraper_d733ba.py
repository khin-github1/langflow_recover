# Recovered Langflow component
# type: MultiWebScraper
# class: MultiWebScraper
# used in 2 flow(s): URL Test, URL Test (Backup)
# json path: node.data.node.template.code.value

import requests
from bs4 import BeautifulSoup # Syntax fixed here
from langflow.custom import CustomComponent
from langflow.schema import Data
from typing import List, Union, Any 

class MultiWebScraper(CustomComponent):
    display_name = "Multi-URL Scraper"
    description = "Scrapes text from multiple URLs and returns a list of Data objects."

    def build_config(self):
        return {
            "urls": {
                "display_name": "URLs",
                "info": "Enter URLs (one per line).",
                "input_types": ["Text", "Message", "Data"],
                "field_type": "textarea",
                "value": "https://ait.ac.th/"
            },
        }

    def build(self, urls: Any, **kwargs) -> List[Data]:
        url_list = []
        
        if isinstance(urls, str):
            url_list = [u.strip() for u in urls.split('\n') if u.strip()]
        elif isinstance(urls, list):
            url_list = [str(u) for u in urls]
        elif hasattr(urls, "text"):
            url_list = [u.strip() for u in urls.text.split('\n') if u.strip()]
        
        results = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        for url in url_list:
            if not url.startswith('http'):
                continue
                
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()

                clean_text = soup.get_text(separator=' ', strip=True)
                results.append(Data(data={"text": clean_text, "source": url}))
            except Exception as e:
                results.append(Data(data={"text": f"Error scraping {url}: {str(e)}", "source": url}))

        return results