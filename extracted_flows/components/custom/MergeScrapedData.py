# Recovered Langflow component
# type: MergeScrapedData
# class: MergeScrapedData
# used in 1 flow(s): 280 Pages
# json path: node.data.node.template.code.value

from langflow.custom import Component
from langflow.io import MessageTextInput, Output, DataInput
from langflow.schema import Data
from typing import List, Union

class MergeScrapedData(Component):
    display_name = "Merge All 135 Pages"
    description = "Flattens all incoming data rows into one single Markdown document."
    icon = "merge"

    inputs = [
        DataInput(
            name="data_list",
            display_name="Data List (All Rows)",
            info="Connect the Scraper 'Scraped Data' output here.",
            is_list=True, # This tells Langflow to collect all 135 rows
        ),
        MessageTextInput(
            name="separator",
            display_name="Separator",
            info="The break between pages.",
            value="\n\n---\n\n",
        ),
    ]

    outputs = [
        Output(display_name="Full Combined Text", name="combined_text", method="combine"),
    ]

    def combine(self) -> str:
        # Check if we actually have data
        if not self.data_list:
            self.status = "No data received."
            return "No data found to merge."
        
        # Ensure we are dealing with a list of Data objects
        input_data = self.data_list if isinstance(self.data_list, list) else [self.data_list]
        
        extracted_texts = []
        for item in input_data:
            # If the item is a Data object, get the 'text' attribute
            if hasattr(item, "data") and "text" in item.data:
                text_content = item.data["text"]
            # If it's already a string or other format
            else:
                text_content = str(item)
            
            if text_content.strip():
                extracted_texts.append(text_content.strip())

        # Join everything into one giant string
        final_string = self.separator.join(extracted_texts)
        
        self.status = f"Successfully merged {len(extracted_texts)} pages."
        return final_string