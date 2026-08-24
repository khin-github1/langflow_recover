# Recovered Langflow component
# type: FullDataMerger
# class: FullDataMerger
# used in 1 flow(s): 280 Pages
# json path: node.data.node.template.code.value

from langflow.custom import Component
from langflow.io import Output, DataInput, MessageTextInput
from langflow.schema import Data
from typing import List, Any

class FullDataMerger(Component):
    display_name = "Full Data Merger"
    description = "Converts a List of Data/Strings into a single Message string."
    icon = "combine"

    inputs = [
        DataInput(
            name="input_data",
            display_name="Input Data (List)",
            info="Connect your Scraper or 'Type Convert' output here.",
            is_list=True,
        ),
        MessageTextInput(
            name="separator",
            display_name="Page Separator",
            value="\n\n---\n\n",
        ),
    ]

    outputs = [
        Output(display_name="Combined Text", name="combined_text", method="merge_all"),
    ]

    def merge_all(self) -> str:
        if not self.input_data:
            return "No data received"

        all_texts = []
        
        # We handle the input whether it's a list of Data objects or a list of strings
        for item in self.input_data:
            if isinstance(item, Data):
                # Try to get 'text' from the Data object dictionary
                val = item.data.get("text", "")
                if isinstance(val, list): # Handle the case where 'text' is a list
                    all_texts.append("\n".join(map(str, val)))
                else:
                    all_texts.append(str(val))
            elif isinstance(item, list):
                # If the item itself is a list (like in your error log)
                all_texts.append("\n".join(map(str, item)))
            else:
                all_texts.append(str(item))

        # Join everything into ONE string
        final_output = self.separator.join(all_texts)
        
        self.status = f"Merged {len(all_texts)} items successfully."
        return final_output