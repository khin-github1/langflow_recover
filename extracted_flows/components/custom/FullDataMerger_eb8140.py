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
    description = "Combines a List of Data/Strings into a single clean String output."
    icon = "combine"

    inputs = [
        DataInput(
            name="input_data",
            display_name="Input Data (List)",
            info="Connect your Scraper output here.",
            is_list=True,
        ),
        MessageTextInput(
            name="separator",
            display_name="Page Separator",
            value="\n\n---\n\n",
        ),
    ]

    outputs = [
        # Changing this to method="merge_all" and ensuring return type is str
        Output(display_name="Combined Text", name="combined_text", method="merge_all"),
    ]

    def merge_all(self) -> str:
        if not self.input_data:
            self.status = "No data received"
            return ""

        all_texts = []
        
        # Normalize the input into a list
        items = self.input_data if isinstance(self.input_data, list) else [self.input_data]
        
        for item in items:
            if isinstance(item, Data):
                val = item.data.get("text", "")
                if isinstance(val, list):
                    all_texts.append("\n".join(map(str, val)))
                else:
                    all_texts.append(str(val))
            elif isinstance(item, list):
                all_texts.append("\n".join(map(str, item)))
            else:
                all_texts.append(str(item))

        # Join into one single string
        final_output = self.separator.join(all_texts)
        
        self.status = f"Merged {len(all_texts)} items into a single string."
        return final_output