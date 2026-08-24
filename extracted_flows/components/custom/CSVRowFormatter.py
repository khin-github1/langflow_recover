# Recovered Langflow component
# type: CSVRowFormatter
# class: CSVRowFormatter
# used in 2 flow(s): Summarization_flow_chunk, Summarization_flow_simple
# json path: node.data.node.template.code.value

from langflow.custom import Component
from langflow.io import HandleInput, Output
from langflow.schema import Data

class CSVRowFormatter(Component):
    display_name = "AIT CSV Formatter"
    description = "Formats AIT summary JSON into a structure compatible with Write File."
    icon = "table"

    inputs = [
        HandleInput(
            name="input_data",
            display_name="Input Data",
            info="Connect the output from your Summarizer here.",
            input_types=["Data"],
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Formatted Data", name="output_data", method="build_row"),
    ]

    def build_row(self) -> Data:
        # 1. Safely extract data from the input
        if hasattr(self.input_data, "data") and isinstance(self.input_data.data, dict):
            raw_dict = self.input_data.data
        else:
            # Fallback if the input is already a string or other format
            raw_dict = getattr(self.input_data, "data", {})

        # 2. Map the keys precisely
        csv_row = {
            "url": raw_dict.get("url", "N/A"),
            "Summary with extractive": raw_dict.get("extractive_summary") or raw_dict.get("Summary with extractive", "N/A"),
            "Summary with abstractive": raw_dict.get("abstractive_summary") or raw_dict.get("summary with abstractive", "N/A")
        }
        
        # 3. THE FIX: 
        # Return a Data object where the 'data' is a dict, 
        # but the specific key 'rows' (or any key) holds the LIST.
        # Most Langflow 'Write File' components look for a list of dicts.
        return Data(data={"records": [csv_row]})