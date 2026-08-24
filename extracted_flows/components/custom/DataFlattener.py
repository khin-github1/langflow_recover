# Recovered Langflow component
# type: DataFlattener
# class: DataFlattener
# used in 2 flow(s): Summarization_flow_chunk, Summarization_flow_simple
# json path: node.data.node.template.code.value

from langflow.custom import Component
from langflow.io import HandleInput, Output
from langflow.schema import Data

class DataFlattener(Component):
    display_name = "CSV Column Flattener"
    description = "Flattens nested 'table' data into top-level columns for CSV writing."
    icon = "table-properties"

    inputs = [
        HandleInput(
            name="input_data",
            display_name="Input Data",
            info="Connect the 'table' output from your Summarizer here.",
            input_types=["Data"],
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Flattened Data", name="output_data", method="flatten_data"),
    ]

    def flatten_data(self) -> Data:
        # 1. Get the raw dictionary
        raw_data = self.input_data.data if hasattr(self.input_data, "data") else {}
        
        # 2. Extract the list from the 'table' key
        # If 'table' exists, use the first item in that list to create columns
        records = raw_data.get("table", [])
        
        if isinstance(records, list) and len(records) > 0:
            # We return the first record as a dictionary. 
            # The Write File component will see these keys as CSV headers.
            return Data(data=records[0])
        
        # Fallback if the format is unexpected
        return self.input_data