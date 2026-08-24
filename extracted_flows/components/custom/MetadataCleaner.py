# Recovered Langflow component
# type: MetadataCleaner
# class: MetadataCleaner
# used in 1 flow(s): with_doc
# json path: node.data.node.template.code.value

from langflow.custom import CustomComponent
from langflow.schema import Data
from typing import Any, List, Union
import pandas as pd

class MetadataCleaner(CustomComponent):
    display_name = "Metadata Cleaner"
    description = "Cleans non-serializable properties from a DataFrame/Record."

    def build_config(self):
        return {
            "input_data": {
                "display_name": "Input Data",
                "info": "Connect the 'Chunks' output (DataFrame) here.",
                "input_types": ["DataFrame", "Message", "Data"] 
            },
        }

    # Added **kwargs to catch the 'code' argument and avoid the TypeError
    def build(self, input_data: Any, **kwargs) -> List[Data]:
        results = []
        
        if input_data is None:
            return []

        # Handle Pandas DataFrame (The Pink Dot data)
        if isinstance(input_data, pd.DataFrame):
            records = input_data.to_dict('records')
        # Handle List of Data/Messages
        elif isinstance(input_data, list):
            records = [item.data if hasattr(item, 'data') else item for item in input_data]
        # Handle Single object
        else:
            records = [input_data.data if hasattr(input_data, 'data') else input_data]

        for row in records:
            cleaned_row = {}
            for key, value in row.items():
                # JSON-safe types for PostgreSQL
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    cleaned_row[key] = value
                else:
                    # Fixes the (builtins.TypeError) by stringifying complex objects
                    cleaned_row[key] = str(value)
            
            results.append(Data(data=cleaned_row))
        
        return results