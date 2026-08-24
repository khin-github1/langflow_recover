# Recovered Langflow component
# type: DataFrameToList
# class: DataFrameToList
# used in 1 flow(s): Summarization_flow_chunk
# json path: node.data.node.template.code.value

from langflow.custom import Component
from langflow.io import HandleInput, Output
from langflow.schema import Data, DataFrame

class DataFrameToList(Component):
    display_name = "AIT Chunk Formatter"
    description = "Converts Split Text DataFrame into a list of {text, source} objects."
    icon = "list"

    inputs = [
        HandleInput(
            name="input_dataframe",
            display_name="Input DataFrame",
            info="Connect the 'Chunks' output from Split Text here.",
            input_types=["DataFrame"],
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Formatted List", name="output_list", method="format_to_list"),
    ]

    def format_to_list(self) -> list[Data]:
        # 1. Ensure we have a valid DataFrame
        df_obj = self.input_dataframe
        if not df_obj or not hasattr(df_obj, "data"):
            return []

        # 2. Extract rows and format into {text, source}
        formatted_results = []
        
        # Langflow DataFrames store a list of Data objects in their .data attribute
        for item in df_obj.data:
            # Extract text from the main text field
            text_content = item.text if hasattr(item, "text") else ""
            
            # Extract source from metadata (usually passed from the scraper)
            source_url = item.data.get("source", "Unknown Source") if hasattr(item, "data") else "Unknown Source"
            
            # Create a new Data object in the requested {text, source} format
            formatted_results.append(Data(data={
                "text": text_content,
                "source": source_url
            }))
            
        return formatted_results