# Recovered Langflow component
# type: ConditionalTableSummarizer
# class: ConditionalTableSummarizer
# used in 1 flow(s): pageindex_json_flow
# json path: node.data.node.template.code.value

import httpx
import json
import re
from langflow.custom import Component
from langflow.io import MessageTextInput, DropdownInput, FloatInput, Output, HandleInput
from langflow.schema import Data
from typing import List

class ConditionalTableSummarizer(Component):
    display_name = "Conditional Table Summarizer"
    description = "Summarizes rows containing HTML tables via LLM; strips HTML from non-table rows."
    icon = "Table"

    inputs = [
        MessageTextInput(
            name="base_url",
            display_name="Ollama API URL",
            value="https://ollama.aitgpt.dev.brain.cs.ait.ac.th",
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=["qwen2.5:7b", "llama3"], # Defaults if refresh fails
            refresh_button=True
        ),
        HandleInput(
            name="input_data",
            display_name="Input Data (DataFrame/List)",
            info="Connect the data containing 'html_content' column here.",
            input_types=["Data"],
            is_list=True 
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
        ),
    ]

    outputs = [
        Output(display_name="Processed Results", name="processed_results", method="process_data"),
    ]

    def strip_html(self, text: str) -> str:
        """Removes HTML tags and normalizes whitespace."""
        # Remove tags
        clean = re.sub(r'<[^>]*>', ' ', text)
        # Collapse multiple spaces/newlines into one
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def has_table(self, text: str) -> bool:
        """Checks for the presence of HTML table structures."""
        table_tags = [r"<table", r"<tr", r"<td"]
        return any(re.search(tag, text, re.IGNORECASE) for tag in table_tags)

    async def process_data(self) -> List[Data]:
        if not self.input_data:
            return []
            
        input_rows = self.input_data if isinstance(self.input_data, list) else [self.input_data]
        results = []

        async with httpx.AsyncClient(timeout=180.0) as client:
            for row in input_rows:
                # Extract the HTML content from the specific column
                html_content = row.data.get("html_content", "")
                current_source = row.data.get("source", "Unknown")
                
                # Check if this row actually contains a table
                if self.has_table(html_content):
                    # --- TABLE DETECTED: SUMMARIZE VIA LLM ---
                    user_prompt = (
                        "The following content contains an HTML table of fees or data. "
                        "Extract the key figures and provide a concise 2-sentence summary.\n\n"
                        f"CONTENT: {html_content}"
                    )

                    payload = {
                        "model": self.model_name,
                        "prompt": user_prompt,
                        "system": "You are a helpful assistant. Output a clear summary.",
                        "stream": False,
                        "options": {"temperature": self.temperature}
                    }

                    try:
                        response = await client.post(f"{self.base_url.rstrip('/')}/api/generate", json=payload)
                        response.raise_for_status()
                        result_text = response.json().get("response", "Could not summarize table.")
                    except Exception as e:
                        result_text = f"Error summarizing table: {str(e)}"
                
                else:
                    # --- NO TABLE: JUST STRIP TAGS ---
                    result_text = self.strip_html(html_content)

                # Create output Data object preserving original metadata
                new_data = row.data.copy()
                new_data["text"] = result_text # Replace text with the new summary or cleaned string
                results.append(Data(data=new_data))
        
        return results