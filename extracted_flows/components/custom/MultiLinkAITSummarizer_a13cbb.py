# Recovered Langflow component
# type: MultiLinkAITSummarizer
# class: MultiLinkAITSummarizer
# used in 1 flow(s): pageindex_json_flow
# json path: node.data.node.template.code.value

import httpx
import json
import re
from langflow.custom import Component
from langflow.io import MessageTextInput, DropdownInput, FloatInput, Output, HandleInput
from langflow.schema import Data
from typing import List, Union, Any

class MultiLinkAITSummarizer(Component):
    display_name = "Conditional AIT Table Summarizer"
    description = "Summarizes HTML tables via LLM; strips tags from plain text rows. Supports DataFrame inputs."
    icon = "table"

    inputs = [
        MessageTextInput(
            name="base_url",
            display_name="Ollama API URL",
            value="https://ollama.aitgpt.dev.brain.cs.ait.ac.th",
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=[],
            refresh_button=True
        ),
        HandleInput(
            name="input_data",
            display_name="Input Data (DataFrame/List)",
            info="Connect your CSV or Scraper output here. Looks for 'html_content' column.",
            input_types=["Data", "DataFrame", "dict", "Message"],
            is_list=True 
        ),
        MessageTextInput(
            name="system_message",
            display_name="System Message",
            value="You are an expert advisor at AIT. Respond ONLY with a valid JSON object.",
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
        ),
    ]

    outputs = [
        Output(display_name="Processed Data", name="summary_list", method="process_summaries"),
    ]

    async def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None):
        if field_name == "model_name" or not build_config["model_name"]["options"]:
            try:
                base_url = build_config["base_url"]["value"].rstrip("/")
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{base_url}/api/tags")
                    if response.status_code == 200:
                        models = [m["name"] for m in response.json().get("models", [])]
                        build_config["model_name"]["options"] = models
            except Exception:
                build_config["model_name"]["options"] = ["qwen2.5:7b", "llama3"]
        return build_config

    def strip_html_tags(self, html_text: str) -> str:
        """Removes HTML tags and cleans up whitespace for non-table rows."""
        if not html_text:
            return ""
        clean_text = re.sub(r'<[^>]*>', ' ', str(html_text))
        return " ".join(clean_text.split())

    def contains_table(self, html_text: str) -> bool:
        """Detects if the HTML string contains table-related tags."""
        if not html_text:
            return False
        text_lower = str(html_text).lower()
        return any(tag in text_lower for tag in ["<table", "<tr>", "<td>"])

    async def process_summaries(self) -> List[Data]:
        if not self.input_data:
            return []
            
        # Unpack input data correctly
        input_rows = []
        if isinstance(self.input_data, list):
            input_rows = self.input_data
        elif hasattr(self.input_data, "value") and isinstance(self.input_data.value, list):
            input_rows = self.input_data.value
        else:
            input_rows = [self.input_data]

        results = []

        async with httpx.AsyncClient(timeout=180.0) as client:
            for row in input_rows:
                # Ensure row is a Data object
                if not isinstance(row, Data):
                    row = Data(data=row) if isinstance(row, dict) else Data(data={"text": str(row)})

                html_val = row.data.get("html_content", "")
                current_url = row.data.get("source", "Unknown")
                
                # Check for table
                if self.contains_table(html_val):
                    # --- FIXED: Use html_val and correct indentation ---
                    user_prompt = (
                        f"Analyze the following HTML table from {current_url}.\n\n"
                        f"CONTENT:\n{str(html_val)[:8000]}\n\n"
                        "TASK: Return a JSON object with this key:\n"
                        "1. 'summary': A 2-sentence summary of the key fees or data.\n"
                        "Format: {\"summary\": \"...\"}"
                    )

                    payload = {
                        "model": self.model_name,
                        "prompt": user_prompt,
                        "system": self.system_message,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": self.temperature, "num_ctx": 8192}
                    }

                    try:
                        response = await client.post(f"{self.base_url.rstrip('/')}/api/generate", json=payload)
                        response.raise_for_status()
                        resp_data = response.json().get("response", "{}")
                        data_dict = json.loads(resp_data)
                        
                        # Fallback logic for key names
                        final_text = data_dict.get("summary") or data_dict.get("abstractive") or "Summary key not found in JSON."
                    except Exception as e:
                        final_text = f"Error summarizing table: {str(e)}"
                
                else:
                    # Strip tags for non-table rows
                    final_text = self.strip_html_tags(html_val)

                # Preserving all original metadata
                output_payload = row.data.copy()
                output_payload["text"] = final_text
                results.append(Data(data=output_payload))
        
        return results