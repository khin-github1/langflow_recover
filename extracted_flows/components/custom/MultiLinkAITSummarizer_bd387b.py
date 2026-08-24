# Recovered Langflow component
# type: MultiLinkAITSummarizer
# class: MultiLinkAITSummarizer
# used in 5 flow(s): Extractive_embed, Method2- Summary, multi_webpage_chunk after summary, pageindex_json_flow, summary with conditional
# json path: node.data.node.template.code.value

import httpx
import json
from langflow.custom import Component
from langflow.io import MessageTextInput, DropdownInput, FloatInput, Output, HandleInput
from langflow.schema import Data
from typing import List

class MultiLinkAITSummarizer(Component):
    display_name = "Multi-Link AIT Summarizer Extractive"
    description = "Summarizes multiple pages and outputs a list of Data objects."
    icon = "Ollama"

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
            display_name="Scraped Data (List)",
            info="Connect the output from your Multi-URL Scraper here.",
            input_types=["Data"],
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
        Output(display_name="Summary List", name="summary_list", method="process_summaries"),
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

    async def process_summaries(self) -> List[Data]:
        # 1. Handle input data correctly
        if not self.input_data:
            return []
            
        input_rows = self.input_data if isinstance(self.input_data, list) else [self.input_data]
        results = []

        async with httpx.AsyncClient(timeout=180.0) as client:
            for row in input_rows:
                # Extract text from the scraper Data object
                current_text = row.data.get("text", "") if hasattr(row, "data") else ""
                current_url = row.data.get("source", "Unknown") if hasattr(row, "data") else "Unknown"

                if not current_text or "Error:" in current_text:
                    continue

                user_prompt = (
                    f"Analyze the content from {current_url}.\n\n"
                    f"CONTENT:\n{current_text[:8000]}\n\n" # Slight reduction for safety
                    "TASK: Return a JSON object with exactly these keys:\n"
                    "1. 'extractive': Key dates, requirements, and contact info.\n"
                    "2. 'abstractive': A 2-sentence summary.\n"
                    "Format: {\"extractive\": \"...\", \"abstractive\": \"...\"}"
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
                    
                    raw_response = response.json().get("response", "{}")
                    data_dict = json.loads(raw_response)
                    
                    # Create a new Data object for the output
                    results.append(Data(data={
                        "text":f"{data_dict.get('extractive', '')}",
                        "source": current_url,
                       # "extractive": data_dict.get("extractive", "N/A"),
                        
                    }))

                except Exception as e:
                    results.append(Data(data={
                        "text": f"Error processing {current_url}: {str(e)}",
                        "source": current_url
                    }))
        
        # This allows the "Split Text" component to see the results f"URL: {current_url}
        return results