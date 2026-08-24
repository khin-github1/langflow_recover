# Recovered Langflow component
# type: MultiLinkAITSummarizer
# class: MultiLinkAITSummarizer
# used in 5 flow(s): Extractive_embed, Method2- Summary, Summarization_flow_chunk, Summarization_flow_simple, abstractive_embed
# json path: node.data.node.template.code.value

import httpx
import json
from langflow.custom import Component
from langflow.io import MessageTextInput, DropdownInput, FloatInput, Output, HandleInput
from langflow.schema import Data

class MultiLinkAITSummarizer(Component):
    display_name = "Multi-Link AIT Summarizer"
    description = "Summarizes multiple scraped AIT pages and outputs them as a list for CSV export."
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
            is_list=True # Crucial for receiving multiple JSON objects
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
                build_config["model_name"]["options"] = ["qwen3:8b", "qwen2.5:7b"]
        return build_config

    async def process_summaries(self) -> list[Data]:
        # Ensure input is a list of Data objects
        input_rows = self.input_data if isinstance(self.input_data, list) else [self.input_data]
        results = []

        async with httpx.AsyncClient(timeout=180.0) as client:
            for row in input_rows:
                # Extract text and URL from each JSON object
                current_text = row.data.get("text", "") if hasattr(row, "data") else ""
                current_url = row.data.get("source", "Unknown") if hasattr(row, "data") else "Unknown"

                if not current_text:
                    continue

                user_prompt = (
                    f"Analyze the content from {current_url}.\n\n"
                    f"CONTENT:\n{current_text[:10000]}\n\n" # Limit text length for stability
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
                    "options": {
                            "temperature": self.temperature,
                            "num_predict": 3000,  # Allows the model to write much longer summaries
                            "top_p": 0.9,
                            "num_ctx": 8192
                                }
                }

                try:
                    response = await client.post(f"{self.base_url.rstrip('/')}/api/generate", json=payload)
                    response.raise_for_status()
                    data_dict = json.loads(response.json().get("response", "{}"))
                    
                    # Map result to your requested CSV column names
                    results.append(Data(data={
                        "url": current_url,
                        "Summary with extractive": data_dict.get("extractive", "N/A"),
                        "Summary with abstractive": data_dict.get("abstractive", "N/A")
                    }))
                except Exception as e:
                    results.append(Data(data={
                        "url": current_url,
                        "Summary with extractive": "Error",
                        "Summary with abstractive": f"Failed: {str(e)}"
                    }))
        
        return results