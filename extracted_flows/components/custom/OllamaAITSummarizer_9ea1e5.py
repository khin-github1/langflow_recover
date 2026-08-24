# Recovered Langflow component
# type: OllamaAITSummarizer
# class: OllamaAITSummarizer
# used in 2 flow(s): Summarization_flow_chunk, Summarization_flow_simple
# json path: node.data.node.template.code.value

import httpx
import json
import re
from langflow.custom import Component
from langflow.io import MessageTextInput, DropdownInput, FloatInput, Output
from langflow.schema import Data

class OllamaAITSummarizer(Component):
    display_name = "V5 Ollama AIT Robust Summarizer"
    description = "Outputs a dictionary containing a list for perfect CSV compatibility."
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
        MessageTextInput(
            name="system_message",
            display_name="System Message",
            value="You are an expert academic advisor at the Asian Institute of Technology (AIT). Respond only in valid JSON.",
        ),
        MessageTextInput(
            name="input_text",
            display_name="Scraped Text",
        ),
        MessageTextInput(
            name="source_url",
            display_name="Source URL",
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            advanced=True
        ),
    ]

    outputs = [
        Output(display_name="CSV Data Row", name="summary_output", method="summarize_text"),
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

    def clean_text(self, text: str) -> str:
        if not text: return "N/A"
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    async def summarize_text(self) -> Data:
        # Define prompt and payload INSIDE the method to avoid 'not defined' errors
        user_prompt = (
            f"Analyze the content from {self.source_url}.\n\n"
            f"CONTENT:\n{self.input_text}\n\n"
            "TASK: Return a JSON object with exactly two keys:\n"
            "1. 'extractive': List key dates, requirements, and contact info.\n"
            "2. 'abstractive': Provide a 2-sentence overview.\n"
            "Format: {\"extractive\": \"...\", \"abstractive\": \"...\"}"
        )

        payload = {
            "model": self.model_name,
            "prompt": user_prompt,
            "system": self.system_message,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature}
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(f"{self.base_url.rstrip('/')}/api/generate", json=payload)
                response.raise_for_status()
                
                raw_response_text = response.json().get("response", "{}")
                data_dict = json.loads(raw_response_text)
                
                row_map = {
                    "url": self.source_url,
                    "Summary with extractive": self.clean_text(data_dict.get("extractive")),
                    "summary with abstractive": self.clean_text(data_dict.get("abstractive"))
                }
                
                # RETURN: A dictionary (satisfies Pydantic) 
                # containing a list (satisfies Write File/Pandas)
                return Data(data={"csv_records": [row_map]})
                
        except Exception as e:
            error_row = {
                "url": self.source_url,
                "Summary with extractive": "Error",
                "summary with abstractive": self.clean_text(f"Failed: {str(e)}")
            }
            return Data(data={"csv_records": [error_row]})