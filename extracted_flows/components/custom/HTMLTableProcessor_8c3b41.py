# Recovered Langflow component
# type: HTMLTableProcessor
# class: HTMLTableProcessorComponent
# used in 1 flow(s): pageindex_json_flow
# json path: node.data.node.template.code.value

"""
Langflow Custom Component: HTML Table Processor with Ollama Summarization
Mirrors MultiLinkAITSummarizer structure exactly so Langflow renders the
text|source table in the Output panel.
"""

import re
import json
import asyncio
from typing import List

import httpx
import pandas as pd

from langflow.custom import Component
from langflow.io import (
    DataFrameInput,
    DropdownInput,
    MessageTextInput,
    FloatInput,
    Output,
)
from langflow.schema import Data


class HTMLTableProcessorComponent(Component):
    display_name = "HTML Table Processor (Ollama)"
    description = "Summarizes tables via Ollama and strips plain HTML. Output mirrors MultiLinkAITSummarizer."
    icon = "Ollama"
    name = "HTMLTableProcessor"

    inputs = [
        DataFrameInput(
            name="input_data",
            display_name="Read File Output (DataFrame)",
            info="Connect the DataFrame output from the Read File (Docling) component here.",
        ),
        MessageTextInput(
            name="base_url",
            display_name="Ollama API URL",
            value="https://ollama.aitgpt.dev.brain.cs.ait.ac.th",
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=[],
            refresh_button=True,
        ),
        MessageTextInput(
            name="system_message",
            display_name="System Message",
            value="You are an expert data analyst. Respond ONLY with a valid JSON object.",
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
        ),
    ]

    outputs = [
        Output(
            display_name="Summary List",
            name="summary_list",
            method="process_summaries",
        ),
    ]

    async def update_build_config(
        self, build_config: dict, field_value: str, field_name: str | None = None
    ) -> dict:
        if field_name == "model_name" or not build_config["model_name"]["options"]:
            try:
                base_url = build_config["base_url"]["value"].rstrip("/")
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(f"{base_url}/api/tags")
                    if response.status_code == 200:
                        models = [m["name"] for m in response.json().get("models", [])]
                        build_config["model_name"]["options"] = models
            except Exception:
                build_config["model_name"]["options"] = ["qwen2.5:7b", "llama3"]
        return build_config

    @staticmethod
    def _contains_table(text: str) -> bool:
        html_table = bool(re.search(r"<table|<tr|<td", text, re.IGNORECASE))
        md_table = "|---|" in text or "|--" in text
        return html_table or md_table

    @staticmethod
    def _strip_html(text: str) -> str:
        stripped = re.sub(r"<[^>]*>", " ", text)
        stripped = re.sub(r"[ \t]+", " ", stripped)
        stripped = re.sub(r"\n{2,}", "\n", stripped)
        return stripped.strip()

    async def process_summaries(self) -> List[Data]:
        # Guard
        if not self.input_data or (isinstance(self.input_data, pd.DataFrame) and self.input_data.empty):
            return []

        # DataFrame → list of dicts (never use row.data on a DataFrame)
        input_rows = self.input_data.to_dict(orient="records")
        results = []

        async with httpx.AsyncClient(timeout=180.0) as client:
            for row in input_rows:
                current_text = row.get("html_content", "")
                current_url  = row.get("source", row.get("file_path", row.get("filename", "unknown")))

                if not current_text:
                    continue

                # --- TABLE branch: summarize via Ollama ---
                if self._contains_table(current_text):
                    user_prompt = (
                        f"Analyze the table content from {current_url}.\n\n"
                        f"CONTENT:\n{current_text[:6000]}\n\n"
                        "TASK: Return a JSON object with exactly these keys:\n"
                        "1. 'extractive': Key data points, numbers, and labels from the table.\n"
                        "2. 'abstractive': A 2-sentence summary of what the table shows.\n"
                        'Format: {"extractive": "...", "abstractive": "..."}'
                    )
                    payload = {
                        "model": self.model_name,
                        "prompt": user_prompt,
                        "system": self.system_message,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": self.temperature, "num_ctx": 8192},
                    }
                    try:
                        response = await client.post(
                            f"{self.base_url.rstrip('/')}/api/generate", json=payload
                        )
                        response.raise_for_status()
                        raw_response = response.json().get("response", "{}")
                        data_dict = json.loads(raw_response)
                        text_out = data_dict.get("extractive", "") or data_dict.get("abstractive", "")
                    except json.JSONDecodeError:
                        text_out = self._strip_html(current_text)
                    except Exception as e:
                        text_out = f"Error summarizing table: {str(e)}"

                # --- PLAIN branch: strip HTML ---
                else:
                    text_out = self._strip_html(current_text)

                if not text_out:
                    continue

                results.append(Data(data={
                    "text": text_out,
                    "source": current_url,
                }))

        return results