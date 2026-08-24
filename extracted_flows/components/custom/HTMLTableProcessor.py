# Recovered Langflow component
# type: HTMLTableProcessor
# class: HTMLTableProcessorComponent
# used in 1 flow(s): pageindex_json_flow
# json path: node.data.node.template.code.value

"""
Langflow Custom Component: HTML Table Processor with Ollama Summarization
=========================================================================
Processes HTML/Markdown content from a DataFrame or List[Data] input.
- Tables → summarized via Ollama API (2-sentence JSON summary)
- Plain HTML → stripped and normalized
"""

import re
import json
import asyncio
from typing import Any

import httpx
import pandas as pd

from langflow.custom import Component
from langflow.io import DataInput, DropdownInput, MessageTextInput, Output
from langflow.schema import Data


class HTMLTableProcessorComponent(Component):
    display_name = "HTML Table Processor"
    description = (
        "Strips HTML from plain content and summarizes tables using Ollama. "
        "Accepts a DataFrame or List[Data] with an `html_content` column/key."
    )
    icon = "table"
    name = "HTMLTableProcessor"

    # ------------------------------------------------------------------ #
    #  Build-time inputs                                                   #
    # ------------------------------------------------------------------ #
    inputs = [
        DataInput(
            name="input_data",
            display_name="Input Data",
            info="A DataFrame or List[Data] containing an `html_content` field.",
            is_list=True,
        ),
        MessageTextInput(
            name="ollama_base_url",
            display_name="Ollama Base URL",
            value="http://localhost:11434",
            info="Base URL of the running Ollama instance.",
        ),
        DropdownInput(
            name="ollama_model",
            display_name="Ollama Model",
            options=["llama3"],          # populated dynamically via update_build_config
            value="llama3",
            info="Model used for table summarization.",
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="ollama_timeout",
            display_name="Ollama Timeout (s)",
            value="60",
            info="Seconds before an Ollama request times out.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Processed Data",
            name="processed_data",
            method="process",
        ),
    ]

    # ------------------------------------------------------------------ #
    #  Dynamic model list                                                  #
    # ------------------------------------------------------------------ #
    async def update_build_config(
        self, build_config: dict, field_value: Any, field_name: str | None = None
    ) -> dict:
        """Fetch available Ollama model tags and populate the dropdown."""
        if field_name in (None, "ollama_base_url", "ollama_model"):
            base_url = (
                build_config.get("ollama_base_url", {}).get("value", "http://localhost:11434")
            ).rstrip("/")
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{base_url}/api/tags")
                    resp.raise_for_status()
                    payload = resp.json()
                    model_names: list[str] = [
                        m.get("name", "") for m in payload.get("models", []) if m.get("name")
                    ]
                    if model_names:
                        build_config["ollama_model"]["options"] = model_names
                        if build_config["ollama_model"]["value"] not in model_names:
                            build_config["ollama_model"]["value"] = model_names[0]
            except Exception as exc:  # noqa: BLE001
                # Non-fatal: keep default options if Ollama is unreachable at config time
                self.log(f"[update_build_config] Could not fetch Ollama models: {exc}")
        return build_config

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _contains_table(text: str) -> bool:
        """Return True if `text` appears to contain an HTML or Markdown table."""
        html_table = bool(re.search(r"<table|<tr|<td", text, re.IGNORECASE))
        md_table = "|---|" in text or "|--" in text
        return html_table or md_table

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        stripped = re.sub(r"<[^>]*>", " ", text)
        stripped = re.sub(r"[ \t]+", " ", stripped)          # collapse spaces/tabs
        stripped = re.sub(r"\n{2,}", "\n", stripped)          # collapse blank lines
        return stripped.strip()

    async def _summarize_table(self, text: str) -> str:
        """
        Send `text` to Ollama /api/generate and return a 2-sentence summary.
        Falls back to stripped plain text on any error.
        """
        base_url = self.ollama_base_url.rstrip("/")
        model = self.ollama_model
        try:
            timeout = float(self.ollama_timeout)
        except ValueError:
            timeout = 60.0

        prompt = (
            "You are a data analyst. Summarize the following table content in exactly "
            "2 concise sentences. Respond ONLY with valid JSON containing a single key "
            '"summary" whose value is the 2-sentence summary string.\n\n'
            f"Table content:\n{text}"
        )

        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{base_url}/api/generate", json=payload)
                resp.raise_for_status()
                raw = resp.json().get("response", "{}")
                try:
                    parsed = json.loads(raw)
                    summary = parsed.get("summary", "").strip()
                    if summary:
                        return summary
                    # Key missing — return the raw response string as fallback
                    return raw.strip()
                except json.JSONDecodeError as jex:
                    self.log(f"[_summarize_table] JSON decode error: {jex}. Raw: {raw!r}")
                    return self._strip_html(text)
        except httpx.TimeoutException:
            self.log("[_summarize_table] Ollama request timed out.")
            return self._strip_html(text)
        except httpx.HTTPStatusError as exc:
            self.log(f"[_summarize_table] HTTP error {exc.response.status_code}: {exc}")
            return self._strip_html(text)
        except Exception as exc:  # noqa: BLE001
            self.log(f"[_summarize_table] Unexpected error: {exc}")
            return self._strip_html(text)

    # ------------------------------------------------------------------ #
    #  Input normalisation                                                 #
    # ------------------------------------------------------------------ #

    def _normalise_input(self, raw_input: Any) -> list[dict]:
        """
        Accept a DataFrame, a single Data object, or a list of Data objects.
        Always returns a list of plain dicts.
        """
        # ---- DataFrame ------------------------------------------------
        if isinstance(raw_input, pd.DataFrame):
            return raw_input.to_dict(orient="records")

        # ---- Wrap a bare Data in a list --------------------------------
        if isinstance(raw_input, Data):
            raw_input = [raw_input]

        # ---- List[Data] ------------------------------------------------
        if isinstance(raw_input, list):
            records: list[dict] = []
            for item in raw_input:
                if isinstance(item, Data):
                    # .data is the underlying dict; never access row.data on a DataFrame!
                    records.append(item.data if isinstance(item.data, dict) else {})
                elif isinstance(item, dict):
                    records.append(item)
                else:
                    self.log(f"[_normalise_input] Skipping unknown item type: {type(item)}")
            return records

        self.log(f"[_normalise_input] Unhandled input type: {type(raw_input)}")
        return []

    # ------------------------------------------------------------------ #
    #  Main processing method                                              #
    # ------------------------------------------------------------------ #

    async def process(self) -> list[Data]:  # type: ignore[override]
        records = self._normalise_input(self.input_data)

        async def _process_record(row: dict) -> Data:
            html_content: str = row.get("html_content") or ""
            source: str = row.get("source", row.get("file_path", "unknown"))

            if not html_content:
                return Data(data={"text": "", "source": source})

            if self._contains_table(html_content):
                text = await self._summarize_table(html_content)
            else:
                text = self._strip_html(html_content)

            return Data(data={"text": text, "source": source})

        results: list[Data] = await asyncio.gather(*[_process_record(r) for r in records])
        self.status = f"Processed {len(results)} record(s)."
        return results