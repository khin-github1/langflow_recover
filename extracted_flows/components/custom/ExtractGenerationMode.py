# Recovered Langflow component
# type: ExtractGenerationMode
# class: ExtractGenerationMode
# used in 23 flow(s): AIT Cognitive RAG, Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup, Cognitive RAG V1.1.0, Cognitive RAG V1.1.0 Clean, Cognitive RAG V1.1.0 Eval, Cognitive RAG V1.1.5 ...
# json path: node.data.node.template.code.value

import json
import re
from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, Output
from lfx.schema.message import Message


class ExtractGenerationMode(Component):
    display_name = "Extract Generation Mode"
    description = "Extracts generation_mode from incoming text/JSON and returns it as a Message."
    icon = "braces"
    name = "ExtractGenerationMode"

    inputs = [
        MessageTextInput(
            name="input_message",
            display_name="Input Message",
            info="Incoming message text or JSON payload.",
            required=True,
        ),
    ]

    outputs = [
        Output(
            name="output_message",
            display_name="Output Message",
            method="build_output",
        ),
    ]

    def _normalize_mode(self, value: Any) -> str:
        if isinstance(value, str) and value.strip().lower() == "reason":
            return "reason"
        return "normal"

    def _strip_code_fences(self, text: str) -> str:
        text = text.strip()

        # Remove ```json ... ``` or ``` ... ```
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        return text.strip()

    def _extract_text(self, value: Any) -> str:
        if value is None:
            return ""

        # If upstream already gave a Message-like object
        if hasattr(value, "text") and getattr(value, "text") is not None:
            return str(getattr(value, "text"))

        # If upstream somehow passed a dict directly
        if isinstance(value, dict):
            return json.dumps(value)

        return str(value)

    def _parse_payload(self, raw_text: str) -> dict:
        raw_text = self._strip_code_fences(raw_text)

        # 1) Direct JSON parse
        try:
            obj = json.loads(raw_text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # 2) Extract first JSON object from surrounding text
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        # 3) Last-resort regex just for generation_mode
        mode_match = re.search(
            r'"generation_mode"\s*:\s*"(reason|normal)"',
            raw_text,
            re.IGNORECASE,
        )
        if mode_match:
            return {"generation_mode": mode_match.group(1).lower()}

        return {}

    def build_output(self) -> Message:
        raw_text = self._extract_text(self.input_message)
        payload = self._parse_payload(raw_text)
        mode = self._normalize_mode(payload.get("generation_mode"))

        result_text = json.dumps({"generation_mode": mode})

        msg = Message(
            text=result_text,
            sender="System",
        )
        self.status = result_text
        return msg