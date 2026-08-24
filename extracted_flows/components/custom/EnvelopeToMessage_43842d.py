# Recovered Langflow component
# type: EnvelopeToMessage
# class: EnvelopeToMessage
# used in 1 flow(s): Cognitive RAG V1.1.0
# json path: node.data.node.template.code.value

import json
from typing import Any, Dict

from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema.data import Data
from langflow.schema.message import Message


class EnvelopeToMessage(Component):
    display_name = "Envelope → Message"
    description = "Extracts parsed_validator_output or text from envelope Data and emits Message."
    icon = "Filter"
    name = "EnvelopeToMessage"

    inputs = [
        DataInput(
            name="envelope",
            display_name="Envelope (Data)",
            info="Data object produced by Ollama Chat (Envelope).",
            required=True,
        )
    ]

    outputs = [
        Output(display_name="Message", name="message", method="build_message"),
    ]

    def build_message(self) -> Message:
        env: Data = self.envelope
        payload: Dict[str, Any] = env.data or {}

        # Best source: normalized parsed validator output
        parsed = payload.get("parsed_validator_output")
        if isinstance(parsed, dict):
            return Message(text=json.dumps(parsed, ensure_ascii=False, indent=2))

        # Next: normalized text field
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            return Message(text=text)

        # Fallback: raw Ollama content
        raw = payload.get("raw", {})
        raw_text = ((raw.get("message") or {}).get("content")) or ""
        if isinstance(raw_text, str) and raw_text.strip():
            return Message(text=raw_text)

        # Final fallback: dump whole payload
        return Message(text=json.dumps(payload, ensure_ascii=False, indent=2))