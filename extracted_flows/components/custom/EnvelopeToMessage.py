# Recovered Langflow component
# type: EnvelopeToMessage
# class: EnvelopeToMessage
# used in 52 flow(s): AIT Cognitive RAG, AITGPT Fees, AITGPT General, AITGPT Program, AITGPT V1.0.0, Baseline Normal AIT, Baseline Normal AIT 123, Baseline Normal Squad ...
# json path: node.data.node.template.code.value

from typing import Any, Dict

from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema.data import Data
from langflow.schema.message import Message


class EnvelopeToMessage(Component):
    display_name = "Envelope → Message"
    description = "Extracts env.data['text'] from the envelope Data and emits Message."
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
        env: Data = self.envelope  # DataInput gives you a Data instance
        payload: Dict[str, Any] = env.data or {}

        text = payload.get("text")
        if not isinstance(text, str):
            # Best-effort fallback if someone sent raw Ollama JSON directly
            raw = payload.get("raw", {})
            text = ((raw.get("message") or {}).get("content")) or ""

        return Message(text=text or "")
