# Recovered Langflow component
# type: JoinedRowToGenerationMode
# class: JoinedRowToGenerationMode
# used in 10 flow(s): Cognitive RAG V1.2.0 GPT, Cognitive RAG V1.2.5 GPT, Cognitive RAG V1.3.0 GPT, Cognitive RAG V1.3.0 GPT Loose AIT , Cognitive RAG V1.3.0 GPT Strict, Cognitive RAG V1.3.0 GPT Strict AIT, Retry RAG V1.3.0 GPT, Retry RAG V1.3.0 GPT Loose AIT ...
# json path: node.data.node.template.code.value

import json
from typing import Any, Dict

from lfx.custom import Component
from lfx.io import DataInput, Output
from lfx.schema import Data, Message


class JoinedRowToGenerationMode(Component):
    display_name = "Joined Row → Generation Mode"
    description = "Extracts matched generation_mode as Message."
    icon = "toggle-right"
    name = "JoinedRowToGenerationMode"

    inputs = [
        DataInput(
            name="joined_record",
            display_name="Joined Record",
            required=True,
        )
    ]

    outputs = [
        Output(
            display_name="Generation Mode Message",
            name="generation_mode_message",
            method="build_generation_mode_message",
        )
    ]

    def _payload(self) -> Dict[str, Any]:
        if isinstance(self.joined_record, Data):
            return self.joined_record.data or {}
        if hasattr(self.joined_record, "data") and isinstance(self.joined_record.data, dict):
            return self.joined_record.data
        if isinstance(self.joined_record, dict):
            return self.joined_record
        return {}

    def build_generation_mode_message(self) -> Message:
        joined = self._payload()
        mode = str(joined.get("generation_mode", "normal") or "normal").strip().lower()
        if mode != "reason":
            mode = "normal"
        payload = {"generation_mode": mode}
        return Message(text=json.dumps(payload, ensure_ascii=False), data=payload)