# Recovered Langflow component
# type: ClarificationExtractor
# class: ClarificationExtractorComponent
# used in 24 flow(s): AIT Cognitive RAG, Cognitive RAG V0.5.2 backup, Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup, Cognitive RAG V1.1.0, Cognitive RAG V1.1.0 Clean, Cognitive RAG V1.1.0 Eval ...
# json path: node.data.node.template.code.value

import ast
import json
from typing import Any

from lfx.custom import Component
from lfx.io import MessageTextInput, Output
from lfx.schema import Message


class ClarificationExtractorComponent(Component):
    display_name = "Clarification Extractor"
    description = "Extracts original question, expected slots, and missing slots from an incoming dict-like message."
    icon = "braces"
    name = "ClarificationExtractor"

    inputs = [
        MessageTextInput(
            name="input_payload",
            display_name="Input Payload",
            info="Incoming payload as a Python dict string or plain dict-like text.",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Extracted Message",
            name="extracted_message",
            method="build_output",
        ),
    ]

    def _parse_payload(self, payload: Any) -> dict:
        if isinstance(payload, dict):
            return payload

        text = str(payload).strip()
        if not text:
            return {}

        # Try Python dict string first
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Try JSON next
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return {}

    def build_output(self) -> Message:
        payload = self._parse_payload(self.input_payload)

        missing_slots = payload.get("missing_slots", {})
        if not isinstance(missing_slots, dict):
            missing_slots = {"general": [], "critical": []}

        result = {
            "original_question": payload.get("original_question", ""),
            "expected_slots": payload.get("expected_general_slots", []),
            "history": payload.get("history_window", ""),
            "missing_slots": {
                "general": missing_slots.get("general", []),
                "critical": missing_slots.get("critical", []),
            },
        }

        json_text = json.dumps(result, ensure_ascii=False, indent=2)

        message = Message(
            text=json_text,
            data=result,
        )

        self.status = message
        return message