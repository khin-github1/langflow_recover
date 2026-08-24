# Recovered Langflow component
# type: ExtractUserQuery
# class: ExtractUserQueryComponent
# used in 24 flow(s): AIT Cognitive RAG, Cognitive RAG V0.5.2 backup, Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup, Cognitive RAG V1.1.0, Cognitive RAG V1.1.0 Clean, Cognitive RAG V1.1.0 Eval ...
# json path: node.data.node.template.code.value

import ast
import json
from typing import Any

from lfx.custom import Component
from lfx.io import MessageInput, Output
from lfx.schema import Message


class ExtractUserQueryComponent(Component):
    display_name = "Extract User Query"
    description = "Extracts the 'User Query' field from a JSON/dict-like Message payload and returns it as a Message."
    icon = "braces"
    name = "ExtractUserQuery"

    inputs = [
        MessageInput(
            name="input_message",
            display_name="Input Message",
            info="Incoming Message containing a JSON or dict-like payload.",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="User Query Message",
            name="user_query_message",
            method="build_output",
        ),
    ]

    def _extract_raw_text(self, value: Any) -> str:
        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()
        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()
        return str(value).strip()

    def _parse_payload(self, raw_text: str) -> dict:
        if not raw_text:
            return {}

        # Try JSON first
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Try Python dict string next
        try:
            parsed = ast.literal_eval(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return {}

    def build_output(self) -> Message:
        raw_text = self._extract_raw_text(self.input_message)
        payload = self._parse_payload(raw_text)

        if not isinstance(payload, dict) or not payload:
            error_result = {
                "_error": "Could not parse input_message into a dictionary payload.",
                "_raw_input": raw_text,
            }
            error_text = json.dumps(error_result, ensure_ascii=False, indent=2)
            self.status = error_text
            return Message(text=error_text, data=error_result)

        user_query = str(payload.get("User Query", "")).strip()

        result = {
            "User Query": user_query
        }

        output_text = json.dumps(result, ensure_ascii=False, indent=2)
        self.status = output_text
        return Message(text=output_text, data=result)