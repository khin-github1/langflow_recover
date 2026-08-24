# Recovered Langflow component
# type: UserQuerySelector
# class: UserQuerySelectorComponent
# used in 24 flow(s): AIT Cognitive RAG, Cognitive RAG V0.5.2 backup, Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup, Cognitive RAG V1.1.0, Cognitive RAG V1.1.0 Clean, Cognitive RAG V1.1.0 Eval ...
# json path: node.data.node.template.code.value

import ast
import json
from typing import Any

from lfx.custom import Component
from lfx.io import MessageInput, Output
from lfx.schema import Message


class UserQuerySelectorComponent(Component):
    display_name = "User Query Selector"
    description = "Adds 'User Query' to the payload using refined_query when refinement.used is true, otherwise original_question."
    icon = "braces"
    name = "UserQuerySelector"

    inputs = [
        MessageInput(
            name="input_message",
            display_name="Input Message",
            info="Incoming analyzer payload as a Message object",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Updated Message",
            name="updated_message",
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

        # Try Python dict string
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

        refinement = payload.get("refinement", {})
        if not isinstance(refinement, dict):
            refinement = {}

        refinement_used = bool(refinement.get("used", False))
        refined_query = str(refinement.get("refined_query", "")).strip()
        original_question = str(payload.get("original_question", "")).strip()

        if refinement_used and refined_query:
            user_query = refined_query
        else:
            user_query = original_question

        payload["User Query"] = user_query

        output_text = json.dumps(payload, ensure_ascii=False, indent=2)
        self.status = output_text
        return Message(text=output_text, data=payload)