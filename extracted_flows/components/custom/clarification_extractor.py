# Recovered Langflow component
# type: clarification_extractor
# class: ClarificationExtractor
# used in 2 flow(s): CRCV V4, Cognitive RAG V0.5.2 backup
# json path: node.data.node.template.code.value

from __future__ import annotations

import ast
import json
import re
from typing import Any

from langflow.custom.custom_component.component import Component
from langflow.io import HandleInput, Output
from langflow.schema.data import Data
from langflow.schema.message import Message


class ClarificationExtractor(Component):
    display_name = "Clarification Extractor"
    description = "Extracts clarification_question from a JSON-like message."
    icon = "help-circle"
    name = "clarification_extractor"

    inputs = [
        HandleInput(
            name="inp",
            display_name="Input (Message/Data)",
            input_types=["Message", "Data"],
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Clarification (Message)", name="out", method="build"),
    ]

    def build(self) -> Message:
        v: Any = self.inp
        if isinstance(v, Data):
            v = v.data
        if isinstance(v, Message):
            v = v.text

        s = "" if v is None else str(v)

        obj = None
        try:
            obj = json.loads(s)
        except Exception:
            try:
                obj = ast.literal_eval(s)  # handles single quotes + True/False/None
            except Exception:
                obj = None

        if isinstance(obj, dict):
            val = obj.get("clarification_question", "")
        else:
            m = re.search(r'"clarification_question"\s*:\s*"([^"]*)"', s, re.DOTALL) \
                or re.search(r"'clarification_question'\s*:\s*'([^']*)'", s, re.DOTALL)
            val = m.group(1) if m else ""

        return Message(text=str(val).strip())
