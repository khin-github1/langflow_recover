# Recovered Langflow component
# type: refined_query_extractor
# class: RefinedQueryExtractor
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


class RefinedQueryExtractor(Component):
    display_name = "Refined Query Extractor"
    description = "Extracts refined_query from a JSON-like message."
    icon = "search"
    name = "refined_query_extractor"

    inputs = [
        HandleInput(
            name="inp",
            display_name="Input (Message/Data)",
            input_types=["Message", "Data"],
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Refined Query (Message)", name="out", method="build"),
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
                obj = ast.literal_eval(s)
            except Exception:
                obj = None

        if isinstance(obj, dict):
            val = obj.get("refined_query", "")
        else:
            m = re.search(r'"refined_query"\s*:\s*"([^"]*)"', s, re.DOTALL) \
                or re.search(r"'refined_query'\s*:\s*'([^']*)'", s, re.DOTALL)
            val = m.group(1) if m else ""

        return Message(text=str(val).strip())
