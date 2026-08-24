# Recovered Langflow component
# type: LoopItemToJoinedRecord
# class: LoopItemToJoinedRecord
# used in 1 flow(s): Cognitive RAG V1.2.5 GPT
# json path: node.data.node.template.code.value

import json
from typing import Any, Dict

from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, Output
from lfx.schema.data import Data


class LoopItemToJoinedRecord(Component):
    display_name = "Loop Item To Joined Record"
    description = "Unwraps one Loop item into a joined_record Data object."
    icon = "filter"
    name = "LoopItemToJoinedRecord"

    inputs = [
        DataInput(
            name="loop_item",
            display_name="Loop Item",
            required=True,
        )
    ]

    outputs = [
        Output(
            display_name="Joined Record",
            name="joined_record",
            method="build_joined_record",
        )
    ]

    def _as_dict(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, Data) and isinstance(value.data, dict):
            return value.data

        if hasattr(value, "data") and isinstance(value.data, dict):
            return value.data

        if isinstance(value, dict):
            return value

        if hasattr(value, "text") and value.text:
            text = str(value.text).strip()
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return {"text": text}

        return {"text": str(value)}

    def build_joined_record(self) -> Data:
        payload = self._as_dict(self.loop_item)

        # Real Loop usually emits the row directly.
        if "custom_id" in payload and "batch_answer_text" in payload:
            joined = payload

        # Fallback wrappers.
        elif isinstance(payload.get("item"), dict):
            joined = payload["item"]

        elif isinstance(payload.get("data"), dict):
            joined = payload["data"]

        elif isinstance(payload.get("text"), dict):
            joined = payload["text"]

        else:
            raise ValueError(
                "Could not unwrap Loop item into joined_record. "
                f"Got keys: {list(payload.keys())}"
            )

        if "custom_id" not in joined:
            raise ValueError("Loop item is missing custom_id.")

        if "batch_answer_text" not in joined:
            raise ValueError("Loop item is missing batch_answer_text.")

        self.status = f"custom_id={joined.get('custom_id')}"

        return Data(
            data=joined,
            text_key="custom_id",
            default_value="",
        )