# Recovered Langflow component
# type: chat_api_payload_collector
# class: ChatAPIPayloadCollector
# used in 1 flow(s): AITGPT V1.0.0
# json path: node.data.node.template.code.value

from __future__ import annotations

from typing import Any, Dict

from langflow.custom.custom_component.component import Component
from langflow.io import HandleInput, Output


class ChatAPIPayloadCollector(Component):
    display_name = "Chat Output (API Payload Collector)"
    description = "Accepts 4 Dict payloads and returns the first non-empty one."
    icon = "cloud-upload"
    name = "chat_api_payload_collector"

    inputs = [
        HandleInput(
            name="payload_1",
            display_name="Payload 1",
            input_types=["Dict"],
            required=True,
        ),
        HandleInput(
            name="payload_2",
            display_name="Payload 2",
            input_types=["Dict"],
            required=True,
        ),
        HandleInput(
            name="payload_3",
            display_name="Payload 3",
            input_types=["Dict"],
            required=True,
        ),
        HandleInput(
            name="payload_4",
            display_name="Payload 4",
            input_types=["Dict"],
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Payload", name="payload", method="build_payload"),
    ]

    @staticmethod
    def _as_dict(x: Any) -> Dict[str, Any]:
        if isinstance(x, dict):
            return x
        try:
            if hasattr(x, "data") and isinstance(getattr(x, "data"), dict):
                return getattr(x, "data")
        except Exception:
            pass
        return {}

    def build_payload(self) -> Dict[str, Any]:
        p1 = self._as_dict(self.payload_1)
        p2 = self._as_dict(self.payload_2)
        p3 = self._as_dict(self.payload_3)
        p4 = self._as_dict(self.payload_4)

        if p1:
            out = dict(p1)
            out["_selected_payload_source"] = "payload_1"
            return out

        if p2:
            out = dict(p2)
            out["_selected_payload_source"] = "payload_2"
            return out

        if p3:
            out = dict(p3)
            out["_selected_payload_source"] = "payload_3"
            return out

        if p4:
            out = dict(p4)
            out["_selected_payload_source"] = "payload_4"
            return out

        return {
            "timestamp_utc": None,
            "user_text": "",
            "assistant_text": "",
            "metrics": {},
            "retrieval_chunks": [],
            "generation_raw": None,
            "model": None,
            "_selected_payload_source": None,
            "_error": "No payload received from any branch.",
        }