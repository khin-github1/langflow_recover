# Recovered Langflow component
# type: chat_api_payload_concat
# class: ChatAPIPayloadConcat
# used in 1 flow(s): AITGPT V1.0.0
# json path: node.data.node.template.code.value

from __future__ import annotations

from typing import Any, Dict, List

from langflow.custom.custom_component.component import Component
from langflow.io import HandleInput, MessageTextInput, Output


class ChatAPIPayloadConcat(Component):
    display_name = "Chat Output (API Payload Concat)"
    description = "Accepts 4 final Dict payloads and returns the first non-empty payload, plus debug info."
    icon = "cloud-upload"
    name = "chat_api_payload_concat"

    inputs = [
        MessageTextInput(name="trigger", display_name="Trigger", required=True),
        HandleInput(name="payload_1", display_name="Payload 1", input_types=["Dict"], required=False),
        HandleInput(name="payload_2", display_name="Payload 2", input_types=["Dict"], required=False),
        HandleInput(name="payload_3", display_name="Payload 3", input_types=["Dict"], required=False),
        HandleInput(name="payload_4", display_name="Payload 4", input_types=["Dict"], required=False),
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
        payloads = [
            self._as_dict(self.payload_1),
            self._as_dict(self.payload_2),
            self._as_dict(self.payload_3),
            self._as_dict(self.payload_4),
        ]
        active = [p for p in payloads if p]

        if active:
            out = dict(active[0])
            out["_debug_active_count"] = len(active)
            out["_debug_all_payloads"] = active
            return out

        return {
            "timestamp_utc": None,
            "user_text": str(self.trigger or ""),
            "assistant_text": "",
            "metrics": {},
            "retrieval_chunks": [],
            "generation_raw": None,
            "model": None,
            "_debug_active_count": 0,
            "_debug_all_payloads": [],
            "_error": "No payload received from any branch."
        }