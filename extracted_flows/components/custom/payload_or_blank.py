# Recovered Langflow component
# type: payload_or_blank
# class: PayloadOrBlank
# used in 1 flow(s): AITGPT V1.0.0
# json path: node.data.node.template.code.value

from __future__ import annotations

from typing import Any, Dict

from langflow.custom.custom_component.component import Component
from langflow.io import HandleInput, MessageTextInput, Output


class PayloadOrBlank(Component):
    display_name = "Payload Or Blank"
    description = "Returns the incoming Dict payload, or {} if nothing arrived."
    icon = "braces"
    name = "payload_or_blank"

    inputs = [
        MessageTextInput(
            name="trigger",
            display_name="Trigger",
            info="Connect ChatInput.message here so this node always runs.",
            required=True,
        ),
        HandleInput(
            name="payload",
            display_name="Payload",
            input_types=["Dict"],
            required=False,
        ),
    ]

    outputs = [
        Output(display_name="Payload", name="payload_out", method="build_payload"),
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
        payload = self._as_dict(self.payload)
        return payload if payload else {}