# Recovered Langflow component
# type: EvaluationPayloadCollectorV2
# class: EvaluationPayloadCollectorV2
# used in 2 flow(s): Cognitive RAG V1.1.0 Eval, Cognitive RAG V1.1.5 Backup
# json path: node.data.node.template.code.value

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from langflow.custom import Component
from langflow.io import HandleInput, MessageTextInput, Output
from langflow.schema.data import Data
from langflow.schema.message import Message


class EvaluationPayloadCollectorV2(Component):
    display_name = "Evaluation Payload Collector V2"
    description = "Minimal typed Data payload collector for testing fresh node registration."
    icon = "package"
    name = "EvaluationPayloadCollectorV2"

    inputs = [
        MessageTextInput(
            name="user_text",
            display_name="User Text",
            required=True,
        ),
        HandleInput(
            name="attempt1_accept",
            display_name="Attempt 1 Accept",
            input_types=["Message", "Data"],
            required=False,
        ),
        HandleInput(
            name="attempt1_reject",
            display_name="Attempt 1 Reject",
            input_types=["Message", "Data"],
            required=False,
        ),
        HandleInput(
            name="attempt2_accept",
            display_name="Attempt 2 Accept",
            input_types=["Message", "Data"],
            required=False,
        ),
        HandleInput(
            name="attempt2_reject",
            display_name="Attempt 2 Reject",
            input_types=["Message", "Data"],
            required=False,
        ),
    ]

    outputs = [
        Output(display_name="Payload", name="payload", method="build_payload"),
    ]

    @staticmethod
    def _now_utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value.strip()

        if isinstance(value, Message):
            return str(value.text or "").strip()

        if isinstance(value, Data):
            if isinstance(value.data, dict):
                for key in ("assistant_text", "text", "content", "message"):
                    v = value.data.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            return str(getattr(value, "text", "") or "").strip()

        if isinstance(value, dict):
            for key in ("assistant_text", "text", "content", "message"):
                v = value.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        if hasattr(value, "text"):
            return str(getattr(value, "text", "") or "").strip()

        return str(value).strip()

    def _resolve_final_output(self) -> Dict[str, str]:
        candidates = [
            ("attempt2_accept", self._safe_text(self.attempt2_accept)),
            ("attempt2_reject", self._safe_text(self.attempt2_reject)),
            ("attempt1_accept", self._safe_text(self.attempt1_accept)),
            ("attempt1_reject", self._safe_text(self.attempt1_reject)),
        ]
        for source, text in candidates:
            if text:
                return {"source": source, "text": text}
        return {"source": "", "text": ""}

    def build_payload(self) -> Data:
        resolved = self._resolve_final_output()

        payload = {
            "timestamp_utc": self._now_utc_iso(),
            "user_text": self._safe_text(self.user_text),
            "assistant_text": resolved["text"],
            "final_output_source": resolved["source"],
            "terminal_outputs": {
                "attempt1_accept": self._safe_text(self.attempt1_accept),
                "attempt1_reject": self._safe_text(self.attempt1_reject),
                "attempt2_accept": self._safe_text(self.attempt2_accept),
                "attempt2_reject": self._safe_text(self.attempt2_reject),
            },
        }

        self.status = f"payload built | source={resolved['source'] or 'none'} | len={len(resolved['text'])}"

        return Data(
            data=payload,
            text_key="assistant_text",
            default_value="",
        )