# Recovered Langflow component
# type: chat_api_payload_output
# class: ChatAPIPayloadOutput
# used in 1 flow(s): CRCV-v1.103 (Normal) Oak Backups
# json path: node.data.node.template.code.value

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, HandleInput, MessageTextInput, Output
from langflow.schema.data import Data
from langflow.schema.message import Message


class ChatAPIPayloadOutput(Component):
    """
    Single-output API payload builder.

    Inputs (all connectable):
      - user: Message/Data (user prompt coming from Chat Input or wherever)
      - assistant: Message/Data (assistant answer OR your Ollama envelope Data)
      - generation: Data (Ollama envelope Data: {text, metrics, raw})
      - retrieval: Data (PGVector extractor envelope Data: {metrics, chunks})

    Output:
      - payload: Data(data=<json payload>)  <-- RED output (connectable)
    """

    display_name = "Chat Output (API Payload)"
    description = "Build one payload Data: {user_text, assistant_text, metrics{...}, retrieval_chunks...}"
    icon = "cloud-upload"
    name = "chat_api_payload_output"

    inputs = [
        HandleInput(
            name="user",
            display_name="User (Message or Data)",
            input_types=["Message", "Data"],
            required=True,
        ),
        HandleInput(
            name="assistant",
            display_name="Assistant (Message or Data)",
            input_types=["Message", "Data"],
            required=True,
        ),
        HandleInput(
            name="generation",
            display_name="Generation Envelope (Data)",
            input_types=["Data"],
            required=False,
        ),
        HandleInput(
            name="retrieval",
            display_name="Retrieval Envelope (Data)",
            input_types=["Data"],
            required=False,
        ),
        # Optional manual override (advanced) if you ever want to type the user prompt manually
        MessageTextInput(
            name="user_text_override",
            display_name="User Text Override",
            info="Optional. If set, overrides extracted user text.",
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="include_retrieval_chunks",
            display_name="Include Retrieval Chunks",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="include_generation_raw",
            display_name="Include Ollama Raw JSON",
            value=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Payload (Data)", name="payload", method="build_payload"),
    ]

    # -----------------------
    # Helpers
    # -----------------------
    @staticmethod
    def _now_utc_iso() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    @staticmethod
    def _as_dict(x: Any) -> Optional[Dict[str, Any]]:
        if isinstance(x, dict):
            return x
        if isinstance(x, Data) and isinstance(x.data, dict):
            return x.data
        try:
            if hasattr(x, "data") and isinstance(getattr(x, "data"), dict):
                return getattr(x, "data")
        except Exception:
            pass
        return None

    @staticmethod
    def _as_text(x: Any) -> str:
        # Message
        if isinstance(x, Message):
            return x.text or ""

        # string
        if isinstance(x, str):
            return x

        # dict-like
        if isinstance(x, dict):
            for k in ("text", "content", "message"):
                v = x.get(k)
                if isinstance(v, str):
                    return v
            return ""

        # Data
        if isinstance(x, Data) and isinstance(x.data, dict):
            d = x.data
            # Common keys:
            for k in ("text", "content", "message"):
                v = d.get(k)
                if isinstance(v, str):
                    return v
            # If this is the Ollama envelope, it has {"text": "...", "metrics": {...}}
            v = d.get("text")
            return v if isinstance(v, str) else ""

        # last resort
        try:
            return str(x)
        except Exception:
            return ""

    @staticmethod
    def _safe_pick_metrics(gen_env: Dict[str, Any]) -> Dict[str, Any]:
        m = gen_env.get("metrics")
        return m if isinstance(m, dict) else {}

    @staticmethod
    def _safe_pick_retrieval_metrics(ret_env: Dict[str, Any]) -> Dict[str, Any]:
        m = ret_env.get("metrics")
        return m if isinstance(m, dict) else {}

    @staticmethod
    def _safe_pick_chunks(ret_env: Dict[str, Any]) -> List[Dict[str, Any]]:
        c = ret_env.get("chunks")
        return c if isinstance(c, list) else []

    # -----------------------
    # Core
    # -----------------------
    def build_payload(self) -> Data:
        # user text
        user_text = (self.user_text_override or "").strip()
        if not user_text:
            user_text = self._as_text(self.user)

        # assistant text
        assistant_text = self._as_text(self.assistant)

        # envelopes
        gen_env = self._as_dict(self.generation) or {}
        ret_env = self._as_dict(self.retrieval) or {}

        gen_metrics = self._safe_pick_metrics(gen_env)
        ret_metrics = self._safe_pick_retrieval_metrics(ret_env)

        payload: Dict[str, Any] = {
            "timestamp_utc": self._now_utc_iso(),
            "user_text": user_text or "",
            "assistant_text": assistant_text or "",
            "metrics": {
                "generation": gen_metrics,
                "retrieval": ret_metrics,
            },
        }

        if bool(self.include_retrieval_chunks):
            payload["retrieval_chunks"] = self._safe_pick_chunks(ret_env)

        if bool(self.include_generation_raw):
            payload["generation_raw"] = gen_env.get("raw")

        # Convenience promotion
        if isinstance(gen_metrics.get("model"), str):
            payload["model"] = gen_metrics["model"]

        # IMPORTANT: return Data so the output is RED and connectable
        return Data(data=payload)
