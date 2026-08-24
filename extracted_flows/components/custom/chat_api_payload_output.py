# Recovered Langflow component
# type: chat_api_payload_output
# class: ChatAPIPayloadOutput
# used in 16 flow(s): AITGPT Fees, AITGPT General, AITGPT Program, AITGPT V1.0.0, Baseline Normal AIT, Baseline Normal AIT 123, Baseline Normal Squad, Baseline Reason AIT ...
# json path: node.data.node.template.code.value

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Union

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, HandleInput, MessageTextInput, Output
from langflow.schema.message import Message
from langflow.schema.data import Data


class ChatAPIPayloadOutput(Component):
    """
    Single-output API payload builder (grey output).

    Input:
      - user_text: the user's prompt (string)
      - assistant: Message or string (assistant answer)
      - generation: dict from Ollama envelope node: {text, metrics, raw}
      - retrieval: dict from PGVector extractor: {metrics, chunks}

    Output:
      - payload: dict you can return to website via LangFlow flow endpoint
    """

    display_name = "Chat Output (API Payload)"
    description = "Builds one JSON payload: {user, assistant, metrics, retrieval_chunks,...} for your website backend."
    icon = "cloud-upload"
    name = "chat_api_payload_output"

    inputs = [
        MessageTextInput(
            name="user_text",
            display_name="User Text",
            info="Raw user prompt text (what the user typed).",
            required=True,
        ),
        HandleInput(
            name="assistant",
            display_name="Assistant (Message or Text)",
            input_types=["Message", "Data"],  # accept Message; Data for safety; also works with raw text
            required=True,
        ),
        HandleInput(
            name="generation",
            display_name="Generation Envelope (dict)",
            input_types=["Data"],  # LangFlow will pass dict-like through the handle system
            required=False,
        ),
        HandleInput(
            name="retrieval",
            display_name="Retrieval Envelope (dict)",
            input_types=["Data"],
            required=False,
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
        Output(display_name="Payload", name="payload", method="build_payload"),
    ]

    # -----------------------
    # Helpers
    # -----------------------
    @staticmethod
    def _now_utc_iso() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    @staticmethod
    def _as_dict(x: Any) -> Optional[Dict[str, Any]]:
        # Accept dict directly
        if isinstance(x, dict):
            return x
        # Accept Data(data={...})
        if isinstance(x, Data) and isinstance(x.data, dict):
            return x.data
        # Some LF versions wrap dict-ish objects; try best-effort
        try:
            if hasattr(x, "data") and isinstance(getattr(x, "data"), dict):
                return getattr(x, "data")
        except Exception:
            pass
        return None

    @staticmethod
    def _assistant_text(assistant: Any) -> str:
        if isinstance(assistant, Message):
            return assistant.text or ""
        if isinstance(assistant, str):
            return assistant
        # Sometimes Message serialized into dict-like
        if isinstance(assistant, dict):
            t = assistant.get("text")
            return t if isinstance(t, str) else ""
        if isinstance(assistant, Data) and isinstance(assistant.data, dict):
            # try common keys
            for k in ("text", "content", "message"):
                v = assistant.data.get(k)
                if isinstance(v, str):
                    return v
        # last resort
        return str(assistant)

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
    def build_payload(self) -> Dict[str, Any]:
        assistant_text = self._assistant_text(self.assistant)

        gen_env = self._as_dict(self.generation) or {}
        ret_env = self._as_dict(self.retrieval) or {}

        gen_metrics = self._safe_pick_metrics(gen_env)
        ret_metrics = self._safe_pick_retrieval_metrics(ret_env)

        payload: Dict[str, Any] = {
            "timestamp_utc": self._now_utc_iso(),
            "user_text": self.user_text or "",
            "assistant_text": assistant_text,
            "metrics": {
                "generation": gen_metrics,
                "retrieval": ret_metrics,
            },
        }

        if self.include_retrieval_chunks:
            payload["retrieval_chunks"] = self._safe_pick_chunks(ret_env)

        if self.include_generation_raw:
            raw = gen_env.get("raw")
            payload["generation_raw"] = raw if isinstance(raw, dict) else raw

        # Convenience: promote a few common fields (optional)
        if isinstance(gen_metrics.get("model"), str):
            payload["model"] = gen_metrics["model"]

        return payload
