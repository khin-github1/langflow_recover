# Recovered Langflow component
# type: sr_policy_enforcer
# class: SRPolicyEnforcer
# used in 2 flow(s): CRCV V4, Cognitive RAG V0.5.2 backup
# json path: node.data.node.template.code.value

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, FloatInput, HandleInput, Output
from langflow.schema.data import Data
from langflow.schema.message import Message


class SRPolicyEnforcer(Component):
    display_name = "SR Policy Enforcer (Deterministic)"
    description = (
        "Deterministically sets action based on support_ratio threshold and whether this is the retry pass. "
        "Keeps verifier fields but overrides `action` to: accept / retry / abstain."
    )
    icon = "shield-check"
    name = "sr_policy_enforcer"

    inputs = [
        HandleInput(
            name="verifier_json",
            display_name="Verifier JSON (Message/Data)",
            input_types=["Message", "Data"],
            required=True,
        ),
        FloatInput(
            name="tau_support",
            display_name="τsupport threshold",
            value=0.8,
            advanced=True,
        ),
        BoolInput(
            name="is_retry_pass",
            display_name="Is retry pass?",
            value=False,
            advanced=True,
        ),
        BoolInput(
            name="keep_llm_action",
            display_name="Keep original LLM action as llm_action",
            value=True,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Enforced JSON (Message)", name="out", method="build"),
    ]

    _re_json = re.compile(r"\{.*\}", re.DOTALL)

    @staticmethod
    def _to_str(v: Any) -> str:
        if isinstance(v, Data):
            v = v.data
        if isinstance(v, Message):
            return (getattr(v, "text", None) or getattr(v, "content", None) or "").strip()
        return "" if v is None else str(v).strip()

    @classmethod
    def _parse_obj(cls, s: str) -> Dict[str, Any]:
        s = (s or "").strip()
        if not s:
            return {}
        # JSON parse
        try:
            o = json.loads(s)
            return o if isinstance(o, dict) else {}
        except Exception:
            pass
        # Python literal (in case)
        try:
            o = ast.literal_eval(s)
            return o if isinstance(o, dict) else {}
        except Exception:
            pass
        # Extract first JSON object
        m = cls._re_json.search(s)
        if m:
            try:
                o = json.loads(m.group(0))
                return o if isinstance(o, dict) else {}
            except Exception:
                pass
        return {}

    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        return max(0.0, min(1.0, v))

    def build(self) -> Message:
        raw = self._to_str(self.verifier_json)
        obj = self._parse_obj(raw)

        # Read SR
        sr = self._clamp01(obj.get("support_ratio", 0.0))

        # Threshold
        tau = float(self.tau_support or 0.8)
        tau = max(0.0, min(1.0, tau))

        retry_pass = bool(self.is_retry_pass)

        # Preserve original action if desired
        if bool(self.keep_llm_action) and "action" in obj:
            obj["llm_action"] = obj.get("action")

        # Deterministic action mapping
        if sr >= tau:
            obj["action"] = "accept"
            # normalize verdict if missing/invalid
            verdict = str(obj.get("verdict", "") or "").strip().lower()
            if verdict not in ("supported", "mixed", "unsupported", "ood"):
                obj["verdict"] = "supported"
        else:
            obj["action"] = "abstain" if retry_pass else "retry"
            # normalize verdict if missing/invalid
            verdict = str(obj.get("verdict", "") or "").strip().lower()
            if verdict not in ("supported", "mixed", "unsupported", "ood"):
                obj["verdict"] = "mixed"

        # Write back clamped SR (optional, but keeps clean)
        obj["support_ratio"] = sr

        return Message(text=json.dumps(obj, ensure_ascii=False))
