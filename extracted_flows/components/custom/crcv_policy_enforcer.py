# Recovered Langflow component
# type: crcv_policy_enforcer
# class: CRCVPolicyEnforcer
# used in 1 flow(s): CRCV V4
# json path: node.data.node.template.code.value

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

from langflow.custom.custom_component.component import Component
from langflow.io import HandleInput, Output
from langflow.schema.data import Data
from langflow.schema.message import Message


class CRCVPolicyEnforcer(Component):
    display_name = "CRCV Policy Enforcer"
    description = (
        "Unwraps controller JSON (supports Ollama envelope), enforces deterministic need_clarify, "
        "adds need_refine, and prevents clarification loops by clearing need_clarify after a prior clarification."
    )
    icon = "shield"
    name = "crcv_policy_enforcer"

    inputs = [
        HandleInput(
            name="controller_json",
            display_name="Controller JSON (Message/Data)",
            input_types=["Message", "Data"],
            required=True,
        ),
        HandleInput(
            name="user_message",
            display_name="User Message (Message/Data)",
            input_types=["Message", "Data"],
            required=True,
        ),
        HandleInput(
            name="history_text",
            display_name="History Text (Message/Data)",
            input_types=["Message", "Data"],
            required=False,
        ),
    ]

    outputs = [
        Output(display_name="Enforced JSON (Message)", name="out", method="build"),
    ]

    _re_json_obj = re.compile(r"\{.*\}", re.DOTALL)

    # -------------------------
    # Basic helpers
    # -------------------------
    @staticmethod
    def _to_str(v: Any) -> str:
        if isinstance(v, Data):
            v = v.data
        if isinstance(v, Message):
            return (getattr(v, "text", None) or getattr(v, "content", None) or "").strip()
        return "" if v is None else str(v).strip()

    @staticmethod
    def _bool(x: Any) -> bool:
        if isinstance(x, bool):
            return x
        if x is None:
            return False
        return str(x).strip().lower() in ("true", "1", "yes")

    @staticmethod
    def _list(x: Any) -> List[str]:
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        return []

    # -------------------------
    # Controller parsing (FIX)
    # -------------------------
    @classmethod
    def _parse_json_like(cls, s: str) -> Optional[Dict[str, Any]]:
        s = (s or "").strip()
        if not s:
            return None

        # Try JSON
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # Try python literal
        try:
            obj = ast.literal_eval(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # Try extracting first {...}
        m = cls._re_json_obj.search(s)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass

        return None

    @classmethod
    def _extract_controller_obj(cls, raw: str) -> Dict[str, Any]:
        """
        Supports:
        1) Direct controller JSON: {"missing_slots":...}
        2) Ollama envelope: {"model":..., "message":{"content":"{...controller json...}"}}
        3) Other envelopes with "content"/"text" fields
        """
        top = cls._parse_json_like(raw)
        if not top:
            return {}

        # Case: already the controller JSON
        if "missing_slots" in top and "need_clarify" in top:
            return top

        # Case: Ollama-style envelope
        msg = top.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                inner = cls._parse_json_like(content)
                if inner and "missing_slots" in inner:
                    return inner

        # Case: other shapes
        for k in ("content", "text", "output", "result"):
            v = top.get(k)
            if isinstance(v, str):
                inner = cls._parse_json_like(v)
                if inner and "missing_slots" in inner:
                    return inner

        return {}

    # -------------------------
    # History inspection (budget)
    # -------------------------
    @staticmethod
    def _last_llm_utterance(history: str) -> str:
        if not history:
            return ""
        matches = list(re.finditer(r"\bLLM:\s*(.*)", history))
        if not matches:
            return ""
        return (matches[-1].group(1) or "").strip()

    @classmethod
    def _clarify_budget_used(cls, history: str) -> bool:
        last_llm = cls._last_llm_utterance(history)
        if not last_llm:
            return False
        # If last assistant message looks like a question / clarification
        if last_llm.endswith("?"):
            return True
        if re.search(r"\b(which|what|could you|can you|please|specify|clarify|confirm)\b", last_llm.lower()):
            return True
        return False

    # -------------------------
    # Minimal slot fill (only removes obvious filled slots)
    # -------------------------
    @staticmethod
    def _msg_has_fee_type(msg: str) -> bool:
        return bool(re.search(r"\b(tuition|fee|fees|academic fee|accommodation|housing|deposit|total|all)\b", msg.lower()))

    # -------------------------
    # Clarification question generator (if controller forgot)
    # -------------------------
    @staticmethod
    def _natural_question_for_slots(slots: List[str]) -> str:
        if not slots:
            return ""
        # Simple natural phrasing; no domain anchoring
        if len(slots) == 1:
            return f"To answer accurately, could you clarify the {slots[0]}?"
        if len(slots) == 2:
            return f"To answer accurately, could you clarify the {slots[0]} and {slots[1]}?"
        # 3+
        head = ", ".join(slots[:-1])
        return f"To answer accurately, could you clarify {head}, and {slots[-1]}?"

    # -------------------------
    # Build
    # -------------------------
    def build(self) -> Message:
        raw = self._to_str(self.controller_json)
        user_msg = self._to_str(self.user_message)
        hist = self._to_str(self.history_text)

        ctrl = self._extract_controller_obj(raw)

        missing = ctrl.get("missing_slots", {}) if isinstance(ctrl.get("missing_slots"), dict) else {}
        critical = self._list(missing.get("critical", []))
        normal = self._list(missing.get("normal", []))

        filled_from_history = self._bool(ctrl.get("filled_from_history", False))
        refined_query = (ctrl.get("refined_query") or "").strip()
        clarification_question = (ctrl.get("clarification_question") or "").strip()

        # --- If controller said fee_type missing but user message already contains it, remove it
        if "fee_type" in critical and self._msg_has_fee_type(user_msg):
            critical = [s for s in critical if s != "fee_type"]

        # --- Deterministic need_clarify from critical slots
        need_clarify = len(critical) > 0

        # --- Clarification loop prevention (no abstain here; just stop asking again)
        budget_used = self._clarify_budget_used(hist)
        should_abstain = False  # keep field for compatibility; you can ignore in routing
        if need_clarify and budget_used:
            # Do NOT ask again; proceed forward
            need_clarify = False
            clarification_question = ""
            # refined_query should be clean and non-empty
            refined_query = refined_query or user_msg

        # --- If we DO need clarify, ensure we have a question
        if need_clarify:
            # keep controller question if present; otherwise generate one
            if not clarification_question:
                clarification_question = self._natural_question_for_slots(critical)
            # refined_query is not used yet; keep clean or empty
            if not refined_query:
                refined_query = ""

        # --- If not clarifying, ensure clarification_question blank and refined_query non-empty
        if not need_clarify:
            clarification_question = ""
            refined_query = refined_query or user_msg

        # --- need_refine: log-only + routing
        need_refine = (not need_clarify) and (filled_from_history or (refined_query.strip() != user_msg.strip()))

        enforced = {
            "missing_slots": {"critical": critical, "normal": normal},
            "filled_from_history": bool(filled_from_history),
            "need_clarify": bool(need_clarify),
            "need_refine": bool(need_refine),
            "should_abstain": bool(should_abstain),
            "refined_query": refined_query,
            "clarification_question": clarification_question,
        }

        return Message(text=json.dumps(enforced, ensure_ascii=False))
