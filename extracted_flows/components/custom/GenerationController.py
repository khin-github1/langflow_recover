# Recovered Langflow component
# type: GenerationController
# class: GenerationControllerComponent
# used in 23 flow(s): AIT Cognitive RAG, Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup, Cognitive RAG V1.1.0, Cognitive RAG V1.1.0 Clean, Cognitive RAG V1.1.0 Eval, Cognitive RAG V1.1.5 ...
# json path: node.data.node.template.code.value

import ast
import json
from typing import Any, Dict

from lfx.custom import Component
from lfx.io import FloatInput, MessageInput, Output
from lfx.schema import Message


class GenerationControllerComponent(Component):
    display_name = "Generation Controller"
    description = "Adds generation_mode using a deterministic 0-1 complexity score."
    icon = "sliders"
    name = "GenerationController"

    inputs = [
        MessageInput(
            name="gate_message",
            display_name="Gate Message",
            info="Gate output containing user_query, rag_results, decomposition_used, SC, ES, CP, and decision.",
            required=True,
        ),
        FloatInput(
            name="weight_decomposition",
            display_name="Weight: Decomposition",
            value=0.40,
            required=True,
        ),
        FloatInput(
            name="weight_evidence_gap",
            display_name="Weight: Evidence Gap (1-ES)",
            value=0.30,
            required=True,
        ),
        FloatInput(
            name="weight_conflict_penalty",
            display_name="Weight: Conflict Penalty",
            value=0.20,
            required=True,
        ),
        FloatInput(
            name="weight_slot_gap",
            display_name="Weight: Slot Gap (1-SC)",
            value=0.10,
            required=True,
        ),
        FloatInput(
            name="complexity_threshold",
            display_name="Complexity Threshold",
            value=0.45,
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Controller Message",
            name="controller_message",
            method="build_output",
        ),
    ]

    def _extract_message_text(self, value: Any) -> str:
        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()
        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()
        if hasattr(value, "data") and isinstance(value.data, dict) and "text" in value.data:
            return str(value.data["text"]).strip()
        return str(value).strip()

    def _parse_payload(self, raw_text: str) -> Dict[str, Any]:
        if not raw_text:
            return {}

        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return {}

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _safe_bool(self, value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "1", "yes"}:
                return True
            if v in {"false", "0", "no"}:
                return False
        return default

    def _clamp_01(self, x: float) -> float:
        return max(0.0, min(1.0, x))

    def build_output(self) -> Message:
        gate_raw = self._extract_message_text(self.gate_message)
        gate_payload = self._parse_payload(gate_raw)

        if not gate_payload:
            error = {
                "_error": "Could not parse gate_message",
                "_raw_gate": gate_raw,
            }
            error_text = json.dumps(error, ensure_ascii=False, indent=2)
            self.status = error_text
            return Message(text=error_text, data=error)

        decision = str(gate_payload.get("decision", "")).strip().lower()
        decomposition_used = self._safe_bool(gate_payload.get("decomposition_used", False), default=False)

        SC = self._clamp_01(self._safe_float(gate_payload.get("slot_coverage_weighted", 0.0), default=0.0))
        ES = self._clamp_01(self._safe_float(gate_payload.get("evidence_sufficiency_score", 0.0), default=0.0))
        CP = self._clamp_01(self._safe_float(gate_payload.get("conflict_penalty", 0.0), default=0.0))

        DQ = 1.0 if decomposition_used else 0.0

        w_dq = self._safe_float(self.weight_decomposition, default=0.40)
        w_eg = self._safe_float(self.weight_evidence_gap, default=0.30)
        w_cp = self._safe_float(self.weight_conflict_penalty, default=0.20)
        w_sg = self._safe_float(self.weight_slot_gap, default=0.10)
        threshold = self._clamp_01(self._safe_float(self.complexity_threshold, default=0.45))

        raw_complexity_score = (
            (w_dq * DQ)
            + (w_eg * (1.0 - ES))
            + (w_cp * CP)
            + (w_sg * (1.0 - SC))
        )
        complexity_score = round(self._clamp_01(raw_complexity_score), 6)

        if decision != "accept":
            generation_mode = "normal"
        else:
            generation_mode = "reason" if complexity_score >= threshold else "normal"

        result = dict(gate_payload)
        result["complexity_score"] = complexity_score
        result["complexity_threshold"] = threshold
        result["complexity_weights"] = {
            "decomposition": w_dq,
            "evidence_gap": w_eg,
            "conflict_penalty": w_cp,
            "slot_gap": w_sg,
        }
        result["generation_mode"] = generation_mode

        output_text = json.dumps(result, ensure_ascii=False, indent=2)
        self.status = output_text
        return Message(text=output_text, data=result)