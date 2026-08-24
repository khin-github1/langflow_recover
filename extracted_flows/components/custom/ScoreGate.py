# Recovered Langflow component
# type: ScoreGate
# class: ScoreGateComponent
# used in 6 flow(s): Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup, Cognitive RAG V1.1.0, Cognitive RAG V1.1.0 Clean, Cognitive RAG V1.1.0 Eval
# json path: node.data.node.template.code.value

import ast
import json
from typing import Any, Dict

from lfx.custom import Component
from lfx.io import FloatInput, MessageInput, Output
from lfx.schema import Message


class ScoreGateComponent(Component):
    display_name = "Score Gate"
    description = "Combines analyzer slot coverage and validator ES/CP into a final accept/abstain decision."
    icon = "shield"
    name = "ScoreGate"

    inputs = [
        MessageInput(
            name="analyzer_message",
            display_name="Analyzer Message",
            info="Analyzer output containing slot_coverage_weighted.",
            required=True,
        ),
        MessageInput(
            name="validator_message",
            display_name="Validator Message",
            info="Validator output containing evidence_sufficiency_score, conflict_penalty, top_kept_evidences, and decomposition fields.",
            required=True,
        ),
        FloatInput(
            name="weight_slot_coverage",
            display_name="Weight: Slot Coverage",
            value=0.35,
            required=True,
        ),
        FloatInput(
            name="weight_evidence_sufficiency",
            display_name="Weight: Evidence Sufficiency",
            value=0.50,
            required=True,
        ),
        FloatInput(
            name="weight_conflict_penalty",
            display_name="Weight: Conflict Penalty",
            value=0.15,
            required=True,
        ),
        FloatInput(
            name="decision_threshold",
            display_name="Decision Threshold",
            value=0.65,
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Gate Message",
            name="gate_message",
            method="build_output",
        ),
    ]

    def _extract_message_text(self, value: Any) -> str:
        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()
        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()
        if hasattr(value, "data") and isinstance(value.data, dict):
            if "text" in value.data:
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

    def build_output(self) -> Message:
        analyzer_raw = self._extract_message_text(self.analyzer_message)
        validator_raw = self._extract_message_text(self.validator_message)

        analyzer_payload = self._parse_payload(analyzer_raw)
        validator_payload = self._parse_payload(validator_raw)

        if not analyzer_payload:
            error = {
                "_error": "Could not parse analyzer_message",
                "_raw_analyzer": analyzer_raw,
            }
            error_text = json.dumps(error, ensure_ascii=False, indent=2)
            self.status = error_text
            return Message(text=error_text, data=error)

        if not validator_payload:
            error = {
                "_error": "Could not parse validator_message",
                "_raw_validator": validator_raw,
            }
            error_text = json.dumps(error, ensure_ascii=False, indent=2)
            self.status = error_text
            return Message(text=error_text, data=error)

        slot_coverage_weighted = self._safe_float(
            analyzer_payload.get("slot_coverage_weighted", 0.0),
            default=0.0,
        )
        evidence_sufficiency_score = self._safe_float(
            validator_payload.get("evidence_sufficiency_score", 0.0),
            default=0.0,
        )
        conflict_penalty = self._safe_float(
            validator_payload.get("conflict_penalty", 0.0),
            default=0.0,
        )

        decomposition_used = self._safe_bool(
            validator_payload.get("decomposition_used", False),
            default=False,
        )
        decomposition_type = str(
            validator_payload.get("decomposition_type", "none")
        ).strip() or "none"

        w_sc = self._safe_float(self.weight_slot_coverage, default=0.35)
        w_es = self._safe_float(self.weight_evidence_sufficiency, default=0.50)
        w_cp = self._safe_float(self.weight_conflict_penalty, default=0.15)
        threshold = self._safe_float(self.decision_threshold, default=0.65)

        final_score = (
            (w_sc * slot_coverage_weighted)
            + (w_es * evidence_sufficiency_score)
            - (w_cp * conflict_penalty)
        )
        final_score = round(final_score, 6)

        decision = "accept" if final_score >= threshold else "abstain"

        result = {
            "user_query": validator_payload.get("original_query", ""),
            "decomposition_used": decomposition_used,
            "decomposition_type": decomposition_type,
            "rag_results": validator_payload.get("top_kept_evidences", []),
            "slot_coverage_weighted": round(slot_coverage_weighted, 6),
            "evidence_sufficiency_score": round(evidence_sufficiency_score, 6),
            "conflict_penalty": round(conflict_penalty, 6),
            "weights": {
                "slot_coverage": w_sc,
                "evidence_sufficiency": w_es,
                "conflict_penalty": w_cp,
            },
            "decision_threshold": threshold,
            "final_score": final_score,
            "decision": decision,
        }

        output_text = json.dumps(result, ensure_ascii=False, indent=2)
        self.status = output_text
        return Message(text=output_text, data=result)