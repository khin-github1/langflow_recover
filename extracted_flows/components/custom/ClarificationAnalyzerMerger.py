# Recovered Langflow component
# type: ClarificationAnalyzerMerger
# class: ClarificationAnalyzerMerger
# used in 16 flow(s): AIT Cognitive RAG, Cognitive RAG V1.1.5, Cognitive RAG V1.2.0 (1), Cognitive RAG V1.2.0 AIT CR, Cognitive RAG V1.2.0 AIT Loose, Cognitive RAG V1.2.0 AIT Normal, Cognitive RAG V1.2.0 AIT RR, Cognitive RAG V1.2.0 AIT Reason ...
# json path: node.data.node.template.code.value

from __future__ import annotations

import ast
import json
from typing import Any, Dict

from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema.message import Message


class ClarificationAnalyzerMerger(Component):
    display_name = "Clarification + Analyzer Merger"
    description = (
        "Merges analyzer output with clarification-tracker output. "
        "If clarification was asked in the last AI turn, clarification_needed is forced to false."
    )
    icon = "merge"
    name = "ClarificationAnalyzerMerger"

    inputs = [
        DataInput(
            name="analyzer_data",
            display_name="Analyzer Data",
            required=True,
        ),
        DataInput(
            name="clarification_tracking_data",
            display_name="Clarification Tracking Data",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Merged Message",
            name="merged_message",
            method="build_output",
        ),
    ]

    def _extract_payload(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        # Data input
        if hasattr(value, "data") and isinstance(value.data, dict):
            return value.data

        # Message input fallback
        if hasattr(value, "text") and value.text:
            txt = str(value.text).strip()
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return {}

    def build_output(self) -> Message:
        analyzer = self._extract_payload(self.analyzer_data)
        tracker = self._extract_payload(self.clarification_tracking_data)

        if not isinstance(analyzer, dict):
            analyzer = {}
        if not isinstance(tracker, dict):
            tracker = {}

        clarification_asked_in_last_ai_turn = bool(
            tracker.get("clarification_asked_in_last_ai_turn", False)
        )
        matched_clarification_type = str(
            tracker.get("matched_clarification_type", "none")
        ).strip() or "none"

        merged = dict(analyzer)

        merged["clarification_tracking"] = {
            "clarification_asked_in_last_ai_turn": clarification_asked_in_last_ai_turn,
            "matched_clarification_type": matched_clarification_type,
        }

        if clarification_asked_in_last_ai_turn:
            merged["clarification_needed"] = False

            clarification_decision = merged.get("clarification_decision", {})
            if not isinstance(clarification_decision, dict):
                clarification_decision = {}

            clarification_decision["clarification_needed"] = False
            clarification_decision["suppressed_because_last_ai_turn_was_clarification"] = True
            clarification_decision["last_turn_clarification_type"] = matched_clarification_type

            merged["clarification_decision"] = clarification_decision
        else:
            clarification_decision = merged.get("clarification_decision", {})
            if isinstance(clarification_decision, dict):
                clarification_decision["suppressed_because_last_ai_turn_was_clarification"] = False
                clarification_decision["last_turn_clarification_type"] = matched_clarification_type
                merged["clarification_decision"] = clarification_decision

        self.status = (
            f"last_turn_clar={int(clarification_asked_in_last_ai_turn)} "
            f"type={matched_clarification_type} "
            f"clar_needed={int(bool(merged.get('clarification_needed', False)))}"
        )

        output_text = json.dumps(merged, ensure_ascii=False, indent=2)
        return Message(text=output_text, data=merged)