# Recovered Langflow component
# type: AnalyzerRefinementMerger
# class: AnalyzerRefinementMerger
# used in 16 flow(s): AIT Cognitive RAG, Cognitive RAG V1.1.5, Cognitive RAG V1.2.0 (1), Cognitive RAG V1.2.0 AIT CR, Cognitive RAG V1.2.0 AIT Loose, Cognitive RAG V1.2.0 AIT Normal, Cognitive RAG V1.2.0 AIT RR, Cognitive RAG V1.2.0 AIT Reason ...
# json path: node.data.node.template.code.value

import ast
import json
from typing import Any, Dict

from lfx.custom import Component
from lfx.io import MessageInput, Output
from lfx.schema import Message


class AnalyzerRefinementMerger(Component):
    display_name = "Analyzer + Refinement Merger"
    description = (
        "Merges the Clarification+Analyzer payload with the Refinement LLM output "
        "into one JSON payload for downstream query selection."
    )
    icon = "merge"
    name = "AnalyzerRefinementMerger"

    inputs = [
        MessageInput(
            name="analyzer_message",
            display_name="Analyzer/Merged Message",
            info="Message from ClarificationAnalyzerMerger",
            required=True,
        ),
        MessageInput(
            name="refinement_message",
            display_name="Refinement Message",
            info="Message from Refinement LLM node",
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

    def _extract_raw_text(self, value: Any) -> str:
        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()
        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()
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

    def build_output(self) -> Message:
        analyzer_raw = self._extract_raw_text(self.analyzer_message)
        refinement_raw = self._extract_raw_text(self.refinement_message)

        analyzer_payload = self._parse_payload(analyzer_raw)
        refinement_payload = self._parse_payload(refinement_raw)

        if not isinstance(analyzer_payload, dict) or not analyzer_payload:
            error_result = {
                "_error": "Could not parse analyzer_message into a dictionary payload.",
                "_raw_analyzer_input": analyzer_raw,
            }
            error_text = json.dumps(error_result, ensure_ascii=False, indent=2)
            self.status = error_text
            return Message(text=error_text, data=error_result)

        if not isinstance(refinement_payload, dict) or not refinement_payload:
            error_result = {
                "_error": "Could not parse refinement_message into a dictionary payload.",
                "_raw_refinement_input": refinement_raw,
            }
            error_text = json.dumps(error_result, ensure_ascii=False, indent=2)
            self.status = error_text
            return Message(text=error_text, data=error_result)

        merged = dict(analyzer_payload)

        merged["refinement"] = {
            "enabled": bool(refinement_payload.get("enabled", True)),
            "used": bool(refinement_payload.get("used", False)),
            "refined_query": str(refinement_payload.get("refined_query", "")).strip(),
            "refinement_type": str(refinement_payload.get("refinement_type", "none")).strip() or "none",
            "used_slots": refinement_payload.get("used_slots", []) if isinstance(refinement_payload.get("used_slots", []), list) else [],
            "llm_metrics": refinement_payload.get("llm_metrics", {}) if isinstance(refinement_payload.get("llm_metrics", {}), dict) else {},
        }

        output_text = json.dumps(merged, ensure_ascii=False, indent=2)

        self.status = (
            f"refine_used={int(bool(merged['refinement'].get('used', False)))} "
            f"type={merged['refinement'].get('refinement_type', 'none')}"
        )

        return Message(text=output_text, data=merged)