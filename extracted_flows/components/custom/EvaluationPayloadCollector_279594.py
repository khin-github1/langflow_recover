# Recovered Langflow component
# type: EvaluationPayloadCollector
# class: EvaluationPayloadCollector
# used in 2 flow(s): Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.1.0
# json path: node.data.node.template.code.value

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langflow.custom import Component
from langflow.io import BoolInput, DataInput, IntInput, MessageTextInput, Output
from langflow.schema.data import Data


class EvaluationPayloadCollector(Component):
    display_name = "Evaluation Payload Collector"
    description = (
        "Collects analyzer / retrieval / generation / verification / decision artifacts, "
        "keeps only needed fields, and outputs one payload for batch evaluation."
    )
    icon = "package"
    name = "EvaluationPayloadCollector"

    inputs = [
        MessageTextInput(name="user_text", display_name="User Text", required=True),
        MessageTextInput(name="assistant_text", display_name="Assistant Text", required=False),

        DataInput(name="analyzer_data", display_name="Analyzer Data", required=False),

        MessageTextInput(name="clarification_raw", display_name="Clarification Raw", required=False),
        MessageTextInput(name="decomposer_raw", display_name="Decomposer Raw", required=False),
        MessageTextInput(name="rag_raw", display_name="RAG Raw", required=False),
        MessageTextInput(name="validator_raw", display_name="Validator Raw", required=False),
        MessageTextInput(name="sufficiency_gate_raw", display_name="Sufficiency Gate Raw", required=False),

        MessageTextInput(name="generation_mode_raw", display_name="Generation Mode Raw", required=False),
        MessageTextInput(name="generator_prompt_raw", display_name="Generator Prompt Raw", required=False),
        MessageTextInput(name="generator_raw", display_name="Generator Raw", required=False),

        MessageTextInput(name="verification_raw", display_name="Verification Raw", required=False),
        MessageTextInput(name="decision_raw", display_name="Decision Raw", required=False),

        BoolInput(
            name="include_retrieval_chunks",
            display_name="Include Retrieval Chunks",
            value=True,
            advanced=True,
        ),
        IntInput(
            name="max_retrieval_chunks",
            display_name="Max Retrieval Chunks",
            value=20,
            advanced=True,
        ),
        BoolInput(
            name="include_raw_blobs",
            display_name="Include Raw Blobs",
            value=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Payload", name="payload", method="build_payload"),
    ]

    # -------------------------
    # Generic parsing helpers
    # -------------------------
    @staticmethod
    def _now_utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    def _parse_json_obj(self, text: str) -> Dict[str, Any]:
        text = self._strip_fences(text)

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        return {}

    def _safe_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if hasattr(value, "text"):
            return str(getattr(value, "text", "") or "").strip()
        return str(value).strip()

    def _safe_dict_from_data(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "data") and isinstance(value.data, dict):
            return value.data
        if isinstance(value, dict):
            return value
        return {}

    def _parse_message_like(self, raw: Any) -> Dict[str, Any]:
        """
        Parses inputs that may be:
        - plain text
        - direct JSON text
        - full Ollama envelope with message.content
        - direct already-parsed dict
        """
        if isinstance(raw, dict):
            obj = raw
        else:
            text = self._safe_text(raw)
            obj = self._parse_json_obj(text)

        if not obj:
            return {
                "raw_obj": {},
                "content_text": self._safe_text(raw),
                "content_json": {},
                "metrics": {},
            }

        content_text = ""
        content_json: Dict[str, Any] = {}

        msg = obj.get("message")
        if isinstance(msg, dict):
            content_text = str(msg.get("content", "") or "").strip()
            content_json = self._parse_json_obj(content_text) if content_text else {}
        else:
            content_text = self._safe_text(raw)

        metrics = {
            "model": obj.get("model"),
            "created_at": obj.get("created_at"),
            "done": obj.get("done"),
            "done_reason": obj.get("done_reason"),
            "total_duration": obj.get("total_duration"),
            "load_duration": obj.get("load_duration"),
            "prompt_eval_count": obj.get("prompt_eval_count"),
            "prompt_eval_duration": obj.get("prompt_eval_duration"),
            "eval_count": obj.get("eval_count"),
            "eval_duration": obj.get("eval_duration"),
        }

        return {
            "raw_obj": obj,
            "content_text": content_text,
            "content_json": content_json,
            "metrics": metrics,
        }

    # -------------------------
    # Selective extractors
    # -------------------------
    def _select_analyzer(self, analyzer: Dict[str, Any]) -> Dict[str, Any]:
        if not analyzer:
            return {}

        return {
            "original_question": analyzer.get("original_question"),
            "intent_type": analyzer.get("intent_type"),
            "request_type": analyzer.get("request_type"),
            "topic": analyzer.get("topic"),
            "expected_general_slots": analyzer.get("expected_general_slots", []),
            "expected_critical_slots": analyzer.get("expected_critical_slots", []),
            "required_general_slots": analyzer.get("required_general_slots", []),
            "required_critical_slots": analyzer.get("required_critical_slots", []),
            "missing_slots": analyzer.get("missing_slots", {"general": [], "critical": []}),
            "slot_coverage_general": analyzer.get("slot_coverage_general"),
            "slot_coverage_critical": analyzer.get("slot_coverage_critical"),
            "slot_coverage_weighted": analyzer.get("slot_coverage_weighted"),
            "counts": analyzer.get("counts", {}),
            "resolved_slots": analyzer.get("resolved_slots", {}),
            "refinement": analyzer.get("refinement", {}),
            "clarification_tracking": analyzer.get("clarification_tracking", {}),
            "clarification_decision": analyzer.get("clarification_decision", {}),
            "clarification_needed": analyzer.get("clarification_needed"),
            "llm_metrics": analyzer.get("llm_metrics", {}),
        }

    def _select_clarification(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        if not parsed:
            return {}
        return {
            "text": parsed.get("content_text", ""),
            "metrics": parsed.get("metrics", {}),
        }

    def _select_decomposer(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        cjson = parsed.get("content_json", {}) or {}
        return {
            "original_query": cjson.get("original_query"),
            "decomposition_used": cjson.get("decomposition_used"),
            "decomposition_type": cjson.get("decomposition_type"),
            "sub_questions": cjson.get("sub_questions", []),
            "final_query_count": cjson.get("final_query_count"),
            "notes": cjson.get("notes"),
            "metrics": parsed.get("metrics", {}),
        }

    def _flatten_retrieval_chunks(self, rag: Dict[str, Any], max_chunks: int) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for group in rag.get("results_by_query", []) or []:
            query_index = group.get("query_index")
            query_text = group.get("query_text")
            for item in group.get("results", []) or []:
                chunks.append(
                    {
                        "query_index": query_index,
                        "query_text": query_text,
                        "rank": item.get("rank"),
                        "score": item.get("score"),
                        "source": (item.get("metadata") or {}).get("source"),
                        "page_content": item.get("page_content"),
                    }
                )
                if len(chunks) >= max_chunks:
                    return chunks
        return chunks

    def _select_rag(self, rag: Dict[str, Any]) -> Dict[str, Any]:
        if not rag:
            return {}

        groups_summary = []
        for group in rag.get("results_by_query", []) or []:
            groups_summary.append(
                {
                    "query_index": group.get("query_index"),
                    "query_text": group.get("query_text"),
                    "n_returned": group.get("n_returned"),
                    "retrieval_wall_time_s": group.get("retrieval_wall_time_s"),
                    "top_sources": [
                        (r.get("metadata") or {}).get("source")
                        for r in (group.get("results", []) or [])[:5]
                    ],
                }
            )

        return {
            "original_query": rag.get("original_query"),
            "decomposition_used": rag.get("decomposition_used"),
            "decomposition_type": rag.get("decomposition_type"),
            "notes": rag.get("notes"),
            "final_query_count": rag.get("final_query_count"),
            "total_results_returned": rag.get("total_results_returned"),
            "results_by_query_summary": groups_summary,
        }

    def _select_validator(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        cjson = parsed.get("content_json", {}) or {}
        top_kept = []
        for item in cjson.get("top_kept_evidences", []) or []:
            top_kept.append(
                {
                    "global_retrieval_index": item.get("global_retrieval_index"),
                    "query_index": item.get("query_index"),
                    "query_text": item.get("query_text"),
                    "validation_score": item.get("validation_score"),
                    "is_answer_bearing": item.get("is_answer_bearing"),
                    "reason_short": item.get("reason_short"),
                }
            )

        return {
            "original_query": cjson.get("original_query"),
            "decomposition_used": cjson.get("decomposition_used"),
            "decomposition_type": cjson.get("decomposition_type"),
            "top_kept_evidences": top_kept,
            "evidence_sufficiency_score": cjson.get("evidence_sufficiency_score"),
            "conflict_penalty": cjson.get("conflict_penalty"),
            "rag_valid": cjson.get("rag_valid"),
            "decision_hint": cjson.get("decision_hint"),
            "notes": cjson.get("notes"),
            "metrics": parsed.get("metrics", {}),
        }

    def _select_sufficiency_gate(self, gate: Dict[str, Any]) -> Dict[str, Any]:
        if not gate:
            return {}

        return {
            "slot_coverage_weighted": gate.get("slot_coverage_weighted"),
            "evidence_sufficiency_score": gate.get("evidence_sufficiency_score"),
            "conflict_penalty": gate.get("conflict_penalty"),
            "weights": gate.get("weights", {}),
            "decision_threshold": gate.get("decision_threshold"),
            "final_score": gate.get("final_score"),
            "decision": gate.get("decision"),
            "rag_results_count": len(gate.get("rag_results", []) or []),
        }

    def _select_generation_mode(self, raw: Any) -> str:
        text = self._safe_text(raw)
        if text.lower() in {"normal", "reason"}:
            return text.lower()

        obj = self._parse_json_obj(text)
        if isinstance(obj, dict):
            mode = str(obj.get("generation_mode", "")).strip().lower()
            if mode in {"normal", "reason"}:
                return mode
        return text

    def _select_generator(self, parsed: Dict[str, Any], assistant_text: str, generation_mode: str, prompt_text: str) -> Dict[str, Any]:
        answer = assistant_text or parsed.get("content_text", "")
        return {
            "generation_mode": generation_mode,
            "prompt": prompt_text,
            "answer": answer,
            "metrics": parsed.get("metrics", {}),
        }

    def _select_verification(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        cjson = parsed.get("content_json", {}) or {}
        claims = cjson.get("claims", []) or []
        slot_coverage = cjson.get("slot_coverage", {}) or {}

        return {
            "claims": claims,
            "slot_coverage": slot_coverage,
            "claims_count": len(claims),
            "critical_slots_count": len((slot_coverage.get("critical", []) or [])) if isinstance(slot_coverage, dict) else 0,
            "metrics": parsed.get("metrics", {}),
        }

    def _select_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        if not decision:
            return {}

        return {
            "decision": decision.get("decision"),
            "generation_mode_in": decision.get("generation_mode_in"),
            "next_generation_mode": decision.get("next_generation_mode"),
            "general_weight": decision.get("general_weight"),
            "critical_weight": decision.get("critical_weight"),
            "final_support_threshold": decision.get("final_support_threshold"),
            "total_claims": decision.get("total_claims"),
            "total_general_claims": decision.get("total_general_claims"),
            "total_critical_claims": decision.get("total_critical_claims"),
            "raw_mean_support_score": decision.get("raw_mean_support_score"),
            "weighted_score_sum": decision.get("weighted_score_sum"),
            "weighted_total": decision.get("weighted_total"),
            "final_support_score": decision.get("final_support_score"),
            "general_slots_total": decision.get("general_slots_total"),
            "general_slots_answered": decision.get("general_slots_answered"),
            "general_slots_missing": decision.get("general_slots_missing"),
            "critical_slots_total": decision.get("critical_slots_total"),
            "critical_slots_answered": decision.get("critical_slots_answered"),
            "critical_slots_missing": decision.get("critical_slots_missing"),
            "claims_per_slot": decision.get("claims_per_slot", {}),
        }

    # -------------------------
    # Main
    # -------------------------
    def build_payload(self) -> Data:
        analyzer = self._safe_dict_from_data(self.analyzer_data)

        clarification_p = self._parse_message_like(self.clarification_raw)
        decomposer_p = self._parse_message_like(self.decomposer_raw)
        rag_obj = self._parse_json_obj(self._safe_text(self.rag_raw))
        validator_p = self._parse_message_like(self.validator_raw)
        suff_gate_obj = self._parse_json_obj(self._safe_text(self.sufficiency_gate_raw))
        generator_p = self._parse_message_like(self.generator_raw)
        verification_p = self._parse_message_like(self.verification_raw)
        decision_obj = self._parse_json_obj(self._safe_text(self.decision_raw))

        user_text = self._safe_text(self.user_text)
        assistant_text = self._safe_text(self.assistant_text)
        generation_mode = self._select_generation_mode(self.generation_mode_raw)
        prompt_text = self._safe_text(self.generator_prompt_raw)

        analyzer_small = self._select_analyzer(analyzer)
        clarification_small = self._select_clarification(clarification_p) if self._safe_text(self.clarification_raw) else {}
        decomposer_small = self._select_decomposer(decomposer_p) if self._safe_text(self.decomposer_raw) else {}
        rag_small = self._select_rag(rag_obj) if rag_obj else {}
        validator_small = self._select_validator(validator_p) if self._safe_text(self.validator_raw) else {}
        suff_gate_small = self._select_sufficiency_gate(suff_gate_obj) if suff_gate_obj else {}
        generator_small = self._select_generator(generator_p, assistant_text, generation_mode, prompt_text)
        verification_small = self._select_verification(verification_p) if self._safe_text(self.verification_raw) else {}
        decision_small = self._select_decision(decision_obj) if decision_obj else {}

        retrieval_chunks = []
        if bool(self.include_retrieval_chunks) and rag_obj:
            retrieval_chunks = self._flatten_retrieval_chunks(rag_obj, int(self.max_retrieval_chunks or 20))

        payload: Dict[str, Any] = {
            # required by your existing BatchRunLangFlow._is_payload()
            "timestamp_utc": self._now_utc_iso(),
            "user_text": user_text,
            "assistant_text": generator_small.get("answer", assistant_text),

            # runner also reads metrics.generation / metrics.retrieval and retrieval_chunks
            "metrics": {
                "analyzer": analyzer_small.get("llm_metrics", {}),
                "retrieval": {
                    "decomposition_used": rag_small.get("decomposition_used"),
                    "decomposition_type": rag_small.get("decomposition_type"),
                    "final_query_count": rag_small.get("final_query_count"),
                    "total_results_returned": rag_small.get("total_results_returned"),
                    "validator_evidence_sufficiency_score": validator_small.get("evidence_sufficiency_score"),
                    "validator_conflict_penalty": validator_small.get("conflict_penalty"),
                    "validator_rag_valid": validator_small.get("rag_valid"),
                    "sufficiency_gate_final_score": suff_gate_small.get("final_score"),
                    "sufficiency_gate_decision": suff_gate_small.get("decision"),
                },
                "generation": generator_small.get("metrics", {}),
                "verification": verification_small.get("metrics", {}),
                "decision": {
                    "decision": decision_small.get("decision"),
                    "final_support_score": decision_small.get("final_support_score"),
                    "next_generation_mode": decision_small.get("next_generation_mode"),
                },
            },

            "retrieval_chunks": retrieval_chunks,

            # compact stage artifacts
            "analyzer": analyzer_small,
            "clarification": clarification_small,
            "decomposer": decomposer_small,
            "rag": rag_small,
            "validator": validator_small,
            "sufficiency_gate": suff_gate_small,
            "generation_mode": generation_mode,
            "generator_prompt": prompt_text,
            "generator": generator_small,
            "verification": verification_small,
            "decision_trace": decision_small,
        }

        if bool(self.include_raw_blobs):
            payload["raw_blobs"] = {
                "clarification_raw": self._safe_text(self.clarification_raw),
                "decomposer_raw": self._safe_text(self.decomposer_raw),
                "rag_raw": self._safe_text(self.rag_raw),
                "validator_raw": self._safe_text(self.validator_raw),
                "sufficiency_gate_raw": self._safe_text(self.sufficiency_gate_raw),
                "generator_raw": self._safe_text(self.generator_raw),
                "verification_raw": self._safe_text(self.verification_raw),
                "decision_raw": self._safe_text(self.decision_raw),
            }

        self.status = (
            f"payload built | decision={decision_small.get('decision')} "
            f"| claims={verification_small.get('claims_count', 0)} "
            f"| retrieval_chunks={len(retrieval_chunks)}"
        )

        return Data(
            data=payload,
            text_key="assistant_text",
            default_value="",
        )