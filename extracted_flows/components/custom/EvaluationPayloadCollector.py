# Recovered Langflow component
# type: EvaluationPayloadCollector
# class: EvaluationPayloadCollector
# used in 1 flow(s): Cognitive RAG V1.1.0 Clean
# json path: node.data.node.template.code.value

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from langflow.custom import Component
from langflow.io import BoolInput, DataInput, IntInput, MessageTextInput, Output
from langflow.schema.data import Data


class EvaluationPayloadCollector(Component):
    display_name = "Evaluation Payload Collector"
    description = (
        "Collects evaluation payloads from a mixed Langflow graph where raw LLM envelopes are Data "
        "and downstream routing/output branches are Message."
    )
    icon = "package"
    name = "EvaluationPayloadCollector"

    inputs = [
        MessageTextInput(name="user_text", display_name="User Text", required=True),

        # Final terminal outputs (replace old final_output_text with these 4)
        MessageTextInput(
            name="attempt1_accept",
            display_name="Attempt 1 Accept",
            info="Connect attempt 1 accepted terminal output here.",
            required=False,
        ),
        MessageTextInput(
            name="attempt1_reject",
            display_name="Attempt 1 Reject",
            info="Connect attempt 1 rejected terminal output here.",
            required=False,
        ),
        MessageTextInput(
            name="attempt2_accept",
            display_name="Attempt 2 Accept",
            info="Connect attempt 2 accepted terminal output here.",
            required=False,
        ),
        MessageTextInput(
            name="attempt2_reject",
            display_name="Attempt 2 Reject",
            info="Connect attempt 2 rejected terminal output here.",
            required=False,
        ),

        # Data: raw analyzer / LLM envelopes
        DataInput(name="analyzer_data", display_name="Analyzer Data", required=False),
        DataInput(name="clarification_raw_data", display_name="Clarification Raw Data", required=False),
        DataInput(name="decomposer_raw_data", display_name="Decomposer Raw Data", required=False),
        DataInput(name="validator_raw_data", display_name="Validator Raw Data", required=False),
        DataInput(name="attempt1_generator_raw_data", display_name="Attempt 1 Generator Raw Data", required=False),
        DataInput(name="attempt1_verification_raw_data", display_name="Attempt 1 Verification Raw Data", required=False),
        DataInput(name="attempt2_generator_raw_data", display_name="Attempt 2 Generator Raw Data", required=False),
        DataInput(name="attempt2_verification_raw_data", display_name="Attempt 2 Verification Raw Data", required=False),

        # Message: normal JSON/text branches
        MessageTextInput(name="rag_message", display_name="RAG Message", required=False),
        MessageTextInput(name="sufficiency_gate_message", display_name="Sufficiency Gate Message", required=False),
        MessageTextInput(
            name="pre_generation_terminal_message",
            display_name="Pre-generation Terminal Message",
            info="Connect the merged pre-generation rejection/clarification terminal output here.",
            required=False,
        ),

        MessageTextInput(name="attempt1_generation_mode_message", display_name="Attempt 1 Generation Mode", required=False),
        MessageTextInput(name="attempt1_prompt_message", display_name="Attempt 1 Prompt Message", required=False),
        MessageTextInput(name="attempt1_decision_message", display_name="Attempt 1 Decision Message", required=False),

        MessageTextInput(name="attempt2_generation_mode_message", display_name="Attempt 2 Generation Mode", required=False),
        MessageTextInput(name="attempt2_prompt_message", display_name="Attempt 2 Prompt Message", required=False),
        MessageTextInput(name="attempt2_decision_message", display_name="Attempt 2 Decision Message", required=False),

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
    # Helpers
    # -------------------------
    @staticmethod
    def _now_utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if hasattr(value, "text"):
            return str(getattr(value, "text", "") or "").strip()
        return str(value).strip()

    @staticmethod
    def _safe_dict_from_data(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "data") and isinstance(value.data, dict):
            return value.data
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    def _parse_json_obj(self, text: str) -> Dict[str, Any]:
        text = self._strip_fences(text)
        if not text:
            return {}

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

    def _parse_message_json(self, raw: Any) -> Dict[str, Any]:
        return self._parse_json_obj(self._safe_text(raw))

    def _parse_data_envelope(self, raw: Any) -> Dict[str, Any]:
        """
        Handles Data outputs from raw LLM nodes.
        Works for:
        - direct raw Ollama envelope stored in Data.data
        - custom component envelope with keys like text/raw/raw_llm_output/metrics
        - verifier envelope with verifier_json already parsed
        """
        data = self._safe_dict_from_data(raw)
        if not data:
            return {
                "data": {},
                "raw_obj": {},
                "content_text": "",
                "content_json": {},
                "metrics": {},
            }

        raw_obj = {}
        if isinstance(data.get("raw"), dict):
            raw_obj = data.get("raw", {})
        elif isinstance(data, dict):
            raw_obj = data

        content_text = ""
        content_json: Dict[str, Any] = {}

        # Best-case: verifier already parsed
        if isinstance(data.get("verifier_json"), dict):
            content_json = data.get("verifier_json", {})
            content_text = str(data.get("text", "") or data.get("raw_llm_output", "")).strip()

        # Usual custom envelope form
        elif isinstance(raw_obj.get("message"), dict):
            content_text = str((raw_obj.get("message") or {}).get("content", "") or "").strip()
            content_json = self._parse_json_obj(content_text) if content_text else {}

        # Fallback text fields
        elif data.get("text"):
            content_text = str(data.get("text", "") or "").strip()
            content_json = self._parse_json_obj(content_text) if content_text else {}
        elif data.get("raw_llm_output"):
            content_text = str(data.get("raw_llm_output", "") or "").strip()
            content_json = self._parse_json_obj(content_text) if content_text else {}

        metrics = {}
        if isinstance(data.get("metrics"), dict):
            metrics = data.get("metrics", {})
        else:
            metrics = {
                "model": raw_obj.get("model"),
                "created_at": raw_obj.get("created_at"),
                "done": raw_obj.get("done"),
                "done_reason": raw_obj.get("done_reason"),
                "total_duration": raw_obj.get("total_duration"),
                "load_duration": raw_obj.get("load_duration"),
                "prompt_eval_count": raw_obj.get("prompt_eval_count"),
                "prompt_eval_duration": raw_obj.get("prompt_eval_duration"),
                "eval_count": raw_obj.get("eval_count"),
                "eval_duration": raw_obj.get("eval_duration"),
            }

        return {
            "data": data,
            "raw_obj": raw_obj,
            "content_text": content_text,
            "content_json": content_json,
            "metrics": metrics,
        }

    def _resolve_final_output_text(self) -> Dict[str, str]:
        """
        Priority:
        1) attempt2_accept
        2) attempt2_reject
        3) attempt1_accept
        4) attempt1_reject
        """
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

    # -------------------------
    # Selectors
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

    def _select_simple_llm_text(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
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

    def _select_rag(self, rag: Dict[str, Any]) -> Dict[str, Any]:
        if not rag:
            return {}

        groups = []
        for group in rag.get("results_by_query", []) or []:
            groups.append(
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
            "results_by_query_summary": groups,
        }

    def _flatten_retrieval_chunks(self, rag: Dict[str, Any], max_chunks: int) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for group in rag.get("results_by_query", []) or []:
            for item in group.get("results", []) or []:
                chunks.append(
                    {
                        "query_index": group.get("query_index"),
                        "query_text": group.get("query_text"),
                        "rank": item.get("rank"),
                        "score": item.get("score"),
                        "source": (item.get("metadata") or {}).get("source"),
                        "page_content": item.get("page_content"),
                    }
                )
                if len(chunks) >= max_chunks:
                    return chunks
        return chunks

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
        if obj:
            mode = str(obj.get("generation_mode", "")).strip().lower()
            if mode in {"normal", "reason"}:
                return mode

        return text

    def _select_generator(self, parsed: Dict[str, Any], generation_mode: str, prompt: str) -> Dict[str, Any]:
        answer = parsed.get("content_text", "") if parsed else ""
        return {
            "generation_mode": generation_mode,
            "prompt": prompt,
            "answer": answer,
            "metrics": parsed.get("metrics", {}) if parsed else {},
        }

    def _extract_verification_json(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        if not parsed:
            return {}

        data = parsed.get("data", {}) or {}
        raw_obj = parsed.get("raw_obj", {}) or {}
        content_json = parsed.get("content_json", {}) or {}

        if isinstance(data.get("verifier_json"), dict):
            return data.get("verifier_json", {})

        if "claims" in raw_obj or "slot_coverage" in raw_obj:
            return raw_obj

        if "claims" in content_json or "slot_coverage" in content_json:
            return content_json

        return {}

    def _select_verification(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        vjson = self._extract_verification_json(parsed)
        claims = vjson.get("claims", []) if isinstance(vjson, dict) else []
        slot_coverage = vjson.get("slot_coverage", {}) if isinstance(vjson, dict) else {}

        return {
            "claims": claims or [],
            "slot_coverage": slot_coverage or {},
            "claims_count": len(claims or []),
            "critical_slots_count": len((slot_coverage.get("critical", []) or [])) if isinstance(slot_coverage, dict) else 0,
            "metrics": parsed.get("metrics", {}) if parsed else {},
        }

    def _select_decision(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        if not obj:
            return {}

        return {
            "decision": obj.get("decision"),
            "generation_mode_in": obj.get("generation_mode_in"),
            "next_generation_mode": obj.get("next_generation_mode"),
            "general_weight": obj.get("general_weight"),
            "critical_weight": obj.get("critical_weight"),
            "final_support_threshold": obj.get("final_support_threshold"),
            "total_claims": obj.get("total_claims"),
            "total_general_claims": obj.get("total_general_claims"),
            "total_critical_claims": obj.get("total_critical_claims"),
            "raw_mean_support_score": obj.get("raw_mean_support_score"),
            "weighted_score_sum": obj.get("weighted_score_sum"),
            "weighted_total": obj.get("weighted_total"),
            "final_support_score": obj.get("final_support_score"),
            "general_slots_total": obj.get("general_slots_total"),
            "general_slots_answered": obj.get("general_slots_answered"),
            "general_slots_missing": obj.get("general_slots_missing"),
            "critical_slots_total": obj.get("critical_slots_total"),
            "critical_slots_answered": obj.get("critical_slots_answered"),
            "critical_slots_missing": obj.get("critical_slots_missing"),
            "claims_per_slot": obj.get("claims_per_slot", {}),
            "claim_rows": obj.get("claim_rows", []),
            "general_slot_rows": obj.get("general_slot_rows", []),
            "critical_slot_rows": obj.get("critical_slot_rows", []),
        }

    def _build_attempt(
        self,
        generation_mode_message: Any,
        prompt_message: Any,
        generator_raw_data: Any,
        verification_raw_data: Any,
        decision_message: Any,
    ) -> Dict[str, Any]:
        generation_mode = self._select_generation_mode(generation_mode_message)
        prompt = self._safe_text(prompt_message)

        generator_parsed = self._parse_data_envelope(generator_raw_data)
        verification_parsed = self._parse_data_envelope(verification_raw_data)
        decision_obj = self._parse_message_json(decision_message)

        generator_small = self._select_generator(generator_parsed, generation_mode, prompt)
        verification_small = self._select_verification(verification_parsed)
        decision_small = self._select_decision(decision_obj)

        used = bool(
            generation_mode
            or prompt
            or generator_small.get("answer")
            or verification_small.get("claims_count", 0) > 0
            or decision_small.get("decision")
        )

        return {
            "used": used,
            "generation_mode": generation_mode,
            "prompt": prompt,
            "generator": generator_small,
            "verification": verification_small,
            "decision": decision_small,
        }

    # -------------------------
    # Main
    # -------------------------
    def build_payload(self) -> Data:
        analyzer_small = self._select_analyzer(self._safe_dict_from_data(self.analyzer_data))

        clarification_small = self._select_simple_llm_text(self._parse_data_envelope(self.clarification_raw_data)) if self._safe_dict_from_data(self.clarification_raw_data) else {}
        decomposer_small = self._select_decomposer(self._parse_data_envelope(self.decomposer_raw_data)) if self._safe_dict_from_data(self.decomposer_raw_data) else {}

        rag_obj = self._parse_message_json(self.rag_message)
        rag_small = self._select_rag(rag_obj) if rag_obj else {}

        validator_small = self._select_validator(self._parse_data_envelope(self.validator_raw_data)) if self._safe_dict_from_data(self.validator_raw_data) else {}
        suff_gate_small = self._select_sufficiency_gate(self._parse_message_json(self.sufficiency_gate_message))

        attempt_1 = self._build_attempt(
            self.attempt1_generation_mode_message,
            self.attempt1_prompt_message,
            self.attempt1_generator_raw_data,
            self.attempt1_verification_raw_data,
            self.attempt1_decision_message,
        )

        attempt_2 = self._build_attempt(
            self.attempt2_generation_mode_message,
            self.attempt2_prompt_message,
            self.attempt2_generator_raw_data,
            self.attempt2_verification_raw_data,
            self.attempt2_decision_message,
        )

        pre_generation_terminal_message = self._safe_text(self.pre_generation_terminal_message)

        resolved_final = self._resolve_final_output_text()
        final_output_source = resolved_final["source"]
        final_output_text = resolved_final["text"]

        retry_used = bool(attempt_2.get("used"))
        attempt_count = 2 if retry_used else (1 if attempt_1.get("used") else 0)
        final_attempt = attempt_2 if retry_used else attempt_1

        final_decision = (
            (attempt_2.get("decision") or {}).get("decision")
            or (attempt_1.get("decision") or {}).get("decision")
            or (suff_gate_small.get("decision") if pre_generation_terminal_message else "")
            or ""
        )

        if pre_generation_terminal_message and not attempt_1.get("used") and not attempt_2.get("used"):
            terminal_stage = "pre_generation"
        elif final_decision:
            terminal_stage = "post_generation"
        else:
            terminal_stage = ""

        assistant_text = (
            final_output_text
            or (final_attempt.get("generator") or {}).get("answer", "")
            or pre_generation_terminal_message
        )

        final_generation_mode = final_attempt.get("generation_mode", "") if final_attempt.get("used") else ""

        final_block = {
            "final_decision": final_decision,
            "retry_used": retry_used,
            "attempt_count": attempt_count,
            "terminal_stage": terminal_stage,
            "final_generation_mode": final_generation_mode,
            "final_answer_shown": assistant_text,
            "final_output_source": final_output_source,
            "terminal_outputs": {
                "attempt1_accept": self._safe_text(self.attempt1_accept),
                "attempt1_reject": self._safe_text(self.attempt1_reject),
                "attempt2_accept": self._safe_text(self.attempt2_accept),
                "attempt2_reject": self._safe_text(self.attempt2_reject),
            },
        }

        retrieval_chunks = []
        if bool(self.include_retrieval_chunks) and rag_obj:
            retrieval_chunks = self._flatten_retrieval_chunks(rag_obj, int(self.max_retrieval_chunks or 20))

        payload: Dict[str, Any] = {
            # required by your batch runner
            "timestamp_utc": self._now_utc_iso(),
            "user_text": self._safe_text(self.user_text),
            "assistant_text": assistant_text,

            # top-level summary
            "final_decision": final_block["final_decision"],
            "retry_used": final_block["retry_used"],
            "attempt_count": final_block["attempt_count"],
            "terminal_stage": final_block["terminal_stage"],
            "final_generation_mode": final_block["final_generation_mode"],

            # runner-friendly metrics
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
                "generation": (final_attempt.get("generator") or {}).get("metrics", {}),
                "verification": (final_attempt.get("verification") or {}).get("metrics", {}),
                "decision": {
                    "decision": final_block["final_decision"],
                    "final_support_score": ((final_attempt.get("decision") or {}).get("final_support_score")),
                    "next_generation_mode": ((final_attempt.get("decision") or {}).get("next_generation_mode")),
                },
            },
            "retrieval_chunks": retrieval_chunks,

            # compact trace
            "pre_generation": {
                "analyzer": analyzer_small,
                "clarification": clarification_small,
                "decomposer": decomposer_small,
                "rag": rag_small,
                "validator": validator_small,
                "sufficiency_gate": suff_gate_small,
                "terminal": bool(pre_generation_terminal_message and not attempt_1.get("used") and not attempt_2.get("used")),
                "terminal_message": pre_generation_terminal_message,
            },
            "attempt_1": attempt_1,
            "attempt_2": attempt_2,
            "final": final_block,
        }

        if bool(self.include_raw_blobs):
            payload["raw_blobs"] = {
                "clarification_raw_data": self._safe_dict_from_data(self.clarification_raw_data),
                "decomposer_raw_data": self._safe_dict_from_data(self.decomposer_raw_data),
                "validator_raw_data": self._safe_dict_from_data(self.validator_raw_data),
                "attempt1_generator_raw_data": self._safe_dict_from_data(self.attempt1_generator_raw_data),
                "attempt1_verification_raw_data": self._safe_dict_from_data(self.attempt1_verification_raw_data),
                "attempt2_generator_raw_data": self._safe_dict_from_data(self.attempt2_generator_raw_data),
                "attempt2_verification_raw_data": self._safe_dict_from_data(self.attempt2_verification_raw_data),
                "rag_message": self._safe_text(self.rag_message),
                "sufficiency_gate_message": self._safe_text(self.sufficiency_gate_message),
                "attempt1_decision_message": self._safe_text(self.attempt1_decision_message),
                "attempt2_decision_message": self._safe_text(self.attempt2_decision_message),
                "attempt1_accept": self._safe_text(self.attempt1_accept),
                "attempt1_reject": self._safe_text(self.attempt1_reject),
                "attempt2_accept": self._safe_text(self.attempt2_accept),
                "attempt2_reject": self._safe_text(self.attempt2_reject),
            }

        self.status = (
            f"payload built | final_decision={final_block['final_decision']} "
            f"| retry_used={int(retry_used)} "
            f"| attempt_count={attempt_count} "
            f"| final_output_source={final_output_source or 'none'} "
            f"| retrieval_chunks={len(retrieval_chunks)}"
        )

        return Data(
            data=payload,
            text_key="assistant_text",
            default_value="",
        )