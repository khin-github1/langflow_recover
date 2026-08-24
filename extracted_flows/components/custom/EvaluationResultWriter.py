# Recovered Langflow component
# type: EvaluationResultWriter
# class: EvaluationResultWriter
# used in 1 flow(s): Cognitive RAG V1.1.5 Backup
# json path: node.data.node.template.code.value

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from langflow.custom import Component
from langflow.io import (
    BoolInput,
    DataInput,
    DropdownInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    Output,
)
from langflow.schema.data import Data
from langflow.schema.message import Message


class EvaluationResultWriter(Component):
    display_name = "Evaluation Result Writer"
    description = "Writes one terminal evaluation record per completed run to a JSONL file."
    icon = "file-text"
    name = "EvaluationResultWriter"

    TERMINAL_CASES = [
        "clarification_requested",
        "pre_generation_terminal",
        "attempt1_accept",
        "attempt1_reject",
        "attempt2_accept",
        "attempt2_reject",
    ]

    inputs = [
        MessageTextInput(
            name="output_path",
            display_name="Output Path",
            required=True,
            value="evaluation_results.jsonl",
            info="JSONL file path. If no extension is provided, .jsonl will be added.",
        ),
        DropdownInput(
            name="terminal_case",
            display_name="Terminal Case",
            options=TERMINAL_CASES,
            value="attempt1_accept",
            real_time_refresh=True,
            required=True,
        ),
        BoolInput(
            name="append_mode",
            display_name="Append Mode",
            value=True,
            advanced=True,
        ),
        MessageTextInput(
            name="run_id",
            display_name="Run ID",
            required=False,
            advanced=True,
            info="Optional unique id for dedup/debug. If empty, one is generated.",
        ),
        MessageTextInput(
            name="user_text",
            display_name="User Text",
            required=True,
        ),
        HandleInput(
            name="terminal_message",
            display_name="Terminal Message",
            input_types=["Message", "Data"],
            required=False,
            info="Final message shown on this terminal branch.",
        ),

        # Early / shared context
        DataInput(name="analyzer_data", display_name="Analyzer Data", required=False),
        DataInput(name="clarification_raw_data", display_name="Clarification Raw Data", required=False),
        DataInput(name="decomposer_raw_data", display_name="Decomposer Raw Data", required=False),
        DataInput(name="validator_raw_data", display_name="Validator Raw Data", required=False),

        HandleInput(
            name="rag_message",
            display_name="RAG Message",
            input_types=["Message", "Data"],
            required=False,
        ),
        HandleInput(
            name="sufficiency_gate_message",
            display_name="Sufficiency Gate Message",
            input_types=["Message", "Data"],
            required=False,
        ),

        # Attempt 1
        HandleInput(
            name="attempt1_generation_mode_message",
            display_name="Attempt 1 Generation Mode",
            input_types=["Message", "Data"],
            required=False,
        ),
        HandleInput(
            name="attempt1_prompt_message",
            display_name="Attempt 1 Prompt Message",
            input_types=["Message", "Data"],
            required=False,
        ),
        DataInput(
            name="attempt1_generator_raw_data",
            display_name="Attempt 1 Generator Raw Data",
            required=False,
        ),
        DataInput(
            name="attempt1_verification_raw_data",
            display_name="Attempt 1 Verification Raw Data",
            required=False,
        ),
        HandleInput(
            name="attempt1_decision_message",
            display_name="Attempt 1 Decision Message",
            input_types=["Message", "Data"],
            required=False,
        ),

        # Attempt 2
        HandleInput(
            name="attempt2_generation_mode_message",
            display_name="Attempt 2 Generation Mode",
            input_types=["Message", "Data"],
            required=False,
        ),
        HandleInput(
            name="attempt2_prompt_message",
            display_name="Attempt 2 Prompt Message",
            input_types=["Message", "Data"],
            required=False,
        ),
        DataInput(
            name="attempt2_generator_raw_data",
            display_name="Attempt 2 Generator Raw Data",
            required=False,
        ),
        DataInput(
            name="attempt2_verification_raw_data",
            display_name="Attempt 2 Verification Raw Data",
            required=False,
        ),
        HandleInput(
            name="attempt2_decision_message",
            display_name="Attempt 2 Decision Message",
            input_types=["Message", "Data"],
            required=False,
        ),

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
    ]

    outputs = [
        Output(display_name="Write Result", name="result", method="write_result"),
    ]

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name != "terminal_case":
            return build_config

        # Hide all case-dependent fields first
        case_fields = [
            "terminal_message",
            "analyzer_data",
            "clarification_raw_data",
            "decomposer_raw_data",
            "validator_raw_data",
            "rag_message",
            "sufficiency_gate_message",
            "attempt1_generation_mode_message",
            "attempt1_prompt_message",
            "attempt1_generator_raw_data",
            "attempt1_verification_raw_data",
            "attempt1_decision_message",
            "attempt2_generation_mode_message",
            "attempt2_prompt_message",
            "attempt2_generator_raw_data",
            "attempt2_verification_raw_data",
            "attempt2_decision_message",
            "include_retrieval_chunks",
            "max_retrieval_chunks",
        ]

        for name in case_fields:
            if name in build_config:
                build_config[name]["show"] = False

        # Always useful
        for name in ["terminal_message", "analyzer_data"]:
            if name in build_config:
                build_config[name]["show"] = True

        case_value = str(field_value or "").strip()

        if case_value == "clarification_requested":
            for name in ["clarification_raw_data"]:
                if name in build_config:
                    build_config[name]["show"] = True

        elif case_value == "pre_generation_terminal":
            for name in [
                "decomposer_raw_data",
                "validator_raw_data",
                "rag_message",
                "sufficiency_gate_message",
                "include_retrieval_chunks",
                "max_retrieval_chunks",
            ]:
                if name in build_config:
                    build_config[name]["show"] = True

        elif case_value in {"attempt1_accept", "attempt1_reject"}:
            for name in [
                "decomposer_raw_data",
                "validator_raw_data",
                "rag_message",
                "sufficiency_gate_message",
                "include_retrieval_chunks",
                "max_retrieval_chunks",
                "attempt1_generation_mode_message",
                "attempt1_prompt_message",
                "attempt1_generator_raw_data",
                "attempt1_verification_raw_data",
                "attempt1_decision_message",
            ]:
                if name in build_config:
                    build_config[name]["show"] = True

        elif case_value in {"attempt2_accept", "attempt2_reject"}:
            for name in [
                "decomposer_raw_data",
                "validator_raw_data",
                "rag_message",
                "sufficiency_gate_message",
                "include_retrieval_chunks",
                "max_retrieval_chunks",
                "attempt1_generation_mode_message",
                "attempt1_prompt_message",
                "attempt1_generator_raw_data",
                "attempt1_verification_raw_data",
                "attempt1_decision_message",
                "attempt2_generation_mode_message",
                "attempt2_prompt_message",
                "attempt2_generator_raw_data",
                "attempt2_verification_raw_data",
                "attempt2_decision_message",
            ]:
                if name in build_config:
                    build_config[name]["show"] = True

        return build_config

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _now_utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _make_run_id() -> str:
        return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value.strip()

        if isinstance(value, Message):
            return str(value.text or "").strip()

        if isinstance(value, Data):
            if isinstance(value.data, dict):
                for key in ("assistant_text", "text", "content", "message", "final_answer_shown"):
                    v = value.data.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            return str(getattr(value, "text", "") or "").strip()

        if isinstance(value, dict):
            for key in ("assistant_text", "text", "content", "message", "final_answer_shown"):
                v = value.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        if hasattr(value, "text"):
            return str(getattr(value, "text", "") or "").strip()

        return str(value).strip()

    @staticmethod
    def _safe_dict_from_data(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, dict):
            return value

        if isinstance(value, Data):
            return value.data if isinstance(value.data, dict) else {}

        if hasattr(value, "data") and isinstance(getattr(value, "data"), dict):
            return getattr(value, "data")

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
        d = self._safe_dict_from_data(raw)
        if d:
            return d
        return self._parse_json_obj(self._safe_text(raw))

    def _parse_data_envelope(self, raw: Any) -> Dict[str, Any]:
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

        if isinstance(data.get("verifier_json"), dict):
            content_json = data.get("verifier_json", {})
            content_text = str(data.get("text", "") or data.get("raw_llm_output", "")).strip()

        elif isinstance(raw_obj.get("message"), dict):
            content_text = str((raw_obj.get("message") or {}).get("content", "") or "").strip()
            content_json = self._parse_json_obj(content_text) if content_text else {}

        elif isinstance(data.get("text"), str):
            content_text = str(data.get("text", "") or "").strip()
            content_json = self._parse_json_obj(content_text) if content_text else {}

        elif isinstance(data.get("raw_llm_output"), str):
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

    def _select_generation_mode(self, raw: Any) -> str:
        text = self._safe_text(raw)
        if text.lower() in {"normal", "reason"}:
            return text.lower()

        obj = self._parse_message_json(raw)
        mode = str(obj.get("generation_mode", "")).strip().lower()
        if mode in {"normal", "reason"}:
            return mode

        return text

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

        answer = generator_parsed.get("content_text", "")
        verification_small = self._select_verification(verification_parsed)

        used = bool(
            generation_mode
            or prompt
            or answer
            or verification_small.get("claims_count", 0) > 0
            or decision_obj.get("decision")
        )

        return {
            "used": used,
            "generation_mode": generation_mode,
            "prompt": prompt,
            "generator": {
                "answer": answer,
                "metrics": generator_parsed.get("metrics", {}),
            },
            "verification": verification_small,
            "decision": decision_obj,
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

    @staticmethod
    def _json_default(obj: Any):
        try:
            return obj.__dict__
        except Exception:
            return str(obj)

    def _normalize_output_path(self) -> Path:
        p = Path((self.output_path or "").strip()).expanduser()
        if not p.suffix:
            p = p.with_suffix(".jsonl")
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -------------------------
    # Main
    # -------------------------
    def write_result(self) -> Message:
        case = str(self.terminal_case or "").strip()
        if case not in self.TERMINAL_CASES:
            raise ValueError(f"Unsupported terminal_case: {case}")

        analyzer_small = self._safe_dict_from_data(self.analyzer_data)
        clarification_small = self._parse_data_envelope(self.clarification_raw_data)
        decomposer_small = self._parse_data_envelope(self.decomposer_raw_data)
        validator_small = self._parse_data_envelope(self.validator_raw_data)

        rag_obj = self._parse_message_json(self.rag_message)
        suff_gate_small = self._parse_message_json(self.sufficiency_gate_message)

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

        terminal_text = self._safe_text(self.terminal_message)
        retrieval_chunks = []
        if bool(self.include_retrieval_chunks) and rag_obj:
            retrieval_chunks = self._flatten_retrieval_chunks(rag_obj, int(self.max_retrieval_chunks or 20))

        run_id = (self.run_id or "").strip() or self._make_run_id()

        # Determine final surface fields by case
        if case == "clarification_requested":
            assistant_text = terminal_text or clarification_small.get("content_text", "")
            final_decision = "clarify"
            retry_used = False
            attempt_count = 0
            terminal_stage = "clarification"
            final_generation_mode = ""

        elif case == "pre_generation_terminal":
            assistant_text = terminal_text
            final_decision = str(suff_gate_small.get("decision") or "pre_generation_terminal")
            retry_used = False
            attempt_count = 0
            terminal_stage = "pre_generation"
            final_generation_mode = ""

        elif case in {"attempt1_accept", "attempt1_reject"}:
            assistant_text = terminal_text or (attempt_1.get("generator") or {}).get("answer", "")
            final_decision = (
                str((attempt_1.get("decision") or {}).get("decision") or "")
                or ("accept" if case.endswith("accept") else "reject")
            )
            retry_used = False
            attempt_count = 1
            terminal_stage = "post_generation"
            final_generation_mode = str(attempt_1.get("generation_mode") or "")

        else:  # attempt2_accept / attempt2_reject
            assistant_text = terminal_text or (attempt_2.get("generator") or {}).get("answer", "")
            final_decision = (
                str((attempt_2.get("decision") or {}).get("decision") or "")
                or ("accept" if case.endswith("accept") else "reject")
            )
            retry_used = True
            attempt_count = 2
            terminal_stage = "post_generation"
            final_generation_mode = str(attempt_2.get("generation_mode") or "")

        record: Dict[str, Any] = {
            "timestamp_utc": self._now_utc_iso(),
            "run_id": run_id,
            "terminal_case": case,
            "user_text": self._safe_text(self.user_text),
            "assistant_text": assistant_text,
            "final_decision": final_decision,
            "retry_used": retry_used,
            "attempt_count": attempt_count,
            "terminal_stage": terminal_stage,
            "final_generation_mode": final_generation_mode,

            "metrics": {
                "analyzer": analyzer_small.get("llm_metrics", {}) if isinstance(analyzer_small, dict) else {},
                "retrieval": {
                    "decomposition_used": rag_obj.get("decomposition_used") if isinstance(rag_obj, dict) else None,
                    "decomposition_type": rag_obj.get("decomposition_type") if isinstance(rag_obj, dict) else None,
                    "final_query_count": rag_obj.get("final_query_count") if isinstance(rag_obj, dict) else None,
                    "total_results_returned": rag_obj.get("total_results_returned") if isinstance(rag_obj, dict) else None,
                    "validator_evidence_sufficiency_score": (validator_small.get("content_json", {}) or {}).get("evidence_sufficiency_score"),
                    "validator_conflict_penalty": (validator_small.get("content_json", {}) or {}).get("conflict_penalty"),
                    "validator_rag_valid": (validator_small.get("content_json", {}) or {}).get("rag_valid"),
                    "sufficiency_gate_final_score": suff_gate_small.get("final_score") if isinstance(suff_gate_small, dict) else None,
                    "sufficiency_gate_decision": suff_gate_small.get("decision") if isinstance(suff_gate_small, dict) else None,
                },
                "generation": (
                    (attempt_2.get("generator") if retry_used else attempt_1.get("generator")) or {}
                ).get("metrics", {}),
                "verification": (
                    (attempt_2.get("verification") if retry_used else attempt_1.get("verification")) or {}
                ).get("metrics", {}),
                "decision": {
                    "decision": final_decision,
                    "final_support_score": (
                        ((attempt_2 if retry_used else attempt_1).get("decision") or {}).get("final_support_score")
                    ),
                    "next_generation_mode": (
                        ((attempt_2 if retry_used else attempt_1).get("decision") or {}).get("next_generation_mode")
                    ),
                },
            },

            "retrieval_chunks": retrieval_chunks,
        }

        # Case-specific detail sections
        if case == "clarification_requested":
            record["clarification"] = {
                "question_text": assistant_text,
                "analyzer": analyzer_small,
                "clarification": {
                    "text": clarification_small.get("content_text", ""),
                    "json": clarification_small.get("content_json", {}),
                    "metrics": clarification_small.get("metrics", {}),
                },
            }

        elif case == "pre_generation_terminal":
            record["pre_generation"] = {
                "analyzer": analyzer_small,
                "decomposer": {
                    "text": decomposer_small.get("content_text", ""),
                    "json": decomposer_small.get("content_json", {}),
                    "metrics": decomposer_small.get("metrics", {}),
                },
                "rag": rag_obj,
                "validator": {
                    "text": validator_small.get("content_text", ""),
                    "json": validator_small.get("content_json", {}),
                    "metrics": validator_small.get("metrics", {}),
                },
                "sufficiency_gate": suff_gate_small,
                "terminal_message": terminal_text,
            }

        elif case in {"attempt1_accept", "attempt1_reject"}:
            record["attempt_1"] = attempt_1
            record["pre_generation"] = {
                "analyzer": analyzer_small,
                "decomposer": {
                    "text": decomposer_small.get("content_text", ""),
                    "json": decomposer_small.get("content_json", {}),
                    "metrics": decomposer_small.get("metrics", {}),
                },
                "rag": rag_obj,
                "validator": {
                    "text": validator_small.get("content_text", ""),
                    "json": validator_small.get("content_json", {}),
                    "metrics": validator_small.get("metrics", {}),
                },
                "sufficiency_gate": suff_gate_small,
            }

        else:  # attempt2_*
            record["attempt_1"] = attempt_1
            record["attempt_2"] = attempt_2
            record["pre_generation"] = {
                "analyzer": analyzer_small,
                "decomposer": {
                    "text": decomposer_small.get("content_text", ""),
                    "json": decomposer_small.get("content_json", {}),
                    "metrics": decomposer_small.get("metrics", {}),
                },
                "rag": rag_obj,
                "validator": {
                    "text": validator_small.get("content_text", ""),
                    "json": validator_small.get("content_json", {}),
                    "metrics": validator_small.get("metrics", {}),
                },
                "sufficiency_gate": suff_gate_small,
            }

        path = self._normalize_output_path()
        mode = "a" if bool(self.append_mode) else "w"

        with path.open(mode, encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=self._json_default) + "\n")

        self.status = (
            f"wrote terminal_case={case} | final_decision={final_decision} | "
            f"attempt_count={attempt_count} | retry_used={int(retry_used)} | path={path}"
        )

        return Message(
            text=(
                f"Saved evaluation record to {path} | "
                f"terminal_case={case} | final_decision={final_decision}"
            )
        )