# Recovered Langflow component
# type: AcceptedResponseStageWriter
# class: AcceptedResponseStageWriter
# used in 12 flow(s): AIT Cognitive RAG, Cognitive RAG V1.2.0 AIT CR, Cognitive RAG V1.2.0 AIT Loose, Cognitive RAG V1.2.0 AIT Normal, Cognitive RAG V1.2.0 AIT RR, Cognitive RAG V1.2.0 AIT Reason, Cognitive RAG V1.2.0 AIT Strict , Cognitive RAG V1.2.0 AIT VaR ...
# json path: node.data.node.template.code.value

import ast
import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import UploadFile

from lfx.custom import Component
from lfx.io import BoolInput, DropdownInput, HandleInput, IntInput, SecretStrInput, StrInput
from lfx.schema import Data, Message
from lfx.services.deps import get_settings_service, get_storage_service, session_scope
from lfx.template.field.base import Output

from langflow.api.v2.files import upload_user_file
from langflow.services.database.models.user.crud import get_user_by_id


class AcceptedResponseStageWriter(Component):
    display_name = "Accepted Response Writer"
    description = (
        "Write an accepted-response stage record from analyzer, clarification tracker, "
        "decomposer, RAG, validator, gate, generation, verification, support gate, "
        "and final accepted response. Supports Local and Google Drive, including JSONL append logging."
    )
    icon = "file-text"
    name = "AcceptedResponseStageWriter"

    LOCAL_FORMAT_CHOICES = ["json", "jsonl", "txt", "markdown"]
    GDRIVE_FORMAT_CHOICES = ["json", "jsonl", "txt", "docs"]
    KNOWN_EXTENSIONS = {"json", "jsonl", "txt", "md", "markdown", "docs"}

    inputs = [
        DropdownInput(
            name="storage_location",
            display_name="Storage Location",
            options=["Local", "Google Drive"],
            value="Local",
            info="Choose where to save the accepted-response stage record.",
            real_time_refresh=True,
        ),
        StrInput(
            name="file_name",
            display_name="File Name",
            value="accepted_response_stage_record",
            required=True,
            info="Saved file name. You may include or omit extension; it will be normalized.",
            tool_mode=True,
        ),
        DropdownInput(
            name="local_format",
            display_name="Local Format",
            options=LOCAL_FORMAT_CHOICES,
            value="jsonl",
            show=False,
        ),
        DropdownInput(
            name="gdrive_format",
            display_name="Google Drive Format",
            options=GDRIVE_FORMAT_CHOICES,
            value="json",
            show=True,
        ),
        BoolInput(
            name="append_mode",
            display_name="Append (Local only)",
            value=True,
            show=False,
            advanced=True,
        ),
        IntInput(
            name="history_char_limit",
            display_name="History Char Limit",
            value=5000,
            advanced=True,
            info="Trim analyzer history_window to this many characters. Use 0 for no limit.",
        ),
        IntInput(
            name="page_content_char_limit",
            display_name="Page Content Char Limit",
            value=1200,
            advanced=True,
            info="Trim retrieved page_content fields to this many characters. Use 0 for no limit.",
        ),
        IntInput(
            name="claim_text_char_limit",
            display_name="Claim Text Char Limit",
            value=1200,
            advanced=True,
            info="Trim claim_text fields to this many characters. Use 0 for no limit.",
        ),
        IntInput(
            name="answer_char_limit",
            display_name="Answer Char Limit",
            value=6000,
            advanced=True,
            info="Trim stored accepted answer text to this many characters. Use 0 for no limit.",
        ),
        BoolInput(
            name="include_raw_envelope_text",
            display_name="Include Raw Envelope Text",
            value=False,
            advanced=True,
            info="Include raw text and envelope keys for debugging wrapped inputs.",
        ),
        HandleInput(
            name="analyzer_output",
            display_name="Analyzer Output",
            input_types=["Data", "Message"],
            required=True,
        ),
        HandleInput(
            name="clarification_tracking",
            display_name="Clarification Tracking",
            input_types=["Data", "Message"],
            required=False,
        ),
        HandleInput(
            name="decomposer_output",
            display_name="Decomposer Output",
            input_types=["Data", "Message"],
            required=False,
        ),
        HandleInput(
            name="rag_output",
            display_name="RAG Output",
            input_types=["Data", "Message"],
            required=False,
        ),
        HandleInput(
            name="validator_output",
            display_name="Validator Output (Raw Envelope)",
            input_types=["Data", "Message"],
            required=False,
        ),
        HandleInput(
            name="validator_combined_output",
            display_name="Validator Combined Output",
            input_types=["Data", "Message"],
            required=False,
        ),
        HandleInput(
            name="gate_output",
            display_name="Gate Output",
            input_types=["Data", "Message"],
            required=False,
        ),
        HandleInput(
            name="generation_output",
            display_name="Generation Output",
            input_types=["Data", "Message"],
            required=True,
        ),
        HandleInput(
            name="verification_output",
            display_name="Verification Output",
            input_types=["Data", "Message"],
            required=True,
        ),
        HandleInput(
            name="support_gate_output",
            display_name="Support Gate Output",
            input_types=["Data", "Message"],
            required=True,
        ),
        HandleInput(
            name="accepted_message",
            display_name="Accepted Message",
            input_types=["Data", "Message"],
            required=True,
        ),
        SecretStrInput(
            name="service_account_key",
            display_name="GCP Credentials Secret Key",
            info="Full Google service account JSON as a secret string.",
            show=True,
            advanced=True,
        ),
        StrInput(
            name="folder_id",
            display_name="Google Drive Folder ID",
            info="Drive folder ID. The folder must be shared with the service account email.",
            required=False,
            show=True,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message", method="write_record"),
    ]

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name != "storage_location":
            return build_config

        storage = field_value or "Local"

        if "local_format" in build_config:
            build_config["local_format"]["show"] = storage == "Local"
        if "append_mode" in build_config:
            build_config["append_mode"]["show"] = storage == "Local"

        if "gdrive_format" in build_config:
            build_config["gdrive_format"]["show"] = storage == "Google Drive"
        if "service_account_key" in build_config:
            build_config["service_account_key"]["show"] = storage == "Google Drive"
        if "folder_id" in build_config:
            build_config["folder_id"]["show"] = storage == "Google Drive"

        return build_config

    # -------------------------
    # Generic extraction helpers
    # -------------------------
    def _secret_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if hasattr(value, "get_secret_value"):
            try:
                return value.get_secret_value()
            except Exception:
                pass
        return str(value)

    def _extract_message_text(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, Message):
            try:
                return str(value.text or "").strip()
            except Exception:
                return str(value).strip()

        if isinstance(value, Data):
            if isinstance(value.data, dict):
                if "text" in value.data:
                    return str(value.data.get("text", "")).strip()
                return json.dumps(value.data, ensure_ascii=False)
            return str(value).strip()

        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()

        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()

        return str(value).strip()

    def _parse_json_like(self, raw_text: Any) -> Optional[Any]:
        if raw_text is None:
            return None

        if isinstance(raw_text, (dict, list)):
            return raw_text

        if not isinstance(raw_text, str):
            return None

        raw = raw_text.strip()
        if not raw:
            return None

        try:
            return json.loads(raw)
        except Exception:
            pass

        try:
            return ast.literal_eval(raw)
        except Exception:
            return None

    def _extract_payload_dict(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, Data) and isinstance(value.data, dict):
            return value.data

        if hasattr(value, "data") and isinstance(value.data, dict):
            return value.data

        if isinstance(value, dict):
            return value

        text = self._extract_message_text(value)
        parsed = self._parse_json_like(text)
        return parsed if isinstance(parsed, dict) else {}

    def _extract_inner_message_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        message_obj = payload.get("message")
        if isinstance(message_obj, dict):
            content = message_obj.get("content")
            parsed = self._parse_json_like(content)
            if isinstance(parsed, dict):
                return parsed

        content = payload.get("content")
        parsed = self._parse_json_like(content)
        if isinstance(parsed, dict):
            return parsed

        return {}

    def _unwrap_payload_dict(
        self,
        value: Any,
        expected_keys: Optional[List[str]] = None,
        wrapper_keys: Optional[List[str]] = None,
        max_depth: int = 6,
    ) -> Dict[str, Any]:
        """
        Robustly unwrap LangFlow/LLM envelopes.

        This is the important fix. Your support gate can arrive as any of these:
        1) {"decision": "accept", "final_support_score": 1.0, ...}
        2) {"message": {"content": "{...support gate json...}"}}
        3) {"data": {"decision": ...}}
        4) {"output": "{...}"}, {"result": "{...}"}, etc.

        The old writer only read case (1), so support_gate fields became null.
        """
        expected_keys = expected_keys or []
        wrapper_keys = wrapper_keys or [
            "message",
            "content",
            "text",
            "data",
            "output",
            "result",
            "response",
            "payload",
            "support_gate",
            "support_gate_output",
        ]

        current: Any = value

        for _ in range(max_depth):
            if isinstance(current, Data):
                current = current.data
                continue

            if isinstance(current, Message):
                current = getattr(current, "text", "")
                continue

            if not isinstance(current, (dict, list)):
                parsed = self._parse_json_like(current)
                if parsed is None:
                    return {}
                current = parsed
                continue

            if isinstance(current, list):
                return {}

            if not isinstance(current, dict):
                return {}

            if expected_keys and any(k in current for k in expected_keys):
                return current

            if not expected_keys and current:
                return current

            moved = False

            message_obj = current.get("message")
            if isinstance(message_obj, dict) and "content" in message_obj:
                current = message_obj.get("content")
                moved = True
            elif isinstance(message_obj, str):
                current = message_obj
                moved = True

            if moved:
                continue

            for key in wrapper_keys:
                if key in current:
                    nested = current.get(key)

                    if isinstance(nested, dict):
                        if expected_keys and any(k in nested for k in expected_keys):
                            return nested
                        current = nested
                        moved = True
                        break

                    if isinstance(nested, str):
                        parsed = self._parse_json_like(nested)
                        if isinstance(parsed, dict):
                            if expected_keys and any(k in parsed for k in expected_keys):
                                return parsed
                            current = parsed
                            moved = True
                            break

            if moved:
                continue

            return current

        return current if isinstance(current, dict) else {}

    def _trim_text(self, text: str, limit: int) -> str:
        limit = int(limit or 0)
        text = text or ""
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit]

    def _trim_history(self, text: str) -> str:
        return self._trim_text(text, int(self.history_char_limit or 0))

    def _trim_page_content(self, text: str) -> str:
        return self._trim_text(text, int(self.page_content_char_limit or 0))

    def _trim_claim_text(self, text: str) -> str:
        return self._trim_text(text, int(self.claim_text_char_limit or 0))

    def _trim_answer_text(self, text: str) -> str:
        return self._trim_text(text, int(self.answer_char_limit or 0))

    def _normalized_base_name(self, file_name: str) -> str:
        raw = (file_name or "").strip()
        path = Path(raw).expanduser()
        suffix = path.suffix.lower().lstrip(".")
        if suffix in self.KNOWN_EXTENSIONS:
            path = path.with_suffix("")
        return str(path)

    def _adjust_local_path(self, file_name: str, fmt: str) -> Path:
        path = Path(self._normalized_base_name(file_name)).expanduser()
        desired = "md" if fmt == "markdown" else fmt
        return path.with_suffix(f".{desired}")

    def _is_plain_text_format(self, fmt: str) -> bool:
        return fmt.lower() in [
            "txt",
            "json",
            "markdown",
            "md",
            "csv",
            "xml",
            "html",
            "yaml",
            "log",
            "tsv",
            "jsonl",
        ]

    def _parse_gcp_credentials(self, service_account_key: str) -> Dict[str, Any]:
        parse_errors = []
        credentials_dict = None

        for label, candidate in [
            ("standard", service_account_key),
            ("stripped", service_account_key.strip()),
            ("newline-fixed", service_account_key.replace("\\n", "\n")),
        ]:
            try:
                credentials_dict = json.loads(candidate, strict=False)
                break
            except json.JSONDecodeError as e:
                parse_errors.append(f"{label}: {e!s}")

        if isinstance(credentials_dict, str):
            try:
                credentials_dict = json.loads(credentials_dict, strict=False)
            except json.JSONDecodeError as e:
                parse_errors.append(f"double-encoded: {e!s}")
                credentials_dict = None

        if credentials_dict is None:
            raise ValueError(
                "Unable to parse service account key JSON. "
                + "; ".join(parse_errors)
            )

        return credentials_dict

    async def _upload_file_to_langflow(self, file_path: Path, append: bool = False) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not getattr(self, "user_id", None):
            raise ValueError("user_id is required to register the file in Langflow Files.")

        with file_path.open("rb") as f:
            async with session_scope() as db:
                current_user = await get_user_by_id(db, self.user_id)

                await upload_user_file(
                    file=UploadFile(
                        filename=file_path.name,
                        file=f,
                        size=file_path.stat().st_size,
                    ),
                    session=db,
                    current_user=current_user,
                    storage_service=get_storage_service(),
                    settings_service=get_settings_service(),
                    append=append,
                )

    async def _upload_bytes_to_langflow(self, filename: str, data: bytes, append: bool) -> None:
        if not getattr(self, "user_id", None):
            raise ValueError("user_id is required to register the file in Langflow Files.")

        file_obj = io.BytesIO(data)

        async with session_scope() as db:
            current_user = await get_user_by_id(db, self.user_id)

            await upload_user_file(
                file=UploadFile(
                    filename=filename,
                    file=file_obj,
                    size=len(data),
                ),
                session=db,
                current_user=current_user,
                storage_service=get_storage_service(),
                settings_service=get_settings_service(),
                append=append,
            )

    # -------------------------
    # Metrics helpers
    # -------------------------
    def _extract_metrics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        metrics = payload.get("llm_metrics")
        if not isinstance(metrics, dict):
            metrics = payload.get("metrics")

        if not isinstance(metrics, dict) and any(
            k in payload
            for k in [
                "model",
                "prompt_eval_count",
                "eval_count",
                "total_duration",
                "total_duration_ns",
            ]
        ):
            metrics = payload

        if not isinstance(metrics, dict):
            return {}

        return {
            "model": metrics.get("model"),
            "created_at": metrics.get("created_at"),
            "done": metrics.get("done"),
            "done_reason": metrics.get("done_reason"),
            "prompt_eval_count": metrics.get("prompt_eval_count"),
            "eval_count": metrics.get("eval_count"),
            "prompt_eval_duration_ns": metrics.get(
                "prompt_eval_duration_ns",
                metrics.get("prompt_eval_duration"),
            ),
            "eval_duration_ns": metrics.get(
                "eval_duration_ns",
                metrics.get("eval_duration"),
            ),
            "total_duration_ns": metrics.get(
                "total_duration_ns",
                metrics.get("total_duration"),
            ),
            "load_duration_ns": metrics.get(
                "load_duration_ns",
                metrics.get("load_duration"),
            ),
            "wall_time_s": metrics.get("wall_time_s"),
            "requested_think": metrics.get("requested_think"),
            "observed_has_thinking_field": metrics.get("observed_has_thinking_field"),
        }

    # -------------------------
    # Field selectors
    # -------------------------
    def _merge_resolved_slots(self, analyzer: Dict[str, Any]) -> Dict[str, Any]:
        resolved = analyzer.get("resolved_slots")
        if isinstance(resolved, dict) and resolved:
            return resolved

        merged: Dict[str, Any] = {}
        user_resolved = analyzer.get("user_resolved_slots", {})
        history_resolved = analyzer.get("history_resolved_slots", {})

        if isinstance(user_resolved, dict):
            merged.update(user_resolved)
        if isinstance(history_resolved, dict):
            merged.update(history_resolved)

        return merged

    def _normalize_missing_slots(self, analyzer: Dict[str, Any]) -> Dict[str, Any]:
        missing_slots = analyzer.get("missing_slots")
        if isinstance(missing_slots, dict):
            return {
                "general": missing_slots.get("general", []),
                "critical": missing_slots.get("critical", []),
            }

        return {
            "general": analyzer.get("missing_general", []),
            "critical": analyzer.get("missing_critical", []),
        }

    def _select_analyzer_fields(self, analyzer: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(analyzer, dict):
            analyzer = {}

        return {
            "original_question": (
                analyzer.get("original_question")
                or analyzer.get("user_query")
                or analyzer.get("original_query")
                or ""
            ),
            "history_window": self._trim_history(str(analyzer.get("history_window", "") or "")),
            "intent_type": analyzer.get("intent_type"),
            "request_type": analyzer.get("request_type"),
            "effective_request_types": analyzer.get("effective_request_types", []),
            "topic": analyzer.get("topic", ""),
            "expected_general_slots": analyzer.get("expected_general_slots", []),
            "expected_critical_slots": analyzer.get("expected_critical_slots", []),
            "required_general_slots": analyzer.get("required_general_slots", []),
            "required_critical_slots": analyzer.get("required_critical_slots", []),
            "missing_slots": self._normalize_missing_slots(analyzer),
            "slot_coverage_general": analyzer.get("slot_coverage_general"),
            "slot_coverage_critical": analyzer.get("slot_coverage_critical"),
            "slot_coverage_weighted": analyzer.get("slot_coverage_weighted"),
            "counts": analyzer.get("counts", {}),
            "resolved_slots": self._merge_resolved_slots(analyzer),
            "refinement": analyzer.get("refinement", {}),
            "clarification_decision": analyzer.get("clarification_decision", {}),
            "clarification_needed": analyzer.get("clarification_needed"),
        }

    def _select_tracker_fields(self, tracker: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(tracker, dict):
            tracker = {}

        clar_last = tracker.get("clarification_asked_in_last_ai_turn")
        if clar_last is None:
            clar_last = tracker.get("clarification_asked_in_history", False)

        result = {
            "clarification_asked_in_last_ai_turn": bool(clar_last),
            "matched_clarification_type": str(
                tracker.get("matched_clarification_type", "none") or "none"
            ),
        }

        if "last_ai_turn_text" in tracker:
            result["last_ai_turn_text"] = str(tracker.get("last_ai_turn_text", "") or "")

        return result

    def _select_decomposer_fields(self, decomposer_value: Any) -> Dict[str, Any]:
        src = self._unwrap_payload_dict(
            decomposer_value,
            expected_keys=[
                "original_query",
                "decomposition_used",
                "sub_questions",
                "final_query_count",
            ],
        )

        result = {
            "original_query": src.get("original_query", ""),
            "decomposition_used": src.get("decomposition_used"),
            "decomposition_type": src.get("decomposition_type"),
            "sub_questions": src.get("sub_questions", []),
            "final_query_count": src.get("final_query_count"),
            "notes": src.get("notes", ""),
        }

        if bool(self.include_raw_envelope_text):
            result["raw_text"] = self._extract_message_text(decomposer_value)

        return result

    def _simplify_rag_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        simplified = []

        for item in results or []:
            if not isinstance(item, dict):
                continue

            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            simplified.append(
                {
                    "rank": item.get("rank"),
                    "score": item.get("score"),
                    "page_content": self._trim_page_content(
                        str(item.get("page_content", "") or "")
                    ),
                    "metadata": {
                        "source": metadata.get("source"),
                    },
                }
            )

        return simplified

    def _select_rag_fields(self, rag_value: Any) -> Dict[str, Any]:
        payload = self._extract_payload_dict(rag_value)

        results_by_query_out = []
        for group in payload.get("results_by_query", []) or []:
            if not isinstance(group, dict):
                continue

            results_by_query_out.append(
                {
                    "query_index": group.get("query_index"),
                    "query_text": group.get("query_text"),
                    "total_queries": group.get("total_queries"),
                    "retrieval_wall_time_s": group.get("retrieval_wall_time_s"),
                    "top_k": group.get("top_k"),
                    "n_returned": group.get("n_returned"),
                    "results": self._simplify_rag_results(group.get("results", [])),
                }
            )

        return {
            "original_query": payload.get("original_query", ""),
            "decomposition_used": payload.get("decomposition_used"),
            "decomposition_type": payload.get("decomposition_type"),
            "notes": payload.get("notes", ""),
            "final_query_count": payload.get("final_query_count"),
            "total_results_returned": payload.get("total_results_returned"),
            "results_by_query": results_by_query_out,
        }

    def _select_validator_raw_fields(self, validator_value: Any) -> Dict[str, Any]:
        src = self._unwrap_payload_dict(
            validator_value,
            expected_keys=[
                "original_query",
                "top_kept_evidences",
                "decomposition_used",
            ],
        )

        top_kept = []
        for ev in src.get("top_kept_evidences", []) or []:
            if isinstance(ev, dict):
                top_kept.append(
                    {
                        "global_retrieval_index": ev.get("global_retrieval_index"),
                        "validation_score": ev.get("validation_score"),
                    }
                )

        result = {
            "original_query": src.get("original_query", ""),
            "decomposition_used": src.get("decomposition_used"),
            "decomposition_type": src.get("decomposition_type"),
            "top_kept_evidences": top_kept,
        }

        if bool(self.include_raw_envelope_text):
            result["raw_text"] = self._extract_message_text(validator_value)

        return result

    def _select_validator_combined_fields(
        self,
        validator_combined_value: Any,
    ) -> Dict[str, Any]:
        payload = self._extract_payload_dict(validator_combined_value)

        top_kept = []
        for ev in payload.get("top_kept_evidences", []) or []:
            if not isinstance(ev, dict):
                continue

            metadata = ev.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            top_kept.append(
                {
                    "global_retrieval_index": ev.get("global_retrieval_index"),
                    "query_index": ev.get("query_index"),
                    "query_text": ev.get("query_text"),
                    "validation_score": ev.get("validation_score"),
                    "is_answer_bearing": ev.get("is_answer_bearing"),
                    "page_content": self._trim_page_content(
                        str(ev.get("page_content", "") or "")
                    ),
                    "metadata": {
                        "source": metadata.get("source"),
                    },
                }
            )

        return {
            "original_query": payload.get("original_query", ""),
            "decomposition_used": payload.get("decomposition_used"),
            "decomposition_type": payload.get("decomposition_type"),
            "top_kept_evidences": top_kept,
            "evidence_sufficiency_score": payload.get("evidence_sufficiency_score"),
            "conflict_penalty": payload.get("conflict_penalty"),
            "rag_valid": payload.get("rag_valid"),
            "decision_hint": payload.get("decision_hint"),
        }

    def _select_gate_fields(self, gate_value: Any) -> Dict[str, Any]:
        payload = self._extract_payload_dict(gate_value)

        rag_results_out = []
        for ev in payload.get("rag_results", []) or []:
            if not isinstance(ev, dict):
                continue

            metadata = ev.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            rag_results_out.append(
                {
                    "global_retrieval_index": ev.get("global_retrieval_index"),
                    "query_index": ev.get("query_index"),
                    "query_text": ev.get("query_text"),
                    "validation_score": ev.get("validation_score"),
                    "is_answer_bearing": ev.get("is_answer_bearing"),
                    "page_content": self._trim_page_content(
                        str(ev.get("page_content", "") or "")
                    ),
                    "metadata": {
                        "source": metadata.get("source"),
                    },
                }
            )

        return {
            "user_query": payload.get("user_query", ""),
            "decomposition_used": payload.get("decomposition_used"),
            "decomposition_type": payload.get("decomposition_type"),
            "slot_coverage_weighted": payload.get("slot_coverage_weighted"),
            "evidence_sufficiency_score": payload.get("evidence_sufficiency_score"),
            "weights": payload.get("weights", {}),
            "decision_threshold": payload.get("decision_threshold"),
            "final_score": payload.get("final_score"),
            "decision": payload.get("decision"),
            "rag_results": rag_results_out,
        }

    def _select_generation_fields(self, generation_value: Any) -> Dict[str, Any]:
        outer = self._extract_payload_dict(generation_value)
        if not isinstance(outer, dict):
            outer = {}

        message_obj = outer.get("message", {})
        if not isinstance(message_obj, dict):
            message_obj = {}

        answer_text = str(
            message_obj.get("content", "")
            or outer.get("content", "")
            or outer.get("text", "")
            or ""
        ).strip()

        result = {
            "model": outer.get("model"),
            "created_at": outer.get("created_at"),
            "done": outer.get("done"),
            "done_reason": outer.get("done_reason"),
            "answer_text": self._trim_answer_text(answer_text),
        }

        if bool(self.include_raw_envelope_text):
            result["raw_text"] = self._extract_message_text(generation_value)

        return result

    def _select_verification_fields(self, verification_value: Any) -> Dict[str, Any]:
        src = self._unwrap_payload_dict(
            verification_value,
            expected_keys=["claims", "slot_coverage"],
        )

        claims_out = []
        for claim in src.get("claims", []) or []:
            if not isinstance(claim, dict):
                continue

            claims_out.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "claim_text": self._trim_claim_text(
                        str(claim.get("claim_text", "") or "")
                    ),
                    "support_score": claim.get("support_score"),
                    "slot_type": claim.get("slot_type"),
                    "slot_importance": claim.get("slot_importance"),
                }
            )

        slot_coverage = src.get("slot_coverage", {})
        if not isinstance(slot_coverage, dict):
            slot_coverage = {}

        def _slot_rows(rows):
            out = []
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                out.append(
                    {
                        "slot_name": row.get("slot_name"),
                        "status": row.get("status"),
                        "support_score": row.get("support_score"),
                    }
                )
            return out

        result = {
            "claims": claims_out,
            "slot_coverage": {
                "general": _slot_rows(slot_coverage.get("general", [])),
                "critical": _slot_rows(slot_coverage.get("critical", [])),
            },
        }

        if bool(self.include_raw_envelope_text):
            outer = self._extract_payload_dict(verification_value)
            result["raw_text"] = self._extract_message_text(verification_value)
            result["outer_keys"] = list(outer.keys()) if isinstance(outer, dict) else []
            result["payload_keys"] = list(src.keys()) if isinstance(src, dict) else []

        return result

    def _select_support_gate_fields(self, support_gate_value: Any) -> Dict[str, Any]:
        src = self._unwrap_payload_dict(
            support_gate_value,
            expected_keys=[
                "decision",
                "final_support_score",
                "claim_rows",
                "raw_mean_support_score",
            ],
        )

        claim_rows_out = []
        for row in src.get("claim_rows", []) or []:
            if not isinstance(row, dict):
                continue

            claim_rows_out.append(
                {
                    "claim_id": row.get("claim_id"),
                    "claim_text": self._trim_claim_text(
                        str(row.get("claim_text", "") or "")
                    ),
                    "slot_type": row.get("slot_type"),
                    "slot_importance": row.get("slot_importance"),
                    "support_score": row.get("support_score"),
                    "weight": row.get("weight"),
                    "weighted_contribution": row.get("weighted_contribution"),
                }
            )

        def _slot_rows(rows):
            out = []
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                out.append(
                    {
                        "slot_name": row.get("slot_name"),
                        "status": row.get("status"),
                        "support_score": row.get("support_score"),
                        "answered": row.get("answered"),
                    }
                )
            return out

        result = {
            "decision": src.get("decision"),
            "generation_mode_in": src.get("generation_mode_in"),
            "next_generation_mode": src.get("next_generation_mode"),
            "general_weight": src.get("general_weight"),
            "critical_weight": src.get("critical_weight"),
            "final_support_threshold": src.get("final_support_threshold"),
            "total_claims": src.get("total_claims"),
            "total_general_claims": src.get("total_general_claims"),
            "total_critical_claims": src.get("total_critical_claims"),
            "raw_mean_support_score": src.get("raw_mean_support_score"),
            "weighted_score_sum": src.get("weighted_score_sum"),
            "weighted_total": src.get("weighted_total"),
            "final_support_score": src.get("final_support_score"),
            "general_slots_total": src.get("general_slots_total"),
            "general_slots_answered": src.get("general_slots_answered"),
            "general_slots_missing": src.get("general_slots_missing"),
            "critical_slots_total": src.get("critical_slots_total"),
            "critical_slots_answered": src.get("critical_slots_answered"),
            "critical_slots_missing": src.get("critical_slots_missing"),
            "claims_per_slot": src.get("claims_per_slot", {}),
            "claim_rows": claim_rows_out,
            "general_slot_rows": _slot_rows(src.get("general_slot_rows", [])),
            "critical_slot_rows": _slot_rows(src.get("critical_slot_rows", [])),
        }

        if bool(self.include_raw_envelope_text):
            outer = self._extract_payload_dict(support_gate_value)
            result["raw_text"] = self._extract_message_text(support_gate_value)
            result["outer_keys"] = list(outer.keys()) if isinstance(outer, dict) else []
            result["payload_keys"] = list(src.keys()) if isinstance(src, dict) else []

        return result

    def _select_final_response_fields(self, response_value: Any) -> Dict[str, Any]:
        payload = self._extract_payload_dict(response_value)
        message_text = ""

        if isinstance(payload, dict) and payload:
            message_text = (
                payload.get("text")
                or payload.get("message")
                or payload.get("response")
                or payload.get("content")
                or ""
            )

            if isinstance(message_text, dict):
                message_text = (
                    message_text.get("content")
                    or message_text.get("text")
                    or json.dumps(message_text, ensure_ascii=False)
                )

        if not message_text:
            message_text = self._extract_message_text(response_value)

        result = {
            "text": self._trim_answer_text(str(message_text or "").strip()),
        }

        if bool(self.include_raw_envelope_text):
            result["raw_text"] = self._extract_message_text(response_value)

        return result

    def _build_record(self) -> Dict[str, Any]:
        analyzer_payload = self._extract_payload_dict(self.analyzer_output)
        tracker_payload = self._extract_payload_dict(self.clarification_tracking)
        decomposer_payload = self._extract_payload_dict(self.decomposer_output)
        rag_payload = self._extract_payload_dict(self.rag_output)
        validator_payload = self._extract_payload_dict(self.validator_output)
        validator_combined_payload = self._extract_payload_dict(self.validator_combined_output)
        gate_payload = self._extract_payload_dict(self.gate_output)
        generation_payload = self._extract_payload_dict(self.generation_output)
        verification_payload = self._extract_payload_dict(self.verification_output)
        support_gate_payload = self._extract_payload_dict(self.support_gate_output)
        accepted_response_payload = self._extract_payload_dict(self.accepted_message)

        return {
            "record_type": "accepted_response_stage_writer",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "analyzer": self._select_analyzer_fields(analyzer_payload),
            "clarification_tracking": self._select_tracker_fields(tracker_payload),
            "decomposer": self._select_decomposer_fields(self.decomposer_output),
            "rag": self._select_rag_fields(self.rag_output),
            "validator": {
                "raw": self._select_validator_raw_fields(self.validator_output),
                "combined": self._select_validator_combined_fields(self.validator_combined_output),
            },
            "gate": self._select_gate_fields(self.gate_output),
            "generation": self._select_generation_fields(self.generation_output),
            "verification": self._select_verification_fields(self.verification_output),
            "support_gate": self._select_support_gate_fields(self.support_gate_output),
            "final_response": self._select_final_response_fields(self.accepted_message),
            "stage_metrics": {
                "analyzer": self._extract_metrics(analyzer_payload),
                "clarification_tracking": self._extract_metrics(tracker_payload),
                "decomposer": self._extract_metrics(decomposer_payload),
                "rag": self._extract_metrics(rag_payload),
                "validator_raw": self._extract_metrics(validator_payload),
                "validator_combined": self._extract_metrics(validator_combined_payload),
                "gate": self._extract_metrics(gate_payload),
                "generation": self._extract_metrics(generation_payload),
                "verification": self._extract_metrics(verification_payload),
                "support_gate": self._extract_metrics(support_gate_payload),
                "final_response": self._extract_metrics(accepted_response_payload),
            },
        }

    # -------------------------
    # Serialization
    # -------------------------
    def _record_to_jsonl_line(self, record: Dict[str, Any]) -> str:
        return json.dumps(record, ensure_ascii=False)

    def _record_to_text(self, record: Dict[str, Any], fmt: str) -> str:
        if fmt == "json":
            return json.dumps(record, ensure_ascii=False, indent=2)

        if fmt == "jsonl":
            return self._record_to_jsonl_line(record)

        sections = [
            ("ANALYZER", record.get("analyzer", {})),
            ("CLARIFICATION TRACKING", record.get("clarification_tracking", {})),
            ("DECOMPOSER", record.get("decomposer", {})),
            ("RAG", record.get("rag", {})),
            ("VALIDATOR", record.get("validator", {})),
            ("GATE", record.get("gate", {})),
            ("GENERATION", record.get("generation", {})),
            ("VERIFICATION", record.get("verification", {})),
            ("SUPPORT GATE", record.get("support_gate", {})),
            ("STAGE METRICS", record.get("stage_metrics", {})),
        ]

        if fmt == "markdown":
            lines = [
                "# Accepted Response Stage Record",
                "",
                f"**Saved At (UTC):** {record.get('saved_at_utc', '')}",
                "",
            ]

            for title, obj in sections:
                lines += [
                    f"## {title.title()}",
                    "```json",
                    json.dumps(obj, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]

            lines += [
                "## Final Response",
                record.get("final_response", {}).get("text", ""),
            ]

            return "\n".join(lines)

        lines = [
            f"Saved At (UTC): {record.get('saved_at_utc', '')}",
            "",
        ]

        for title, obj in sections:
            lines += [
                f"=== {title} ===",
                json.dumps(obj, ensure_ascii=False, indent=2),
                "",
            ]

        lines += [
            "=== FINAL RESPONSE ===",
            record.get("final_response", {}).get("text", ""),
        ]

        return "\n".join(lines)

    # -------------------------
    # Local write helpers
    # -------------------------
    def _write_json(self, path: Path, record: Dict[str, Any], should_append: bool) -> str:
        if should_append:
            existing_data = []
            try:
                existing_content = path.read_text(encoding="utf-8").strip()
                if existing_content:
                    parsed = json.loads(existing_content)
                    if isinstance(parsed, dict):
                        existing_data = [parsed]
                    elif isinstance(parsed, list):
                        existing_data = parsed
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []

            existing_data.append(record)
            path.write_text(
                json.dumps(existing_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        action = "appended to" if should_append else "saved successfully as"
        return f"Accepted response stage record {action} '{path}'"

    def _write_jsonl(self, path: Path, record: Dict[str, Any], should_append: bool) -> str:
        line = self._record_to_jsonl_line(record)

        if should_append:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            path.write_text(line + "\n", encoding="utf-8")

        action = "appended to" if should_append else "saved successfully as"
        return f"Accepted response stage record {action} '{path}'"

    def _write_textlike(self, path: Path, content: str, should_append: bool) -> str:
        if should_append:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            sep = "\n\n---\n\n" if existing else ""
            path.write_text(existing + sep + content, encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")

        action = "appended to" if should_append else "saved successfully as"
        return f"Accepted response stage record {action} '{path}'"

    # -------------------------
    # Save backends
    # -------------------------
    async def _save_local(self, record: Dict[str, Any]) -> Message:
        fmt = self.local_format or "jsonl"
        path = self._adjust_local_path(self.file_name, fmt)

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        append_mode = bool(getattr(self, "append_mode", False))
        should_append = append_mode and path.exists() and self._is_plain_text_format(fmt)

        if fmt == "json":
            confirmation = self._write_json(path, record, should_append)
            await self._upload_file_to_langflow(path, append=False)

        elif fmt == "jsonl":
            confirmation = self._write_jsonl(path, record, should_append)
            line_bytes = (self._record_to_jsonl_line(record) + "\n").encode("utf-8")

            if should_append:
                await self._upload_bytes_to_langflow(path.name, line_bytes, append=True)
            else:
                await self._upload_file_to_langflow(path, append=False)

        elif fmt in {"txt", "markdown"}:
            content = self._record_to_text(record, fmt)
            confirmation = self._write_textlike(path, content, should_append)

            if should_append:
                delta = ("\n\n---\n\n" + content).encode("utf-8")
                await self._upload_bytes_to_langflow(path.name, delta, append=True)
            else:
                await self._upload_file_to_langflow(path, append=False)

        else:
            raise ValueError(f"Unsupported local format: {fmt}")

        final_path = Path.cwd() / path if not path.is_absolute() else path
        return Message(text=f"{confirmation} at {final_path}")

    async def _save_google_drive(self, record: Dict[str, Any]) -> Message:
        service_account_key = self._secret_to_text(self.service_account_key)
        if not service_account_key:
            raise ValueError("GCP Credentials Secret Key is required for Google Drive storage.")
        if not self.folder_id:
            raise ValueError("Google Drive Folder ID is required for Google Drive storage.")

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
        except ImportError as e:
            raise ImportError("Google API client libraries are not installed.") from e

        credentials_dict = self._parse_gcp_credentials(service_account_key)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        drive_service = build("drive", "v3", credentials=credentials)

        fmt = self.gdrive_format or "json"

        if fmt == "docs":
            return await self._save_google_doc(record, credentials)

        content = self._record_to_text(record, fmt)
        suffix = "jsonl" if fmt == "jsonl" else ("json" if fmt == "json" else "txt")
        base_name = Path(self._normalized_base_name(self.file_name)).name
        file_name = f"{base_name}.{suffix}"

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=f".{suffix}",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            if fmt == "jsonl" and not content.endswith("\n"):
                temp_file.write("\n")
            temp_path = temp_file.name

        try:
            file_metadata = {
                "name": file_name,
                "parents": [self.folder_id],
            }
            media = MediaFileUpload(temp_path, resumable=True)
            uploaded_file = drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id",
            ).execute()

            file_id = uploaded_file.get("id")
            file_url = f"https://drive.google.com/file/d/{file_id}/view"
            return Message(text=f"Accepted response stage record uploaded to Google Drive: {file_url}")

        finally:
            temp_file_path = Path(temp_path)
            if temp_file_path.exists():
                temp_file_path.unlink()

    async def _save_google_doc(self, record: Dict[str, Any], credentials) -> Message:
        try:
            from googleapiclient.discovery import build
        except ImportError as e:
            raise ImportError("Google API client libraries are not installed.") from e

        drive_service = build("drive", "v3", credentials=credentials)
        docs_service = build("docs", "v1", credentials=credentials)

        content = self._record_to_text(record, "txt")
        base_name = Path(self._normalized_base_name(self.file_name)).name

        file_metadata = {
            "name": base_name,
            "mimeType": "application/vnd.google-apps.document",
            "parents": [self.folder_id],
        }

        created_file = drive_service.files().create(
            body=file_metadata,
            fields="id",
        ).execute()
        document_id = created_file["id"]

        requests = [
            {
                "insertText": {
                    "location": {"index": 1},
                    "text": content,
                }
            }
        ]
        docs_service.documents().batchUpdate(
            documentId=document_id,
            body={"requests": requests},
        ).execute()

        file_url = f"https://docs.google.com/document/d/{document_id}/edit"
        return Message(text=f"Accepted response stage record created in Google Docs: {file_url}")

    async def write_record(self) -> Message:
        record = self._build_record()
        location = (self.storage_location or "Local").strip()

        if location == "Local":
            result = await self._save_local(record)
        elif location == "Google Drive":
            result = await self._save_google_drive(record)
        else:
            raise ValueError(f"Unsupported storage location: {location}")

        self.status = result
        return result