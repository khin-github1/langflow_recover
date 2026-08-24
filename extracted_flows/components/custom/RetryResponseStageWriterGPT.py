# Recovered Langflow component
# type: RetryResponseStageWriterGPT
# class: RetryResponseStageWriterGPT
# used in 10 flow(s): Cognitive RAG V1.2.0 GPT, Cognitive RAG V1.2.5 GPT, Cognitive RAG V1.3.0 GPT, Cognitive RAG V1.3.0 GPT Loose AIT , Cognitive RAG V1.3.0 GPT Strict, Cognitive RAG V1.3.0 GPT Strict AIT, Retry RAG V1.3.0 GPT, Retry RAG V1.3.0 GPT Loose AIT ...
# json path: node.data.node.template.code.value

import ast
import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import UploadFile

from lfx.custom import Component
from lfx.io import BoolInput, DropdownInput, HandleInput, IntInput, SecretStrInput, StrInput
from lfx.schema import Data, Message
from lfx.services.deps import get_settings_service, get_storage_service, session_scope
from lfx.template.field.base import Output

from langflow.api.v2.files import upload_user_file
from langflow.services.database.models.user.crud import get_user_by_id


class RetryResponseStageWriterGPT(Component):
    display_name = "Retry Response Writer GPT"
    description = (
        "Write a retry-response stage record for the GPT retry path using only the "
        "retry joined record, retry verification/support gate outputs, and final retry message."
    )
    icon = "file-text"
    name = "RetryResponseStageWriterGPT"

    LOCAL_FORMAT_CHOICES = ["json", "jsonl", "txt", "markdown"]
    GDRIVE_FORMAT_CHOICES = ["json", "jsonl", "txt", "docs"]
    KNOWN_EXTENSIONS = {"json", "jsonl", "txt", "md", "markdown", "docs"}

    inputs = [
        DropdownInput(
            name="storage_location",
            display_name="Storage Location",
            options=["Local", "Google Drive"],
            value="Local",
            info="Choose where to save the retry-response stage record.",
            real_time_refresh=True,
        ),
        StrInput(
            name="file_name",
            display_name="File Name",
            value="retry_response_stage_record_gpt",
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
            info="Trim prompt/context-like fields to this many characters. Use 0 for no limit.",
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
            info="Trim stored answer text to this many characters. Use 0 for no limit.",
        ),
        BoolInput(
            name="include_raw_envelope_text",
            display_name="Include Raw Envelope Text",
            value=False,
            advanced=True,
        ),

        HandleInput(
            name="second_joined_record",
            display_name="Retry Joined Record",
            input_types=["Data", "Message"],
            required=True,
        ),
        HandleInput(
            name="second_verification_output",
            display_name="Retry Verification Output",
            input_types=["Data", "Message"],
            required=True,
        ),
        HandleInput(
            name="second_support_gate_output",
            display_name="Retry Support Gate Output",
            input_types=["Data", "Message"],
            required=True,
        ),
        HandleInput(
            name="retry_message",
            display_name="Retry Message",
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
    # Generic helpers
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

    def _parse_json_like(self, raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text:
            return None

        raw = raw_text.strip()

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return None

    def _extract_payload_dict(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, Data) and isinstance(value.data, dict):
            return value.data

        if hasattr(value, "data") and isinstance(value.data, dict):
            return value.data

        text = self._extract_message_text(value)
        parsed = self._parse_json_like(text)
        return parsed or {}

    def _extract_inner_message_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        message_obj = payload.get("message")
        if isinstance(message_obj, dict):
            content = message_obj.get("content")
            if isinstance(content, str):
                parsed = self._parse_json_like(content)
                if isinstance(parsed, dict):
                    return parsed

        content = payload.get("content")
        if isinstance(content, str):
            parsed = self._parse_json_like(content)
            if isinstance(parsed, dict):
                return parsed

        return {}

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
        plain_text_formats = ["txt", "json", "markdown", "md", "csv", "xml", "html", "yaml", "log", "tsv", "jsonl"]
        return fmt.lower() in plain_text_formats

    def _parse_gcp_credentials(self, service_account_key: str) -> Dict[str, Any]:
        parse_errors = []
        credentials_dict = None

        try:
            credentials_dict = json.loads(service_account_key, strict=False)
        except json.JSONDecodeError as e:
            parse_errors.append(f"Standard parse: {e!s}")

        if credentials_dict is None:
            try:
                cleaned_key = service_account_key.strip()
                credentials_dict = json.loads(cleaned_key, strict=False)
            except json.JSONDecodeError as e:
                parse_errors.append(f"Stripped parse: {e!s}")

        if credentials_dict is None:
            try:
                decoded_once = json.loads(service_account_key, strict=False)
                if isinstance(decoded_once, str):
                    credentials_dict = json.loads(decoded_once, strict=False)
                else:
                    credentials_dict = decoded_once
            except json.JSONDecodeError as e:
                parse_errors.append(f"Double-encoded parse: {e!s}")

        if credentials_dict is None:
            try:
                fixed_key = service_account_key.replace("\\n", "\n")
                credentials_dict = json.loads(fixed_key, strict=False)
            except json.JSONDecodeError as e:
                parse_errors.append(f"Newline-fixed parse: {e!s}")

        if credentials_dict is None:
            error_details = "; ".join(parse_errors)
            raise ValueError(
                "Unable to parse service account key JSON. "
                f"Tried multiple strategies: {error_details}"
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
        if isinstance(metrics, dict):
            return {
                "model": metrics.get("model"),
                "created_at": metrics.get("created_at"),
                "done": metrics.get("done"),
                "done_reason": metrics.get("done_reason"),
                "prompt_eval_count": metrics.get("prompt_eval_count"),
                "eval_count": metrics.get("eval_count"),
                "prompt_eval_duration_ns": metrics.get("prompt_eval_duration_ns"),
                "eval_duration_ns": metrics.get("eval_duration_ns"),
                "total_duration_ns": metrics.get("total_duration_ns"),
                "load_duration_ns": metrics.get("load_duration_ns"),
                "wall_time_s": metrics.get("wall_time_s"),
                "requested_think": metrics.get("requested_think"),
                "observed_has_thinking_field": metrics.get("observed_has_thinking_field"),
                "total_tokens": metrics.get("total_tokens"),
                "reasoning_tokens": metrics.get("reasoning_tokens"),
                "status_code": metrics.get("status_code"),
                "request_id": metrics.get("request_id"),
            }

        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            return {
                "model": metrics.get("model"),
                "created_at": metrics.get("created_at"),
                "done": metrics.get("done"),
                "done_reason": metrics.get("done_reason"),
                "prompt_eval_count": metrics.get("prompt_eval_count"),
                "eval_count": metrics.get("eval_count"),
                "prompt_eval_duration_ns": metrics.get("prompt_eval_duration_ns"),
                "eval_duration_ns": metrics.get("eval_duration_ns"),
                "total_duration_ns": metrics.get("total_duration_ns"),
                "load_duration_ns": metrics.get("load_duration_ns"),
                "wall_time_s": metrics.get("wall_time_s"),
                "requested_think": metrics.get("requested_think"),
                "observed_has_thinking_field": metrics.get("observed_has_thinking_field"),
                "total_tokens": metrics.get("total_tokens"),
                "reasoning_tokens": metrics.get("reasoning_tokens"),
                "status_code": metrics.get("status_code"),
                "request_id": metrics.get("request_id"),
            }

        if any(
            k in payload
            for k in [
                "model",
                "prompt_eval_count",
                "eval_count",
                "total_duration",
                "total_duration_ns",
                "total_tokens",
                "reasoning_tokens",
                "status_code",
                "request_id",
            ]
        ):
            return {
                "model": payload.get("model"),
                "created_at": payload.get("created_at"),
                "done": payload.get("done"),
                "done_reason": payload.get("done_reason"),
                "prompt_eval_count": payload.get("prompt_eval_count"),
                "eval_count": payload.get("eval_count"),
                "prompt_eval_duration_ns": payload.get("prompt_eval_duration_ns", payload.get("prompt_eval_duration")),
                "eval_duration_ns": payload.get("eval_duration_ns", payload.get("eval_duration")),
                "total_duration_ns": payload.get("total_duration_ns", payload.get("total_duration")),
                "load_duration_ns": payload.get("load_duration_ns", payload.get("load_duration")),
                "wall_time_s": payload.get("wall_time_s"),
                "requested_think": payload.get("requested_think"),
                "observed_has_thinking_field": payload.get("observed_has_thinking_field"),
                "total_tokens": payload.get("total_tokens"),
                "reasoning_tokens": payload.get("reasoning_tokens"),
                "status_code": payload.get("status_code"),
                "request_id": payload.get("request_id"),
            }

        return {}

    # -------------------------
    # Field selection
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

    def _select_joined_fields(self, joined: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(joined, dict):
            joined = {}

        batch_record = joined.get("batch_output_record", {})
        if not isinstance(batch_record, dict):
            batch_record = {}

        response = batch_record.get("response", {})
        if not isinstance(response, dict):
            response = {}

        body = response.get("body", {})
        if not isinstance(body, dict):
            body = {}

        usage = body.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}

        completion_details = usage.get("completion_tokens_details", {})
        if not isinstance(completion_details, dict):
            completion_details = {}

        answer_text = str(joined.get("batch_answer_text", "") or "").strip()
        prompt_text = str(joined.get("generator_prompt_text", "") or "").strip()
        generation_mode = str(joined.get("generation_mode", "normal") or "normal").strip().lower()
        if generation_mode != "reason":
            generation_mode = "normal"

        out = {
            "record_index": joined.get("record_index"),
            "custom_id": joined.get("custom_id"),
            "generation_mode": generation_mode,
            "generator_prompt_text": self._trim_page_content(prompt_text),
            "generation": {
                "model": body.get("model"),
                "created_at": body.get("created"),
                "done": response.get("status_code") == 200,
                "done_reason": (
                    body.get("choices", [{}])[0].get("finish_reason")
                    if isinstance(body.get("choices"), list) and body.get("choices")
                    else None
                ),
                "answer_text": self._trim_answer_text(answer_text),
            },
            "batch_usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "reasoning_tokens": completion_details.get("reasoning_tokens"),
            },
            "batch_meta": {
                "status_code": response.get("status_code"),
                "request_id": response.get("request_id"),
                "batch_response_id": body.get("id"),
            },
        }

        if bool(self.include_raw_envelope_text):
            out["raw_batch_output_record"] = batch_record

        return out

    def _select_verification_fields(self, verification_value: Any) -> Dict[str, Any]:
        outer = self._extract_payload_dict(verification_value)
        inner = self._extract_inner_message_json(outer)
        src = inner if inner else outer

        if not isinstance(src, dict):
            src = {}

        claims_out = []
        for claim in src.get("claims", []) or []:
            if not isinstance(claim, dict):
                continue
            claims_out.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "claim_text": self._trim_claim_text(str(claim.get("claim_text", "") or "")),
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
            result["raw_text"] = self._extract_message_text(verification_value)

        return result

    def _select_support_gate_fields(self, support_gate_value: Any) -> Dict[str, Any]:
        payload = self._extract_payload_dict(support_gate_value)
        if not isinstance(payload, dict):
            payload = {}

        claim_rows_out = []
        for row in payload.get("claim_rows", []) or []:
            if not isinstance(row, dict):
                continue
            claim_rows_out.append(
                {
                    "claim_id": row.get("claim_id"),
                    "claim_text": self._trim_claim_text(str(row.get("claim_text", "") or "")),
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

        return {
            "decision": payload.get("decision"),
            "generation_mode_in": payload.get("generation_mode_in"),
            "next_generation_mode": payload.get("next_generation_mode"),
            "general_weight": payload.get("general_weight"),
            "critical_weight": payload.get("critical_weight"),
            "final_support_threshold": payload.get("final_support_threshold"),
            "total_claims": payload.get("total_claims"),
            "total_general_claims": payload.get("total_general_claims"),
            "total_critical_claims": payload.get("total_critical_claims"),
            "raw_mean_support_score": payload.get("raw_mean_support_score"),
            "weighted_score_sum": payload.get("weighted_score_sum"),
            "weighted_total": payload.get("weighted_total"),
            "final_support_score": payload.get("final_support_score"),
            "general_slots_total": payload.get("general_slots_total"),
            "general_slots_answered": payload.get("general_slots_answered"),
            "general_slots_missing": payload.get("general_slots_missing"),
            "critical_slots_total": payload.get("critical_slots_total"),
            "critical_slots_answered": payload.get("critical_slots_answered"),
            "critical_slots_missing": payload.get("critical_slots_missing"),
            "claims_per_slot": payload.get("claims_per_slot", {}),
            "claim_rows": claim_rows_out,
            "general_slot_rows": _slot_rows(payload.get("general_slot_rows", [])),
            "critical_slot_rows": _slot_rows(payload.get("critical_slot_rows", [])),
        }

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

        if not message_text:
            message_text = self._extract_message_text(response_value)

        result = {
            "text": self._trim_answer_text(str(message_text or "").strip()),
        }

        if bool(self.include_raw_envelope_text):
            result["raw_text"] = self._extract_message_text(response_value)

        return result

    def _build_attempt_metrics_from_joined(self, joined_fields: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model": joined_fields.get("generation", {}).get("model"),
            "created_at": joined_fields.get("generation", {}).get("created_at"),
            "done": joined_fields.get("generation", {}).get("done"),
            "done_reason": joined_fields.get("generation", {}).get("done_reason"),
            "prompt_eval_count": joined_fields.get("batch_usage", {}).get("prompt_tokens"),
            "eval_count": joined_fields.get("batch_usage", {}).get("completion_tokens"),
            "total_tokens": joined_fields.get("batch_usage", {}).get("total_tokens"),
            "reasoning_tokens": joined_fields.get("batch_usage", {}).get("reasoning_tokens"),
            "requested_think": joined_fields.get("generation_mode") == "reason",
            "status_code": joined_fields.get("batch_meta", {}).get("status_code"),
            "request_id": joined_fields.get("batch_meta", {}).get("request_id"),
        }

    def _build_record(self) -> Dict[str, Any]:
        retry_joined_payload = self._extract_payload_dict(self.second_joined_record)
        retry_verification_payload = self._extract_payload_dict(self.second_verification_output)
        retry_support_gate_payload = self._extract_payload_dict(self.second_support_gate_output)
        retry_message_payload = self._extract_payload_dict(self.retry_message)

        analyzer_payload = retry_joined_payload.get("analyzer_data", {})
        if not isinstance(analyzer_payload, dict):
            analyzer_payload = {}

        retry_joined_fields = self._select_joined_fields(retry_joined_payload)

        return {
            "record_type": "retry_response_stage_writer_gpt",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "analyzer": self._select_analyzer_fields(analyzer_payload),
            "retry_attempt": {
                "joined": {
                    "record_index": retry_joined_fields.get("record_index"),
                    "custom_id": retry_joined_fields.get("custom_id"),
                    "generation_mode": retry_joined_fields.get("generation_mode"),
                    "generator_prompt_text": retry_joined_fields.get("generator_prompt_text"),
                    "batch_usage": retry_joined_fields.get("batch_usage", {}),
                    "batch_meta": retry_joined_fields.get("batch_meta", {}),
                },
                "generation": retry_joined_fields.get("generation", {}),
                "verification": self._select_verification_fields(self.second_verification_output),
                "support_gate": self._select_support_gate_fields(self.second_support_gate_output),
            },
            "final_response": self._select_final_response_fields(self.retry_message),
            "stage_metrics": {
                "retry_generation": self._build_attempt_metrics_from_joined(retry_joined_fields),
                "retry_verification": self._extract_metrics(retry_verification_payload),
                "retry_support_gate": self._extract_metrics(retry_support_gate_payload),
                "retry_message": self._extract_metrics(retry_message_payload),
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

        analyzer = record.get("analyzer", {})
        retry_attempt = record.get("retry_attempt", {})
        final_response = record.get("final_response", {})
        metrics = record.get("stage_metrics", {})

        lines = [
            f"Saved At (UTC): {record.get('saved_at_utc', '')}",
            "",
            "=== ANALYZER ===",
            json.dumps(analyzer, ensure_ascii=False, indent=2),
            "",
            "=== RETRY ATTEMPT ===",
            json.dumps(retry_attempt, ensure_ascii=False, indent=2),
            "",
            "=== FINAL RETRY MESSAGE ===",
            final_response.get("text", ""),
            "",
            "=== STAGE METRICS ===",
            json.dumps(metrics, ensure_ascii=False, indent=2),
        ]

        if fmt == "markdown":
            md_lines = [
                "# Retry Response Stage Record GPT",
                "",
                f"**Saved At (UTC):** {record.get('saved_at_utc', '')}",
                "",
                "## Analyzer",
                "```json",
                json.dumps(analyzer, ensure_ascii=False, indent=2),
                "```",
                "",
                "## Retry Attempt",
                "```json",
                json.dumps(retry_attempt, ensure_ascii=False, indent=2),
                "```",
                "",
                "## Final Retry Message",
                final_response.get("text", ""),
                "",
                "## Stage Metrics",
                "```json",
                json.dumps(metrics, ensure_ascii=False, indent=2),
                "```",
            ]
            return "\n".join(md_lines)

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
            path.write_text(json.dumps(existing_data, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        action = "appended to" if should_append else "saved successfully as"
        return f"Retry GPT response stage record {action} '{path}'"

    def _write_jsonl(self, path: Path, record: Dict[str, Any], should_append: bool) -> str:
        line = self._record_to_jsonl_line(record)

        if should_append:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            path.write_text(line + "\n", encoding="utf-8")

        action = "appended to" if should_append else "saved successfully as"
        return f"Retry GPT response stage record {action} '{path}'"

    def _write_textlike(self, path: Path, content: str, should_append: bool) -> str:
        if should_append:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            sep = "\n\n---\n\n" if existing else ""
            path.write_text(existing + sep + content, encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")

        action = "appended to" if should_append else "saved successfully as"
        return f"Retry GPT response stage record {action} '{path}'"

    # -------------------------
    # Save backends
    # -------------------------
    async def _save_local(self, record: Dict[str, Any]) -> Message:
        fmt = self.local_format or "json"
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
            return Message(text=f"Retry GPT response stage record uploaded to Google Drive: {file_url}")
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

        created_file = drive_service.files().create(body=file_metadata, fields="id").execute()
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
        return Message(text=f"Retry GPT response stage record created in Google Docs: {file_url}")

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