# Recovered Langflow component
# type: ClarificationStageWriter
# class: ClarificationStageWriter
# used in 16 flow(s): AIT Cognitive RAG, Cognitive RAG V1.1.5, Cognitive RAG V1.2.0 (1), Cognitive RAG V1.2.0 AIT CR, Cognitive RAG V1.2.0 AIT Loose, Cognitive RAG V1.2.0 AIT Normal, Cognitive RAG V1.2.0 AIT RR, Cognitive RAG V1.2.0 AIT Reason ...
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


class ClarificationStageWriter(Component):
    display_name = "Clarification Stage Writer"
    description = (
        "Write a compact clarification-stage record from analyzer output, "
        "clarification tracker output, and clarification message envelope. "
        "Supports Local and Google Drive, including JSONL for appendable logs."
    )
    icon = "file-text"
    name = "ClarificationStageWriter"

    LOCAL_FORMAT_CHOICES = ["json", "jsonl", "txt", "markdown"]
    GDRIVE_FORMAT_CHOICES = ["json", "jsonl", "txt", "docs"]
    KNOWN_EXTENSIONS = {"json", "jsonl", "txt", "md", "markdown", "docs"}

    inputs = [
        DropdownInput(
            name="storage_location",
            display_name="Storage Location",
            options=["Local", "Google Drive"],
            value="Google Drive",
            info="Choose where to save the clarification-stage record.",
            real_time_refresh=True,
        ),
        StrInput(
            name="file_name",
            display_name="File Name",
            value="clarification_stage_record",
            required=True,
            info="Saved file name. You may include or omit extension; it will be normalized.",
            tool_mode=True,
        ),
        DropdownInput(
            name="local_format",
            display_name="Local Format",
            options=LOCAL_FORMAT_CHOICES,
            value="json",
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
            value=False,
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
        BoolInput(
            name="include_raw_envelope_text",
            display_name="Include Raw Envelope Text",
            value=False,
            advanced=True,
            info="Include raw envelope text fields in the saved record.",
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
            required=True,
        ),
        HandleInput(
            name="clarification_message_envelope",
            display_name="Clarification Message Envelope",
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

    def _trim_history(self, text: str) -> str:
        limit = int(self.history_char_limit or 0)
        if limit <= 0:
            return text or ""
        text = text or ""
        if len(text) <= limit:
            return text
        return text[-limit:]

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
    # Field selection
    # -------------------------
    def _extract_metrics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        metrics = payload.get("llm_metrics")
        if isinstance(metrics, dict):
            return {
                "model": metrics.get("model"),
                "prompt_eval_count": metrics.get("prompt_eval_count"),
                "eval_count": metrics.get("eval_count"),
                "prompt_eval_duration_ns": metrics.get("prompt_eval_duration_ns"),
                "eval_duration_ns": metrics.get("eval_duration_ns"),
                "total_duration_ns": metrics.get("total_duration_ns"),
                "load_duration_ns": metrics.get("load_duration_ns"),
                "wall_time_s": metrics.get("wall_time_s"),
                "requested_think": metrics.get("requested_think"),
                "observed_has_thinking_field": metrics.get("observed_has_thinking_field"),
            }

        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            return {
                "model": metrics.get("model"),
                "prompt_eval_count": metrics.get("prompt_eval_count"),
                "eval_count": metrics.get("eval_count"),
                "prompt_eval_duration_ns": metrics.get("prompt_eval_duration_ns"),
                "eval_duration_ns": metrics.get("eval_duration_ns"),
                "total_duration_ns": metrics.get("total_duration_ns"),
                "load_duration_ns": metrics.get("load_duration_ns"),
                "wall_time_s": metrics.get("wall_time_s"),
                "requested_think": metrics.get("requested_think"),
                "observed_has_thinking_field": metrics.get("observed_has_thinking_field"),
            }

        return {}

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

        matched_type = tracker.get("matched_clarification_type", "none")

        result = {
            "clarification_asked_in_last_ai_turn": bool(clar_last),
            "matched_clarification_type": str(matched_type or "none"),
        }

        if "last_ai_turn_text" in tracker:
            result["last_ai_turn_text"] = str(tracker.get("last_ai_turn_text", "") or "")

        return result

    def _select_clarification_message_fields(self, envelope_value: Any) -> Dict[str, Any]:
        payload = self._extract_payload_dict(envelope_value)
        message_text = ""

        if isinstance(payload, dict) and payload:
            message_text = (
                payload.get("clarification_question")
                or payload.get("question")
                or payload.get("message")
                or payload.get("text")
                or ""
            )

        if not message_text:
            message_text = self._extract_message_text(envelope_value)

        result = {
            "text": str(message_text or "").strip(),
        }

        if bool(self.include_raw_envelope_text):
            result["raw_text"] = self._extract_message_text(envelope_value)

        return result

    def _build_record(self) -> Dict[str, Any]:
        analyzer_payload = self._extract_payload_dict(self.analyzer_output)
        tracker_payload = self._extract_payload_dict(self.clarification_tracking)
        clarification_envelope_payload = self._extract_payload_dict(self.clarification_message_envelope)

        return {
            "record_type": "clarification_stage_writer",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "analyzer": self._select_analyzer_fields(analyzer_payload),
            "clarification_tracking": self._select_tracker_fields(tracker_payload),
            "clarification_message": self._select_clarification_message_fields(
                self.clarification_message_envelope
            ),
            "stage_metrics": {
                "analyzer": self._extract_metrics(analyzer_payload),
                "clarification_tracking": self._extract_metrics(tracker_payload),
                "clarification_message": self._extract_metrics(clarification_envelope_payload),
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
        tracker = record.get("clarification_tracking", {})
        clarification_message = record.get("clarification_message", {})
        metrics = record.get("stage_metrics", {})

        lines = [
            f"Saved At (UTC): {record.get('saved_at_utc', '')}",
            "",
            "=== ANALYZER ===",
            f"Original Question: {analyzer.get('original_question', '')}",
            f"History Window: {analyzer.get('history_window', '')}",
            f"Intent Type: {analyzer.get('intent_type', '')}",
            f"Request Type: {analyzer.get('request_type', '')}",
            f"Effective Request Types: {analyzer.get('effective_request_types', [])}",
            f"Topic: {analyzer.get('topic', '')}",
            f"Expected General Slots: {analyzer.get('expected_general_slots', [])}",
            f"Expected Critical Slots: {analyzer.get('expected_critical_slots', [])}",
            f"Required General Slots: {analyzer.get('required_general_slots', [])}",
            f"Required Critical Slots: {analyzer.get('required_critical_slots', [])}",
            f"Missing Slots: {analyzer.get('missing_slots', {})}",
            f"Slot Coverage General: {analyzer.get('slot_coverage_general', None)}",
            f"Slot Coverage Critical: {analyzer.get('slot_coverage_critical', None)}",
            f"Slot Coverage Weighted: {analyzer.get('slot_coverage_weighted', None)}",
            f"Counts: {analyzer.get('counts', {})}",
            f"Resolved Slots: {analyzer.get('resolved_slots', {})}",
            f"Refinement: {analyzer.get('refinement', {})}",
            f"Clarification Decision: {analyzer.get('clarification_decision', {})}",
            f"Clarification Needed: {analyzer.get('clarification_needed', None)}",
            "",
            "=== CLARIFICATION TRACKING ===",
            f"Clarification Asked In Last AI Turn: {tracker.get('clarification_asked_in_last_ai_turn', False)}",
            f"Matched Clarification Type: {tracker.get('matched_clarification_type', 'none')}",
            f"Last AI Turn Text: {tracker.get('last_ai_turn_text', '')}",
            "",
            "=== CLARIFICATION MESSAGE ===",
            clarification_message.get("text", ""),
            "",
            "=== STAGE METRICS ===",
            f"Analyzer Metrics: {metrics.get('analyzer', {})}",
            f"Clarification Tracking Metrics: {metrics.get('clarification_tracking', {})}",
            f"Clarification Message Metrics: {metrics.get('clarification_message', {})}",
        ]

        if fmt == "markdown":
            md_lines = [
                "# Clarification Stage Record",
                "",
                f"**Saved At (UTC):** {record.get('saved_at_utc', '')}",
                "",
                "## Analyzer",
                f"- **Original Question:** {analyzer.get('original_question', '')}",
                f"- **History Window:** {analyzer.get('history_window', '')}",
                f"- **Intent Type:** `{analyzer.get('intent_type', '')}`",
                f"- **Request Type:** `{analyzer.get('request_type', '')}`",
                f"- **Effective Request Types:** `{analyzer.get('effective_request_types', [])}`",
                f"- **Topic:** `{analyzer.get('topic', '')}`",
                f"- **Expected General Slots:** `{analyzer.get('expected_general_slots', [])}`",
                f"- **Expected Critical Slots:** `{analyzer.get('expected_critical_slots', [])}`",
                f"- **Required General Slots:** `{analyzer.get('required_general_slots', [])}`",
                f"- **Required Critical Slots:** `{analyzer.get('required_critical_slots', [])}`",
                f"- **Missing Slots:** `{analyzer.get('missing_slots', {})}`",
                f"- **Slot Coverage General:** `{analyzer.get('slot_coverage_general', None)}`",
                f"- **Slot Coverage Critical:** `{analyzer.get('slot_coverage_critical', None)}`",
                f"- **Slot Coverage Weighted:** `{analyzer.get('slot_coverage_weighted', None)}`",
                f"- **Counts:** `{analyzer.get('counts', {})}`",
                f"- **Resolved Slots:** `{analyzer.get('resolved_slots', {})}`",
                f"- **Refinement:** `{analyzer.get('refinement', {})}`",
                f"- **Clarification Decision:** `{analyzer.get('clarification_decision', {})}`",
                f"- **Clarification Needed:** `{analyzer.get('clarification_needed', None)}`",
                "",
                "## Clarification Tracking",
                f"- **Clarification Asked In Last AI Turn:** `{tracker.get('clarification_asked_in_last_ai_turn', False)}`",
                f"- **Matched Clarification Type:** `{tracker.get('matched_clarification_type', 'none')}`",
                f"- **Last AI Turn Text:** {tracker.get('last_ai_turn_text', '')}",
                "",
                "## Clarification Message",
                clarification_message.get("text", ""),
                "",
                "## Stage Metrics",
                f"- **Analyzer:** `{metrics.get('analyzer', {})}`",
                f"- **Clarification Tracking:** `{metrics.get('clarification_tracking', {})}`",
                f"- **Clarification Message:** `{metrics.get('clarification_message', {})}`",
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
        return f"Clarification stage record {action} '{path}'"

    def _write_jsonl(self, path: Path, record: Dict[str, Any], should_append: bool) -> str:
        line = self._record_to_jsonl_line(record)

        if should_append:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            path.write_text(line + "\n", encoding="utf-8")

        action = "appended to" if should_append else "saved successfully as"
        return f"Clarification stage record {action} '{path}'"

    def _write_textlike(self, path: Path, content: str, should_append: bool) -> str:
        if should_append:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            sep = "\n\n---\n\n" if existing else ""
            path.write_text(existing + sep + content, encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")

        action = "appended to" if should_append else "saved successfully as"
        return f"Clarification stage record {action} '{path}'"

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
            # JSON append is handled correctly locally, but Langflow Files does not support
            # safe structured overwrite via append. So we re-upload the whole file.
            await self._upload_file_to_langflow(path, append=False)

        elif fmt == "jsonl":
            confirmation = self._write_jsonl(path, record, should_append)

            line_bytes = (self._record_to_jsonl_line(record) + "\n").encode("utf-8")
            if should_append:
                # Append only the delta to the same Langflow-managed file
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
            return Message(text=f"Clarification stage record uploaded to Google Drive: {file_url}")
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
        return Message(text=f"Clarification stage record created in Google Docs: {file_url}")

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