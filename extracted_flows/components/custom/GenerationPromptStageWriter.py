# Recovered Langflow component
# type: GenerationPromptStageWriter
# class: GenerationPromptStageWriter
# used in 1 flow(s): Cognitive RAG V1.1.5
# json path: node.data.node.template.code.value

import ast
import io
import json
import re
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


class GenerationPromptStageWriter(Component):
    display_name = "Generation Prompt Stage Writer"
    description = (
        "Writes the generation-mode output and generator prompt input "
        "into a structured local or Google Drive record. Supports JSONL append."
    )
    icon = "file-text"
    name = "GenerationPromptStageWriter"

    LOCAL_FORMAT_CHOICES = ["json", "jsonl", "txt", "markdown"]
    GDRIVE_FORMAT_CHOICES = ["json", "jsonl", "txt", "docs"]
    KNOWN_EXTENSIONS = {"json", "jsonl", "txt", "md", "markdown", "docs"}

    inputs = [
        DropdownInput(
            name="storage_location",
            display_name="Storage Location",
            options=["Local", "Google Drive"],
            value="Local",
            info="Choose where to save the record.",
            real_time_refresh=True,
        ),
        StrInput(
            name="file_name",
            display_name="File Name",
            value="generation_prompt_stage_record",
            required=True,
            info="Saved file name. Extension will be normalized.",
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
            name="prompt_char_limit",
            display_name="Prompt Char Limit",
            value=12000,
            advanced=True,
            info="Trim stored raw prompt text to this many characters. Use 0 for no limit.",
        ),
        IntInput(
            name="context_chunk_char_limit",
            display_name="Context Chunk Char Limit",
            value=2000,
            advanced=True,
            info="Trim each stored context chunk to this many characters. Use 0 for no limit.",
        ),
        BoolInput(
            name="include_raw_prompt_text",
            display_name="Include Raw Prompt Text",
            value=True,
            advanced=True,
        ),
        HandleInput(
            name="generation_mode_input",
            display_name="Generation Mode Input",
            input_types=["Data", "Message"],
            required=True,
        ),
        HandleInput(
            name="generator_prompt_input",
            display_name="Generator Prompt Input",
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

    def _trim_text(self, text: str, limit: int) -> str:
        limit = int(limit or 0)
        text = text or ""
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit]

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
    # Extraction / parsing
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

    def _select_generation_mode_fields(self, mode_value: Any) -> Dict[str, Any]:
        payload = self._extract_payload_dict(mode_value)
        raw_text = self._extract_message_text(mode_value)

        if isinstance(payload, dict) and payload:
            generation_mode = str(payload.get("generation_mode", "") or "").strip()
            if generation_mode:
                return {"generation_mode": generation_mode}

        parsed = self._parse_json_like(raw_text)
        if isinstance(parsed, dict):
            generation_mode = str(parsed.get("generation_mode", "") or "").strip()
            if generation_mode:
                return {"generation_mode": generation_mode}

        return {"generation_mode": raw_text.strip()}

    def _parse_generator_prompt(self, prompt_text: str) -> Dict[str, Any]:
        prompt_text = prompt_text or ""
        user_query = ""
        context_blocks: List[Dict[str, Any]] = []

        m = re.search(r"User Query:\s*(.*?)\n\s*\nContext:\s*", prompt_text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            user_query = m.group(1).strip()
            context_text = prompt_text[m.end():]
        else:
            m2 = re.search(r"User Query:\s*(.*)", prompt_text, flags=re.IGNORECASE)
            if m2:
                user_query = m2.group(1).strip()
            context_text = ""

        if context_text:
            block_matches = list(
                re.finditer(r"(?m)^\[(\d+)\]\s*", context_text)
            )

            for i, match in enumerate(block_matches):
                chunk_id = int(match.group(1))
                start = match.end()
                end = block_matches[i + 1].start() if i + 1 < len(block_matches) else len(context_text)
                block_text = context_text[start:end].strip()

                source = ""
                source_match = re.search(r"Source:\s*(\S+)", block_text, flags=re.IGNORECASE)
                if source_match:
                    source = source_match.group(1).strip()

                content_only = re.sub(r"\n?Source:\s*\S+\s*$", "", block_text, flags=re.IGNORECASE).strip()

                context_blocks.append(
                    {
                        "index": chunk_id,
                        "page_content": self._trim_text(content_only, int(self.context_chunk_char_limit or 0)),
                        "source": source,
                    }
                )

        return {
            "user_query": user_query,
            "context_block_count": len(context_blocks),
            "context_blocks": context_blocks,
            "sources": [b["source"] for b in context_blocks if b.get("source")],
        }

    def _select_generator_prompt_fields(self, prompt_value: Any) -> Dict[str, Any]:
        prompt_payload = self._extract_payload_dict(prompt_value)
        prompt_text = ""

        if isinstance(prompt_payload, dict) and prompt_payload:
            prompt_text = (
                str(prompt_payload.get("text", "")).strip()
                or str(prompt_payload.get("prompt", "")).strip()
                or ""
            )

        if not prompt_text:
            prompt_text = self._extract_message_text(prompt_value)

        parsed = self._parse_generator_prompt(prompt_text)

        result = {
            "user_query": parsed.get("user_query", ""),
            "context_block_count": parsed.get("context_block_count", 0),
            "context_blocks": parsed.get("context_blocks", []),
            "sources": parsed.get("sources", []),
        }

        if bool(self.include_raw_prompt_text):
            result["raw_prompt_text"] = self._trim_text(prompt_text, int(self.prompt_char_limit or 0))

        return result

    def _build_record(self) -> Dict[str, Any]:
        generation_mode_payload = self._extract_payload_dict(self.generation_mode_input)
        generator_prompt_payload = self._extract_payload_dict(self.generator_prompt_input)

        return {
            "record_type": "generation_prompt_stage_writer",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation_mode": self._select_generation_mode_fields(self.generation_mode_input),
            "generator_prompt": self._select_generator_prompt_fields(self.generator_prompt_input),
            "stage_metrics": {
                "generation_mode": self._extract_metrics(generation_mode_payload),
                "generator_prompt": self._extract_metrics(generator_prompt_payload),
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

        generation_mode = record.get("generation_mode", {})
        generator_prompt = record.get("generator_prompt", {})
        metrics = record.get("stage_metrics", {})

        lines = [
            f"Saved At (UTC): {record.get('saved_at_utc', '')}",
            "",
            "=== GENERATION MODE ===",
            f"Generation Mode: {generation_mode.get('generation_mode', '')}",
            "",
            "=== GENERATOR PROMPT ===",
            f"User Query: {generator_prompt.get('user_query', '')}",
            f"Context Block Count: {generator_prompt.get('context_block_count', 0)}",
            f"Sources: {generator_prompt.get('sources', [])}",
            "",
            "=== CONTEXT BLOCKS ===",
            json.dumps(generator_prompt.get("context_blocks", []), ensure_ascii=False, indent=2),
            "",
            "=== STAGE METRICS ===",
            f"Generation Mode Metrics: {metrics.get('generation_mode', {})}",
            f"Generator Prompt Metrics: {metrics.get('generator_prompt', {})}",
        ]

        if generator_prompt.get("raw_prompt_text"):
            lines.extend(
                [
                    "",
                    "=== RAW PROMPT TEXT ===",
                    generator_prompt.get("raw_prompt_text", ""),
                ]
            )

        if fmt == "markdown":
            md_lines = [
                "# Generation Prompt Stage Record",
                "",
                f"**Saved At (UTC):** {record.get('saved_at_utc', '')}",
                "",
                "## Generation Mode",
                f"- **Generation Mode:** `{generation_mode.get('generation_mode', '')}`",
                "",
                "## Generator Prompt",
                f"- **User Query:** {generator_prompt.get('user_query', '')}",
                f"- **Context Block Count:** `{generator_prompt.get('context_block_count', 0)}`",
                f"- **Sources:** `{generator_prompt.get('sources', [])}`",
                "",
                "## Context Blocks",
                "```json",
                json.dumps(generator_prompt.get("context_blocks", []), ensure_ascii=False, indent=2),
                "```",
                "",
                "## Stage Metrics",
                f"- **Generation Mode:** `{metrics.get('generation_mode', {})}`",
                f"- **Generator Prompt:** `{metrics.get('generator_prompt', {})}`",
            ]

            if generator_prompt.get("raw_prompt_text"):
                md_lines.extend(
                    [
                        "",
                        "## Raw Prompt Text",
                        "```text",
                        generator_prompt.get("raw_prompt_text", ""),
                        "```",
                    ]
                )

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
        return f"Generation prompt stage record {action} '{path}'"

    def _write_jsonl(self, path: Path, record: Dict[str, Any], should_append: bool) -> str:
        line = self._record_to_jsonl_line(record)

        if should_append:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            path.write_text(line + "\n", encoding="utf-8")

        action = "appended to" if should_append else "saved successfully as"
        return f"Generation prompt stage record {action} '{path}'"

    def _write_textlike(self, path: Path, content: str, should_append: bool) -> str:
        if should_append:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            sep = "\n\n---\n\n" if existing else ""
            path.write_text(existing + sep + content, encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")

        action = "appended to" if should_append else "saved successfully as"
        return f"Generation prompt stage record {action} '{path}'"

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
            return Message(text=f"Generation prompt stage record uploaded to Google Drive: {file_url}")
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
        return Message(text=f"Generation prompt stage record created in Google Docs: {file_url}")

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