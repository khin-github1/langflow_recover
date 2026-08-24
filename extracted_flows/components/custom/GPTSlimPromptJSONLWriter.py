# Recovered Langflow component
# type: GPTSlimPromptJSONLWriter
# class: GPTSlimPromptJSONLWriter
# used in 3 flow(s): Cognitive RAG V1.2.0 GPT, Cognitive RAG V1.2.5 GPT, Cognitive RAG V1.3.0 GPT Ingest
# json path: node.data.node.template.code.value

import ast
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import UploadFile

from lfx.custom import Component
from lfx.io import BoolInput, HandleInput, Output, StrInput
from lfx.schema import Data, Message
from lfx.services.deps import get_settings_service, get_storage_service, session_scope

from langflow.api.v2.files import upload_user_file
from langflow.services.database.models.user.crud import get_user_by_id


class GPTSlimPromptJSONLWriter(Component):
    display_name = "GPT Slim Prompt JSONL Writer"
    description = (
        "Extract only generation_mode and generator_prompt.raw_prompt_text from a large "
        "record payload and save them into a new JSONL file."
    )
    icon = "file-text"
    name = "GPTSlimPromptJSONLWriter"

    inputs = [
        HandleInput(
            name="input_payload",
            display_name="Input Payload",
            input_types=["Data", "Message"],
            required=True,
            info="Either a single saved record or a wrapper containing records[].",
        ),
        StrInput(
            name="file_name",
            display_name="Base File Name",
            value="gpt_ready_prompts",
            required=True,
            tool_mode=True,
            info="Output file name. Extension optional. If split_by_mode=true, _normal/_reason will be added.",
        ),
        BoolInput(
            name="append_mode",
            display_name="Append Mode",
            value=True,
            required=False,
        ),
        BoolInput(
            name="split_by_mode",
            display_name="Split By Mode",
            value=False,
            required=False,
            info="If true, save separate files for normal and reason modes.",
        ),
        BoolInput(
            name="skip_invalid_records",
            display_name="Skip Invalid Records",
            value=True,
            required=False,
            info="If true, records missing generation_mode or raw_prompt_text are skipped instead of raising an error.",
        ),
    ]

    outputs = [
        Output(
            display_name="Message",
            name="message",
            method="write_jsonl",
        ),
    ]

    # -------------------------
    # Generic helpers
    # -------------------------
    def _extract_raw_text(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, Message):
            try:
                return str(value.text or "").strip()
            except Exception:
                return str(value).strip()

        if isinstance(value, Data):
            if isinstance(value.data, dict):
                return json.dumps(value.data, ensure_ascii=False)
            return str(value).strip()

        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()

        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()

        if hasattr(value, "data") and isinstance(value.data, dict):
            return json.dumps(value.data, ensure_ascii=False)

        return str(value).strip()

    def _parse_json_like(self, raw_text: str) -> Dict[str, Any]:
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

    def _extract_payload_dict(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, Data) and isinstance(value.data, dict):
            return value.data

        if hasattr(value, "data") and isinstance(value.data, dict):
            return value.data

        raw_text = self._extract_raw_text(value)
        return self._parse_json_like(raw_text)

    def _normalize_mode(self, value: Any) -> str:
        mode = str(value or "").strip().lower()
        return "reason" if mode == "reason" else "normal"

    def _base_path(self) -> Path:
        raw = str(self.file_name or "").strip()
        if not raw:
            raise ValueError("file_name is required.")

        p = Path(raw).expanduser()
        if p.suffix.lower() == ".jsonl":
            p = p.with_suffix("")
        return p

    def _path_for_mode(self, mode: str) -> Path:
        base = self._base_path()
        if bool(self.split_by_mode):
            return base.with_name(f"{base.name}_{mode}").with_suffix(".jsonl")
        return base.with_suffix(".jsonl")

    # -------------------------
    # Record extraction
    # -------------------------
    def _get_records_list(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict) or not payload:
            return []

        # case 1: wrapper payload from reader {record_count, records:[...]}
        records = payload.get("records")
        if isinstance(records, list):
            return [r for r in records if isinstance(r, dict)]

        # case 2: single record directly
        if "generator_prompt" in payload or "generation_mode" in payload:
            return [payload]

        return []

    def _extract_slim_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        generation_mode_block = record.get("generation_mode", {})
        if isinstance(generation_mode_block, dict):
            generation_mode = self._normalize_mode(generation_mode_block.get("generation_mode"))
        else:
            generation_mode = self._normalize_mode(generation_mode_block)

        generator_prompt = record.get("generator_prompt", {})
        raw_prompt_text = ""

        if isinstance(generator_prompt, dict):
            raw_prompt_text = str(generator_prompt.get("raw_prompt_text", "") or "").strip()
        elif isinstance(generator_prompt, str):
            raw_prompt_text = generator_prompt.strip()

        if not raw_prompt_text:
            raise ValueError("Missing generator_prompt.raw_prompt_text")

        return {
            "generation_mode": generation_mode,
            "raw_prompt_text": raw_prompt_text,
        }

    def _build_slim_records(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        payload = self._extract_payload_dict(self.input_payload)
        records = self._get_records_list(payload)

        if not records:
            raise ValueError(
                "No valid records found. Input must be either a single record or a wrapper containing records[]."
            )

        slim_records: List[Dict[str, Any]] = []
        skipped: List[str] = []

        for idx, record in enumerate(records, start=1):
            try:
                slim = self._extract_slim_record(record)
                slim_records.append(slim)
            except Exception as e:
                msg = f"record_{idx}: {e}"
                if bool(self.skip_invalid_records):
                    skipped.append(msg)
                    continue
                raise ValueError(msg) from e

        if not slim_records:
            raise ValueError(f"All records were invalid. Skipped: {skipped}")

        return slim_records, skipped

    def _group_lines_by_path(self, slim_records: List[Dict[str, Any]]) -> Dict[Path, List[str]]:
        grouped: Dict[Path, List[str]] = {}

        for rec in slim_records:
            mode = self._normalize_mode(rec.get("generation_mode"))
            path = self._path_for_mode(mode)
            line = json.dumps(rec, ensure_ascii=False)
            grouped.setdefault(path, []).append(line)

        return grouped

    # -------------------------
    # Langflow file registration
    # -------------------------
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
    # Writer
    # -------------------------
    async def write_jsonl(self) -> Message:
        slim_records, skipped = self._build_slim_records()
        grouped = self._group_lines_by_path(slim_records)

        saved_files = []
        total_written = 0

        for path, lines in grouped.items():
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

            content = "\n".join(lines) + "\n"
            should_append = bool(self.append_mode) and path.exists()

            if should_append:
                with path.open("a", encoding="utf-8") as f:
                    f.write(content)
                await self._upload_bytes_to_langflow(
                    path.name,
                    content.encode("utf-8"),
                    append=True,
                )
            else:
                path.write_text(content, encoding="utf-8")
                await self._upload_file_to_langflow(path, append=False)

            total_written += len(lines)
            final_path = Path.cwd() / path if not path.is_absolute() else path
            saved_files.append(str(final_path))

        result = {
            "written_records": total_written,
            "saved_files": saved_files,
            "split_by_mode": bool(self.split_by_mode),
            "skipped_records": skipped,
        }

        self.status = json.dumps(result, ensure_ascii=False)
        return Message(
            text=f"Saved {total_written} slim GPT prompt record(s) into {len(saved_files)} file(s).",
            data=result,
        )