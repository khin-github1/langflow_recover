# Recovered Langflow component
# type: BaselineRAGFinalWriter
# class: BaselineRAGFinalWriter
# used in 7 flow(s): Baseline Normal AIT, Baseline Normal AIT 123, Baseline Normal Squad, Baseline Reason AIT, Baseline Reason Squad, FAQ System A, FAQ System B
# json path: node.data.node.template.code.value

import io
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import UploadFile

from lfx.custom import Component
from lfx.io import BoolInput, HandleInput, StrInput
from lfx.schema import Data, Message
from lfx.services.deps import get_settings_service, get_storage_service, session_scope
from lfx.template.field.base import Output

from langflow.api.v2.files import upload_user_file
from langflow.services.database.models.user.crud import get_user_by_id


class BaselineRAGFinalWriter(Component):
    display_name = "Baseline RAG Final Writer"
    description = "Writes {user_message, assistant_text, generation, retrieval} to JSONL and passes assistant_text forward."
    icon = "file-text"
    name = "BaselineRAGFinalWriter"

    inputs = [
        HandleInput(
            name="user_message",
            display_name="User Message / Incoming Question",
            input_types=["Message", "Data"],
            required=True,
        ),
        HandleInput(
            name="assistant_message",
            display_name="Assistant Message / Final Answer",
            input_types=["Message", "Data"],
            required=True,
        ),
        HandleInput(
            name="generation_envelope",
            display_name="Generation Envelope",
            input_types=["Data", "Message"],
            required=True,
        ),
        HandleInput(
            name="retrieval_envelope",
            display_name="Retrieval Envelope / PGVector Output",
            input_types=["Data", "Message"],
            required=True,
        ),
        StrInput(
            name="file_name",
            display_name="Output JSONL File",
            value="baseline_rag_results",
            required=True,
            advanced=True,
            tool_mode=True,
        ),
        BoolInput(
            name="append_mode",
            display_name="Append Mode",
            value=True,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Assistant Message", name="assistant_output", method="write_record"),
    ]

    def _parse_json_text(self, text: Any) -> Optional[Any]:
        if not isinstance(text, str):
            return None
        raw = text.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _message_text(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, Message):
            return str(value.text or "").strip()

        if isinstance(value, Data):
            data = value.data
            if isinstance(data, dict):
                for key in ["text", "content", "message", "response", "answer", "input", "question"]:
                    v = data.get(key)
                    if isinstance(v, str):
                        return v.strip()
                    if isinstance(v, dict):
                        content = v.get("content") or v.get("text") or v.get("message")
                        if isinstance(content, str):
                            return content.strip()
                return json.dumps(data, ensure_ascii=False)
            return str(data or "").strip()

        if isinstance(value, dict):
            for key in ["text", "content", "message", "response", "answer", "input", "question"]:
                v = value.get(key)
                if isinstance(v, str):
                    return v.strip()
                if isinstance(v, dict):
                    content = v.get("content") or v.get("text") or v.get("message")
                    if isinstance(content, str):
                        return content.strip()
            return json.dumps(value, ensure_ascii=False)

        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()

        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()

        return str(value).strip()

    def _to_plain(self, value: Any) -> Any:
        if value is None:
            return {}

        if isinstance(value, Data):
            return self._to_plain(value.data)

        if isinstance(value, Message):
            text = str(value.text or "").strip()
            parsed = self._parse_json_text(text)
            return parsed if parsed is not None else {"text": text}

        if isinstance(value, dict):
            return {str(k): self._to_plain(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._to_plain(v) for v in value]

        if hasattr(value, "data"):
            try:
                return self._to_plain(value.data)
            except Exception:
                pass

        if hasattr(value, "text"):
            try:
                text = str(value.text or "").strip()
                parsed = self._parse_json_text(text)
                return parsed if parsed is not None else {"text": text}
            except Exception:
                pass

        if hasattr(value, "content"):
            try:
                text = str(value.content or "").strip()
                parsed = self._parse_json_text(text)
                return parsed if parsed is not None else {"text": text}
            except Exception:
                pass

        if isinstance(value, str):
            parsed = self._parse_json_text(value)
            return parsed if parsed is not None else value

        return value

    def _jsonl_path(self) -> Path:
        raw = (self.file_name or "baseline_rag_results").strip()
        path = Path(raw).expanduser()
        if path.suffix.lower() != ".jsonl":
            path = path.with_suffix(".jsonl")
        return path

    async def _upload_file_to_langflow(self, file_path: Path, append: bool = False) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not getattr(self, "user_id", None):
            raise ValueError("user_id is required to register the file in LangFlow Files.")

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
            raise ValueError("user_id is required to register the file in LangFlow Files.")

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

    async def _write_jsonl_and_register(self, record: Dict[str, Any]) -> Path:
        path = self._jsonl_path()

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(record, ensure_ascii=False) + "\n"
        should_append = bool(self.append_mode) and path.exists()

        if should_append:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)

            await self._upload_bytes_to_langflow(
                filename=path.name,
                data=line.encode("utf-8"),
                append=True,
            )
        else:
            path.write_text(line, encoding="utf-8")
            await self._upload_file_to_langflow(path, append=False)

        return path

    async def write_record(self) -> Message:
        assistant_text = self._message_text(self.assistant_message)

        record = {
            "user_message": self._message_text(self.user_message),
            "assistant_text": assistant_text,
            "generation": self._to_plain(self.generation_envelope),
            "retrieval": self._to_plain(self.retrieval_envelope),
        }

        path = await self._write_jsonl_and_register(record)
        final_path = Path.cwd() / path if not path.is_absolute() else path

        self.status = Message(text=f"Baseline RAG final record written to {final_path}")

        return Message(text=assistant_text)