# Recovered Langflow component
# type: BaselineRAGResponseWriter
# class: BaselineRAGResponseWriter
# used in 7 flow(s): Baseline Normal AIT, Baseline Normal AIT 123, Baseline Normal Squad, Baseline Reason AIT, Baseline Reason Squad, FAQ System A, FAQ System B
# json path: node.data.node.template.code.value

import ast
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import UploadFile

from lfx.custom import Component
from lfx.io import BoolInput, DropdownInput, HandleInput, IntInput, StrInput
from lfx.schema import Data, Message
from lfx.services.deps import get_settings_service, get_storage_service, session_scope
from lfx.template.field.base import Output

from langflow.api.v2.files import upload_user_file
from langflow.services.database.models.user.crud import get_user_by_id


class BaselineRAGResponseWriter(Component):
    display_name = "Baseline RAG Response Writer"
    description = (
        "Writes one baseline/general RAG benchmark record per question. "
        "Stores sample id, question, retrieved context, prompt, generation output, "
        "final answer, and model/latency metrics as JSONL."
    )
    icon = "file-text"
    name = "BaselineRAGResponseWriter"

    LOCAL_FORMAT_CHOICES = ["jsonl", "json", "txt", "markdown"]
    KNOWN_EXTENSIONS = {"json", "jsonl", "txt", "md", "markdown"}

    inputs = [
        StrInput(
            name="file_name",
            display_name="File Name",
            value="baseline_rag_results",
            required=True,
            info="Saved file name. You may include or omit extension.",
            tool_mode=True,
        ),
        DropdownInput(
            name="local_format",
            display_name="Local Format",
            options=LOCAL_FORMAT_CHOICES,
            value="jsonl",
        ),
        BoolInput(
            name="append_mode",
            display_name="Append",
            value=True,
            advanced=True,
            info="Append one JSONL row per run. Recommended ON for benchmark runs.",
        ),
        StrInput(
            name="run_name",
            display_name="Run Name",
            value="baseline_rag",
            required=False,
            info="Optional run label, e.g. baseline_rag_qwen, baseline_rag_ollama.",
            tool_mode=True,
        ),
        StrInput(
            name="dataset_name",
            display_name="Dataset Name",
            value="",
            required=False,
            info="Optional dataset label, e.g. squad, ragtruth, ait_faq.",
            tool_mode=True,
        ),
        StrInput(
            name="sample_id",
            display_name="Sample ID",
            value="",
            required=False,
            info="Optional benchmark sample id. If blank, writer tries to infer it from inputs.",
            tool_mode=True,
        ),
        StrInput(
            name="session_id",
            display_name="Session ID",
            value="",
            required=False,
            info="Optional session id. Use fresh session id per sample for fair baseline testing.",
            tool_mode=True,
        ),
        IntInput(
            name="page_content_char_limit",
            display_name="Page Content Char Limit",
            value=1600,
            advanced=True,
            info="Trim retrieved page_content/context fields to this many characters. Use 0 for no limit.",
        ),
        IntInput(
            name="prompt_char_limit",
            display_name="Prompt Char Limit",
            value=8000,
            advanced=True,
            info="Trim stored prompt text to this many characters. Use 0 for no limit.",
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
            info="Turn ON only for debugging if wrapped LangFlow messages are not being parsed.",
        ),

        HandleInput(
            name="user_question",
            display_name="User Question / Chat Input",
            input_types=["Data", "Message"],
            required=True,
        ),
        HandleInput(
            name="retrieval_output",
            display_name="Retriever / PGVector Output",
            input_types=["Data", "Message"],
            required=False,
        ),
        HandleInput(
            name="formatted_context",
            display_name="Formatted RAG Context",
            input_types=["Data", "Message"],
            required=False,
        ),
        HandleInput(
            name="prompt_output",
            display_name="Final Prompt Output",
            input_types=["Data", "Message"],
            required=False,
        ),
        HandleInput(
            name="generation_output",
            display_name="Generator / Ollama Output",
            input_types=["Data", "Message"],
            required=True,
        ),
        HandleInput(
            name="final_message",
            display_name="Final Answer / Chat Output Message",
            input_types=["Data", "Message"],
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message", method="write_record"),
    ]

    # -------------------------
    # Generic helpers
    # -------------------------
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
                for key in ["text", "content", "message", "response", "answer", "input"]:
                    if key in value.data and value.data.get(key) is not None:
                        obj = value.data.get(key)
                        if isinstance(obj, dict):
                            return json.dumps(obj, ensure_ascii=False)
                        return str(obj).strip()
                return json.dumps(value.data, ensure_ascii=False)
            return str(value).strip()

        if isinstance(value, dict):
            for key in ["text", "content", "message", "response", "answer", "input"]:
                if key in value and value.get(key) is not None:
                    obj = value.get(key)
                    if isinstance(obj, dict):
                        inner = obj.get("content") or obj.get("text") or obj.get("message")
                        if inner is not None:
                            return str(inner).strip()
                        return json.dumps(obj, ensure_ascii=False)
                    return str(obj).strip()
            return json.dumps(value, ensure_ascii=False)

        if isinstance(value, list):
            return json.dumps(self._to_plain(value), ensure_ascii=False)

        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()

        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()

        return str(value).strip()

    def _to_plain(self, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, Data):
            return self._to_plain(value.data)

        if isinstance(value, Message):
            return {"text": self._extract_message_text(value)}

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
                return {"text": str(value.text or "")}
            except Exception:
                pass

        return value

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

    def _extract_payload_any(self, value: Any) -> Any:
        if value is None:
            return {}

        plain = self._to_plain(value)

        if isinstance(plain, (dict, list)):
            return plain

        text = self._extract_message_text(value)
        parsed = self._parse_json_like(text)
        if parsed is not None:
            return parsed

        return text

    def _unwrap_payload_dict(
        self,
        value: Any,
        expected_keys: Optional[List[str]] = None,
        wrapper_keys: Optional[List[str]] = None,
        max_depth: int = 8,
    ) -> Dict[str, Any]:
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
            "answer",
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
                if key not in current:
                    continue

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
        text = text or ""
        limit = int(limit or 0)
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit]

    def _trim_page_content(self, text: str) -> str:
        return self._trim_text(text, int(self.page_content_char_limit or 0))

    def _trim_prompt(self, text: str) -> str:
        return self._trim_text(text, int(self.prompt_char_limit or 0))

    def _trim_answer(self, text: str) -> str:
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
        return fmt.lower() in ["jsonl", "json", "txt", "markdown", "md"]

    async def _upload_file_to_langflow(self, file_path: Path, append: bool = False) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not getattr(self, "user_id", None):
            # In some LangFlow executions user_id is not available.
            # Local file is still written; skip registration instead of failing the whole run.
            return

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
            return

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
    # Metrics extraction
    # -------------------------
    def _extract_metrics(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            payload = self._extract_payload_dict(payload)

        if not isinstance(payload, dict):
            return {}

        metrics = payload.get("llm_metrics")
        if not isinstance(metrics, dict):
            metrics = payload.get("metrics")

        if not isinstance(metrics, dict) and any(
            k in payload
            for k in [
                "model",
                "created_at",
                "done",
                "done_reason",
                "prompt_eval_count",
                "eval_count",
                "total_duration",
                "total_duration_ns",
                "wall_time_s",
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
    # Field extraction
    # -------------------------
    def _extract_sample_id(self, *payloads: Any) -> str:
        explicit = str(getattr(self, "sample_id", "") or "").strip()
        if explicit:
            return explicit

        candidate_keys = [
            "sample_id",
            "id",
            "qid",
            "question_id",
            "example_id",
            "record_id",
        ]

        for payload in payloads:
            plain = self._extract_payload_any(payload)

            if isinstance(plain, dict):
                for key in candidate_keys:
                    value = plain.get(key)
                    if value is not None and str(value).strip():
                        return str(value).strip()

                metadata = plain.get("metadata")
                if isinstance(metadata, dict):
                    for key in candidate_keys:
                        value = metadata.get(key)
                        if value is not None and str(value).strip():
                            return str(value).strip()

        return ""

    def _select_question_fields(self, question_value: Any) -> Dict[str, Any]:
        payload = self._extract_payload_any(question_value)
        text = ""

        if isinstance(payload, dict):
            for key in ["question", "user_query", "input", "text", "content", "message"]:
                if key in payload and payload.get(key) is not None:
                    obj = payload.get(key)
                    if isinstance(obj, dict):
                        text = str(
                            obj.get("content")
                            or obj.get("text")
                            or obj.get("message")
                            or json.dumps(obj, ensure_ascii=False)
                        )
                    else:
                        text = str(obj)
                    break

        if not text:
            text = self._extract_message_text(question_value)

        return {
            "sample_id": self._extract_sample_id(question_value),
            "text": text.strip(),
            "session_id": str(self.session_id or "").strip(),
        }

    def _simplify_retrieval_item(self, item: Dict[str, Any], rank_fallback: Optional[int] = None) -> Dict[str, Any]:
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        page_content = (
            item.get("page_content")
            or item.get("content")
            or item.get("text")
            or item.get("document")
            or ""
        )

        return {
            "rank": item.get("rank", rank_fallback),
            "score": item.get("score", item.get("similarity", item.get("distance"))),
            "page_content": self._trim_page_content(str(page_content or "")),
            "metadata": {
                "source": metadata.get("source") or metadata.get("file_name") or metadata.get("url"),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id") or metadata.get("id"),
            },
        }

    def _select_retrieval_fields(self, retrieval_value: Any) -> Dict[str, Any]:
        payload = self._extract_payload_any(retrieval_value)

        results: List[Dict[str, Any]] = []

        if isinstance(payload, list):
            for idx, item in enumerate(payload, start=1):
                if isinstance(item, dict):
                    results.append(self._simplify_retrieval_item(item, idx))
                elif isinstance(item, str):
                    results.append(
                        {
                            "rank": idx,
                            "score": None,
                            "page_content": self._trim_page_content(item),
                            "metadata": {},
                        }
                    )

        elif isinstance(payload, dict):
            raw_results = None

            for key in ["results", "documents", "data", "chunks", "retrieved_documents"]:
                if isinstance(payload.get(key), list):
                    raw_results = payload.get(key)
                    break

            if raw_results is None and isinstance(payload.get("results_by_query"), list):
                for group in payload.get("results_by_query") or []:
                    if isinstance(group, dict) and isinstance(group.get("results"), list):
                        for item in group.get("results") or []:
                            if isinstance(item, dict):
                                item_copy = dict(item)
                                item_copy.setdefault("query_text", group.get("query_text"))
                                results.append(self._simplify_retrieval_item(item_copy, len(results) + 1))

            elif raw_results is not None:
                for idx, item in enumerate(raw_results, start=1):
                    if isinstance(item, dict):
                        results.append(self._simplify_retrieval_item(item, idx))
                    elif isinstance(item, str):
                        results.append(
                            {
                                "rank": idx,
                                "score": None,
                                "page_content": self._trim_page_content(item),
                                "metadata": {},
                            }
                        )

            else:
                # Single document-like dict.
                if any(k in payload for k in ["page_content", "content", "text", "document"]):
                    results.append(self._simplify_retrieval_item(payload, 1))

        return {
            "n_results": len(results),
            "results": results,
        }

    def _select_context_fields(self, context_value: Any) -> Dict[str, Any]:
        text = self._extract_message_text(context_value)
        payload = self._extract_payload_any(context_value)

        if isinstance(payload, dict):
            for key in ["context", "formatted_context", "text", "content", "message"]:
                value = payload.get(key)
                if value:
                    if isinstance(value, dict):
                        text = json.dumps(value, ensure_ascii=False)
                    else:
                        text = str(value)
                    break

        return {
            "text": self._trim_page_content(text),
        }

    def _select_prompt_fields(self, prompt_value: Any) -> Dict[str, Any]:
        text = self._extract_message_text(prompt_value)
        payload = self._extract_payload_any(prompt_value)

        if isinstance(payload, dict):
            for key in ["prompt", "text", "content", "message"]:
                value = payload.get(key)
                if value:
                    if isinstance(value, dict):
                        text = str(
                            value.get("content")
                            or value.get("text")
                            or json.dumps(value, ensure_ascii=False)
                        )
                    else:
                        text = str(value)
                    break

        return {
            "text": self._trim_prompt(text),
        }

    def _select_generation_fields(self, generation_value: Any) -> Dict[str, Any]:
        payload = self._extract_payload_dict(generation_value)
        unwrapped = self._unwrap_payload_dict(
            generation_value,
            expected_keys=["message", "content", "response", "model", "eval_count"],
        )

        src = payload if payload else unwrapped
        if not isinstance(src, dict):
            src = {}

        message_obj = src.get("message")
        answer_text = ""

        if isinstance(message_obj, dict):
            answer_text = str(message_obj.get("content", "") or "").strip()

        if not answer_text:
            answer_text = str(
                src.get("content")
                or src.get("response")
                or src.get("answer")
                or src.get("text")
                or ""
            ).strip()

        if not answer_text:
            answer_text = self._extract_message_text(generation_value)

        result = {
            "model": src.get("model"),
            "created_at": src.get("created_at"),
            "done": src.get("done"),
            "done_reason": src.get("done_reason"),
            "answer_text": self._trim_answer(answer_text),
        }

        if bool(self.include_raw_envelope_text):
            result["raw_text"] = self._extract_message_text(generation_value)
            result["payload_keys"] = list(src.keys()) if isinstance(src, dict) else []

        return result

    def _select_final_answer_fields(self, final_value: Any, generation_answer: str = "") -> Dict[str, Any]:
        text = self._extract_message_text(final_value)
        payload = self._extract_payload_any(final_value)

        if isinstance(payload, dict):
            for key in ["answer", "response", "text", "content", "message"]:
                value = payload.get(key)
                if value:
                    if isinstance(value, dict):
                        text = str(
                            value.get("content")
                            or value.get("text")
                            or value.get("message")
                            or json.dumps(value, ensure_ascii=False)
                        )
                    else:
                        text = str(value)
                    break

        if not text and generation_answer:
            text = generation_answer

        result = {
            "text": self._trim_answer(text.strip()),
        }

        if bool(self.include_raw_envelope_text):
            result["raw_text"] = self._extract_message_text(final_value)

        return result

    def _build_record(self) -> Dict[str, Any]:
        question = self._select_question_fields(self.user_question)
        sample_id = self._extract_sample_id(
            self.user_question,
            self.retrieval_output,
            self.formatted_context,
            self.prompt_output,
            self.generation_output,
            self.final_message,
        )

        if sample_id:
            question["sample_id"] = sample_id

        retrieval = self._select_retrieval_fields(self.retrieval_output)
        context = self._select_context_fields(self.formatted_context)
        prompt = self._select_prompt_fields(self.prompt_output)
        generation = self._select_generation_fields(self.generation_output)
        final_answer = self._select_final_answer_fields(
            self.final_message,
            generation_answer=generation.get("answer_text", ""),
        )

        generation_payload = self._extract_payload_dict(self.generation_output)
        final_payload = self._extract_payload_dict(self.final_message)

        return {
            "record_type": "baseline_rag_response_writer",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": str(self.run_name or "").strip(),
            "dataset_name": str(self.dataset_name or "").strip(),
            "sample_id": question.get("sample_id", ""),
            "session_id": question.get("session_id", ""),
            "question": {
                "text": question.get("text", ""),
            },
            "retrieval": retrieval,
            "formatted_context": context,
            "prompt": prompt,
            "generation": generation,
            "final_response": final_answer,
            "stage_metrics": {
                "generation": self._extract_metrics(generation_payload),
                "final_message": self._extract_metrics(final_payload),
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

        if fmt == "markdown":
            return "\n".join(
                [
                    "# Baseline RAG Response Record",
                    "",
                    f"**Saved At (UTC):** {record.get('saved_at_utc', '')}",
                    f"**Run:** {record.get('run_name', '')}",
                    f"**Dataset:** {record.get('dataset_name', '')}",
                    f"**Sample ID:** {record.get('sample_id', '')}",
                    "",
                    "## Question",
                    record.get("question", {}).get("text", ""),
                    "",
                    "## Final Response",
                    record.get("final_response", {}).get("text", ""),
                    "",
                    "## JSON Record",
                    "```json",
                    json.dumps(record, ensure_ascii=False, indent=2),
                    "```",
                ]
            )

        return "\n".join(
            [
                f"Saved At (UTC): {record.get('saved_at_utc', '')}",
                f"Run: {record.get('run_name', '')}",
                f"Dataset: {record.get('dataset_name', '')}",
                f"Sample ID: {record.get('sample_id', '')}",
                "",
                "=== QUESTION ===",
                record.get("question", {}).get("text", ""),
                "",
                "=== FINAL RESPONSE ===",
                record.get("final_response", {}).get("text", ""),
                "",
                "=== FULL JSON ===",
                json.dumps(record, ensure_ascii=False, indent=2),
            ]
        )

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
        return f"Baseline RAG response record {action} '{path}'"

    def _write_jsonl(self, path: Path, record: Dict[str, Any], should_append: bool) -> str:
        line = self._record_to_jsonl_line(record)

        if should_append:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            path.write_text(line + "\n", encoding="utf-8")

        action = "appended to" if should_append else "saved successfully as"
        return f"Baseline RAG response record {action} '{path}'"

    def _write_textlike(self, path: Path, content: str, should_append: bool) -> str:
        if should_append:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            sep = "\n\n---\n\n" if existing else ""
            path.write_text(existing + sep + content, encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")

        action = "appended to" if should_append else "saved successfully as"
        return f"Baseline RAG response record {action} '{path}'"

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

    async def write_record(self) -> Message:
        record = self._build_record()
        result = await self._save_local(record)
        self.status = result
        return result