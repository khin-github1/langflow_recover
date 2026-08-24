# Recovered Langflow component
# type: OpenAIFlexibleBatchRunner
# class: OpenAIFlexibleBatchRunner
# used in 3 flow(s): Cognitive RAG V1.2.0 GPT, Cognitive RAG V1.2.5 GPT, Cognitive RAG V1.3.0 GPT Ingest
# json path: node.data.node.template.code.value

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from fastapi import UploadFile

from lfx.custom import Component
from lfx.io import (
    BoolInput,
    FileInput,
    IntInput,
    Output,
    SecretStrInput,
    StrInput,
)
from lfx.schema import Message
from lfx.services.deps import get_settings_service, get_storage_service, session_scope

from langflow.api.v2.files import upload_user_file
from langflow.services.database.models.user.crud import get_user_by_id


class OpenAIFlexibleBatchRunner(Component):
    display_name = "OpenAI Flexible Batch Runner"
    description = (
        "Accepts either slim CRCV JSONL records "
        "(generation_mode + raw_prompt_text/prompt_text) or already-built "
        "OpenAI batch JSONL records (custom_id + method + url + body), "
        "then submits them to OpenAI Batch, optionally polls, downloads outputs, "
        "and saves summaries."
    )
    icon = "upload"
    name = "OpenAIFlexibleBatchRunner"

    inputs = [
        FileInput(
            name="input_jsonl_file",
            display_name="Input JSONL File",
            file_types=["jsonl", "txt"],
            required=True,
            info=(
                "Can be either:\n"
                "1) slim records with generation_mode + raw_prompt_text/prompt_text\n"
                "2) prebuilt OpenAI batch lines with custom_id/method/url/body"
            ),
        ),
        StrInput(
            name="base_url",
            display_name="Base URL",
            value="https://api.openai.com",
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="OpenAI API Key",
            required=True,
        ),
        StrInput(
            name="model_name",
            display_name="Model Name",
            value="gpt-5-mini",
            required=True,
            info="Used only when input file is the slim format.",
        ),
        StrInput(
            name="output_base_name",
            display_name="Output Base Name",
            value="openai_batch_run",
            required=True,
            tool_mode=True,
        ),
        StrInput(
            name="system_prompt_normal",
            display_name="System Prompt (Normal)",
            value=(
                "You are a careful FAQ answer generator. "
                "Answer using only the provided context. "
                "Be concise, factual, and direct. "
                "If the context is insufficient, say so clearly. "
                "Do not invent facts."
            ),
            required=False,
        ),
        StrInput(
            name="system_prompt_reason",
            display_name="System Prompt (Reason)",
            value=(
                "You are a careful FAQ answer generator. "
                "Use the provided context to reason carefully before answering. "
                "Synthesize across multiple evidence snippets when needed. "
                "Be factual, explicit, and grounded in the context. "
                "If the context is insufficient or conflicting, say so clearly. "
                "Do not invent facts."
            ),
            required=False,
        ),
        StrInput(
            name="reason_effort_mode",
            display_name="Reason Mode Effort",
            value="medium",
            required=True,
        ),
        StrInput(
            name="normal_effort_preference",
            display_name="Normal Mode Effort Preference",
            value="minimal",
            required=True,
        ),
        StrInput(
            name="temperature",
            display_name="Temperature",
            value="1",
            required=False,
            info="This runner forces temperature to 1.0 for compatibility with this model.",
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            value=0,
            required=False,
            info="0 means omit max_tokens.",
        ),
        BoolInput(
            name="json_mode",
            display_name="JSON Mode",
            value=False,
            required=False,
        ),
        BoolInput(
            name="wait_for_completion",
            display_name="Wait For Completion",
            value=False,
            required=False,
        ),
        IntInput(
            name="poll_interval_s",
            display_name="Poll Interval (seconds)",
            value=30,
            required=False,
        ),
        IntInput(
            name="max_wait_s",
            display_name="Max Wait (seconds)",
            value=1800,
            required=False,
        ),
        BoolInput(
            name="save_downloaded_output",
            display_name="Save Downloaded Output",
            value=True,
            required=False,
        ),
        BoolInput(
            name="register_artifacts_in_langflow",
            display_name="Register Artifacts in Langflow",
            value=True,
            required=False,
        ),
        StrInput(
            name="batch_description",
            display_name="Batch Description",
            value="CRCV GPT batch run",
            required=False,
        ),
        IntInput(
            name="http_timeout_s",
            display_name="HTTP Timeout (seconds)",
            value=300,
            required=False,
        ),
    ]

    outputs = [
        Output(
            display_name="Message",
            name="message",
            method="run_batch",
        ),
    ]

    # -------------------------
    # Basic helpers
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

    def _normalized_base_url(self, url: str) -> str:
        url = (url or "").strip().rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3].rstrip("/")
        return url

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _http_timeout(self) -> int:
        try:
            return max(30, int(self.http_timeout_s or 300))
        except Exception:
            return 300

    def _resolve_uploaded_file_path(self) -> Path:
        uploaded = getattr(self, "input_jsonl_file", None)
        if not uploaded:
            raise ValueError("input_jsonl_file is required.")

        if isinstance(uploaded, list):
            if not uploaded:
                raise ValueError("input_jsonl_file list is empty.")
            uploaded = uploaded[0]

        path = Path(str(uploaded)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input JSONL file not found: {path}")
        return path

    def _base_output_path(self) -> Path:
        raw = str(self.output_base_name or "").strip()
        if not raw:
            raise ValueError("output_base_name is required.")
        return Path(raw).expanduser()

    def _auth_headers(self) -> Dict[str, str]:
        api_key = self._secret_to_text(self.api_key).strip()
        if not api_key:
            raise ValueError("api_key is required.")

        suspicious_markers = [
            "import ",
            "from ",
            "class ",
            "def ",
            "httpx",
            "UploadFile",
            "OpenAIFlexibleBatchRunner",
        ]
        if "\n" in api_key or any(marker in api_key for marker in suspicious_markers):
            raise ValueError(
                "API Key field contains pasted code/text, not an OpenAI key. "
                "Paste only the raw sk-... key into the API Key input."
            )

        if not api_key.startswith("sk-"):
            raise ValueError(
                "OpenAI API key looks invalid. Expected a key starting with sk-."
            )

        return {"Authorization": f"Bearer {api_key}"}

    def _raise_for_status_with_body(self, resp: httpx.Response, context: str) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            try:
                body_preview = resp.text[:3000]
            except Exception:
                body_preview = "<unable to read response body>"
            raise RuntimeError(
                f"{context} failed: HTTP {resp.status_code}\n"
                f"URL: {resp.request.url}\n"
                f"Response body:\n{body_preview}"
            ) from e

    # -------------------------
    # Mode/prompt helpers
    # -------------------------
    def _normalize_reason_effort(self, effort: str) -> str:
        effort = str(effort or "").strip().lower()
        if effort not in {"minimal", "low", "medium", "high"}:
            return "medium"
        return effort

    def _normalize_normal_effort(self, effort: str) -> str:
        effort = str(effort or "").strip().lower()
        if effort == "none":
            return "minimal"
        if effort not in {"minimal", "low", "medium"}:
            return "minimal"
        return effort

    def _normalize_temperature(self, value: Any) -> float:
        # This model only supports the default temperature behavior.
        # Always coerce to 1.0 to prevent batch failures.
        return 1.0

    def _normalize_mode(self, mode: Any) -> str:
        if isinstance(mode, dict):
            mode = mode.get("generation_mode", "")
        mode = str(mode or "").strip().lower()
        return "reason" if mode == "reason" else "normal"

    def _extract_generation_mode(self, record: Dict[str, Any]) -> str:
        if "generation_mode" in record:
            return self._normalize_mode(record.get("generation_mode"))
        if "mode" in record:
            return self._normalize_mode(record.get("mode"))
        return "normal"

    def _extract_prompt_text(self, record: Dict[str, Any]) -> str:
        candidate_keys = [
            "raw_prompt_text",
            "prompt_text",
            "final_prompt",
            "raw_prompt",
            "prompt",
            "input_text",
            "user_input",
            "question",
            "text",
            "content",
        ]

        for key in candidate_keys:
            value = record.get(key)
            if value is None:
                continue

            if isinstance(value, str):
                value = value.strip()
                if value:
                    return value

            if isinstance(value, dict):
                for nested_key in ["raw_prompt_text", "text", "content", "value", "prompt"]:
                    nested_val = value.get(nested_key)
                    if isinstance(nested_val, str) and nested_val.strip():
                        return nested_val.strip()

            try:
                text = str(value).strip()
                if text and text not in {"{}", "[]", "None"}:
                    return text
            except Exception:
                pass

        generator_prompt = record.get("generator_prompt")
        if isinstance(generator_prompt, dict):
            for nested_key in ["raw_prompt_text", "prompt_text", "text", "content", "prompt"]:
                nested_val = generator_prompt.get(nested_key)
                if isinstance(nested_val, str) and nested_val.strip():
                    return nested_val.strip()

        for parent_key in ["message", "payload", "data"]:
            parent_val = record.get(parent_key)
            if isinstance(parent_val, dict):
                for nested_key in candidate_keys + ["raw_prompt_text"]:
                    nested_val = parent_val.get(nested_key)
                    if isinstance(nested_val, str) and nested_val.strip():
                        return nested_val.strip()

        available_keys = sorted(record.keys())
        raise ValueError(
            "Record missing prompt field. "
            f"Tried direct and nested prompt keys. Available keys={available_keys}"
        )

    # -------------------------
    # File format detection
    # -------------------------
    def _is_prebuilt_batch_record(self, record: Dict[str, Any]) -> bool:
        return all(k in record for k in ["custom_id", "method", "url", "body"])

    def _is_slim_record(self, record: Dict[str, Any]) -> bool:
        if "generation_mode" in record:
            return True
        slim_prompt_keys = {
            "raw_prompt_text",
            "prompt_text",
            "final_prompt",
            "raw_prompt",
            "prompt",
            "input_text",
            "user_input",
            "question",
            "text",
            "content",
            "generator_prompt",
        }
        return any(k in record for k in slim_prompt_keys)

    # -------------------------
    # Read / normalize input
    # -------------------------
    def _read_jsonl_objects(self, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception as e:
                    raise ValueError(f"Invalid JSONL at line {lineno}: {e}") from e

                if not isinstance(obj, dict):
                    raise ValueError(f"Line {lineno} is valid JSON but not an object.")

                records.append(obj)

        if not records:
            raise ValueError("No valid JSONL records found in input file.")

        return records

    def _build_batch_line_from_slim(
        self,
        record: Dict[str, Any],
        idx: int,
        model_name: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        mode = self._extract_generation_mode(record)
        prompt_text = self._extract_prompt_text(record)

        if mode == "reason":
            reasoning_effort = self._normalize_reason_effort(self.reason_effort_mode)
            system_prompt = str(self.system_prompt_reason or "").strip()
        else:
            reasoning_effort = self._normalize_normal_effort(self.normal_effort_preference)
            system_prompt = str(self.system_prompt_normal or "").strip()

        body: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": self._normalize_temperature(self.temperature),
            "reasoning_effort": reasoning_effort,
        }

        max_tokens = int(self.max_tokens or 0)
        if max_tokens > 0:
            body["max_tokens"] = max_tokens

        if bool(self.json_mode):
            body["response_format"] = {"type": "json_object"}

        custom_id = f"{mode}_{idx:06d}"

        batch_line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        manifest_row = {
            "custom_id": custom_id,
            "generation_mode": mode,
            "reasoning_effort_used": reasoning_effort,
            "prompt_preview": prompt_text[:300],
            "source_format": "slim",
        }

        return batch_line, manifest_row

    def _normalize_prebuilt_batch_record(
        self,
        record: Dict[str, Any],
        idx: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        missing = [k for k in ["custom_id", "method", "url", "body"] if k not in record]
        if missing:
            raise ValueError(f"Prebuilt batch record missing keys: {missing}")

        if not isinstance(record.get("body"), dict):
            raise ValueError("Prebuilt batch record field 'body' must be an object.")

        model_name = str(record["body"].get("model", "") or "").strip()
        if not model_name:
            raise ValueError("Prebuilt batch record missing body.model")

        # Force temperature=1.0 even for prebuilt records, to avoid unsupported-value errors.
        record = dict(record)
        record["body"] = dict(record["body"])
        record["body"]["temperature"] = 1.0

        custom_id = str(record.get("custom_id", "") or "")
        if custom_id.startswith("reason_"):
            mode = "reason"
        elif custom_id.startswith("normal_"):
            mode = "normal"
        else:
            mode = "unknown"

        manifest_row = {
            "custom_id": custom_id or f"record_{idx:06d}",
            "generation_mode": mode,
            "reasoning_effort_used": (
                record["body"].get("reasoning_effort")
                if isinstance(record["body"], dict)
                else None
            ),
            "prompt_preview": "",
            "source_format": "prebuilt",
        }

        return record, manifest_row

    def _normalize_input_to_batch_file(
        self,
        input_path: Path,
    ) -> Tuple[Path, Path, Dict[str, Any]]:
        raw_records = self._read_jsonl_objects(input_path)

        base = self._base_output_path()
        batch_input_path = base.with_name(f"{base.name}_batch_input").with_suffix(".jsonl")
        manifest_path = base.with_name(f"{base.name}_manifest").with_suffix(".json")

        if not batch_input_path.parent.exists():
            batch_input_path.parent.mkdir(parents=True, exist_ok=True)

        normalized_records: List[Dict[str, Any]] = []
        manifest_rows: List[Dict[str, Any]] = []
        models = set()
        endpoints = set()
        mode_counts = {"normal": 0, "reason": 0, "unknown": 0}
        source_formats = {"slim": 0, "prebuilt": 0}

        for idx, record in enumerate(raw_records, start=1):
            if self._is_prebuilt_batch_record(record):
                batch_line, manifest_row = self._normalize_prebuilt_batch_record(record, idx)
                source_formats["prebuilt"] += 1
            elif self._is_slim_record(record):
                batch_line, manifest_row = self._build_batch_line_from_slim(
                    record,
                    idx,
                    str(self.model_name or "").strip(),
                )
                source_formats["slim"] += 1
            else:
                available_keys = sorted(record.keys())
                raise ValueError(
                    f"Line {idx} is neither a slim record nor a prebuilt batch record. "
                    f"Available keys={available_keys}"
                )

            body = batch_line.get("body", {})
            model_name = str(body.get("model", "") or "").strip()
            endpoint = str(batch_line.get("url", "") or "").strip()

            if not model_name:
                raise ValueError(f"Line {idx} missing body.model after normalization.")
            if not endpoint:
                raise ValueError(f"Line {idx} missing url after normalization.")

            models.add(model_name)
            endpoints.add(endpoint)

            mode = manifest_row.get("generation_mode", "unknown")
            if mode not in mode_counts:
                mode = "unknown"
            mode_counts[mode] += 1

            normalized_records.append(batch_line)
            manifest_rows.append(manifest_row)

        if len(models) > 1:
            raise ValueError(
                f"Normalized batch file contains multiple models: {sorted(models)}. "
                "Use a single model per batch input file."
            )

        if len(endpoints) > 1:
            raise ValueError(
                f"Normalized batch file contains multiple endpoints: {sorted(endpoints)}. "
                "Use a single endpoint per batch input file."
            )

        batch_lines = [json.dumps(r, ensure_ascii=False) for r in normalized_records]
        batch_input_path.write_text("\n".join(batch_lines) + "\n", encoding="utf-8")

        manifest_payload = {
            "record_count": len(normalized_records),
            "model_name": next(iter(models)),
            "endpoint": next(iter(endpoints)),
            "mode_counts": mode_counts,
            "source_formats": source_formats,
            "rows": manifest_rows,
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.status = (
            f"Loaded {manifest_payload['record_count']} record(s). "
            f"formats={source_formats} "
            f"model={manifest_payload['model_name']} "
            f"endpoint={manifest_payload['endpoint']}"
        )

        return batch_input_path, manifest_path, manifest_payload

    # -------------------------
    # HTTP calls
    # -------------------------
    def _submit_batch_http(self, batch_input_path: Path, endpoint: str, model_name: str) -> Dict[str, Any]:
        base_url = self._normalized_base_url(self.base_url)
        if not base_url:
            raise ValueError("base_url is required.")

        headers = self._auth_headers()
        timings: Dict[str, float] = {}

        with httpx.Client(timeout=self._http_timeout()) as client:
            t0 = time.perf_counter()
            with batch_input_path.open("rb") as f:
                files = {
                    "file": (batch_input_path.name, f, "application/jsonl"),
                }
                data = {"purpose": "batch"}

                upload_resp = client.post(
                    f"{base_url}/v1/files",
                    headers=headers,
                    files=files,
                    data=data,
                )
                self._raise_for_status_with_body(upload_resp, "File upload")
                uploaded_file = upload_resp.json()
            timings["upload_wall_time_s"] = time.perf_counter() - t0

            input_file_id = uploaded_file.get("id")
            if not input_file_id:
                raise ValueError("Upload succeeded but no input file ID was returned.")

            batch_body = {
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": "24h",
                "metadata": {
                    "batch_description": str(self.batch_description or "CRCV GPT batch run"),
                    "model_name": model_name,
                },
            }

            t1 = time.perf_counter()
            batch_resp = client.post(
                f"{base_url}/v1/batches",
                headers={**headers, "Content-Type": "application/json"},
                json=batch_body,
            )
            self._raise_for_status_with_body(batch_resp, "Batch creation")
            batch_obj = batch_resp.json()
            timings["create_batch_wall_time_s"] = time.perf_counter() - t1

        return {
            "uploaded_file": uploaded_file,
            "batch": batch_obj,
            "timings": timings,
        }

    def _retrieve_batch_http(self, batch_id: str) -> Dict[str, Any]:
        base_url = self._normalized_base_url(self.base_url)
        headers = self._auth_headers()

        with httpx.Client(timeout=max(120, self._http_timeout())) as client:
            resp = client.get(
                f"{base_url}/v1/batches/{batch_id}",
                headers={**headers, "Content-Type": "application/json"},
            )
            self._raise_for_status_with_body(resp, "Batch retrieve")
            return resp.json()

    def _download_file_content_http(self, file_id: str) -> str:
        base_url = self._normalized_base_url(self.base_url)
        headers = self._auth_headers()

        with httpx.Client(timeout=max(300, self._http_timeout())) as client:
            resp = client.get(
                f"{base_url}/v1/files/{file_id}/content",
                headers=headers,
            )
            self._raise_for_status_with_body(resp, "File content download")
            return resp.text

    # -------------------------
    # Poll / parse output
    # -------------------------
    def _wait_for_terminal_batch(self, batch_id: str) -> Tuple[Dict[str, Any], float]:
        poll_interval_s = max(1, int(self.poll_interval_s or 30))
        max_wait_s = max(1, int(self.max_wait_s or 1800))

        start = time.perf_counter()
        while True:
            batch_obj = self._retrieve_batch_http(batch_id)
            status = str(batch_obj.get("status", "") or "").strip()

            if status in {"completed", "failed", "expired", "cancelled"}:
                return batch_obj, time.perf_counter() - start

            if (time.perf_counter() - start) >= max_wait_s:
                return batch_obj, time.perf_counter() - start

            time.sleep(poll_interval_s)

    def _parse_output_jsonl(self, output_text: str) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        per_mode = {
            "normal": {
                "requests": 0,
                "successful": 0,
                "failed": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
            },
            "reason": {
                "requests": 0,
                "successful": 0,
                "failed": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
            },
            "unknown": {
                "requests": 0,
                "successful": 0,
                "failed": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
            },
        }

        for line in (output_text or "").splitlines():
            raw = line.strip()
            if not raw:
                continue

            try:
                obj = json.loads(raw)
            except Exception:
                continue

            custom_id = str(obj.get("custom_id", "") or "")
            if custom_id.startswith("reason_"):
                mode = "reason"
            elif custom_id.startswith("normal_"):
                mode = "normal"
            else:
                mode = "unknown"

            per_mode[mode]["requests"] += 1

            response = obj.get("response")
            error = obj.get("error")

            if error:
                per_mode[mode]["failed"] += 1
                rows.append(
                    {
                        "custom_id": custom_id,
                        "mode": mode,
                        "status_code": None,
                        "error": error,
                    }
                )
                continue

            if not isinstance(response, dict):
                per_mode[mode]["failed"] += 1
                rows.append(
                    {
                        "custom_id": custom_id,
                        "mode": mode,
                        "status_code": None,
                        "error": {"message": "Missing response object"},
                    }
                )
                continue

            status_code = response.get("status_code")
            body = response.get("body", {}) if isinstance(response.get("body"), dict) else {}
            usage = body.get("usage", {}) if isinstance(body.get("usage"), dict) else {}

            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", 0) or 0)

            reasoning_tokens = 0
            if isinstance(usage.get("reasoning_tokens"), int):
                reasoning_tokens = int(usage.get("reasoning_tokens") or 0)
            elif isinstance(usage.get("completion_tokens_details"), dict):
                reasoning_tokens = int(
                    usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
                )
            elif isinstance(usage.get("output_tokens_details"), dict):
                reasoning_tokens = int(
                    usage.get("output_tokens_details", {}).get("reasoning_tokens", 0) or 0
                )

            if int(status_code or 0) == 200:
                per_mode[mode]["successful"] += 1
            else:
                per_mode[mode]["failed"] += 1

            per_mode[mode]["prompt_tokens"] += prompt_tokens
            per_mode[mode]["completion_tokens"] += completion_tokens
            per_mode[mode]["total_tokens"] += total_tokens
            per_mode[mode]["reasoning_tokens"] += reasoning_tokens

            assistant_text = ""
            choices = body.get("choices", [])
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    msg = first_choice.get("message", {})
                    if isinstance(msg, dict):
                        assistant_text = str(msg.get("content", "") or "")

            rows.append(
                {
                    "custom_id": custom_id,
                    "mode": mode,
                    "status_code": status_code,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "assistant_preview": assistant_text[:300],
                }
            )

        aggregate = {
            "requests": sum(v["requests"] for v in per_mode.values()),
            "successful": sum(v["successful"] for v in per_mode.values()),
            "failed": sum(v["failed"] for v in per_mode.values()),
            "prompt_tokens": sum(v["prompt_tokens"] for v in per_mode.values()),
            "completion_tokens": sum(v["completion_tokens"] for v in per_mode.values()),
            "total_tokens": sum(v["total_tokens"] for v in per_mode.values()),
            "reasoning_tokens": sum(v["reasoning_tokens"] for v in per_mode.values()),
        }

        return {
            "aggregate_usage_from_output": aggregate,
            "usage_by_mode_from_output": per_mode,
            "result_rows": rows,
        }

    # -------------------------
    # Optional Langflow artifact registration
    # -------------------------
    async def _upload_file_to_langflow(self, file_path: Path, append: bool = False) -> None:
        if not bool(self.register_artifacts_in_langflow):
            return
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not getattr(self, "user_id", None):
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

    async def _upload_bytes_to_langflow(self, filename: str, data: bytes, append: bool = False) -> None:
        if not bool(self.register_artifacts_in_langflow):
            return
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
    # Main run
    # -------------------------
    async def run_batch(self) -> Message:
        t_all_start = time.perf_counter()

        input_path = self._resolve_uploaded_file_path()
        batch_input_path, manifest_path, manifest_payload = self._normalize_input_to_batch_file(input_path)

        await self._upload_file_to_langflow(batch_input_path, append=False)
        await self._upload_file_to_langflow(manifest_path, append=False)

        submit_result = self._submit_batch_http(
            batch_input_path=batch_input_path,
            endpoint=manifest_payload["endpoint"],
            model_name=manifest_payload["model_name"],
        )

        batch_obj = submit_result["batch"]
        timings = submit_result["timings"]

        result: Dict[str, Any] = {
            "input_jsonl_path": str(input_path),
            "built_batch_input_jsonl_path": str(batch_input_path),
            "manifest_json_path": str(manifest_path),
            "model_name": manifest_payload["model_name"],
            "endpoint": manifest_payload["endpoint"],
            "mode_counts": manifest_payload["mode_counts"],
            "source_formats": manifest_payload["source_formats"],
            "record_count": manifest_payload["record_count"],
            "batch_id": batch_obj.get("id"),
            "batch_status": batch_obj.get("status"),
            "input_file_id": batch_obj.get("input_file_id"),
            "output_file_id": batch_obj.get("output_file_id"),
            "error_file_id": batch_obj.get("error_file_id"),
            "request_counts": batch_obj.get("request_counts"),
            "upload_wall_time_s": timings.get("upload_wall_time_s"),
            "create_batch_wall_time_s": timings.get("create_batch_wall_time_s"),
        }

        final_batch_obj = batch_obj

        if bool(self.wait_for_completion):
            final_batch_obj, poll_wall_time_s = self._wait_for_terminal_batch(batch_obj["id"])
            result["batch_status"] = final_batch_obj.get("status")
            result["request_counts"] = final_batch_obj.get("request_counts")
            result["output_file_id"] = final_batch_obj.get("output_file_id")
            result["error_file_id"] = final_batch_obj.get("error_file_id")
            result["poll_wall_time_s"] = poll_wall_time_s

            output_file_id = final_batch_obj.get("output_file_id")
            if output_file_id and bool(self.save_downloaded_output):
                output_text = self._download_file_content_http(output_file_id)

                output_jsonl_path = self._base_output_path().with_name(
                    f"{self._base_output_path().name}_batch_output"
                ).with_suffix(".jsonl")
                output_jsonl_path.write_text(output_text, encoding="utf-8")
                await self._upload_file_to_langflow(output_jsonl_path, append=False)
                result["downloaded_output_jsonl_path"] = str(output_jsonl_path)

                output_parse_summary = self._parse_output_jsonl(output_text)
                result.update(output_parse_summary)

            error_file_id = final_batch_obj.get("error_file_id")
            if error_file_id and bool(self.save_downloaded_output):
                try:
                    error_text = self._download_file_content_http(error_file_id)
                    error_jsonl_path = self._base_output_path().with_name(
                        f"{self._base_output_path().name}_batch_errors"
                    ).with_suffix(".jsonl")
                    error_jsonl_path.write_text(error_text, encoding="utf-8")
                    await self._upload_file_to_langflow(error_jsonl_path, append=False)
                    result["downloaded_error_jsonl_path"] = str(error_jsonl_path)
                except Exception as e:
                    result["error_file_download_error"] = str(e)

        if isinstance(final_batch_obj.get("usage"), dict):
            result["batch_usage_from_api"] = final_batch_obj.get("usage")

        result["server_timestamps"] = {
            "created_at": final_batch_obj.get("created_at"),
            "in_progress_at": final_batch_obj.get("in_progress_at"),
            "finalizing_at": final_batch_obj.get("finalizing_at"),
            "completed_at": final_batch_obj.get("completed_at"),
            "failed_at": final_batch_obj.get("failed_at"),
            "expired_at": final_batch_obj.get("expired_at"),
            "cancelled_at": final_batch_obj.get("cancelled_at"),
        }

        result["local_total_wall_time_s"] = time.perf_counter() - t_all_start

        summary_path = self._base_output_path().with_name(
            f"{self._base_output_path().name}_summary"
        ).with_suffix(".json")
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        await self._upload_file_to_langflow(summary_path, append=False)
        result["summary_json_path"] = str(summary_path)

        self.status = (
            f"batch_id={result.get('batch_id')} "
            f"status={result.get('batch_status')} "
            f"records={result.get('record_count', 0)}"
        )

        if bool(self.wait_for_completion):
            text = (
                f"Batch submitted and checked. "
                f"batch_id={result.get('batch_id')} "
                f"status={result.get('batch_status')}"
            )
        else:
            text = (
                f"Batch submitted. "
                f"batch_id={result.get('batch_id')} "
                f"status={result.get('batch_status')}. "
                f"Set wait_for_completion=true to poll and download results."
            )

        return Message(text=text, data=result)