# Recovered Langflow component
# type: OllamaRAGValidatorEnvelopeNoCP
# class: OllamaRAGValidatorEnvelopeNoCP
# used in 17 flow(s): AIT Cognitive RAG, Cognitive RAG V1.1.5, Cognitive RAG V1.1.5 Backup, Cognitive RAG V1.2.0 (1), Cognitive RAG V1.2.0 AIT CR, Cognitive RAG V1.2.0 AIT Loose, Cognitive RAG V1.2.0 AIT Normal, Cognitive RAG V1.2.0 AIT RR ...
# json path: node.data.node.template.code.value

from __future__ import annotations

import ast
import json
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx

from langflow.custom import Component
from langflow.io import (
    BoolInput,
    DictInput,
    DropdownInput,
    FloatInput,
    IntInput,
    MessageInput,
    MessageTextInput,
    Output,
    SliderInput,
)
from langflow.field_typing.range_spec import RangeSpec
from langflow.logging import logger
from langflow.schema.data import Data

HTTP_STATUS_OK = 200


class OllamaRAGValidatorEnvelopeNoCP(Component):
    """
    Minimal single-pass evidence validator for bounded FAQ / RAG pipelines.

    Design:
    - One LLM call only
    - No retry
    - No conflict penalty
    - LLM returns only global_retrieval_index + validation_score
    - Python reconstructs query_index/query_text/is_answer_bearing
    - Python computes evidence_sufficiency_score deterministically
    """

    display_name = "Ollama RAG Validator (No CP)"
    description = (
        "Single-pass validator. LLM returns only evidence indices and validation scores. "
        "Python reconstructs remaining fields and computes ES deterministically."
    )
    icon = "Ollama"
    name = "OllamaRAGValidatorEnvelopeNoCP"

    JSON_MODELS_KEY = "models"
    JSON_NAME_KEY = "name"
    JSON_CAPABILITIES_KEY = "capabilities"
    DESIRED_CAPABILITY = "completion"
    TOOL_CALLING_CAPABILITY = "tools"

    DEFAULT_SYSTEM = (
        "You are an EVIDENCE RANKER.\n"
        "Your job is ONLY to score and filter retrieved evidences.\n"
        "You are NOT allowed to answer the user query.\n"
        "You are NOT allowed to answer any decomposed sub-query.\n"
        "You are NOT allowed to generate question-answer pairs.\n"
        "You are NOT allowed to output keys like \"query_1\", \"query_2\", \"question\", or \"answer\".\n\n"
        "Return ONLY one valid JSON object.\n\n"
        "Required process:\n"
        "1. Review all evidences.\n"
        "2. Assign validation_score to evidences.\n"
        "3. Keep only the best evidences, maximum KEEP_LIMIT.\n"
        "4. Sort top_kept_evidences by validation_score descending.\n"
        "5. Output ONLY the required JSON schema.\n\n"
        "Scoring rules:\n"
        "- validation_score = 1.0 means directly answer-bearing and highly specific\n"
        "- validation_score = 0.8 means relevant and useful but slightly less specific\n"
        "- validation_score = 0.5 means partially useful\n"
        "- validation_score = 0.2 means weakly relevant\n"
        "- validation_score = 0.0 means irrelevant\n\n"
        "Rules:\n"
        "- Return ONLY valid JSON.\n"
        "- No markdown.\n"
        "- No prose.\n"
        "- No extra fields.\n"
        "- No omitted fields.\n"
        "- top_kept_evidences must contain at most KEEP_LIMIT items.\n"
        "- top_kept_evidences must be sorted by validation_score descending.\n"
        "- All scores must be between 0 and 1.\n"
        "- decomposition_type must be exactly one of: \"parallel\", \"sequential\", \"none\".\n"
        "- For each kept evidence, return ONLY: global_retrieval_index and validation_score.\n\n"
        "Forbidden output patterns:\n"
        "- {\"query_1\": {...}}\n"
        "- {\"question\": \"...\", \"answer\": \"...\"}\n"
        "- any final answer to the user"
    )

    inputs = [
        MessageTextInput(
            name="base_url",
            display_name="Base URL",
            value="http://localhost:11434",
            real_time_refresh=True,
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=[],
            refresh_button=True,
            real_time_refresh=True,
        ),
        MessageInput(
            name="retrieval_message",
            display_name="Retrieval Message",
            info="Grouped retrieval results from your decomposer-aware retriever.",
            required=True,
        ),
        MessageTextInput(
            name="system",
            display_name="System",
            advanced=True,
            value=DEFAULT_SYSTEM,
        ),
        IntInput(
            name="max_kept_results",
            display_name="Max Kept Results",
            value=8,
            advanced=False,
        ),
        IntInput(
            name="snippet_char_limit",
            display_name="Snippet Char Limit",
            value=450,
            advanced=False,
        ),
        FloatInput(
            name="answer_bearing_threshold",
            display_name="Answer-Bearing Threshold",
            value=0.5,
            advanced=False,
        ),
        BoolInput(
            name="use_fallback_output",
            display_name="Use Fallback Output",
            value=True,
            advanced=False,
        ),
        BoolInput(
            name="include_thinking_in_envelope",
            display_name="Include Thinking in Envelope",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="enable_thinking",
            display_name="Enable Thinking (Ollama think)",
            value=False,
        ),
        BoolInput(
            name="keep_alive_enabled",
            display_name="Keep Alive Enabled",
            value=False,
            advanced=True,
        ),
        MessageTextInput(
            name="keep_alive",
            display_name="Keep Alive",
            value="10m",
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.0,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        IntInput(name="seed", display_name="Seed", advanced=True, value=42),
        IntInput(name="num_ctx", display_name="num_ctx", advanced=True),
        IntInput(name="num_predict", display_name="num_predict", advanced=True),
        IntInput(name="top_k", display_name="top_k", advanced=True),
        FloatInput(name="top_p", display_name="top_p", advanced=True),
        FloatInput(name="repeat_penalty", display_name="repeat_penalty", advanced=True),
        IntInput(name="repeat_last_n", display_name="repeat_last_n", advanced=True),
        MessageTextInput(
            name="stop_tokens",
            display_name="Stop Tokens",
            advanced=True,
        ),
        MessageTextInput(
            name="format",
            display_name="Format",
            info="Optional format such as json.",
            advanced=True,
            value="json",
        ),
        DictInput(
            name="extra_options",
            display_name="Extra options (dict)",
            value={},
            advanced=True,
        ),
        BoolInput(
            name="tool_model_enabled",
            display_name="Tool Model Enabled",
            value=True,
            real_time_refresh=True,
            advanced=True,
        ),
        IntInput(
            name="timeout_s",
            display_name="Timeout (seconds)",
            value=120,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Envelope", name="envelope", method="build_envelope"),
    ]

    @staticmethod
    def _none_if_empty(v: Any) -> Optional[Any]:
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @classmethod
    def _as_int(cls, v: Any) -> Optional[int]:
        v = cls._none_if_empty(v)
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            raise ValueError(f"Expected int, got: {v!r}")

    @classmethod
    def _as_float(cls, v: Any) -> Optional[float]:
        v = cls._none_if_empty(v)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            raise ValueError(f"Expected float, got: {v!r}")

    @staticmethod
    def _clamp_score(v: Any, default: float = 0.0) -> float:
        try:
            x = float(v)
        except Exception:
            x = default
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        return round(x, 4)

    @staticmethod
    def _to_bool(v: Any, default: bool = False) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            t = v.strip().lower()
            if t in {"true", "1", "yes"}:
                return True
            if t in {"false", "0", "no"}:
                return False
        return default

    @staticmethod
    def _extract_message_text(value: Any) -> str:
        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()
        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()
        if hasattr(value, "data") and isinstance(value.data, dict):
            if "text" in value.data:
                return str(value.data["text"]).strip()
        return str(value).strip()

    @classmethod
    def _extract_json_dict(cls, raw_text: str) -> Optional[Dict[str, Any]]:
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

        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            chunk = match.group(0)
            try:
                parsed = json.loads(chunk)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(chunk)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return None

    @staticmethod
    def _safe_decomposition_type(v: Any) -> str:
        s = str(v or "none").strip().lower()
        if s not in {"parallel", "sequential", "none"}:
            return "none"
        return s

    @staticmethod
    def _parse_payload(raw_text: str) -> Dict[str, Any]:
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

    def _flatten_retrieval_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        original_query = str(payload.get("original_query", "")).strip()
        decomposition_used = bool(payload.get("decomposition_used", False))
        decomposition_type = self._safe_decomposition_type(payload.get("decomposition_type", "none"))
        results_by_query = payload.get("results_by_query", [])
        if not isinstance(results_by_query, list):
            results_by_query = []

        flattened: List[Dict[str, Any]] = []
        global_rank = 0
        snippet_limit = int(self.snippet_char_limit or 450)

        for group in results_by_query:
            query_index = group.get("query_index")
            query_text = str(group.get("query_text", "")).strip()
            results = group.get("results", [])
            if not isinstance(results, list):
                results = []

            for item in results:
                global_rank += 1
                full_page = str(item.get("page_content", "") or "")
                snippet = re.sub(r"\s+", " ", full_page).strip()
                if snippet_limit > 0:
                    snippet = snippet[:snippet_limit]

                flattened.append(
                    {
                        "global_retrieval_index": global_rank,
                        "query_index": query_index,
                        "query_text": query_text,
                        "retrieval_rank_within_query": item.get("rank"),
                        "pgvector_score": item.get("score"),
                        "page_content_snippet": snippet,
                        "metadata": item.get("metadata", {}) or {},
                    }
                )

        return {
            "original_query": original_query,
            "decomposition_used": decomposition_used,
            "decomposition_type": decomposition_type,
            "flattened_evidences": flattened,
        }

    def _build_validator_prompt(self, normalized: Dict[str, Any]) -> str:
        keep_limit = int(self.max_kept_results or 8)

        prompt = {
            "task": "score_and_filter_evidences_only",
            "original_query": normalized.get("original_query", ""),
            "decomposition_used": normalized.get("decomposition_used", False),
            "decomposition_type": normalized.get("decomposition_type", "none"),
            "keep_limit": keep_limit,
            "forbidden_output_patterns": [
                {"query_1": {"question": "...", "answer": "..."}},
                {"question": "...", "answer": "..."},
            ],
            "required_output_schema": {
                "original_query": normalized.get("original_query", ""),
                "decomposition_used": normalized.get("decomposition_used", False),
                "decomposition_type": normalized.get("decomposition_type", "none"),
                "top_kept_evidences": [
                    {
                        "global_retrieval_index": 0,
                        "validation_score": 0.0,
                    }
                ]
            },
            "instructions": [
                "Do not answer the question.",
                "Do not create question-answer output.",
                f"Keep at most {keep_limit} evidences.",
                "Sort top_kept_evidences by validation_score descending.",
                "Return only global_retrieval_index and validation_score for each kept evidence.",
                "Do not include query_text.",
                "Do not include reason_short.",
                "Do not include is_answer_bearing.",
                "Do not include evidence_sufficiency_score.",
                "Do not include conflict_penalty."
            ],
            "valid_output_example": {
                "original_query": normalized.get("original_query", ""),
                "decomposition_used": normalized.get("decomposition_used", False),
                "decomposition_type": normalized.get("decomposition_type", "none"),
                "top_kept_evidences": [
                    {
                        "global_retrieval_index": 1,
                        "validation_score": 1.0,
                    },
                    {
                        "global_retrieval_index": 3,
                        "validation_score": 0.8,
                    }
                ]
            },
            "invalid_output_example": {
                "query_1": {
                    "question": "bad pattern",
                    "answer": "bad pattern"
                }
            },
            "evidences": normalized.get("flattened_evidences", []),
        }

        return json.dumps(prompt, ensure_ascii=False)

    @staticmethod
    def _contains_forbidden_qa_shape(obj: Dict[str, Any]) -> bool:
        forbidden_top = {"question", "answer"}
        if any(k in obj for k in forbidden_top):
            return True

        for key, value in obj.items():
            if isinstance(key, str) and key.startswith("query_"):
                if isinstance(value, dict) and ("question" in value or "answer" in value):
                    return True

        return False

    def _is_valid_validator_payload(self, obj: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(obj, dict):
            return False

        if self._contains_forbidden_qa_shape(obj):
            return False

        if "top_kept_evidences" not in obj:
            return False

        if not isinstance(obj.get("top_kept_evidences"), list):
            return False

        return True

    def _normalize_validator_payload(
        self,
        obj: Dict[str, Any],
        normalized_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        keep_limit = int(self.max_kept_results or 8)
        answer_bearing_threshold = float(self.answer_bearing_threshold or 0.5)

        flattened = normalized_input.get("flattened_evidences", [])
        known_by_idx: Dict[int, Dict[str, Any]] = {}
        for ev in flattened:
            try:
                known_by_idx[int(ev["global_retrieval_index"])] = ev
            except Exception:
                pass

        decomposition_type = self._safe_decomposition_type(
            obj.get("decomposition_type", normalized_input.get("decomposition_type", "none"))
        )

        kept = obj.get("top_kept_evidences", [])
        if not isinstance(kept, list):
            kept = []

        cleaned_kept: List[Dict[str, Any]] = []

        for item in kept:
            if not isinstance(item, dict):
                continue

            global_idx = item.get("global_retrieval_index", None)
            try:
                global_idx = int(global_idx)
            except Exception:
                continue

            matched_ev = known_by_idx.get(global_idx)
            if matched_ev is None:
                continue

            score = self._clamp_score(item.get("validation_score", 0.0))

            cleaned_kept.append(
                {
                    "global_retrieval_index": global_idx,
                    "query_index": matched_ev.get("query_index"),
                    "query_text": str(matched_ev.get("query_text", "")).strip(),
                    "validation_score": score,
                    "is_answer_bearing": bool(score >= answer_bearing_threshold),
                }
            )

        cleaned_kept.sort(key=lambda x: x["validation_score"], reverse=True)
        cleaned_kept = cleaned_kept[:keep_limit]

        return {
            "original_query": str(
                obj.get("original_query", normalized_input.get("original_query", ""))
            ).strip(),
            "decomposition_used": self._to_bool(
                obj.get("decomposition_used", normalized_input.get("decomposition_used", False)),
                default=bool(normalized_input.get("decomposition_used", False)),
            ),
            "decomposition_type": decomposition_type,
            "top_kept_evidences": cleaned_kept,
        }

    def _compute_es_deterministic(self, kept: List[Dict[str, Any]]) -> float:
        if not kept:
            return 0.0

        scores = [self._clamp_score(x.get("validation_score", 0.0)) for x in kept]
        avg_score = sum(scores) / len(scores)

        answer_bearing_count = sum(1 for x in kept if bool(x.get("is_answer_bearing", False)))
        answer_bearing_ratio = answer_bearing_count / len(kept) if kept else 0.0

        es = (0.75 * avg_score) + (0.25 * answer_bearing_ratio)
        es = min(1.0, max(0.0, es))
        return round(es, 4)

    async def is_valid_ollama_url(self, url: str) -> bool:
        try:
            base = (url or "").rstrip("/") + "/"
            async with httpx.AsyncClient() as client:
                r = await client.get(urljoin(base, "api/tags"))
                return r.status_code == HTTP_STATUS_OK
        except httpx.RequestError:
            return False

    async def get_models(self, base_url_value: str, *, tool_model_enabled: bool | None = None) -> List[str]:
        base_url = (base_url_value or "").rstrip("/") + "/"
        tags_url = urljoin(base_url, "api/tags")
        show_url = urljoin(base_url, "api/show")

        model_ids: List[str] = []
        async with httpx.AsyncClient() as client:
            tags_response = await client.get(tags_url)
            tags_response.raise_for_status()
            models = tags_response.json()

            for model in models.get(self.JSON_MODELS_KEY, []):
                model_name = model.get(self.JSON_NAME_KEY)
                if not model_name:
                    continue

                show_response = await client.post(show_url, json={"model": model_name})
                show_response.raise_for_status()
                json_data = show_response.json()

                capabilities = json_data.get(self.JSON_CAPABILITIES_KEY, [])
                if self.DESIRED_CAPABILITY in capabilities and (
                    not tool_model_enabled or self.TOOL_CALLING_CAPABILITY in capabilities
                ):
                    model_ids.append(model_name)

        return model_ids

    async def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None):
        if field_name in {"base_url", "model_name", "tool_model_enabled"}:
            try:
                base = build_config.get("base_url", {}).get("value", self.base_url)
                tool_on = build_config.get("tool_model_enabled", {}).get("value", self.tool_model_enabled)
                if base and await self.is_valid_ollama_url(base):
                    build_config["model_name"]["options"] = await self.get_models(base, tool_model_enabled=tool_on)
                else:
                    build_config["model_name"]["options"] = []
            except Exception:
                build_config["model_name"]["options"] = []
        return build_config

    def _call_chat_sync(self, user_prompt: str) -> Dict[str, Any]:
        base_url = (self.base_url or "").rstrip("/") + "/"
        if not base_url:
            raise ValueError("base_url is required.")
        if not self.model_name:
            raise ValueError("model_name is required.")

        messages = []
        sys_text = (self.system or "").strip()
        if sys_text:
            sys_text = sys_text.replace("KEEP_LIMIT", str(int(self.max_kept_results or 8)))
            messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": user_prompt})

        options: Dict[str, Any] = {}

        temp = self._as_float(self.temperature)
        if temp is not None:
            options["temperature"] = temp

        seed = self._as_int(self.seed)
        if seed is not None:
            options["seed"] = seed

        num_ctx = self._as_int(self.num_ctx)
        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        num_predict = self._as_int(self.num_predict)
        if num_predict is not None:
            options["num_predict"] = num_predict

        top_k = self._as_int(self.top_k)
        if top_k is not None:
            options["top_k"] = top_k

        top_p = self._as_float(self.top_p)
        if top_p is not None:
            options["top_p"] = top_p

        repeat_penalty = self._as_float(self.repeat_penalty)
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty

        repeat_last_n = self._as_int(self.repeat_last_n)
        if repeat_last_n is not None:
            options["repeat_last_n"] = repeat_last_n

        if self.stop_tokens:
            stops = [t.strip() for t in self.stop_tokens.split(",") if t.strip()]
            if stops:
                options["stop"] = stops

        if isinstance(self.extra_options, dict) and self.extra_options:
            options.update(self.extra_options)

        body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": options,
            "think": bool(self.enable_thinking),
        }

        if bool(self.keep_alive_enabled):
            ka = (self.keep_alive or "").strip()
            if ka:
                body["keep_alive"] = ka

        fmt = (self.format or "").strip()
        if fmt:
            body["format"] = fmt

        url = urljoin(base_url, "api/chat")
        timeout_s = self._as_float(self.timeout_s)
        if timeout_s is None:
            timeout_s = 120.0

        t0 = time.perf_counter()
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            raw = resp.json()
        wall_s = time.perf_counter() - t0

        msg = raw.get("message") or {}
        text = msg.get("content") or ""
        thinking = msg.get("thinking") or ""
        observed_has_thinking_field = bool(thinking)

        metrics = {
            "model": raw.get("model"),
            "created_at": raw.get("created_at"),
            "done": raw.get("done"),
            "done_reason": raw.get("done_reason"),
            "total_duration_ns": raw.get("total_duration"),
            "load_duration_ns": raw.get("load_duration"),
            "prompt_eval_count": raw.get("prompt_eval_count"),
            "prompt_eval_duration_ns": raw.get("prompt_eval_duration"),
            "eval_count": raw.get("eval_count"),
            "eval_duration_ns": raw.get("eval_duration"),
            "wall_time_s": wall_s,
            "requested_think": bool(self.enable_thinking),
            "observed_has_thinking_field": observed_has_thinking_field,
        }

        env: Dict[str, Any] = {"text": text, "metrics": metrics, "raw": raw}
        if bool(self.include_thinking_in_envelope):
            env["thinking"] = thinking

        env["parsed_validator_output"] = self._extract_json_dict(text)
        return env

    def build_envelope(self) -> Data:
        raw_text = self._extract_message_text(self.retrieval_message)
        parsed_payload = self._parse_payload(raw_text)

        if not parsed_payload:
            error_env = {
                "text": json.dumps(
                    {
                        "_error": "Could not parse retrieval_message",
                        "_raw_input": raw_text,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "thinking": "",
                "metrics": {},
                "raw": {},
                "parsed_validator_output": None,
                "validation": {
                    "initial_output_valid": False,
                    "used_fallback_output": False,
                },
            }
            return Data(text_key="text", data=error_env, default_value="")

        normalized_input = self._flatten_retrieval_payload(parsed_payload)
        prompt = self._build_validator_prompt(normalized_input)
        env = self._call_chat_sync(prompt)

        parsed = env.get("parsed_validator_output")
        initial_valid = self._is_valid_validator_payload(parsed)

        if initial_valid:
            normalized_output = self._normalize_validator_payload(parsed, normalized_input)
            normalized_output["evidence_sufficiency_score"] = self._compute_es_deterministic(
                normalized_output.get("top_kept_evidences", [])
            )

            env["parsed_validator_output"] = normalized_output
            env["text"] = json.dumps(normalized_output, ensure_ascii=False, indent=2)
            env["validation"] = {
                "initial_output_valid": True,
                "used_fallback_output": False,
            }
        else:
            if bool(self.use_fallback_output):
                fallback_output = {
                    "original_query": normalized_input.get("original_query", ""),
                    "decomposition_used": normalized_input.get("decomposition_used", False),
                    "decomposition_type": normalized_input.get("decomposition_type", "none"),
                    "top_kept_evidences": [],
                    "evidence_sufficiency_score": 0.0,
                }
                env["parsed_validator_output"] = fallback_output
                env["text"] = json.dumps(fallback_output, ensure_ascii=False, indent=2)
                env["validation"] = {
                    "initial_output_valid": False,
                    "used_fallback_output": True,
                }
            else:
                error_output = {
                    "_error": "Validator output was invalid.",
                    "original_query": normalized_input.get("original_query", ""),
                    "decomposition_used": normalized_input.get("decomposition_used", False),
                    "decomposition_type": normalized_input.get("decomposition_type", "none"),
                    "raw_model_output": env.get("text", ""),
                }
                env["parsed_validator_output"] = None
                env["text"] = json.dumps(error_output, ensure_ascii=False, indent=2)
                env["validation"] = {
                    "initial_output_valid": False,
                    "used_fallback_output": False,
                }

        try:
            logger.debug(f"RAG Validator (No CP) envelope metrics: {env.get('metrics')}")
        except Exception:
            pass

        return Data(
            text_key="text",
            data=env,
            default_value="",
        )