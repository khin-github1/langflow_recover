# Recovered Langflow component
# type: OllamaRAGValidatorEnvelope
# class: OllamaRAGValidatorEnvelope
# used in 5 flow(s): Cognitive RAG V0.5.2 backup, Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup, Cognitive RAG V1.1.0
# json path: node.data.node.template.code.value

import ast
import json
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
    MessageTextInput,
    MessageInput,
    Output,
    SliderInput,
)
from langflow.field_typing.range_spec import RangeSpec
from langflow.logging import logger
from langflow.schema.data import Data

HTTP_STATUS_OK = 200


class OllamaRAGValidatorEnvelope(Component):
    """
    Input:
      retrieval_message: grouped retrieval JSON from your decomposer-aware PGVector retriever

    Output:
      Data.data = {
        "text": <validator_json_as_string>,
        "thinking": <assistant_thinking_or_empty>,
        "metrics": {...},
        "raw": <full_ollama_json>,
        "parsed_validator_output": {...optional parsed json...}
      }

    Validator logic requested by user:
      1) validate ALL retrieved evidences
      2) rerank by answerability
      3) keep top <= 8
      4) compute ES and CP on those kept 8
    """

    display_name = "Ollama RAG Validator (Envelope)"
    description = "LLM-based validator that validates all retrieved evidences, reranks them, keeps top 8, and computes ES/CP."
    icon = "Ollama"
    name = "OllamaRAGValidatorEnvelope"

    JSON_MODELS_KEY = "models"
    JSON_NAME_KEY = "name"
    JSON_CAPABILITIES_KEY = "capabilities"
    DESIRED_CAPABILITY = "completion"
    TOOL_CALLING_CAPABILITY = "tools"

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
            info="Grouped retrieval results from your decomposer-aware PGVector retriever.",
            required=True,
        ),
        MessageTextInput(
            name="system",
            display_name="System",
            advanced=True,
            value=(
                "You are the RAG VALIDATOR for a bounded FAQ QA system.\n"
                "You must validate ALL retrieved evidences first, rerank them by how well they can answer the user's question, "
                "keep at most the best 8 evidences, then compute Evidence Sufficiency (ES) and Conflict Penalty (CP).\n"
                "Return ONLY valid JSON. No markdown. No extra text."
            ),
        ),
        IntInput(
            name="max_kept_results",
            display_name="Max Kept Results",
            value=8,
            advanced=False,
        ),
        BoolInput(
            name="enable_thinking",
            display_name="Enable Thinking (Ollama think)",
            value=False,
        ),
        BoolInput(
            name="include_thinking_in_envelope",
            display_name="Include Thinking in Envelope",
            value=True,
            advanced=True,
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
            value=0.1,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        IntInput(name="seed", display_name="Seed", advanced=True),
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

    # ---------------------------
    # Safe parsing helpers
    # ---------------------------
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
    def _extract_message_text(value: Any) -> str:
        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()
        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()
        if hasattr(value, "data") and isinstance(value.data, dict):
            if "text" in value.data:
                return str(value.data["text"]).strip()
        return str(value).strip()

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

    @staticmethod
    def _flatten_retrieval_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        original_query = str(payload.get("original_query", "")).strip()
        decomposition_used = bool(payload.get("decomposition_used", False))
        decomposition_type = str(payload.get("decomposition_type", "none")).strip() or "none"
        notes = str(payload.get("notes", "")).strip()
        results_by_query = payload.get("results_by_query", [])
        if not isinstance(results_by_query, list):
            results_by_query = []

        flattened: List[Dict[str, Any]] = []
        global_rank = 0

        for group in results_by_query:
            query_index = group.get("query_index")
            query_text = str(group.get("query_text", "")).strip()
            results = group.get("results", [])
            if not isinstance(results, list):
                results = []

            for item in results:
                global_rank += 1
                flattened.append(
                    {
                        "global_retrieval_index": global_rank,
                        "query_index": query_index,
                        "query_text": query_text,
                        "retrieval_rank_within_query": item.get("rank"),
                        "pgvector_score": item.get("score"),
                        "page_content": item.get("page_content", ""),
                        "metadata": item.get("metadata", {}) or {},
                    }
                )

        return {
            "original_query": original_query,
            "decomposition_used": decomposition_used,
            "decomposition_type": decomposition_type,
            "notes": notes,
            "flattened_evidences": flattened,
        }

    def _build_validator_prompt(self, normalized: Dict[str, Any]) -> str:
        max_kept = int(self.max_kept_results or 8)

        schema = {
            "original_query": normalized.get("original_query", ""),
            "decomposition_used": normalized.get("decomposition_used", False),
            "decomposition_type": normalized.get("decomposition_type", "none"),
            "notes": normalized.get("notes", ""),
            "instructions": {
                "task_order": [
                    "validate_all_evidences",
                    "rerank_by_answerability",
                    f"keep_top_{max_kept}",
                    "compute_evidence_sufficiency_score",
                    "compute_conflict_penalty",
                ],
                "definition_of_validation": (
                    "For each evidence, judge how well it can help answer the user's original query and "
                    "its assigned sub-question. Consider relevance, specificity, answer-bearingness, and usefulness."
                ),
                "definition_of_es": (
                    "Evidence Sufficiency Score (0 to 1): based ONLY on the kept evidences after reranking. "
                    "Higher means the kept evidences are collectively enough to answer safely."
                ),
                "definition_of_cp": (
                    "Conflict Penalty (0 to 1): based ONLY on the kept evidences after reranking. "
                    "Higher means the kept evidences contain conflicting answer-bearing information such as dates, "
                    "numbers, fees, yes/no contradictions, or policy contradictions."
                ),
                "keep_limit": max_kept,
            },
            "return_json_schema": {
                "original_query": "string",
                "decomposition_used": "bool",
                "decomposition_type": "string",
                "validated_evidences_all": [
                    {
                        "global_retrieval_index": "int",
                        "query_index": "int_or_null",
                        "query_text": "string",
                        "validation_score": "float_0_to_1",
                        "is_answer_bearing": "bool",
                        "reason_short": "string"
                    }
                ],
                "top_kept_evidences": [
                    {
                        "global_retrieval_index": "int",
                        "query_index": "int_or_null",
                        "query_text": "string",
                        "validation_score": "float_0_to_1",
                        "is_answer_bearing": "bool",
                        "reason_short": "string"
                    }
                ],
                "evidence_sufficiency_score": "float_0_to_1",
                "conflict_penalty": "float_0_to_1",
                "rag_valid": "bool",
                "decision_hint": "accept_or_retry_or_clarify_or_abstain",
                "notes": "short_string"
            },
            "evidences": normalized.get("flattened_evidences", []),
        }

        return json.dumps(schema, ensure_ascii=False)

    # ---------------------------
    # Model list refresh
    # ---------------------------
    async def is_valid_ollama_url(self, url: str) -> bool:
        try:
            base = (url or "").rstrip("/") + "/"
            async with httpx.AsyncClient() as client:
                r = await client.get(urljoin(base, "api/tags"))
                return r.status_code == HTTP_STATUS_OK
        except httpx.RequestError:
            return False

    async def get_models(self, base_url_value: str, *, tool_model_enabled: bool | None = None) -> list[str]:
        base_url = (base_url_value or "").rstrip("/") + "/"
        tags_url = urljoin(base_url, "api/tags")
        show_url = urljoin(base_url, "api/show")

        model_ids: list[str] = []
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

    # ---------------------------
    # Core call
    # ---------------------------
    def _call_chat_sync(self, user_prompt: str) -> Dict[str, Any]:
        base_url = (self.base_url or "").rstrip("/") + "/"
        if not base_url:
            raise ValueError("base_url is required.")
        if not self.model_name:
            raise ValueError("model_name is required.")

        messages = []
        sys_text = (self.system or "").strip()
        if sys_text:
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

        try:
            eval_s = (metrics["eval_duration_ns"] or 0) / 1e9
            out_toks = metrics["eval_count"] or 0
            metrics["tokens_per_second_out"] = (out_toks / eval_s) if eval_s > 0 else None
        except Exception:
            pass

        env: Dict[str, Any] = {"text": text, "metrics": metrics, "raw": raw}
        if bool(self.include_thinking_in_envelope):
            env["thinking"] = thinking

        try:
            env["parsed_validator_output"] = json.loads(text)
        except Exception:
            env["parsed_validator_output"] = None

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
            }
            return Data(text_key="text", data=error_env, default_value="")

        normalized = self._flatten_retrieval_payload(parsed_payload)
        user_prompt = self._build_validator_prompt(normalized)

        env = self._call_chat_sync(user_prompt)

        try:
            logger.debug(f"RAG Validator envelope metrics: {env.get('metrics')}")
        except Exception:
            pass

        return Data(
            text_key="text",
            data=env,
            default_value="",
        )