# Recovered Langflow component
# type: OllamaModeAwareGeneratorEnvelope
# class: OllamaModeAwareGeneratorEnvelope
# used in 23 flow(s): AIT Cognitive RAG, Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup, Cognitive RAG V1.1.0, Cognitive RAG V1.1.0 Clean, Cognitive RAG V1.1.0 Eval, Cognitive RAG V1.1.5 ...
# json path: node.data.node.template.code.value

import ast
import json
import time
from typing import Any, Dict, Optional
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


class OllamaModeAwareGeneratorEnvelope(Component):
    """
    Input:
      - controller_message: contains generation_mode = normal | reason
      - prompt_message: plain-text formatted generator prompt
        (e.g. "User Query: ...\\n\\nContext:\\n...")

    Behavior:
      - if generation_mode == "reason" -> think = True
      - else -> think = False

    Output:
      Data.data = {
        "text": <assistant_text>,
        "thinking": <assistant_thinking_or_empty>,
        "metrics": {...},
        "raw": <full_ollama_json>,
        "generation_mode": "normal" | "reason",
        "thinking_used": true | false,     # pipeline/control decision
        "used_reasoning": true | false,    # kept for compatibility
        "effective_think": true | false,
        "has_thinking": true | false,      # whether Ollama returned non-empty thinking
        "used_prompt": <prompt text sent to model>
      }
    """

    display_name = "Ollama Generator (Mode-Aware Envelope)"
    description = "Mode-aware Ollama generator: enables think automatically when generation_mode = reason."
    icon = "Ollama"
    name = "OllamaModeAwareGeneratorEnvelope"

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
            name="controller_message",
            display_name="Controller Message",
            info="Controller output containing generation_mode.",
            required=True,
        ),
        MessageInput(
            name="prompt_message",
            display_name="Prompt Message",
            info="Formatted generator prompt message (plain text).",
            required=True,
        ),
        MessageTextInput(
            name="system_normal",
            display_name="System Prompt (Normal)",
            value=(
                "You are a careful FAQ answer generator.\n"
                "Answer using only the provided context.\n"
                "Be concise, factual, and direct.\n"
                "If the context is insufficient, say so clearly.\n"
                "Do not invent facts."
            ),
            advanced=False,
        ),
        MessageTextInput(
            name="system_reason",
            display_name="System Prompt (Reason)",
            value=(
                "You are a careful FAQ answer generator.\n"
                "Use the provided context to reason carefully before answering.\n"
                "Synthesize across multiple evidence snippets when needed.\n"
                "Be factual, explicit, and grounded in the context.\n"
                "If the context is insufficient or conflicting, say so clearly.\n"
                "Do not invent facts."
            ),
            advanced=False,
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
        except Exception as e:
            raise ValueError(f"Expected int, got: {v!r}") from e

    @classmethod
    def _as_float(cls, v: Any) -> Optional[float]:
        v = cls._none_if_empty(v)
        if v is None:
            return None
        try:
            return float(v)
        except Exception as e:
            raise ValueError(f"Expected float, got: {v!r}") from e

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

    def _call_chat_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        effective_think: bool,
        generation_mode: str,
    ) -> Dict[str, Any]:
        base_url = (self.base_url or "").rstrip("/") + "/"
        if not base_url:
            raise ValueError("base_url is required.")
        if not self.model_name:
            raise ValueError("model_name is required.")
        if user_prompt is None:
            raise ValueError("prompt_message is required.")

        messages = []
        sys_text = (system_prompt or "").strip()
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
            "think": bool(effective_think),
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

        # Runtime observation: did Ollama actually return a non-empty thinking field?
        has_thinking = bool(str(thinking).strip())

        # Pipeline/control decision: did this run use thinking mode?
        # This is the main flag your later verifier should use.
        thinking_used = bool(generation_mode == "reason")

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
            "requested_think": bool(effective_think),
            "thinking_used": thinking_used,
            "observed_has_thinking_field": has_thinking,
        }

        try:
            eval_s = (metrics["eval_duration_ns"] or 0) / 1e9
            out_toks = metrics["eval_count"] or 0
            metrics["tokens_per_second_out"] = (out_toks / eval_s) if eval_s > 0 else None
        except Exception:
            metrics["tokens_per_second_out"] = None

        env: Dict[str, Any] = {
            "text": text,
            "metrics": metrics,
            "raw": raw,
            "generation_mode": generation_mode,
            "thinking_used": thinking_used,
            "used_reasoning": thinking_used,   # compatibility alias
            "has_thinking": has_thinking,
            "effective_think": bool(effective_think),
            "used_prompt": user_prompt,
        }

        if bool(self.include_thinking_in_envelope):
            env["thinking"] = thinking

        return env

    def build_envelope(self) -> Data:
        controller_raw = self._extract_message_text(self.controller_message)
        prompt_text = self._extract_message_text(self.prompt_message)

        controller_payload = self._parse_payload(controller_raw)
        if not controller_payload:
            error_env = {
                "text": json.dumps(
                    {
                        "_error": "Could not parse controller_message",
                        "_raw_controller": controller_raw,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "thinking": "",
                "metrics": {},
                "raw": {},
                "generation_mode": None,
                "thinking_used": False,
                "used_reasoning": False,
                "effective_think": False,
                "has_thinking": False,
                "used_prompt": prompt_text,
            }
            return Data(text_key="text", data=error_env, default_value="")

        generation_mode = str(controller_payload.get("generation_mode", "normal")).strip().lower()
        if generation_mode not in {"normal", "reason"}:
            generation_mode = "normal"

        effective_think = generation_mode == "reason"
        system_prompt = self.system_reason if effective_think else self.system_normal

        env = self._call_chat_sync(
            system_prompt=system_prompt,
            user_prompt=prompt_text,
            effective_think=effective_think,
            generation_mode=generation_mode,
        )

        try:
            logger.debug(f"Mode-aware generator metrics: {env.get('metrics')}")
        except Exception:
            pass

        return Data(
            text_key="text",
            data=env,
            default_value="",
        )