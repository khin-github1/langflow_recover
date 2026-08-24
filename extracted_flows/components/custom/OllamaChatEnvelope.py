# Recovered Langflow component
# type: OllamaChatEnvelope
# class: OllamaChatEnvelope
# used in 4 flow(s): CRCV-v1.103 (Normal) Khin, CRCV-v1.103 (Normal) Oak (Backup), CRCV-v1.103 (Normal) Oak Backups, CRCV-v1.103 (Reasoning) Experiment
# json path: node.data.node.template.code.value

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
    MessageTextInput,
    Output,
    SliderInput,
)
from langflow.field_typing.range_spec import RangeSpec
from langflow.logging import logger
from langflow.schema.data import Data  # IMPORTANT: real LangFlow Data type

HTTP_STATUS_OK = 200


class OllamaChatEnvelope(Component):
    """
    Single-output generator (Data):
      Data.data = {
        "text": <assistant_text>,
        "metrics": {...},
        "raw": <full_ollama_json>
      }

    This lets you fan-out to multiple downstream nodes reliably.
    """

    display_name = "Ollama Chat (Envelope)"
    description = "One output (Data): {text, metrics, raw} from Ollama /api/chat."
    icon = "Ollama"
    name = "OllamaChatEnvelope"

    # For model dropdown refresh
    JSON_MODELS_KEY = "models"
    JSON_NAME_KEY = "name"
    JSON_CAPABILITIES_KEY = "capabilities"
    DESIRED_CAPABILITY = "completion"
    TOOL_CALLING_CAPABILITY = "tools"

    inputs = [
        MessageTextInput(
            name="base_url",
            display_name="Base URL",
            info="Ollama endpoint, e.g. http://localhost:11434",
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
        MessageTextInput(
            name="user_prompt",
            display_name="User Prompt",
        ),
        MessageTextInput(
            name="system",
            display_name="System",
            info="Optional system prompt.",
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        IntInput(name="num_ctx", display_name="num_ctx", advanced=True),
        IntInput(name="num_predict", display_name="num_predict", advanced=True),
        IntInput(name="top_k", display_name="top_k", advanced=True),
        FloatInput(name="top_p", display_name="top_p", advanced=True),
        FloatInput(name="repeat_penalty", display_name="repeat_penalty", advanced=True),
        IntInput(name="repeat_last_n", display_name="repeat_last_n", advanced=True),
        MessageTextInput(
            name="stop_tokens",
            display_name="Stop Tokens",
            info="Comma-separated -> options.stop",
            advanced=True,
        ),
        MessageTextInput(
            name="format",
            display_name="Format",
            info="Optional (e.g., json).",
            advanced=True,
        ),
        DictInput(
            name="extra_options",
            display_name="Extra options (dict)",
            value={},
            info="Merged into Ollama request 'options'.",
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
    def _call_chat_sync(self) -> Dict[str, Any]:
        base_url = (self.base_url or "").rstrip("/") + "/"
        if not base_url:
            raise ValueError("base_url is required.")
        if not self.model_name:
            raise ValueError("model_name is required.")
        if self.user_prompt is None:
            raise ValueError("user_prompt is required.")

        messages = []
        sys_text = (self.system or "").strip()
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": self.user_prompt})

        options: Dict[str, Any] = {}

        temp = self._as_float(self.temperature)
        if temp is not None:
            options["temperature"] = temp

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
        }

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

        text = ((raw.get("message") or {}).get("content")) or ""

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
        }

        try:
            eval_s = (metrics["eval_duration_ns"] or 0) / 1e9
            out_toks = metrics["eval_count"] or 0
            metrics["tokens_per_second_out"] = (out_toks / eval_s) if eval_s > 0 else None
        except Exception:
            pass

        return {"text": text, "metrics": metrics, "raw": raw}

    def build_envelope(self) -> Data:
        env = self._call_chat_sync()
        try:
            logger.debug(f"Ollama envelope metrics: {env.get('metrics')}")
        except Exception:
            pass

        # IMPORTANT: return LangFlow Data (typed), not plain dict
        return Data(
            text_key="text",
            data=env,
            default_value="",
        )
