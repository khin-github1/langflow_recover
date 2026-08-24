# Recovered Langflow component
# type: ClarificationTrackingExtractorOllama
# class: ClarificationTrackingExtractorOllama
# used in 1 flow(s): Cognitive RAG V1.1.5 Backup
# json path: node.data.node.template.code.value

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx

from langflow.custom import Component
from langflow.io import (
    BoolInput,
    DictInput,
    DropdownInput,
    IntInput,
    MessageTextInput,
    Output,
    SliderInput,
)
from langflow.field_typing.range_spec import RangeSpec
from langflow.schema.data import Data

HTTP_STATUS_OK = 200


class ClarificationTrackingExtractorOllama(Component):
    display_name = "Clarification Tracking Extractor [Ollama]"
    description = "Parallel analyzer node that only detects whether a clarification was asked and whether USER is answering it."
    icon = "search"
    name = "ClarificationTrackingExtractorOllama"

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
        BoolInput(
            name="tool_model_enabled",
            display_name="Tool Model Enabled",
            value=True,
            real_time_refresh=True,
            advanced=True,
        ),
        MessageTextInput(
            name="user_input",
            display_name="User Input",
            required=True,
        ),
        MessageTextInput(
            name="history_text",
            display_name="History (raw text)",
            required=False,
        ),
        IntInput(
            name="max_history_chars",
            display_name="Max history chars sent",
            value=6000,
            advanced=True,
        ),
        MessageTextInput(
            name="system",
            display_name="System",
            advanced=True,
            value=(
                "You are a CLARIFICATION TRACKING EXTRACTOR.\n"
                "Return ONLY valid JSON with exactly these fields:\n"
                "{\n"
                '  "clarification_asked_in_history": false,\n'
                '  "user_is_answering_clarification": false\n'
                "}"
            ),
        ),
        MessageTextInput(
            name="template",
            display_name="Template",
            advanced=True,
            value=(
                "Return ONLY valid JSON.\n\n"
                "USER: {{USER}}\n"
                "HISTORY: {{HISTORY}}\n\n"
                "TASK:\n"
                "1) clarification_asked_in_history = whether HISTORY contains a clarification question from the system.\n"
                "2) user_is_answering_clarification = whether USER looks like a direct answer to that clarification.\n\n"
                "OUTPUT JSON:\n"
                "{\n"
                '  "clarification_asked_in_history": false,\n'
                '  "user_is_answering_clarification": false\n'
                "}"
            ),
        ),
        BoolInput(
            name="enable_thinking",
            display_name="Enable Thinking",
            value=False,
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.0,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        IntInput(
            name="timeout_s",
            display_name="Timeout (seconds)",
            value=120,
            advanced=True,
        ),
        DictInput(
            name="extra_options",
            display_name="Extra options (dict)",
            value={},
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Clarification Tracking", name="clarification_data", method="build_output"),
    ]

    _re_json_obj = re.compile(r"\{.*\}", re.DOTALL)
    _re_degree = re.compile(r"\b(master(?:'s)?|msc|m\.?sc\.?|phd|doctor(?:al)?|bachelor(?:'s)?|undergraduate)\b", re.I)
    _re_intake = re.compile(r"\b((?:aug(?:ust)?|jan(?:uary)?|sep(?:tember)?|spring|fall|autumn|summer)\s+\d{4}|\d{4}\s+intake)\b", re.I)

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

    @classmethod
    def _extract_json(cls, text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        m = cls._re_json_obj.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None

    def _fallback_tracking(self, user: str, history: str) -> Dict[str, bool]:
        h = (history or "").strip().lower()
        u = (user or "").strip().lower()

        clarification_asked = bool(
            h and (
                "which one" in h
                or "which program" in h
                or "which intake" in h
                or "which degree" in h
                or "do you mean" in h
                or "could you clarify" in h
                or "what do you mean" in h
                or "which court" in h
            )
        )

        user_answering = bool(
            clarification_asked and u and (
                len(u) <= 80
                or u.startswith("the one")
                or u.startswith("for ")
                or u.startswith("it is")
                or u.startswith("it's")
                or bool(self._re_intake.search(u))
                or bool(self._re_degree.search(u))
            )
        )

        return {
            "clarification_asked_in_history": clarification_asked,
            "user_is_answering_clarification": user_answering,
        }

    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        base = (self.base_url or "").rstrip("/") + "/"
        url = urljoin(base, "api/chat")

        messages = []
        sys_text = (self.system or "").strip()
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(self.temperature or 0.0),
                **(self.extra_options or {}),
            },
            "think": bool(self.enable_thinking),
            "format": "json",
        }

        timeout_s = int(self.timeout_s or 120)
        t0 = time.perf_counter()
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(url, json=body)
            r.raise_for_status()
            raw = r.json()
        wall_s = time.perf_counter() - t0

        msg = raw.get("message") or {}
        text = msg.get("content") or ""
        thinking = msg.get("thinking") or ""

        metrics = {
            "model": raw.get("model"),
            "wall_time_s": wall_s,
            "prompt_eval_count": raw.get("prompt_eval_count"),
            "eval_count": raw.get("eval_count"),
            "requested_think": bool(self.enable_thinking),
            "observed_has_thinking_field": bool(thinking),
        }
        return {"text": text, "metrics": metrics}

    def build_output(self) -> Data:
        user = (self.user_input or "").strip()
        history = (self.history_text or "").strip()
        max_hist = int(self.max_history_chars or 6000)
        hist_send = history if (max_hist <= 0 or len(history) <= max_hist) else (history[:max_hist] + "\n[TRUNCATED]")

        prompt = (self.template or "")
        prompt = prompt.replace("{{USER}}", user)
        prompt = prompt.replace("{{HISTORY}}", hist_send)

        env = self._call_ollama(prompt)
        obj = self._extract_json(env.get("text", "")) or {}

        tracking = {
            "clarification_asked_in_history": bool(obj.get("clarification_asked_in_history", False)),
            "user_is_answering_clarification": bool(obj.get("user_is_answering_clarification", False)),
        }

        if not tracking["clarification_asked_in_history"] and not tracking["user_is_answering_clarification"]:
            tracking = self._fallback_tracking(user, hist_send)

        self.status = (
            f"clarHist={int(tracking['clarification_asked_in_history'])} "
            f"clarReply={int(tracking['user_is_answering_clarification'])}"
        )

        return Data(text_key="clarification", data=tracking, default_value="")