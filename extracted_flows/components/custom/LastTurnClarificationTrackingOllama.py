# Recovered Langflow component
# type: LastTurnClarificationTrackingOllama
# class: LastTurnClarificationTrackingOllama
# used in 16 flow(s): AIT Cognitive RAG, Cognitive RAG V1.1.5, Cognitive RAG V1.2.0 (1), Cognitive RAG V1.2.0 AIT CR, Cognitive RAG V1.2.0 AIT Loose, Cognitive RAG V1.2.0 AIT Normal, Cognitive RAG V1.2.0 AIT RR, Cognitive RAG V1.2.0 AIT Reason ...
# json path: node.data.node.template.code.value

from __future__ import annotations

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
    IntInput,
    MessageTextInput,
    Output,
    SliderInput,
)
from langflow.field_typing.range_spec import RangeSpec
from langflow.schema.data import Data

HTTP_STATUS_OK = 200


class LastTurnClarificationTrackingOllama(Component):
    display_name = "Last-Turn Clarification Tracking [Ollama]"
    description = (
        "LLM-powered detector that checks whether the last AI/Assistant turn in history "
        "is a clarification question. Python first extracts the last AI turn deterministically."
    )
    icon = "search"
    name = "LastTurnClarificationTrackingOllama"

    JSON_MODELS_KEY = "models"
    JSON_NAME_KEY = "name"
    JSON_CAPABILITIES_KEY = "capabilities"
    DESIRED_CAPABILITY = "completion"
    TOOL_CALLING_CAPABILITY = "tools"

    DEFAULT_SYSTEM = (
        "You are a clarification detector.\n"
        "Your job is ONLY to inspect the provided LAST_AI_TURN and decide "
        "whether it is a clarification question.\n\n"
        "Return ONLY valid JSON.\n"
        "Do not explain.\n"
        "Do not output markdown.\n"
        "Be conservative.\n"
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
        BoolInput(
            name="tool_model_enabled",
            display_name="Tool Model Enabled",
            value=True,
            real_time_refresh=True,
            advanced=True,
        ),
        MessageTextInput(
            name="history_text",
            display_name="History (raw text)",
            required=True,
        ),
        IntInput(
            name="max_history_chars",
            display_name="Max history chars sent",
            value=4000,
            advanced=True,
        ),
        MessageTextInput(
            name="system",
            display_name="System",
            advanced=True,
            value=DEFAULT_SYSTEM,
        ),
        MessageTextInput(
            name="template",
            display_name="Template",
            advanced=True,
            value=(
                "Return ONLY valid JSON.\n\n"
                "LAST_AI_TURN: {{LAST_AI_TURN}}\n\n"
                "TASK:\n"
                "Decide whether LAST_AI_TURN is a clarification question.\n\n"
                "Allowed clarification types:\n"
                "- intake_year_or_term\n"
                "- program\n"
                "- degree_level\n"
                "- program_or_degree\n"
                "- fee_type\n"
                "- scholarship_type\n"
                "- accommodation_type\n"
                "- generic\n"
                "- none\n\n"
                "OUTPUT JSON SCHEMA:\n"
                "{\n"
                '  "clarification_asked_in_last_ai_turn": false,\n'
                '  "matched_clarification_type": "none"\n'
                "}\n\n"
                "RULES:\n"
                "- matched_clarification_type must be exactly one of the allowed values.\n"
                "- If LAST_AI_TURN is not a clarification question, use \"none\".\n"
                "- Treat questions like \"Which intake are you referring to?\" as clarification.\n"
                "- Treat questions like \"Which program do you mean?\" as clarification.\n"
                "- Treat questions like \"Do you mean tuition fee or application fee?\" as clarification.\n"
            ),
        ),
        BoolInput(
            name="enable_thinking",
            display_name="Enable Thinking (Ollama think)",
            value=False,
            advanced=False,
        ),
        BoolInput(
            name="use_fallback_output",
            display_name="Use Fallback Output",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="use_deterministic_fallback",
            display_name="Use Deterministic Fallback",
            value=True,
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
        Output(display_name="Clarification Tracking", name="tracking", method="run"),
    ]

    _re_json_obj = re.compile(r"\{.*\}", re.DOTALL)

    _re_intake_question = re.compile(
        r"\b(which|what)\s+(intake|term|semester|year|batch)\b|\bwhich intake are you referring to\b",
        re.I,
    )
    _re_program_question = re.compile(
        r"\bwhich\s+(program|degree|course|department|major)\b|\bwhat program\b|\bwhich one are you referring to\b",
        re.I,
    )
    _re_degree_question = re.compile(
        r"\bwhich\s+degree\b|\bwhat degree level\b|\bmaster'?s or phd\b",
        re.I,
    )
    _re_fee_type_question = re.compile(
        r"\bwhich\s+(fee|fees|cost|type of fee)\b|\bdo you mean tuition\b|\bwhat kind of fee\b",
        re.I,
    )
    _re_scholarship_type_question = re.compile(
        r"\bwhich\s+scholarship\b|\bwhat scholarship\b|\bwhich scholarship are you referring to\b",
        re.I,
    )
    _re_accommodation_question = re.compile(
        r"\bwhich\s+(room|dorm|accommodation|housing)\b|\bwhat type of accommodation\b",
        re.I,
    )
    _re_generic_clarify = re.compile(
        r"\b(can you clarify|could you clarify|please clarify|what do you mean|which one do you mean|do you mean)\b",
        re.I,
    )

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

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        m = self._re_json_obj.search(raw)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def _trim_history(self, text: str) -> str:
        text = text or ""
        limit = int(self.max_history_chars or 0)
        if limit <= 0 or len(text) <= limit:
            return text
        return text[-limit:]

    def _extract_last_ai_turn(self, history: str) -> str:
        if not history:
            return ""

        lines = [ln.rstrip() for ln in history.splitlines() if ln.strip()]
        ai_lines: List[str] = []

        for ln in lines:
            stripped = ln.strip()
            lower = stripped.lower()

            if lower.startswith("ai:") or lower.startswith("assistant:"):
                ai_lines.append(stripped.split(":", 1)[1].strip())
            elif lower.startswith("assistant -") or lower.startswith("ai -"):
                ai_lines.append(stripped.split("-", 1)[1].strip())

        if ai_lines:
            return ai_lines[-1].strip()

        # fallback: if no explicit AI markers, assume alternating conversation and try to grab last question-like line
        for ln in reversed(lines):
            if "?" in ln:
                return ln.strip()

        return ""

    def _deterministic_type_from_last_turn(self, last_ai_turn: str) -> str:
        t = last_ai_turn or ""
        if not t:
            return "none"
        if self._re_intake_question.search(t):
            return "intake_year_or_term"
        if self._re_program_question.search(t):
            return "program_or_degree"
        if self._re_degree_question.search(t):
            return "degree_level"
        if self._re_fee_type_question.search(t):
            return "fee_type"
        if self._re_scholarship_type_question.search(t):
            return "scholarship_type"
        if self._re_accommodation_question.search(t):
            return "accommodation_type"
        if self._re_generic_clarify.search(t):
            return "generic"
        if "?" in t:
            return "generic"
        return "none"

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

        timeout_s = float(self.timeout_s or 120)
        t0 = time.perf_counter()
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(url, json=body)
            r.raise_for_status()
            raw = r.json()
        wall_s = time.perf_counter() - t0

        msg = raw.get("message") or {}
        text = msg.get("content") or ""

        metrics = {
            "model": raw.get("model"),
            "created_at": raw.get("created_at"),
            "done": raw.get("done"),
            "done_reason": raw.get("done_reason"),
            "prompt_eval_count": raw.get("prompt_eval_count"),
            "eval_count": raw.get("eval_count"),
            "prompt_eval_duration_ns": raw.get("prompt_eval_duration"),
            "eval_duration_ns": raw.get("eval_duration"),
            "total_duration_ns": raw.get("total_duration"),
            "load_duration_ns": raw.get("load_duration"),
            "wall_time_s": wall_s,
            "requested_think": bool(self.enable_thinking),
            "observed_has_thinking_field": bool((msg.get("thinking") or "")),
        }

        return {"text": text, "metrics": metrics}

    def _normalize_output(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        allowed_types = {
            "intake_year_or_term",
            "program",
            "degree_level",
            "program_or_degree",
            "fee_type",
            "scholarship_type",
            "accommodation_type",
            "generic",
            "none",
        }

        clarification_asked = bool(obj.get("clarification_asked_in_last_ai_turn", False))
        matched_type = str(obj.get("matched_clarification_type", "none")).strip()
        if matched_type not in allowed_types:
            matched_type = "none"

        if not clarification_asked:
            matched_type = "none"

        return {
            "clarification_asked_in_last_ai_turn": clarification_asked,
            "matched_clarification_type": matched_type,
        }

    def _fallback_output(self, last_ai_turn: str) -> Dict[str, Any]:
        if bool(self.use_deterministic_fallback):
            matched_type = self._deterministic_type_from_last_turn(last_ai_turn)
            return {
                "clarification_asked_in_last_ai_turn": matched_type != "none",
                "matched_clarification_type": matched_type,
            }

        return {
            "clarification_asked_in_last_ai_turn": False,
            "matched_clarification_type": "none",
        }

    def run(self) -> Data:
        history = self._trim_history((self.history_text or "").strip())
        last_ai_turn = self._extract_last_ai_turn(history)

        prompt = (self.template or "")
        prompt = prompt.replace("{{LAST_AI_TURN}}", last_ai_turn)

        env = self._call_ollama(prompt)
        raw_text = env.get("text", "")
        parsed = self._extract_json(raw_text)

        if isinstance(parsed, dict):
            payload = self._normalize_output(parsed)

            # If model missed an obvious clarification, deterministic fallback can override
            if bool(self.use_deterministic_fallback):
                det = self._fallback_output(last_ai_turn)
                if det["clarification_asked_in_last_ai_turn"] and not payload["clarification_asked_in_last_ai_turn"]:
                    payload = det
        else:
            if bool(self.use_fallback_output):
                payload = self._fallback_output(last_ai_turn)
            else:
                payload = {
                    "_error": "Clarification tracking output was invalid.",
                    "raw_model_output": raw_text,
                }

        self.status = (
            f"clarAsked={int(bool(payload.get('clarification_asked_in_last_ai_turn', False)))} "
            f"type={payload.get('matched_clarification_type', 'none')} "
            f"lastAI={'yes' if last_ai_turn else 'no'}"
            if "_error" not in payload else "clarification_tracking_error"
        )

        return Data(text_key="clarification_tracking", data=payload, default_value="")