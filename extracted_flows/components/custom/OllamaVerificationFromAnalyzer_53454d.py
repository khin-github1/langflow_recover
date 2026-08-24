# Recovered Langflow component
# type: OllamaVerificationFromAnalyzer
# class: OllamaVerificationFromAnalyzer
# used in 13 flow(s): Cognitive RAG V1.2.0 AIT CR, Cognitive RAG V1.2.0 AIT Loose, Cognitive RAG V1.2.0 AIT Normal, Cognitive RAG V1.2.0 AIT RR, Cognitive RAG V1.2.0 AIT Reason, Cognitive RAG V1.2.0 AIT Strict , Cognitive RAG V1.2.0 AIT VaR, Cognitive RAG V1.2.0 AIT VeR ...
# json path: node.data.node.template.code.value

import json
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx

from langflow.custom import Component
from langflow.io import (
    BoolInput,
    DataInput,
    DictInput,
    DropdownInput,
    IntInput,
    MessageTextInput,
    Output, 
    SliderInput,
)
from langflow.field_typing.range_spec import RangeSpec
from langflow.logging import logger
from langflow.schema.data import Data

HTTP_STATUS_OK = 200


class OllamaVerificationFromAnalyzer(Component):
    """
    Single-output verifier (Data).

    Reads expected_general_slots / expected_critical_slots from analyzer_data.
    Sends generated_answer + rag_context + analyzer context to Ollama.
    Expects JSON-only verifier output.
    Collects raw claim scores and slot coverage scores only.
    No deterministic scoring or decision logic here.
    """

    display_name = "Ollama Verification From Analyzer"
    description = "Verifier that uses expected slots from analyzer_data and collects raw claim/slot scores."
    icon = "Ollama"
    name = "OllamaVerificationFromAnalyzer"

    JSON_MODELS_KEY = "models"
    JSON_NAME_KEY = "name"
    JSON_CAPABILITIES_KEY = "capabilities"
    DESIRED_CAPABILITY = "completion"
    TOOL_CALLING_CAPABILITY = "tools"

    DEFAULT_SYSTEM_MESSAGE = """
You are a verification model for a bounded RAG FAQ system.

You MUST use the expected slots provided by the analyzer.
Do NOT invent your own slots.

Your job:
1. Split GENERATED_ANSWER into atomic factual claims.
2. Verify each claim ONLY against RAG_CONTEXT.
3. Give each claim a support_score in [0, 1].
4. Assign slot_type for each claim when possible.
5. Assign slot_importance as general, critical, or other.
6. Check slot coverage ONLY for analyzer-provided slots.
7. Return ONLY valid JSON. No markdown. No explanation.

Rules:
- Use ONLY RAG_CONTEXT.
- Do NOT use outside knowledge.
- Keep claims atomic.
- Do NOT return reasons.
- Do NOT return evidence snippets.
- Do NOT return supported true/false.
- Only return the exact JSON schema requested.
""".strip()

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
            name="generated_answer",
            display_name="Generated Answer",
            info="Answer produced by the generator.",
        ),
        MessageTextInput(
            name="rag_context",
            display_name="RAG Context",
            info="Retrieved evidence used for verification.",
        ),
        DataInput(
            name="analyzer_data",
            display_name="Analyzer Data",
            info="Analyzer/controller output as Data containing expected_general_slots and expected_critical_slots.",
            required=True,
        ),
        MessageTextInput(
            name="system",
            display_name="System Message",
            info="Full system message used by the verifier.",
            value=DEFAULT_SYSTEM_MESSAGE,
            advanced=False,
        ),
        BoolInput(
            name="enable_thinking",
            display_name="Enable Thinking (Ollama think)",
            value=False,
            advanced=True,
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
            value=0.0,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        IntInput(name="seed", display_name="Seed", advanced=True),
        IntInput(name="num_ctx", display_name="num_ctx", advanced=True),
        IntInput(name="num_predict", display_name="num_predict", advanced=True),
        IntInput(name="top_k", display_name="top_k", advanced=True),
        BoolInput(
            name="tool_model_enabled",
            display_name="Tool Model Enabled",
            value=True,
            real_time_refresh=True,
            advanced=True,
        ),
        DictInput(
            name="extra_options",
            display_name="Extra options (dict)",
            value={},
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
        Output(display_name="Verification", name="verification", method="build_verification"),
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

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    @classmethod
    def _parse_json_obj(cls, text: str) -> Dict[str, Any]:
        text = cls._strip_fences(text)

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        return {}

    def _parse_analyzer(self) -> Dict[str, Any]:
        analyzer = {}

        if hasattr(self.analyzer_data, "data") and isinstance(self.analyzer_data.data, dict):
            analyzer = self.analyzer_data.data
        elif isinstance(self.analyzer_data, dict):
            analyzer = self.analyzer_data

        if not analyzer:
            return {}

        expected_general_slots = analyzer.get("expected_general_slots", [])
        expected_critical_slots = analyzer.get("expected_critical_slots", [])

        if not isinstance(expected_general_slots, list):
            expected_general_slots = []
        if not isinstance(expected_critical_slots, list):
            expected_critical_slots = []

        analyzer["expected_general_slots"] = [
            str(x).strip().lower() for x in expected_general_slots if str(x).strip()
        ]
        analyzer["expected_critical_slots"] = [
            str(x).strip().lower() for x in expected_critical_slots if str(x).strip()
        ]
        return analyzer

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
                    build_config["model_name"]["options"] = await self.get_models(
                        base,
                        tool_model_enabled=tool_on,
                    )
                else:
                    build_config["model_name"]["options"] = []
            except Exception:
                build_config["model_name"]["options"] = []
        return build_config

    def _build_messages(self, analyzer: Dict[str, Any]) -> List[Dict[str, str]]:
        original_question = str(analyzer.get("original_question", "")).strip()
        expected_general_slots = analyzer.get("expected_general_slots", [])
        expected_critical_slots = analyzer.get("expected_critical_slots", [])

        base_system = (self.system or "").strip()
        if not base_system:
            base_system = self.DEFAULT_SYSTEM_MESSAGE

        user_prompt = f"""
Return EXACTLY this JSON schema:

{{
  "claims": [
    {{
      "claim_id": "c1",
      "claim_text": "string",
      "support_score": 0.0,
      "slot_type": "deadline|eligibility|required_documents|fee|intake|application_process|scholarship|other",
      "slot_importance": "general|critical|other"
    }}
  ],
  "slot_coverage": {{
    "general": [
      {{
        "slot_name": "string",
        "status": "answered|missing|not_applicable",
        "support_score": 0.0
      }}
    ],
    "critical": [
      {{
        "slot_name": "string",
        "status": "answered|missing|not_applicable",
        "support_score": 0.0
      }}
    ]
  }}
}}

ORIGINAL_QUESTION:
{original_question}

ANALYZER_EXPECTED_GENERAL_SLOTS:
{json.dumps(expected_general_slots, ensure_ascii=False)}

ANALYZER_EXPECTED_CRITICAL_SLOTS:
{json.dumps(expected_critical_slots, ensure_ascii=False)}

GENERATED_ANSWER:
{self.generated_answer or ""}

RAG_CONTEXT:
{self.rag_context or ""}
""".strip()

        return [
            {"role": "system", "content": base_system},
            {"role": "user", "content": user_prompt},
        ]

    def _call_verifier_sync(self) -> Dict[str, Any]:
        base_url = (self.base_url or "").rstrip("/") + "/"
        if not base_url:
            raise ValueError("base_url is required.")
        if not self.model_name:
            raise ValueError("model_name is required.")
        if self.generated_answer is None:
            raise ValueError("generated_answer is required.")
        if self.rag_context is None:
            raise ValueError("rag_context is required.")
        if self.analyzer_data is None:
            raise ValueError("analyzer_data is required.")

        analyzer = self._parse_analyzer()
        if not analyzer:
            raise ValueError("analyzer_data could not be parsed.")

        messages = self._build_messages(analyzer)

        options: Dict[str, Any] = {}

        temp = self._none_if_empty(self.temperature)
        if temp is not None:
            options["temperature"] = float(temp)

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

        if isinstance(self.extra_options, dict) and self.extra_options:
            options.update(self.extra_options)

        body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": options,
            "think": bool(self.enable_thinking),
            "format": "json",
        }

        if bool(self.keep_alive_enabled):
            ka = (self.keep_alive or "").strip()
            if ka:
                body["keep_alive"] = ka

        url = urljoin(base_url, "api/chat")
        timeout_s = self._as_int(self.timeout_s) or 120

        t0 = time.perf_counter()
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            raw = resp.json()
        wall_s = time.perf_counter() - t0

        msg = raw.get("message") or {}
        llm_text = msg.get("content") or ""
        thinking = msg.get("thinking") or ""

        verifier_json = self._parse_json_obj(llm_text)

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
            "observed_has_thinking_field": bool(thinking),
            "verifier_json_parse_ok": bool(verifier_json),
        }

        env: Dict[str, Any] = {
            "text": llm_text,
            "verifier_json": verifier_json,
            "claims": verifier_json.get("claims", []) if verifier_json else [],
            "slot_coverage": verifier_json.get("slot_coverage", {}) if verifier_json else {},
            "metrics": metrics,
            "raw_llm_output": llm_text,
            "analyzer_used": analyzer,
            "raw": raw,
            "parse_error": not bool(verifier_json),
        }

        if bool(self.include_thinking_in_envelope):
            env["thinking"] = thinking

        return env

    def build_verification(self) -> Data:
        env = self._call_verifier_sync()
        try:
            logger.debug(f"Ollama verifier metrics: {env.get('metrics')}")
        except Exception:
            pass

        return Data(
            text_key="text",
            data=env,
            default_value="",
        )