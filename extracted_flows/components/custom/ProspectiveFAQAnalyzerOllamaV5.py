# Recovered Langflow component
# type: ProspectiveFAQAnalyzerOllamaV5
# class: ProspectiveFAQAnalyzerOllamaV5
# used in 1 flow(s): Cognitive RAG V0.5.2 backup
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
    FloatInput,
    IntInput,
    MessageTextInput,
    Output,
    SliderInput,
)
from langflow.field_typing.range_spec import RangeSpec
from langflow.schema.data import Data

HTTP_STATUS_OK = 200


class ProspectiveFAQAnalyzerOllamaV5(Component):
    """
    Analyzer (1 Ollama call, minimal output):
    - Select request_type
    - For that request_type, compute missing_general/missing_critical (LLM)
    - ALSO return resolved slot values + their source (user/history) for expected slots only (LLM)
    - Code computes slot coverage + triggers refinement deterministically by building refined_query
      when a critical (or relevant) slot is supplied from history and the user question is ambiguous.
    """

    display_name = "Analyzer (Prospective FAQ, minimal + deterministic refine) [Ollama] V5"
    description = "Minimal analyzer: request_type-dependent expected slots, missing slots, weighted coverage, and deterministic refinement using history-resolved slot values."
    icon = "search"
    name = "ProspectiveFAQAnalyzerOllamaV5"

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

        MessageTextInput(name="user_input", display_name="User Input", required=True),
        MessageTextInput(name="history_text", display_name="History (raw text)", advanced=True),
        IntInput(name="max_history_chars", display_name="Max history chars sent", value=6000, advanced=True),

        # request_type -> expected slots
        MessageTextInput(
            name="slot_schema_json",
            display_name="Request-type slot schema JSON",
            info="JSON dict: {request_type:{general:[...],critical:[...]}, ...}",
            value=json.dumps(
                {
                    "university_overview": {"general": [], "critical": []},
                    "general_info": {"general": [], "critical": []},

                    "program_info": {"general": ["program"], "critical": ["program"]},
                    "admission_requirements": {"general": ["program", "degree_level"], "critical": ["program", "degree_level"]},

                    # If you want program required for fees, keep it critical.
                    "fees": {"general": ["program", "degree_level", "fee_type"], "critical": ["program", "fee_type"]},

                    "deadlines": {"general": ["program", "degree_level", "intake_year_or_term"], "critical": ["intake_year_or_term"]},
                    "scholarship": {"general": ["scholarship_type", "degree_level"], "critical": ["scholarship_type"]},

                    "dorm_overview": {"general": [], "critical": []},
                    "dorm_fees": {"general": ["accommodation_type", "fee_type"], "critical": ["accommodation_type"]},

                    "unknown": {"general": [], "critical": []},
                },
                ensure_ascii=False,
                indent=2,
            ),
        ),

        FloatInput(name="weight_general", display_name="Weight general", value=1.0, advanced=True),
        FloatInput(name="weight_critical", display_name="Weight critical", value=2.0, advanced=True),

        BoolInput(name="enable_smalltalk_bypass", display_name="Smalltalk bypass", value=True, advanced=True),

        MessageTextInput(
            name="system",
            display_name="System (expandable)",
            value="You are a careful analyzer. Return ONLY valid JSON. No code fences. No extra text.",
            advanced=True,
        ),
        MessageTextInput(
            name="analyzer_template",
            display_name="Analyzer Template (expandable)",
            value=(
                "Return ONLY valid JSON.\n\n"
                "You are the ANALYZER for a bounded RAG FAQ system (prospective student).\n"
                "You output signals ONLY. Do NOT decide actions (no need_clarify, no abstain).\n\n"
                "USER: {{USER}}\n"
                "HISTORY: {{HISTORY}}\n\n"
                "REQUEST_TYPES (choose exactly one):\n{{REQUEST_TYPES}}\n\n"
                "SLOT_SCHEMA (reference):\n{{SLOT_SCHEMA}}\n\n"
                "TASKS\n"
                "1) intent_type: chitchat|faq|other\n"
                "2) Choose request_type from REQUEST_TYPES.\n"
                "3) For the chosen request_type:\n"
                "   - missing_general: slots in schema[request_type].general NOT explicitly present in USER+HISTORY\n"
                "   - missing_critical: slots in schema[request_type].critical NOT explicitly present in USER+HISTORY\n"
                "4) resolved_slots: For slots in schema[request_type].general ∪ critical that ARE present, extract:\n"
                "   - value: short string\n"
                "   - source: user|history\n"
                "   Only include slots you are confident are explicitly stated.\n"
                "5) clarification_tracking:\n"
                "   - clarification_asked_in_history: true if assistant previously asked for missing info\n"
                "   - user_is_answering_clarification: true if USER looks like an answer to that\n\n"
                "OUTPUT JSON\n"
                "{\n"
                "  \"intent_type\": \"faq\",\n"
                "  \"request_type\": \"unknown\",\n"
                "  \"topic\": \"\",\n"
                "  \"missing_general\": [],\n"
                "  \"missing_critical\": [],\n"
                "  \"resolved_slots\": {\n"
                "     \"intake_year_or_term\": {\"value\":\"Aug 2026\", \"source\":\"history\"}\n"
                "  },\n"
                "  \"clarification_tracking\": {\n"
                "    \"clarification_asked_in_history\": false,\n"
                "    \"user_is_answering_clarification\": false\n"
                "  }\n"
                "}\n\n"
                "RULES\n"
                "- Do NOT invent values.\n"
                "- Be conservative: if unclear, treat as missing / omit from resolved_slots.\n"
                "- missing_* must contain ONLY slots from the schema for that request_type.\n"
            ),
            advanced=True,
        ),

        BoolInput(name="enable_thinking", display_name="Enable Thinking (Ollama think)", value=False, advanced=True),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        IntInput(name="timeout_s", display_name="Timeout (seconds)", value=120, advanced=True),
        DictInput(name="extra_options", display_name="Extra options (dict)", value={}, advanced=True),
    ]

    outputs = [Output(display_name="Analyzer (Data)", name="analyzer", method="run")]

    _re_smalltalk = re.compile(
        r"^\s*(hi|hello|hey|thanks|thank you|thx|ok|okay|good morning|good afternoon|good evening)\b",
        re.IGNORECASE,
    )
    _re_json_obj = re.compile(r"\{.*\}", re.DOTALL)

    # ---------- Dropdown refresh ----------
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

    # ---------- Helpers ----------
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

    @staticmethod
    def _safe_list(x: Any) -> List[str]:
        if not isinstance(x, list):
            return []
        out = []
        for it in x:
            s = str(it).strip()
            if s:
                out.append(s)
        return out

    @staticmethod
    def _coverage_from_missing(expected: List[str], missing: List[str]) -> float:
        if not expected:
            return 1.0
        mset = set(missing)
        miss = sum(1 for s in expected if s in mset)
        return max(0.0, min(1.0, 1.0 - (miss / len(expected))))

    @staticmethod
    def _weighted_coverage(eg: List[str], ec: List[str], mg: List[str], mc: List[str], wg: float, wc: float) -> Dict[str, Any]:
        eg = list(dict.fromkeys(eg))
        ec = list(dict.fromkeys(ec))

        cov_g = ProspectiveFAQAnalyzerOllamaV5._coverage_from_missing(eg, mg)
        cov_c = ProspectiveFAQAnalyzerOllamaV5._coverage_from_missing(ec, mc)

        g_total = len(eg)
        c_total = len(ec)
        g_missing = sum(1 for s in eg if s in set(mg))
        c_missing = sum(1 for s in ec if s in set(mc))
        g_present = g_total - g_missing
        c_present = c_total - c_missing

        denom = wg * g_total + wc * c_total
        num = wg * g_present + wc * c_present
        cov_w = (num / denom) if denom > 0 else 1.0

        return {
            "slot_coverage_general": round(float(cov_g), 4),
            "slot_coverage_critical": round(float(cov_c), 4),
            "slot_coverage_weighted": round(float(cov_w), 4),
            "counts": {
                "general_total": g_total,
                "critical_total": c_total,
                "general_missing": g_missing,
                "critical_missing": c_missing,
            },
        }

    @staticmethod
    def _build_refined_query(request_type: str, user: str, resolved: Dict[str, Dict[str, str]]) -> str:
        """
        Deterministic refinement (no extra LLM).
        Uses only resolved slot values to make query more specific.
        """
        # pull values
        def val(k: str) -> str:
            r = resolved.get(k, {})
            return str(r.get("value", "")).strip()

        program = val("program")
        degree = val("degree_level")
        intake = val("intake_year_or_term")
        fee_type = val("fee_type")
        schol = val("scholarship_type")
        accom = val("accommodation_type")

        parts: List[str] = ["AIT"]

        if request_type in ("fees",):
            parts.append("fees")
            if fee_type:
                parts.append(fee_type)
            if program:
                parts.append(program)
            if degree:
                parts.append(degree)

        elif request_type in ("deadlines",):
            parts.append("application deadline")
            if intake:
                parts.append(intake)
            if program:
                parts.append(program)
            if degree:
                parts.append(degree)

        elif request_type in ("scholarship",):
            parts.append("scholarship")
            if schol:
                parts.append(schol)
            if degree:
                parts.append(degree)
            if program:
                parts.append(program)

        elif request_type in ("program_info", "admission_requirements"):
            if program:
                parts.append(program)
            if degree:
                parts.append(degree)
            parts.append("information" if request_type == "program_info" else "admission requirements")

        elif request_type in ("dorm_fees",):
            parts.append("dormitory cost")
            if accom:
                parts.append(accom)
            if fee_type:
                parts.append(fee_type)

        else:
            # fallback: just append any resolved values
            for k in ("program", "degree_level", "intake_year_or_term", "fee_type", "scholarship_type"):
                v = val(k)
                if v:
                    parts.append(v)

        refined = " ".join([p for p in parts if p]).strip()
        return refined if refined else user

    # ---------- Ollama call ----------
    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        base = (self.base_url or "").rstrip("/") + "/"
        if not base:
            raise ValueError("base_url is required")
        if not self.model_name:
            raise ValueError("model_name is required")

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
            "options": {"temperature": float(self.temperature or 0.1), **(self.extra_options or {})},
            "think": bool(self.enable_thinking),
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
        thinking = msg.get("thinking") or ""

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
            "observed_has_thinking_field": bool(thinking),
        }
        return {"text": text, "metrics": metrics}

    def run(self) -> Data:
        user = (self.user_input or "").strip()
        history = (self.history_text or "").strip()

        # smalltalk bypass
        if bool(self.enable_smalltalk_bypass) and self._re_smalltalk.match(user) and len(user) <= 80:
            payload = {
                "original_question": user,
                "intent_type": "chitchat",
                "request_type": "general_info",
                "topic": "",
                "expected_general_slots": [],
                "expected_critical_slots": [],
                "missing_slots": {"general": [], "critical": []},
                "slot_coverage_general": 1.0,
                "slot_coverage_critical": 1.0,
                "slot_coverage_weighted": 1.0,
                "counts": {"general_total": 0, "critical_total": 0, "general_missing": 0, "critical_missing": 0},
                "resolved_slots": {},
                "refinement": {"used": False, "refined_query": "", "used_slots": []},
                "clarification_tracking": {"clarification_asked_in_history": False, "user_is_answering_clarification": False},
                "llm_metrics": {},
            }
            self.status = "chitchat_bypass"
            return Data(text_key="analyzer", data=payload, default_value="")

        # parse schema
        try:
            schema = json.loads(self.slot_schema_json or "{}")
        except Exception:
            schema = {}
        if not isinstance(schema, dict):
            schema = {}
        if "unknown" not in schema:
            schema["unknown"] = {"general": [], "critical": []}

        request_types = sorted(list(schema.keys()))

        max_hist = int(self.max_history_chars or 6000)
        hist_send = history if (max_hist <= 0 or len(history) <= max_hist) else (history[:max_hist] + "\n[TRUNCATED]")

        prompt = (self.analyzer_template or "")
        prompt = prompt.replace("{{USER}}", user)
        prompt = prompt.replace("{{HISTORY}}", hist_send)
        prompt = prompt.replace("{{REQUEST_TYPES}}", json.dumps(request_types, ensure_ascii=False))
        prompt = prompt.replace("{{SLOT_SCHEMA}}", json.dumps(schema, ensure_ascii=False))

        env = self._call_ollama(prompt)
        obj = self._extract_json(env.get("text", "")) or {}
        metrics = env.get("metrics") or {}

        intent_type = str(obj.get("intent_type") or "faq").strip().lower()
        request_type = str(obj.get("request_type") or "unknown").strip()
        if request_type not in schema:
            request_type = "unknown"
        topic = str(obj.get("topic") or "").strip()

        expected_general = self._safe_list(schema.get(request_type, {}).get("general", []))
        expected_critical = self._safe_list(schema.get(request_type, {}).get("critical", []))

        missing_general = self._safe_list(obj.get("missing_general", []))
        missing_critical = self._safe_list(obj.get("missing_critical", []))

        # filter missing to schema
        eg_set = set(expected_general)
        ec_set = set(expected_critical)
        missing_general = [s for s in missing_general if s in eg_set]
        missing_critical = [s for s in missing_critical if s in ec_set]

        # resolved_slots: only expected slots; each entry {value, source}
        resolved_slots_raw = obj.get("resolved_slots", {})
        resolved_slots: Dict[str, Dict[str, str]] = {}
        if isinstance(resolved_slots_raw, dict):
            for k, v in resolved_slots_raw.items():
                kk = str(k).strip()
                if kk not in set(expected_general + expected_critical):
                    continue
                if not isinstance(v, dict):
                    continue
                val = str(v.get("value", "")).strip()
                src = str(v.get("source", "")).strip().lower()
                if not val or src not in ("user", "history"):
                    continue
                resolved_slots[kk] = {"value": val, "source": src}

        # clarification tracking
        ct = obj.get("clarification_tracking") if isinstance(obj.get("clarification_tracking"), dict) else {}
        clarification_tracking = {
            "clarification_asked_in_history": bool(ct.get("clarification_asked_in_history", False)),
            "user_is_answering_clarification": bool(ct.get("user_is_answering_clarification", False)),
        }

        wg = float(self.weight_general or 1.0)
        wc = float(self.weight_critical or 2.0)
        cov = self._weighted_coverage(expected_general, expected_critical, missing_general, missing_critical, wg, wc)

        # ---------- Deterministic refinement trigger ----------
        # If the query is short/ambiguous (e.g., "When is the deadline?") and
        # a critical slot value is resolved from HISTORY, then refine.
        history_resolved_critical = [
            s for s in expected_critical
            if s in resolved_slots and resolved_slots[s].get("source") == "history"
        ]

        ambiguous = (len(user) <= 40) or user.lower().strip() in ("when is the deadline?", "what is the deadline?", "how much is the fee?", "how much are the fees?")
        refine_used = bool(history_resolved_critical and ambiguous)

        refined_query = self._build_refined_query(request_type, user, resolved_slots) if refine_used else ""
        used_slots = history_resolved_critical if refine_used else []

        payload = {
            "original_question": user,
            "history_window": hist_send,

            "intent_type": intent_type,
            "request_type": request_type,
            "topic": topic,

            "expected_general_slots": expected_general,
            "expected_critical_slots": expected_critical,
            "missing_slots": {"general": missing_general, "critical": missing_critical},

            **cov,

            "resolved_slots": resolved_slots,  # minimal, only for expected slots present (needed for refinement)
            "refinement": {"used": refine_used, "refined_query": refined_query, "used_slots": used_slots},

            "clarification_tracking": clarification_tracking,
            "llm_metrics": metrics,
        }

        self.status = (
            f"{request_type} | covW={payload['slot_coverage_weighted']:.2f} "
            f"| missC={len(missing_critical)} missG={len(missing_general)} "
            f"| refine={int(refine_used)} | in={metrics.get('prompt_eval_count')} out={metrics.get('eval_count')}"
        )
        return Data(text_key="analyzer", data=payload, default_value="")