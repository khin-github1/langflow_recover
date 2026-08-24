# Recovered Langflow component
# type: ProspectiveFAQAnalyzerOllamaV7
# class: ProspectiveFAQAnalyzerOllamaV7
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


class ProspectiveFAQAnalyzerOllamaV7(Component):
    """
    Analyzer (1 Ollama call, minimal output) with refinement toggle for ablation:
    - enable_refinement=True:
        LLM prompt includes refined_question (optional), and Python may trigger refinement deterministically.
    - enable_refinement=False:
        LLM prompt contains ZERO refinement instructions (no refined_question field),
        and Python forces refinement.used=false always.

    Added:
    - clarification_threshold input
    - clarification_needed final boolean computed after weighted coverage
    - suppression of clarification_needed when the user is answering a clarification already asked in history
    """

    display_name = "Analyzer (Prospective FAQ) [Ollama] V7 (Refinement Toggle)"
    description = "Request-type schema + missing/resolved slots + weighted coverage + optional deterministic refinement (toggleable for ablation)."
    icon = "search"
    name = "ProspectiveFAQAnalyzerOllamaV7"

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

        # Ablation toggle
        BoolInput(
            name="enable_refinement",
            display_name="Enable Refinement (Ablation Toggle)",
            value=True,
            advanced=False,
        ),

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

        # NEW: clarification threshold
        FloatInput(
            name="clarification_threshold",
            display_name="Clarification Threshold",
            value=0.75,
            advanced=False,
            info="If weighted slot coverage is below this threshold, clarification_needed becomes true unless the user is already answering a prior clarification.",
        ),

        BoolInput(name="enable_smalltalk_bypass", display_name="Smalltalk bypass", value=True, advanced=True),

        MessageTextInput(
            name="system",
            display_name="System (expandable)",
            value="You are a careful analyzer. Return ONLY valid JSON. No code fences. No extra text.",
            advanced=True,
        ),

        # Template WITHOUT refinement
        MessageTextInput(
            name="template_no_refine",
            display_name="Analyzer Template (NO refinement) (expandable)",
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
                "3) For chosen request_type:\n"
                "   - missing_general: schema[request_type].general NOT explicitly present in USER+HISTORY\n"
                "   - missing_critical: schema[request_type].critical NOT explicitly present in USER+HISTORY\n"
                "4) resolved_slots: for slots in schema[request_type].general ∪ critical that ARE present, extract:\n"
                "   - value (short string)\n"
                "   - source: user|history\n"
                "   Only include slots you are confident are explicitly stated.\n"
                "5) clarification_tracking:\n"
                "   - clarification_asked_in_history (bool)\n"
                "   - user_is_answering_clarification (bool)\n\n"
                "OUTPUT JSON\n"
                "{\n"
                "  \"intent_type\": \"faq\",\n"
                "  \"request_type\": \"unknown\",\n"
                "  \"topic\": \"\",\n"
                "  \"missing_general\": [],\n"
                "  \"missing_critical\": [],\n"
                "  \"resolved_slots\": {\n"
                "    \"intake_year_or_term\": {\"value\":\"Aug 2026\", \"source\":\"history\"}\n"
                "  },\n"
                "  \"clarification_tracking\": {\n"
                "    \"clarification_asked_in_history\": false,\n"
                "    \"user_is_answering_clarification\": false\n"
                "  }\n"
                "}\n\n"
                "RULES\n"
                "- Do NOT invent values.\n"
                "- Be conservative: if unclear, treat as missing / omit from resolved_slots.\n"
                "- missing_* must contain ONLY slots from schema for that request_type.\n"
            ),
            advanced=True,
        ),

        # Template WITH refinement
        MessageTextInput(
            name="template_with_refine",
            display_name="Analyzer Template (WITH refinement) (expandable)",
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
                "3) For chosen request_type:\n"
                "   - missing_general: schema[request_type].general NOT explicitly present in USER+HISTORY\n"
                "   - missing_critical: schema[request_type].critical NOT explicitly present in USER+HISTORY\n"
                "4) resolved_slots: for slots in schema[request_type].general ∪ critical that ARE present, extract:\n"
                "   - value (short string)\n"
                "   - source: user|history\n"
                "   Only include slots you are confident are explicitly stated.\n"
                "5) refined_question (OPTIONAL):\n"
                "   - If USER is a generic question and HISTORY provides a resolved CRITICAL slot value,\n"
                "     rewrite into ONE clear natural-language QUESTION using those values.\n"
                "   - refined_question MUST be one sentence and MUST end with '?'.\n"
                "   - Otherwise refined_question must be \"\".\n"
                "6) clarification_tracking:\n"
                "   - clarification_asked_in_history (bool)\n"
                "   - user_is_answering_clarification (bool)\n\n"
                "OUTPUT JSON\n"
                "{\n"
                "  \"intent_type\": \"faq\",\n"
                "  \"request_type\": \"unknown\",\n"
                "  \"topic\": \"\",\n"
                "  \"missing_general\": [],\n"
                "  \"missing_critical\": [],\n"
                "  \"resolved_slots\": {\n"
                "    \"intake_year_or_term\": {\"value\":\"Aug 2026\", \"source\":\"history\"}\n"
                "  },\n"
                "  \"refined_question\": \"\",\n"
                "  \"clarification_tracking\": {\n"
                "    \"clarification_asked_in_history\": false,\n"
                "    \"user_is_answering_clarification\": false\n"
                "  }\n"
                "}\n\n"
                "RULES\n"
                "- Do NOT invent values.\n"
                "- Be conservative: if unclear, treat as missing / omit from resolved_slots.\n"
                "- missing_* must contain ONLY slots from schema for that request_type.\n"
                "- refined_question must be \"\" or a single sentence ending with '?'.\n"
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

    # Generic/ambiguous question patterns by request type
    _re_deadline_generic = re.compile(r"^\s*(when\s+is\s+the\s+deadline\??|deadline\??)\s*$", re.I)
    _re_fee_generic = re.compile(r"^\s*(how\s+much\s+is\s+the\s+fee\??|fees\??|tuition\??|cost\??)\s*$", re.I)
    _re_scholar_generic = re.compile(r"^\s*(what\s+are\s+the\s+requirements\??|scholarship\??)\s*$", re.I)

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
    def _weighted_coverage(
        eg: List[str],
        ec: List[str],
        mg: List[str],
        mc: List[str],
        wg: float,
        wc: float,
    ) -> Dict[str, Any]:
        eg = list(dict.fromkeys(eg))
        ec = list(dict.fromkeys(ec))

        cov_g = ProspectiveFAQAnalyzerOllamaV7._coverage_from_missing(eg, mg)
        cov_c = ProspectiveFAQAnalyzerOllamaV7._coverage_from_missing(ec, mc)

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
    def _ensure_question(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        if "\n" in s:
            s = s.split("\n")[0].strip()
        if not s.endswith("?"):
            s = s.rstrip(".") + "?"
        if s.count("?") > 1:
            s = s.split("?")[0].strip() + "?"
        return s

    def _fallback_refined_question(self, request_type: str, resolved: Dict[str, Dict[str, str]]) -> str:
        def val(k: str) -> str:
            r = resolved.get(k, {})
            return str(r.get("value", "")).strip()

        program = val("program")
        degree = val("degree_level")
        intake = val("intake_year_or_term")
        fee_type = val("fee_type")
        schol = val("scholarship_type")
        accom = val("accommodation_type")

        if request_type == "deadlines":
            if intake:
                if program and degree:
                    return f"When is the AIT application deadline for {degree} {program} ({intake} intake)?"
                if program:
                    return f"When is the AIT application deadline for {program} ({intake} intake)?"
                if degree:
                    return f"When is the AIT application deadline for {degree} programs ({intake} intake)?"
                return f"When is the AIT application deadline for the {intake} intake?"
            return "When is the AIT application deadline?"

        if request_type == "fees":
            ft = fee_type or "tuition fee"
            if program and degree:
                return f"What is the {ft} for {degree} admission in {program} at AIT?"
            if degree:
                return f"What is the {ft} for {degree} admission at AIT?"
            if program:
                return f"What is the {ft} for {program} at AIT?"
            return f"What is the {ft} at AIT?"

        if request_type == "scholarship":
            if schol:
                if degree:
                    return f"What are the requirements for the {schol} scholarship for {degree} programs at AIT?"
                return f"What are the requirements for the {schol} scholarship at AIT?"
            return "What are the scholarship requirements at AIT?"

        if request_type == "dorm_fees":
            if accom:
                return f"What is the monthly cost for {accom} accommodation at AIT dormitory?"
            return "What is the dormitory accommodation cost at AIT?"

        return ""

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

        if bool(self.enable_smalltalk_bypass) and self._re_smalltalk.match(user) and len(user) <= 80:
            payload = {
                "original_question": user,
                "history_window": "",
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
                "refinement": {"enabled": bool(self.enable_refinement), "used": False, "refined_query": "", "used_slots": []},
                "clarification_tracking": {"clarification_asked_in_history": False, "user_is_answering_clarification": False},
                "clarification_decision": {
                    "threshold": float(self.clarification_threshold or 0.75),
                    "coverage_weighted": 1.0,
                    "has_missing_slots": False,
                    "suppressed_because_answering_prior_clarification": False,
                    "clarification_needed": False,
                },
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

        # IMPORTANT: choose prompt template based on enable_refinement
        template = self.template_with_refine if bool(self.enable_refinement) else self.template_no_refine
        prompt = (template or "")
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

        # filter missing lists to schema
        eg_set = set(expected_general)
        ec_set = set(expected_critical)
        missing_general = [s for s in missing_general if s in eg_set]
        missing_critical = [s for s in missing_critical if s in ec_set]

        # resolved_slots
        resolved_slots_raw = obj.get("resolved_slots", {})
        resolved_slots: Dict[str, Dict[str, str]] = {}
        if isinstance(resolved_slots_raw, dict):
            allowed = set(expected_general + expected_critical)
            for k, v in resolved_slots_raw.items():
                kk = str(k).strip()
                if kk not in allowed:
                    continue
                if not isinstance(v, dict):
                    continue
                val = str(v.get("value", "")).strip()
                src = str(v.get("source", "")).strip().lower()
                if val and src in ("user", "history"):
                    resolved_slots[kk] = {"value": val, "source": src}

        # coverage
        wg = float(self.weight_general or 1.0)
        wc = float(self.weight_critical or 2.0)
        cov = self._weighted_coverage(expected_general, expected_critical, missing_general, missing_critical, wg, wc)

        # clarification tracking
        ct = obj.get("clarification_tracking") if isinstance(obj.get("clarification_tracking"), dict) else {}
        clarification_tracking = {
            "clarification_asked_in_history": bool(ct.get("clarification_asked_in_history", False)),
            "user_is_answering_clarification": bool(ct.get("user_is_answering_clarification", False)),
        }

        # ---------- FINAL clarification decision ----------
        clarification_threshold = float(self.clarification_threshold or 0.75)
        coverage_weighted = float(cov.get("slot_coverage_weighted", 1.0))
        has_missing_slots = bool(missing_general or missing_critical)

        # Base rule:
        # Ask for clarification only when coverage is below threshold AND something is still missing.
        clarification_needed = bool((coverage_weighted < clarification_threshold) and has_missing_slots)

        # Suppress clarification if history already asked one and current user turn is answering it.
        suppressed_because_answering_prior_clarification = bool(
            clarification_tracking["clarification_asked_in_history"]
            and clarification_tracking["user_is_answering_clarification"]
        )
        if suppressed_because_answering_prior_clarification:
            clarification_needed = False

        clarification_decision = {
            "threshold": round(clarification_threshold, 4),
            "coverage_weighted": round(coverage_weighted, 4),
            "has_missing_slots": has_missing_slots,
            "suppressed_because_answering_prior_clarification": suppressed_because_answering_prior_clarification,
            "clarification_needed": clarification_needed,
        }

        # ---------- Refinement logic (toggleable) ----------
        refine_enabled = bool(self.enable_refinement)
        refine_used = False
        refined_query = ""
        used_slots: List[str] = []

        if refine_enabled:
            history_resolved_critical = [
                s for s in expected_critical
                if s in resolved_slots and resolved_slots[s].get("source") == "history"
            ]

            u = user.strip()
            if request_type == "deadlines":
                ambiguous = bool(self._re_deadline_generic.match(u)) or len(u) <= 40
            elif request_type == "fees":
                ambiguous = bool(self._re_fee_generic.match(u)) or len(u) <= 45
            elif request_type == "scholarship":
                ambiguous = bool(self._re_scholar_generic.match(u)) or len(u) <= 55
            else:
                ambiguous = len(u) <= 40

            refine_used = bool(history_resolved_critical and ambiguous)
            if refine_used:
                llm_refined_q = self._ensure_question(str(obj.get("refined_question", "") or ""))
                fallback_q = self._ensure_question(self._fallback_refined_question(request_type, resolved_slots))
                refined_query = llm_refined_q or fallback_q
                used_slots = history_resolved_critical

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

            "resolved_slots": resolved_slots,

            # Always output refinement object for pipeline stability
            "refinement": {
                "enabled": refine_enabled,
                "used": refine_used,
                "refined_query": refined_query,
                "used_slots": used_slots,
            },

            "clarification_tracking": clarification_tracking,
            "clarification_decision": clarification_decision,
            "clarification_needed": clarification_needed,

            "llm_metrics": metrics,
        }

        self.status = (
            f"{request_type} | covW={payload['slot_coverage_weighted']:.2f} "
            f"| clar={int(clarification_needed)}@{clarification_threshold:.2f} "
            f"| missC={len(missing_critical)} missG={len(missing_general)} "
            f"| refine_en={int(refine_enabled)} refine_used={int(refine_used)} "
            f"| in={metrics.get('prompt_eval_count')} out={metrics.get('eval_count')}"
        )
        return Data(text_key="analyzer", data=payload, default_value="")