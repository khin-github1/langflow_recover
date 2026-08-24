# Recovered Langflow component
# type: SlotGapRefinementExtractorOllamaV2
# class: SlotGapRefinementExtractorOllamaV2
# used in 1 flow(s): Cognitive RAG V1.1.5 Backup
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


class SlotGapRefinementExtractorOllamaV2(Component):
    display_name = "Slot Gap + Refinement Extractor [Ollama] V2"
    description = (
        "Infers expected answer slots from USER, extracts USER/HISTORY prerequisite slots, "
        "computes missing slots from USER only, and proposes refined query if HISTORY fills the gap."
    )
    icon = "search"
    name = "SlotGapRefinementExtractorOllamaV2"

    JSON_MODELS_KEY = "models"
    JSON_NAME_KEY = "name"
    JSON_CAPABILITIES_KEY = "capabilities"
    DESIRED_CAPABILITY = "completion"
    TOOL_CALLING_CAPABILITY = "tools"

    # IMPORTANT:
    # This map is intentionally SOFTER than the old one.
    # tuition_fees does NOT require fee_type because "tuition" is already encoded by the expected slot.
    # scholarship_opportunities does NOT require scholarship_type because the user can ask broadly.
    # deadline does NOT require intake by default; only if explicitly present can it refine.
    DEFAULT_EXPECTED_TO_REQUIRED_MAP = {
        "contact_information": {"general": [], "critical": []},
        "university_overview": {"general": [], "critical": []},
        "program_info": {"general": ["program"], "critical": []},
        "admission_requirements": {"general": ["program"], "critical": []},
        "eligibility": {"general": ["program"], "critical": []},
        "required_documents": {"general": ["program"], "critical": []},
        "application_process": {"general": ["program"], "critical": []},
        "deadline": {"general": ["program", "degree_level"], "critical": []},
        "fees": {"general": ["program"], "critical": []},
        "tuition_fees": {"general": ["program"], "critical": []},
        "scholarship": {"general": ["program", "degree_level"], "critical": []},
        "scholarship_opportunities": {"general": ["program", "degree_level"], "critical": []},
        "dorm_overview": {"general": [], "critical": []},
        "dorm_fees": {"general": ["accommodation_type"], "critical": []},
        "location": {"general": [], "critical": []},
        "hours": {"general": [], "critical": []},
        "other": {"general": [], "critical": []},
    }

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
            advanced=True,
        ),
        IntInput(
            name="max_history_chars",
            display_name="Max history chars sent",
            value=6000,
            advanced=True,
        ),
        MessageTextInput(
            name="expected_to_required_map_json",
            display_name="Expected→Required Slot Map JSON",
            value=json.dumps(DEFAULT_EXPECTED_TO_REQUIRED_MAP, ensure_ascii=False, indent=2),
            advanced=True,
        ),
        BoolInput(
            name="enable_refinement",
            display_name="Enable Refinement",
            value=True,
            advanced=False,
        ),
        MessageTextInput(
            name="system",
            display_name="System",
            advanced=True,
            value=(
                "You are a SLOT GAP AND REFINEMENT EXTRACTOR.\n"
                "Tasks:\n"
                "1) infer expected answer slots from USER,\n"
                "2) extract prerequisite slot values explicitly stated in USER,\n"
                "3) extract slot values from HISTORY that can fill gaps missing from USER,\n"
                "4) propose one refined query if HISTORY supplies missing useful details.\n"
                "Return ONLY valid JSON."
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
                "EXPECTED_TO_REQUIRED_MAP:\n{{EXPECTED_TO_REQUIRED_MAP}}\n\n"
                "ANSWER_SLOT_CATALOG:\n{{ANSWER_SLOT_CATALOG}}\n\n"
                "GLOBAL_QUERY_SLOT_CATALOG:\n{{GLOBAL_QUERY_SLOT_CATALOG}}\n\n"
                "TASKS:\n"
                "1) expected_general_slots / expected_critical_slots:\n"
                "   - infer from USER what information the answer should cover.\n"
                "2) user_resolved_slots:\n"
                "   - extract only slot values explicitly stated in USER.\n"
                "3) history_resolved_slots:\n"
                "   - extract only slot values from HISTORY that help fill USER-missing info.\n"
                "4) refined_query:\n"
                "   - if USER is generic/underspecified and HISTORY supplies useful missing details,\n"
                "     rewrite USER into one clearer natural-language question.\n"
                "   - otherwise return empty string.\n\n"
                "RULES:\n"
                "- Do NOT invent values.\n"
                "- Be conservative.\n"
                "- Use only slot names from the provided catalogs.\n"
                "- refined_query must be either empty string or one sentence ending with '?'.\n\n"
                "OUTPUT JSON:\n"
                "{\n"
                '  "expected_general_slots": [],\n'
                '  "expected_critical_slots": [],\n'
                '  "user_resolved_slots": {},\n'
                '  "history_resolved_slots": {},\n'
                '  "refined_query": ""\n'
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
        Output(display_name="Slot Gap Data", name="slot_gap_data", method="build_output"),
    ]

    _re_json_obj = re.compile(r"\{.*\}", re.DOTALL)

    _re_program_1 = re.compile(
        r"\b(master(?:'s)?|phd|bachelor(?:'s)?)\s+in\s+([A-Za-z0-9&/\- ,()]+?)(?:\s+at\s+AIT|\?|$)",
        re.I,
    )
    _re_program_2 = re.compile(
        r"\bprogram\s+in\s+([A-Za-z0-9&/\- ,()]+?)(?:\s+at\s+AIT|\?|$)",
        re.I,
    )
    _re_degree = re.compile(
        r"\b(master(?:'s)?|msc|m\.?sc\.?|phd|doctor(?:al)?|bachelor(?:'s)?|undergraduate)\b",
        re.I,
    )
    _re_intake = re.compile(
        r"\b((?:aug(?:ust)?|jan(?:uary)?|sep(?:tember)?|spring|fall|autumn|summer)\s+\d{4}|\d{4}\s+intake)\b",
        re.I,
    )
    _re_fee_type = re.compile(
        r"\b(tuition|registration|application|living|dorm|accommodation)\b",
        re.I,
    )
    _re_accommodation = re.compile(
        r"\b(single room|shared room|double room|twin room|dormitory|dorm)\b",
        re.I,
    )
    _re_scholarship = re.compile(
        r"\b([A-Za-z0-9&()'’\- ,]+?Scholarship(?: Fund)?)\b",
        re.I,
    )

    _re_deadline_generic = re.compile(r"^\s*(when\s+is\s+the\s+deadline\??|deadline\??)\s*$", re.I)
    _re_fee_generic = re.compile(r"^\s*(how\s+much\s+is\s+the\s+fee\??|fees\??|tuition\??|cost\??)\s*$", re.I)
    _re_scholar_generic = re.compile(r"^\s*(scholarship\??|what\s+are\s+the\s+scholarships\??)\s*$", re.I)

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

    @staticmethod
    def _safe_list(x: Any) -> List[str]:
        if not isinstance(x, list):
            return []
        out: List[str] = []
        for it in x:
            s = str(it).strip()
            if s:
                out.append(s)
        return out

    @staticmethod
    def _normalize_answer_slots(slots: List[str]) -> List[str]:
        mapping = {
            "contact_info": "contact_information",
            "contacts": "contact_information",
            "contact": "contact_information",
            "contact_information": "contact_information",
            "admin_contact": "contact_information",

            "deadline": "deadline",
            "deadlines": "deadline",
            "application_deadline": "deadline",

            "fee": "fees",
            "fees": "fees",
            "tuition": "tuition_fees",
            "tuition_fee": "tuition_fees",
            "tuition_fees": "tuition_fees",
            "cost": "fees",

            "scholarship": "scholarship",
            "scholarships": "scholarship_opportunities",
            "scholarship_opportunities": "scholarship_opportunities",

            "requirements": "admission_requirements",
            "admission_requirement": "admission_requirements",
            "admission_requirements": "admission_requirements",

            "eligibility": "eligibility",

            "documents": "required_documents",
            "required_documents": "required_documents",

            "apply": "application_process",
            "application": "application_process",
            "application_process": "application_process",

            "program": "program_info",
            "program_info": "program_info",

            "overview": "university_overview",
            "university_overview": "university_overview",

            "dorm": "dorm_overview",
            "dormitory": "dorm_overview",
            "dorm_overview": "dorm_overview",
            "dorm_fee": "dorm_fees",
            "dorm_fees": "dorm_fees",

            "location": "location",
            "hours": "hours",
            "other": "other",
        }

        out: List[str] = []
        for s in slots:
            k = str(s).strip().lower()
            if not k:
                continue
            out.append(mapping.get(k, k))
        return list(dict.fromkeys(out))

    @staticmethod
    def _union_required_slots(
        expected_general: List[str],
        expected_critical: List[str],
        expected_to_required_map: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        required_general: List[str] = []
        required_critical: List[str] = []

        for slot in expected_general + expected_critical:
            cfg = expected_to_required_map.get(slot, {})
            if isinstance(cfg, dict):
                required_general.extend(
                    [str(x).strip() for x in cfg.get("general", []) if str(x).strip()]
                )
                required_critical.extend(
                    [str(x).strip() for x in cfg.get("critical", []) if str(x).strip()]
                )

        return {
            "required_general": list(dict.fromkeys(required_general)),
            "required_critical": list(dict.fromkeys(required_critical)),
        }

    def _fallback_expected_slots(self, user: str) -> Dict[str, List[str]]:
        u = (user or "").lower()
        general: List[str] = []
        critical: List[str] = []

        if any(k in u for k in ["contact", "phone", "email", "address"]):
            critical.append("contact_information")
        if "deadline" in u:
            critical.append("deadline")
        if any(k in u for k in ["tuition"]):
            critical.append("tuition_fees")
        elif any(k in u for k in ["fee", "fees", "cost"]):
            critical.append("fees")
        if "scholar" in u:
            critical.append("scholarship_opportunities")
        if any(k in u for k in ["requirement", "requirements"]):
            critical.append("admission_requirements")
        if any(k in u for k in ["eligibility", "eligible"]):
            critical.append("eligibility")
        if any(k in u for k in ["document", "documents"]):
            critical.append("required_documents")
        if any(k in u for k in ["apply", "application process", "how to apply"]):
            critical.append("application_process")
        if any(k in u for k in ["program", "curriculum", "course", "specialization"]):
            general.append("program_info")
        if any(k in u for k in ["dorm", "dormitory", "accommodation"]):
            general.append("dorm_overview")
        if any(k in u for k in ["where", "location"]):
            general.append("location")
        if "hours" in u:
            general.append("hours")

        return {
            "expected_general_slots": list(dict.fromkeys(self._normalize_answer_slots(general))),
            "expected_critical_slots": list(dict.fromkeys(self._normalize_answer_slots(critical))),
        }

    def _extract_slots_from_text(self, text: str, allowed_slots: List[str], source: str) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        allowed = set(allowed_slots)
        txt = text or ""

        def add(slot: str, value: str) -> None:
            if slot in allowed and value and slot not in out:
                out[slot] = {"value": value.strip(), "source": source}

        if "program" in allowed:
            m = self._re_program_1.search(txt)
            if m:
                degree = m.group(1).strip()
                prog = m.group(2).strip(" .?")
                add("program", f"{degree} in {prog}")
            else:
                m2 = self._re_program_2.search(txt)
                if m2:
                    prog = m2.group(1).strip(" .?")
                    add("program", prog)

        if "degree_level" in allowed:
            m = self._re_degree.search(txt)
            if m:
                raw = m.group(1).strip().lower()
                if raw.startswith("master") or raw in {"msc", "m.sc.", "m.sc"}:
                    add("degree_level", "Master's")
                elif raw.startswith("phd") or raw.startswith("doctor"):
                    add("degree_level", "PhD")
                elif raw.startswith("bachelor") or raw.startswith("undergraduate"):
                    add("degree_level", "Bachelor's")

        if "intake_year_or_term" in allowed:
            m = self._re_intake.search(txt)
            if m:
                add("intake_year_or_term", m.group(1))

        if "fee_type" in allowed:
            m = self._re_fee_type.search(txt)
            if m:
                add("fee_type", m.group(1).lower())

        if "accommodation_type" in allowed:
            m = self._re_accommodation.search(txt)
            if m:
                add("accommodation_type", m.group(1))

        if "scholarship_type" in allowed:
            m = self._re_scholarship.search(txt)
            if m:
                add("scholarship_type", m.group(1).strip())

        return out

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

    def _fallback_refined_query(self, user: str, history_resolved_slots: Dict[str, Dict[str, str]], expected_critical: List[str]) -> str:
        u = (user or "").strip()
        program = history_resolved_slots.get("program", {}).get("value", "")
        degree = history_resolved_slots.get("degree_level", {}).get("value", "")
        intake = history_resolved_slots.get("intake_year_or_term", {}).get("value", "")
        accommodation = history_resolved_slots.get("accommodation_type", {}).get("value", "")

        if "deadline" in expected_critical and self._re_deadline_generic.match(u):
            if program and degree and intake:
                return f"When is the application deadline for {degree} {program} at AIT for {intake}?"
            if program and degree:
                return f"When is the application deadline for {degree} {program} at AIT?"
            if intake:
                return f"When is the application deadline at AIT for the {intake} intake?"

        if ("fees" in expected_critical or "tuition_fees" in expected_critical) and self._re_fee_generic.match(u):
            if program and degree:
                return f"What are the tuition fees for {degree} {program} at AIT?"
            if program:
                return f"What are the tuition fees for {program} at AIT?"

        if ("scholarship" in expected_critical or "scholarship_opportunities" in expected_critical) and self._re_scholar_generic.match(u):
            if program and degree:
                return f"What scholarship opportunities are available for {degree} {program} at AIT?"
            if program:
                return f"What scholarship opportunities are available for {program} at AIT?"

        if "dorm_fees" in expected_critical and accommodation:
            return f"What is the cost of {accommodation} accommodation at AIT dormitory?"

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

        try:
            expected_to_required_map = json.loads(self.expected_to_required_map_json or "{}")
            if not isinstance(expected_to_required_map, dict):
                expected_to_required_map = self.DEFAULT_EXPECTED_TO_REQUIRED_MAP
        except Exception:
            expected_to_required_map = self.DEFAULT_EXPECTED_TO_REQUIRED_MAP

        answer_slot_catalog = list(expected_to_required_map.keys())
        global_query_slot_catalog: List[str] = []
        for _, cfg in expected_to_required_map.items():
            if isinstance(cfg, dict):
                for key in ("general", "critical"):
                    vals = cfg.get(key, [])
                    if isinstance(vals, list):
                        for x in vals:
                            sx = str(x).strip()
                            if sx:
                                global_query_slot_catalog.append(sx)
        global_query_slot_catalog = list(dict.fromkeys(global_query_slot_catalog))

        prompt = (self.template or "")
        prompt = prompt.replace("{{USER}}", user)
        prompt = prompt.replace("{{HISTORY}}", hist_send)
        prompt = prompt.replace("{{EXPECTED_TO_REQUIRED_MAP}}", json.dumps(expected_to_required_map, ensure_ascii=False))
        prompt = prompt.replace("{{ANSWER_SLOT_CATALOG}}", json.dumps(answer_slot_catalog, ensure_ascii=False))
        prompt = prompt.replace("{{GLOBAL_QUERY_SLOT_CATALOG}}", json.dumps(global_query_slot_catalog, ensure_ascii=False))

        env = self._call_ollama(prompt)
        obj = self._extract_json(env.get("text", "")) or {}

        expected_general = self._normalize_answer_slots(self._safe_list(obj.get("expected_general_slots", [])))
        expected_critical = self._normalize_answer_slots(self._safe_list(obj.get("expected_critical_slots", [])))

        if not expected_general and not expected_critical:
            fallback = self._fallback_expected_slots(user)
            expected_general = fallback["expected_general_slots"]
            expected_critical = fallback["expected_critical_slots"]

        required = self._union_required_slots(expected_general, expected_critical, expected_to_required_map)
        allowed_required = list(dict.fromkeys(required["required_general"] + required["required_critical"]))

        # LLM extraction
        llm_user_slots = obj.get("user_resolved_slots", {})
        llm_history_slots = obj.get("history_resolved_slots", {})

        user_resolved_slots: Dict[str, Dict[str, str]] = {}
        history_resolved_slots: Dict[str, Dict[str, str]] = {}

        if isinstance(llm_user_slots, dict):
            for k, v in llm_user_slots.items():
                kk = str(k).strip()
                if kk not in allowed_required or not isinstance(v, dict):
                    continue
                val = str(v.get("value", "")).strip()
                src = str(v.get("source", "")).strip().lower()
                if val and src == "user":
                    user_resolved_slots[kk] = {"value": val, "source": "user"}

        if isinstance(llm_history_slots, dict):
            for k, v in llm_history_slots.items():
                kk = str(k).strip()
                if kk not in allowed_required or not isinstance(v, dict):
                    continue
                val = str(v.get("value", "")).strip()
                src = str(v.get("source", "")).strip().lower()
                if val and src == "history":
                    history_resolved_slots[kk] = {"value": val, "source": "history"}

        # Deterministic fallback extraction from raw text
        fallback_user = self._extract_slots_from_text(user, allowed_required, "user")
        fallback_history = self._extract_slots_from_text(hist_send, allowed_required, "history")

        for k, v in fallback_user.items():
            if k not in user_resolved_slots:
                user_resolved_slots[k] = v
        for k, v in fallback_history.items():
            if k not in history_resolved_slots and k not in user_resolved_slots:
                history_resolved_slots[k] = v

        missing_general = [s for s in required["required_general"] if s not in user_resolved_slots]
        missing_critical = [s for s in required["required_critical"] if s not in user_resolved_slots]

        missing_all = set(missing_general + missing_critical)
        history_resolved_slots = {k: v for k, v in history_resolved_slots.items() if k in missing_all}

        refined_query = ""
        refine_used = False
        used_slots: List[str] = []

        if bool(self.enable_refinement):
            llm_refined = self._ensure_question(str(obj.get("refined_query", "") or ""))
            fallback_refined = self._ensure_question(
                self._fallback_refined_query(user, history_resolved_slots, expected_critical)
            )
            refined_query = llm_refined or fallback_refined
            used_slots = list(history_resolved_slots.keys())
            refine_used = bool(refined_query and used_slots)

        payload = {
            "expected_general_slots": expected_general,
            "expected_critical_slots": expected_critical,
            "missing_general": missing_general,           # missing from USER only
            "missing_critical": missing_critical,         # missing from USER only
            "user_resolved_slots": user_resolved_slots,
            "history_resolved_slots": history_resolved_slots,
            "refinement": {
                "enabled": bool(self.enable_refinement),
                "used": refine_used,
                "refined_query": refined_query,
                "used_slots": used_slots,
            },
        }

        self.status = (
            f"expG={len(expected_general)} expC={len(expected_critical)} "
            f"missG={len(missing_general)} missC={len(missing_critical)} "
            f"userResolved={len(user_resolved_slots)} histFill={len(history_resolved_slots)} "
            f"refine={int(refine_used)}"
        )

        return Data(text_key="slot_gap", data=payload, default_value="")