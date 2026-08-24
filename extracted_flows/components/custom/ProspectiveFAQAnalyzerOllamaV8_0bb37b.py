# Recovered Langflow component
# type: ProspectiveFAQAnalyzerOllamaV8
# class: ProspectiveFAQAnalyzerOllamaV8
# used in 1 flow(s): Cognitive RAG V1.1.5
# json path: node.data.node.template.code.value

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
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


class ProspectiveFAQAnalyzerOllamaV8(Component):
    """
    Small-model-hardened analyzer for prospective-student FAQ routing.

    What changed from the previous version:
    - clarification-reply detection removed from this analyzer
    - no clarification_tracking in the LLM prompt
    - no clarification_tracking in the output
    - clarification_needed is computed only from weighted coverage + missing slots
    - refinement remains optional
    """

    display_name = "Analyzer (Prospective FAQ) [Ollama] V8 (No Clarification Tracking)"
    description = (
        "Expected answer slots + deterministic query-slot extraction + "
        "multi-intent required-slot union + optional refinement. "
        "Clarification reply detection removed from this node."
    )
    icon = "search"
    name = "ProspectiveFAQAnalyzerOllamaV8"

    JSON_MODELS_KEY = "models"
    JSON_NAME_KEY = "name"
    JSON_CAPABILITIES_KEY = "capabilities"
    DESIRED_CAPABILITY = "completion"
    TOOL_CALLING_CAPABILITY = "tools"

    ANSWER_SLOT_CATALOG = [
        "contact_information",
        "university_overview",
        "program_info",
        "admission_requirements",
        "eligibility",
        "required_documents",
        "application_process",
        "deadline",
        "fees",
        "tuition_fees",
        "scholarship",
        "scholarship_opportunities",
        "dorm_overview",
        "dorm_fees",
        "location",
        "hours",
        "other",
    ]

    MONTHS = {
        "jan": "January",
        "january": "January",
        "feb": "February",
        "february": "February",
        "mar": "March",
        "march": "March",
        "apr": "April",
        "april": "April",
        "may": "May",
        "jun": "June",
        "june": "June",
        "jul": "July",
        "july": "July",
        "aug": "August",
        "august": "August",
        "sep": "September",
        "sept": "September",
        "september": "September",
        "oct": "October",
        "october": "October",
        "nov": "November",
        "november": "November",
        "dec": "December",
        "december": "December",
    }

    REQUEST_TYPE_ALIASES = {
        "deadline": "deadlines",
        "deadlines": "deadlines",
        "fee": "fees",
        "fees": "fees",
        "tuition": "fees",
        "scholarship": "scholarship",
        "scholarships": "scholarship",
        "scholarship_opportunities": "scholarship",
        "contact": "contact_info",
        "contact_info": "contact_info",
        "contact_information": "contact_info",
        "program": "program_info",
        "program_info": "program_info",
        "requirements": "admission_requirements",
        "admission_requirement": "admission_requirements",
        "admission_requirements": "admission_requirements",
        "dorm": "dorm_overview",
        "dorm_overview": "dorm_overview",
        "dorm_fee": "dorm_fees",
        "dorm_fees": "dorm_fees",
        "overview": "university_overview",
        "university_overview": "university_overview",
        "general_info": "general_info",
        "unknown": "unknown",
        "multi_admissions_info": "multi_admissions_info",
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
        MessageTextInput(name="user_input", display_name="User Input", required=True),
        MessageTextInput(name="history_text", display_name="History (raw text)", advanced=True),
        IntInput(name="max_history_chars", display_name="Max history chars sent", value=6000, advanced=True),

        BoolInput(
            name="enable_refinement",
            display_name="Enable Refinement (Ablation Toggle)",
            value=True,
            advanced=False,
        ),

        MessageTextInput(
            name="slot_schema_json",
            display_name="Required Query Slot Schema JSON",
            info=(
                "JSON dict for clarification/retrieval requirements only. "
                "These are NOT answer-target slots."
            ),
            value=json.dumps(
                {
                    "university_overview": {"general": [], "critical": []},
                    "general_info": {"general": [], "critical": []},
                    "contact_info": {"general": [], "critical": []},
                    "program_info": {"general": ["program"], "critical": ["program"]},
                    "admission_requirements": {
                        "general": ["program", "degree_level"],
                        "critical": ["program", "degree_level"],
                    },
                    "fees": {
                        "general": ["program", "degree_level", "fee_type"],
                        "critical": ["program", "fee_type"],
                    },
                    "deadlines": {
                        "general": ["program", "degree_level", "intake_year_or_term"],
                        "critical": ["intake_year_or_term"],
                    },
                    "scholarship": {
                        "general": ["degree_level"],
                        "critical": [],
                    },
                    "dorm_overview": {"general": [], "critical": []},
                    "dorm_fees": {
                        "general": ["accommodation_type", "fee_type"],
                        "critical": ["accommodation_type"],
                    },
                    "multi_admissions_info": {
                        "general": ["program", "degree_level", "intake_year_or_term", "fee_type"],
                        "critical": ["program", "degree_level", "intake_year_or_term"],
                    },
                    "unknown": {"general": [], "critical": []},
                },
                ensure_ascii=False,
                indent=2,
            ),
        ),

        FloatInput(name="weight_general", display_name="Weight general", value=1.0, advanced=True),
        FloatInput(name="weight_critical", display_name="Weight critical", value=2.0, advanced=True),

        FloatInput(
            name="clarification_threshold",
            display_name="Clarification Threshold",
            value=0.75,
            advanced=False,
            info=(
                "If weighted coverage over required query slots is below this threshold, "
                "clarification_needed becomes true."
            ),
        ),

        BoolInput(name="enable_smalltalk_bypass", display_name="Smalltalk bypass", value=True, advanced=True),

        MessageTextInput(
            name="system",
            display_name="System (expandable)",
            value="You are a careful analyzer. Return ONLY valid JSON. No code fences. No extra text.",
            advanced=True,
        ),

        MessageTextInput(
            name="template_no_refine",
            display_name="Analyzer Template (NO refinement) (expandable)",
            value=(
                "Return ONLY valid JSON.\n\n"
                "You are the ANALYZER for a bounded RAG FAQ system (prospective student).\n"
                "You output signals ONLY. Do NOT decide actions.\n\n"
                "USER: {{USER}}\n"
                "HISTORY: {{HISTORY}}\n\n"
                "REQUEST_TYPES (choose one best-fit type):\n{{REQUEST_TYPES}}\n\n"
                "REQUIRED_QUERY_SLOT_SCHEMA (reference):\n{{SLOT_SCHEMA}}\n\n"
                "ANSWER_SLOT_CATALOG (canonical names only):\n{{ANSWER_SLOT_CATALOG}}\n\n"
                "TASKS\n"
                "1) intent_type: chitchat|faq|other\n"
                "2) Choose request_type from REQUEST_TYPES.\n"
                "3) expected_general_slots / expected_critical_slots:\n"
                "   - These are the information items the ANSWER must cover.\n"
                "   - Use only names from ANSWER_SLOT_CATALOG.\n"
                "4) resolved_slots:\n"
                "   - Only for explicit query-side slots seen in USER or HISTORY.\n"
                "   - Format: {slot_name: {\"value\": \"...\", \"source\": \"user|history\"}}\n\n"
                "OUTPUT JSON\n"
                "{\n"
                "  \"intent_type\": \"faq\",\n"
                "  \"request_type\": \"unknown\",\n"
                "  \"topic\": \"\",\n"
                "  \"expected_general_slots\": [],\n"
                "  \"expected_critical_slots\": [],\n"
                "  \"resolved_slots\": {}\n"
                "}\n\n"
                "RULES\n"
                "- Do NOT invent values.\n"
                "- Be conservative.\n"
                "- Do NOT output missing_general or missing_critical.\n"
                "- Do NOT output explanation text.\n"
            ),
            advanced=True,
        ),

        MessageTextInput(
            name="template_with_refine",
            display_name="Analyzer Template (WITH refinement) (expandable)",
            value=(
                "Return ONLY valid JSON.\n\n"
                "You are the ANALYZER for a bounded RAG FAQ system (prospective student).\n"
                "You output signals ONLY. Do NOT decide actions.\n\n"
                "USER: {{USER}}\n"
                "HISTORY: {{HISTORY}}\n\n"
                "REQUEST_TYPES (choose one best-fit type):\n{{REQUEST_TYPES}}\n\n"
                "REQUIRED_QUERY_SLOT_SCHEMA (reference):\n{{SLOT_SCHEMA}}\n\n"
                "ANSWER_SLOT_CATALOG (canonical names only):\n{{ANSWER_SLOT_CATALOG}}\n\n"
                "TASKS\n"
                "1) intent_type: chitchat|faq|other\n"
                "2) Choose request_type from REQUEST_TYPES.\n"
                "3) expected_general_slots / expected_critical_slots:\n"
                "   - These are the information items the ANSWER must cover.\n"
                "   - Use only names from ANSWER_SLOT_CATALOG.\n"
                "4) resolved_slots:\n"
                "   - Only for explicit query-side slots seen in USER or HISTORY.\n"
                "   - Format: {slot_name: {\"value\": \"...\", \"source\": \"user|history\"}}\n"
                "5) refined_question:\n"
                "   - OPTIONAL.\n"
                "   - If USER is vague and HISTORY supplies enough slots, rewrite into ONE concise natural question.\n"
                "   - refined_question must end with '?', or be empty.\n\n"
                "OUTPUT JSON\n"
                "{\n"
                "  \"intent_type\": \"faq\",\n"
                "  \"request_type\": \"unknown\",\n"
                "  \"topic\": \"\",\n"
                "  \"expected_general_slots\": [],\n"
                "  \"expected_critical_slots\": [],\n"
                "  \"resolved_slots\": {},\n"
                "  \"refined_question\": \"\"\n"
                "}\n\n"
                "RULES\n"
                "- Do NOT invent values.\n"
                "- Be conservative.\n"
                "- Do NOT output missing_general or missing_critical.\n"
                "- Do NOT output explanation text.\n"
            ),
            advanced=True,
        ),

        BoolInput(
            name="enable_thinking",
            display_name="Enable Thinking (Ollama think)",
            info="Send Ollama /api/chat with think=true or think=false.",
            value=False,
            advanced=False,
            real_time_refresh=True,
        ),
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

    _re_deadline_generic = re.compile(r"^\s*(when\s+is\s+the\s+deadline\??|deadline\??)\s*$", re.I)
    _re_fee_generic = re.compile(r"^\s*(how\s+much\s+is\s+the\s+fee\??|fees\??|tuition\??|cost\??)\s*$", re.I)
    _re_scholar_generic = re.compile(
        r"^\s*(what\s+are\s+the\s+requirements\??|scholarship\??|scholarship opportunities\??)\s*$",
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
    def _dedupe(seq: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for s in seq:
            if s and s not in seen:
                out.append(s)
                seen.add(s)
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

        cov_g = ProspectiveFAQAnalyzerOllamaV8._coverage_from_missing(eg, mg)
        cov_c = ProspectiveFAQAnalyzerOllamaV8._coverage_from_missing(ec, mc)

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

    @staticmethod
    def _norm_text(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    def _is_smalltalk_only(self, user: str) -> bool:
        u = (user or "").strip().lower()
        if not u:
            return False
        if not self._re_smalltalk.match(u):
            return False

        request_markers = [
            "?",
            "can you",
            "could you",
            "tell me",
            "what",
            "when",
            "where",
            "how",
            "who",
            "why",
            "contact",
            "deadline",
            "fee",
            "tuition",
            "requirement",
            "scholarship",
            "program",
            "admission",
            "apply",
            "application",
        ]
        return not any(m in u for m in request_markers)

    def _normalize_answer_slots(self, slots: List[str]) -> List[str]:
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
        return self._dedupe(out)

    def _normalize_request_type(self, request_type: str, schema: Dict[str, Any]) -> str:
        rt = str(request_type or "").strip().lower()
        rt = self.REQUEST_TYPE_ALIASES.get(rt, rt)
        return rt if rt in schema else "unknown"

    def _fallback_expected_slots(self, user: str, request_type: str) -> Dict[str, List[str]]:
        u = (user or "").lower()
        general: List[str] = []
        critical: List[str] = []

        if any(k in u for k in ["contact", "phone", "email", "address"]):
            critical.append("contact_information")
        if "deadline" in u:
            critical.append("deadline")
        if any(k in u for k in ["fee", "fees", "tuition", "cost"]):
            critical.extend(["fees", "tuition_fees"])
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
        if any(k in u for k in ["program", "curriculum", "course", "specialization", "master's", "phd", "bachelor"]):
            general.append("program_info")
        if any(k in u for k in ["dorm", "dormitory", "accommodation"]):
            general.append("dorm_overview")
        if any(k in u for k in ["where", "location"]):
            general.append("location")
        if "hours" in u:
            general.append("hours")

        rt_map = {
            "contact_info": {"general": [], "critical": ["contact_information"]},
            "program_info": {"general": ["program_info"], "critical": []},
            "admission_requirements": {"general": [], "critical": ["admission_requirements"]},
            "fees": {"general": [], "critical": ["fees", "tuition_fees"]},
            "deadlines": {"general": [], "critical": ["deadline"]},
            "scholarship": {"general": [], "critical": ["scholarship_opportunities"]},
            "dorm_overview": {"general": ["dorm_overview"], "critical": []},
            "dorm_fees": {"general": [], "critical": ["dorm_fees"]},
            "university_overview": {"general": ["university_overview"], "critical": []},
        }
        if request_type in rt_map:
            general.extend(rt_map[request_type]["general"])
            critical.extend(rt_map[request_type]["critical"])

        return {
            "general": self._dedupe(self._normalize_answer_slots(general)),
            "critical": self._dedupe(self._normalize_answer_slots(critical)),
        }

    def _all_query_slots_from_schema(self, schema: Dict[str, Any]) -> List[str]:
        slots: List[str] = []
        for spec in schema.values():
            if isinstance(spec, dict):
                slots.extend(self._safe_list(spec.get("general", [])))
                slots.extend(self._safe_list(spec.get("critical", [])))
        return self._dedupe(slots)

    def _extract_degree_level_from_text(self, text: str) -> str:
        t = text or ""
        patterns = [
            (r"\b(master'?s|masters|m\.?sc|m\.?eng|m\.?a)\b", "Master's"),
            (r"\b(phd|ph\.?d|doctorate|doctoral)\b", "PhD"),
            (r"\b(bachelor'?s|bachelors|b\.?sc|b\.?eng|undergraduate)\b", "Bachelor's"),
        ]
        for pat, label in patterns:
            if re.search(pat, t, re.I):
                return label
        return ""

    def _extract_fee_type_from_text(self, text: str) -> str:
        t = text or ""
        if re.search(r"\btuition fees?\b|\btuition\b", t, re.I):
            return "tuition"
        if re.search(r"\bapplication fees?\b|\bapplication fee\b", t, re.I):
            return "application"
        if re.search(r"\bdorm fees?\b|\baccommodation fees?\b", t, re.I):
            return "accommodation"
        return ""

    def _extract_scholarship_type_from_text(self, text: str) -> str:
        t = text or ""
        m = re.search(r"\b(king'?s scholarship|ait scholarship|government scholarship)\b", t, re.I)
        return m.group(1).strip() if m else ""

    def _extract_accommodation_type_from_text(self, text: str) -> str:
        t = text or ""
        patterns = [
            r"\b(single room)\b",
            r"\b(double room)\b",
            r"\b(shared room)\b",
            r"\bstudio\b",
            r"\bdormitory\b",
        ]
        for pat in patterns:
            m = re.search(pat, t, re.I)
            if m:
                return m.group(1).strip()
        return ""

    def _extract_intake_from_text(self, text: str) -> str:
        t = text or ""

        m1 = re.search(
            r"\b("
            r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
            r")\s+((?:20|19)\d{2})\b",
            t,
            re.I,
        )
        if m1:
            month = self.MONTHS.get(m1.group(1).lower(), m1.group(1).title())
            year = m1.group(2)
            return f"{month} {year}"

        m2 = re.search(
            r"\b((?:20|19)\d{2})\s+("
            r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
            r")\b",
            t,
            re.I,
        )
        if m2:
            year = m2.group(1)
            month = self.MONTHS.get(m2.group(2).lower(), m2.group(2).title())
            return f"{month} {year}"

        return ""

    def _cleanup_program_phrase(self, s: str) -> str:
        s = re.sub(r"\s+", " ", (s or "")).strip(" ,.-")
        s = re.sub(r"\bfor\s+\d{4}\s+\w+\s+intake\b", "", s, flags=re.I).strip(" ,.-")
        s = re.sub(r"\bfor\s+\w+\s+\d{4}\s+intake\b", "", s, flags=re.I).strip(" ,.-")
        s = re.sub(r"\bintake\b", "", s, flags=re.I).strip(" ,.-")
        return s

    def _extract_program_from_text(self, text: str) -> str:
        t = text or ""

        patterns = [
            r"\bfor\s+(?:the\s+)?(.+?)\s+at\s+AIT\b",
            r"\bfor\s+(?:the\s+)?(.+?)\s+(?:for\s+(?:the\s+)?)?(?:\d{4}|\bjan|\bfeb|\bmar|\bapr|\bmay|\bjun|\bjul|\baug|\bsep|\boct|\bnov|\bdec|\bat\s+AIT|\?|,|$)",
            r"\b((?:Master'?s|Masters|PhD|Doctoral|Bachelor'?s)\s+in\s+[A-Za-z0-9&/\-\s]+)\b",
        ]
        for pat in patterns:
            m = re.search(pat, t, re.I)
            if m:
                candidate = self._cleanup_program_phrase(m.group(1))
                if candidate and len(candidate) >= 4:
                    return candidate
        return ""

    def _deterministic_resolve_slots_from_text(self, text: str, source: str) -> Dict[str, Dict[str, str]]:
        resolved: Dict[str, Dict[str, str]] = {}

        program = self._extract_program_from_text(text)
        if program:
            resolved["program"] = {"value": program, "source": source}

        degree = self._extract_degree_level_from_text(text)
        if degree:
            resolved["degree_level"] = {"value": degree, "source": source}

        intake = self._extract_intake_from_text(text)
        if intake:
            resolved["intake_year_or_term"] = {"value": intake, "source": source}

        fee_type = self._extract_fee_type_from_text(text)
        if fee_type:
            resolved["fee_type"] = {"value": fee_type, "source": source}

        scholarship_type = self._extract_scholarship_type_from_text(text)
        if scholarship_type:
            resolved["scholarship_type"] = {"value": scholarship_type, "source": source}

        accom = self._extract_accommodation_type_from_text(text)
        if accom:
            resolved["accommodation_type"] = {"value": accom, "source": source}

        return resolved

    def _compose_program_phrase(self, program: str, degree: str) -> str:
        program = (program or "").strip()
        degree = (degree or "").strip()

        if program and degree:
            if self._norm_text(degree) in self._norm_text(program):
                return program
            return f"{degree} {program}"

        if program:
            return program
        if degree:
            return f"{degree} programs"
        return ""

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

        program_phrase = self._compose_program_phrase(program, degree)

        if request_type in {"deadlines", "multi_admissions_info"}:
            if intake:
                if program_phrase:
                    return f"When is the AIT application deadline for {program_phrase} ({intake} intake)?"
                return f"When is the AIT application deadline for the {intake} intake?"
            if program_phrase:
                return f"When is the AIT application deadline for {program_phrase} at AIT?"
            return "When is the AIT application deadline at AIT?"

        if request_type == "fees":
            ft = fee_type or "tuition fee"
            if program_phrase:
                return f"What is the {ft} for {program_phrase} at AIT?"
            if degree:
                return f"What is the {ft} for {degree} admission at AIT?"
            return f"What is the {ft} at AIT?"

        if request_type == "scholarship":
            if schol:
                if degree:
                    return f"What are the requirements for the {schol} scholarship for {degree} programs at AIT?"
                return f"What are the requirements for the {schol} scholarship at AIT?"
            if degree:
                return f"What scholarship opportunities are available for {degree} programs at AIT?"
            return "What scholarship opportunities are available at AIT?"

        if request_type == "dorm_fees":
            if accom:
                return f"What is the monthly cost for {accom} accommodation at AIT dormitory?"
            return "What is the dormitory accommodation cost at AIT?"

        if request_type == "admission_requirements":
            if program_phrase:
                return f"What are the admission requirements for {program_phrase} at AIT?"
            return "What are the admission requirements at AIT?"

        return ""

    def _map_expected_slots_to_request_types(
        self,
        expected_general: List[str],
        expected_critical: List[str],
    ) -> List[str]:
        slots = set(expected_general + expected_critical)
        out: List[str] = []

        if "contact_information" in slots:
            out.append("contact_info")
        if "program_info" in slots:
            out.append("program_info")
        if "admission_requirements" in slots or "eligibility" in slots or "required_documents" in slots:
            out.append("admission_requirements")
        if "fees" in slots or "tuition_fees" in slots:
            out.append("fees")
        if "deadline" in slots:
            out.append("deadlines")
        if "scholarship" in slots or "scholarship_opportunities" in slots:
            out.append("scholarship")
        if "dorm_overview" in slots:
            out.append("dorm_overview")
        if "dorm_fees" in slots:
            out.append("dorm_fees")
        if "university_overview" in slots:
            out.append("university_overview")

        return self._dedupe(out)

    def _derive_required_slots(
        self,
        expected_general: List[str],
        expected_critical: List[str],
        model_request_type: str,
        schema: Dict[str, Any],
    ) -> Tuple[str, List[str], List[str], List[str]]:
        from_expected = self._map_expected_slots_to_request_types(expected_general, expected_critical)

        effective_types: List[str] = []
        if model_request_type != "unknown":
            effective_types.append(model_request_type)
        effective_types.extend(from_expected)
        effective_types = self._dedupe([t for t in effective_types if t in schema and t != "unknown"])

        if len(effective_types) == 0:
            return "unknown", [], [], []
        if len(effective_types) == 1:
            rt = effective_types[0]
            return (
                rt,
                effective_types,
                self._safe_list(schema.get(rt, {}).get("general", [])),
                self._safe_list(schema.get(rt, {}).get("critical", [])),
            )

        req_g: List[str] = []
        req_c: List[str] = []
        for rt in effective_types:
            req_g.extend(self._safe_list(schema.get(rt, {}).get("general", [])))
            req_c.extend(self._safe_list(schema.get(rt, {}).get("critical", [])))

        return "multi_admissions_info", effective_types, self._dedupe(req_g), self._dedupe(req_c)

    def _merge_resolved_slots(
        self,
        model_resolved: Dict[str, Dict[str, str]],
        user_det: Dict[str, Dict[str, str]],
        history_det: Dict[str, Dict[str, str]],
        allowed_slots: List[str],
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
        allowed = set(allowed_slots)

        user_resolved: Dict[str, Dict[str, str]] = {}
        history_resolved: Dict[str, Dict[str, str]] = {}

        for k, v in model_resolved.items():
            if k not in allowed:
                continue
            src = v.get("source", "")
            if src == "user":
                user_resolved[k] = v
            elif src == "history":
                history_resolved[k] = v

        for k, v in history_det.items():
            if k in allowed:
                history_resolved[k] = v

        for k, v in user_det.items():
            if k in allowed:
                user_resolved[k] = v

        merged = {}
        merged.update(history_resolved)
        merged.update(user_resolved)

        return merged, user_resolved, history_resolved

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

        if bool(self.enable_smalltalk_bypass) and self._is_smalltalk_only(user) and len(user) <= 80:
            payload = {
                "original_question": user,
                "history_window": "",
                "intent_type": "chitchat",
                "request_type": "general_info",
                "effective_request_types": [],
                "topic": "",
                "expected_general_slots": [],
                "expected_critical_slots": [],
                "required_general_slots": [],
                "required_critical_slots": [],
                "missing_slots": {"general": [], "critical": []},
                "slot_coverage_general": 1.0,
                "slot_coverage_critical": 1.0,
                "slot_coverage_weighted": 1.0,
                "counts": {"general_total": 0, "critical_total": 0, "general_missing": 0, "critical_missing": 0},
                "resolved_slots": {},
                "user_resolved_slots": {},
                "history_resolved_slots": {},
                "refinement": {
                    "enabled": bool(self.enable_refinement),
                    "used": False,
                    "refined_query": "",
                    "used_slots": [],
                },
                "clarification_decision": {
                    "threshold": float(self.clarification_threshold or 0.75),
                    "coverage_weighted": 1.0,
                    "has_missing_slots": False,
                    "clarification_needed": False,
                },
                "clarification_needed": False,
                "llm_metrics": {},
            }
            self.status = "chitchat_bypass"
            return Data(text_key="analyzer", data=payload, default_value="")

        try:
            schema = json.loads(self.slot_schema_json or "{}")
        except Exception:
            schema = {}
        if not isinstance(schema, dict):
            schema = {}
        if "unknown" not in schema:
            schema["unknown"] = {"general": [], "critical": []}
        if "multi_admissions_info" not in schema:
            schema["multi_admissions_info"] = {
                "general": ["program", "degree_level", "intake_year_or_term", "fee_type"],
                "critical": ["program", "degree_level", "intake_year_or_term"],
            }

        request_types = sorted(list(schema.keys()))

        max_hist = int(self.max_history_chars or 6000)
        hist_send = history if (max_hist <= 0 or len(history) <= max_hist) else (history[:max_hist] + "\n[TRUNCATED]")

        template = self.template_with_refine if bool(self.enable_refinement) else self.template_no_refine
        prompt = (template or "")
        prompt = prompt.replace("{{USER}}", user)
        prompt = prompt.replace("{{HISTORY}}", hist_send)
        prompt = prompt.replace("{{REQUEST_TYPES}}", json.dumps(request_types, ensure_ascii=False))
        prompt = prompt.replace("{{SLOT_SCHEMA}}", json.dumps(schema, ensure_ascii=False))
        prompt = prompt.replace("{{ANSWER_SLOT_CATALOG}}", json.dumps(self.ANSWER_SLOT_CATALOG, ensure_ascii=False))

        env = self._call_ollama(prompt)
        obj = self._extract_json(env.get("text", "")) or {}
        metrics = env.get("metrics") or {}

        intent_type = str(obj.get("intent_type") or "faq").strip().lower()
        model_request_type = self._normalize_request_type(str(obj.get("request_type") or "unknown"), schema)
        topic = str(obj.get("topic") or "").strip()

        model_expected_general = self._normalize_answer_slots(self._safe_list(obj.get("expected_general_slots", [])))
        model_expected_critical = self._normalize_answer_slots(self._safe_list(obj.get("expected_critical_slots", [])))

        fallback = self._fallback_expected_slots(user, model_request_type)
        expected_general = self._dedupe(model_expected_general + fallback["general"])
        expected_critical = self._dedupe(model_expected_critical + fallback["critical"])

        request_type, effective_request_types, required_general, required_critical = self._derive_required_slots(
            expected_general,
            expected_critical,
            model_request_type,
            schema,
        )

        all_query_slots = self._all_query_slots_from_schema(schema)

        model_resolved_raw = obj.get("resolved_slots", {})
        model_resolved: Dict[str, Dict[str, str]] = {}
        if isinstance(model_resolved_raw, dict):
            for k, v in model_resolved_raw.items():
                kk = str(k).strip()
                if kk not in all_query_slots:
                    continue
                if not isinstance(v, dict):
                    continue
                val = str(v.get("value", "")).strip()
                src = str(v.get("source", "")).strip().lower()
                if val and src in ("user", "history"):
                    model_resolved[kk] = {"value": val, "source": src}

        user_det = self._deterministic_resolve_slots_from_text(user, "user")
        history_det = self._deterministic_resolve_slots_from_text(history, "history")

        resolved_slots, user_resolved_slots, history_resolved_slots = self._merge_resolved_slots(
            model_resolved,
            user_det,
            history_det,
            allowed_slots=self._dedupe(required_general + required_critical),
        )

        missing_general = [
            s for s in required_general
            if not str((resolved_slots.get(s, {}) or {}).get("value", "")).strip()
        ]
        missing_critical = [
            s for s in required_critical
            if not str((resolved_slots.get(s, {}) or {}).get("value", "")).strip()
        ]

        wg = float(self.weight_general or 1.0)
        wc = float(self.weight_critical or 2.0)
        cov = self._weighted_coverage(required_general, required_critical, missing_general, missing_critical, wg, wc)

        clarification_threshold = float(self.clarification_threshold or 0.75)
        coverage_weighted = float(cov.get("slot_coverage_weighted", 1.0))
        has_missing_slots = bool(missing_general or missing_critical)

        clarification_needed = bool((coverage_weighted < clarification_threshold) and has_missing_slots)

        clarification_decision = {
            "threshold": round(clarification_threshold, 4),
            "coverage_weighted": round(coverage_weighted, 4),
            "has_missing_slots": has_missing_slots,
            "clarification_needed": clarification_needed,
        }

        refine_enabled = bool(self.enable_refinement)
        refine_used = False
        refined_query = ""
        used_slots: List[str] = []

        if refine_enabled:
            history_resolved_critical = [
                s for s in required_critical
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

                qnorm = self._norm_text(llm_refined_q)
                bad_dup = ("masters masters" in qnorm) or ("phd phd" in qnorm) or ("bachelor bachelor" in qnorm)

                refined_query = fallback_q if (not llm_refined_q or bad_dup) else llm_refined_q
                used_slots = history_resolved_critical

        payload = {
            "original_question": user,
            "history_window": hist_send,

            "intent_type": intent_type,
            "request_type": request_type,
            "effective_request_types": effective_request_types,
            "topic": topic,

            "expected_general_slots": expected_general,
            "expected_critical_slots": expected_critical,

            "required_general_slots": required_general,
            "required_critical_slots": required_critical,
            "missing_slots": {"general": missing_general, "critical": missing_critical},

            **cov,

            "resolved_slots": resolved_slots,
            "user_resolved_slots": user_resolved_slots,
            "history_resolved_slots": history_resolved_slots,

            "refinement": {
                "enabled": refine_enabled,
                "used": refine_used,
                "refined_query": refined_query,
                "used_slots": used_slots,
            },

            "clarification_decision": clarification_decision,
            "clarification_needed": clarification_needed,

            "llm_metrics": metrics,
        }

        self.status = (
            f"{request_type} | eff={','.join(effective_request_types) if effective_request_types else 'none'} "
            f"| exp={len(expected_general) + len(expected_critical)} "
            f"| reqW={payload['slot_coverage_weighted']:.2f} "
            f"| clar={int(clarification_needed)}@{clarification_threshold:.2f} "
            f"| missC={len(missing_critical)} missG={len(missing_general)} "
            f"| refine_en={int(refine_enabled)} refine_used={int(refine_used)} "
            f"| think={int(bool(self.enable_thinking))} "
            f"| in={metrics.get('prompt_eval_count')} out={metrics.get('eval_count')}"
        )

        return Data(text_key="analyzer", data=payload, default_value="")