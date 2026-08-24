# Recovered Langflow component
# type: RefinementLLMFromMerged
# class: RefinementLLMFromMerged
# used in 16 flow(s): AIT Cognitive RAG, Cognitive RAG V1.1.5, Cognitive RAG V1.2.0 (1), Cognitive RAG V1.2.0 AIT CR, Cognitive RAG V1.2.0 AIT Loose, Cognitive RAG V1.2.0 AIT Normal, Cognitive RAG V1.2.0 AIT RR, Cognitive RAG V1.2.0 AIT Reason ...
# json path: node.data.node.template.code.value

from __future__ import annotations

import ast
import json
import re
import time
from typing import Any, Dict, List, Optional

import httpx

from lfx.custom import Component
from lfx.io import BoolInput, DictInput, IntInput, MessageInput, Output, StrInput
from lfx.schema import Message


class RefinementLLMFromMerged(Component):
    display_name = "Refinement LLM (Merged Input)"
    description = (
        "Takes merged analyzer+clarification payload, decides whether query refinement is needed, "
        "and outputs only refinement JSON."
    )
    icon = "edit"
    name = "RefinementLLMFromMerged"

    inputs = [
        MessageInput(
            name="input_message",
            display_name="Merged Input Message",
            info="Output from ClarificationAnalyzerMerger",
            required=True,
        ),
        StrInput(
            name="base_url",
            display_name="Ollama Base URL",
            value="http://localhost:11434",
            required=True,
        ),
        StrInput(
            name="model_name",
            display_name="Model Name",
            value="qwen3:4b-instruct",
            required=True,
        ),
        BoolInput(
            name="enable_refinement",
            display_name="Enable Refinement",
            value=True,
        ),
        BoolInput(
            name="enable_thinking",
            display_name="Enable Thinking",
            value=False,
            advanced=True,
        ),
        IntInput(
            name="timeout_s",
            display_name="Timeout (seconds)",
            value=120,
            advanced=True,
        ),
        IntInput(
            name="max_history_chars",
            display_name="Max History Characters",
            value=5000,
            advanced=True,
        ),
        StrInput(
            name="system_prompt",
            display_name="System Prompt",
            value=(
                "You are a careful query refiner for a bounded FAQ/RAG system. "
                "Your job is ONLY to decide whether the current user query must be rewritten "
                "using conversation history, and if so, output one standalone refined query. "
                "Return ONLY valid JSON. Do not output markdown. Do not explain. "
                "Do not invent facts. Preserve user intent exactly."
            ),
            advanced=True,
        ),
        StrInput(
            name="user_prompt_template",
            display_name="User Prompt Template",
            value=(
                "Return ONLY valid JSON.\n\n"
                "MERGED_PAYLOAD_JSON:\n{{MERGED_PAYLOAD_JSON}}\n\n"
                "Your task is to decide whether original_question must be rewritten into a "
                "standalone retrieval-ready query using history_window.\n\n"
                "IMPORTANT:\n"
                "If the current query is NOT fully understandable on its own without history, "
                "you MUST set used=true and produce refined_query.\n\n"
                "You MUST use refinement for:\n"
                "1. pronouns or references like: it, this, that, that one, these, those\n"
                "2. short follow-up questions that depend on history\n"
                "3. clarification replies that only provide a missing slot\n"
                "4. elliptical fragments such as: 'for August intake', 'the master's one', "
                "'homework submission for CS', 'what about scholarships'\n\n"
                "You MUST NOT use refinement only when original_question is already a clear "
                "standalone retrieval query.\n\n"
                "Allowed refinement_type values:\n"
                "- none\n"
                "- history_followup\n"
                "- pronoun_resolution\n"
                "- clarification_reply\n"
                "- ellipsis\n"
                "- other\n\n"
                "OUTPUT JSON SCHEMA:\n"
                "{\n"
                '  "used": false,\n'
                '  "refined_query": "",\n'
                '  "refinement_type": "none",\n'
                '  "used_slots": []\n'
                "}\n\n"
                "RULES:\n"
                "- If used=false, refined_query must be empty.\n"
                "- If used=true, refined_query must be one standalone natural-language question.\n"
                "- Do not answer the question.\n"
                "- Do not add information not grounded in merged payload.\n"
                "- If original_question contains 'it', 'this', 'that', 'these', or 'those', "
                "and history is needed, you MUST set used=true.\n"
                "- If original_question is a short fragment or clarification reply, and history "
                "is needed, you MUST set used=true.\n"
            ),
            advanced=True,
        ),
        DictInput(
            name="extra_options",
            display_name="Extra Options",
            value={},
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Refinement Message",
            name="refinement_message",
            method="build_output",
        ),
    ]

    _re_json_obj = re.compile(r"\{.*\}", re.DOTALL)
    _re_pronoun = re.compile(r"\b(it|this|that|these|those|that one|this one)\b", re.I)

    def _extract_raw_text(self, value: Any) -> str:
        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()
        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()
        return str(value).strip()

    def _parse_payload(self, raw_text: str) -> Dict[str, Any]:
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

    def _dedupe(self, seq: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for s in seq:
            if s and s not in seen:
                out.append(s)
                seen.add(s)
        return out

    def _trim_history(self, text: str) -> str:
        text = text or ""
        limit = int(self.max_history_chars or 0)
        if limit <= 0 or len(text) <= limit:
            return text
        return text[-limit:]

    def _ensure_question(self, s: str) -> str:
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

    def _norm_text(self, s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    def _extract_last_user_before_last_ai(self, history: str) -> str:
        if not history:
            return ""

        turns = []
        for ln in history.splitlines():
            line = (ln or "").strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("user:"):
                turns.append(("user", line.split(":", 1)[1].strip()))
            elif lower.startswith("ai:") or lower.startswith("assistant:"):
                turns.append(("ai", line.split(":", 1)[1].strip()))

        last_ai_idx = -1
        for i in range(len(turns) - 1, -1, -1):
            if turns[i][0] == "ai":
                last_ai_idx = i
                break

        if last_ai_idx <= 0:
            return ""

        for i in range(last_ai_idx - 1, -1, -1):
            if turns[i][0] == "user":
                return turns[i][1]

        return ""

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

    def _looks_like_fragment(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        if "?" in t:
            return False
        return len(t.split()) <= 8

    def _needs_force_refinement(self, payload: Dict[str, Any]) -> bool:
        user = str(payload.get("original_question", "") or "").strip()
        history = str(payload.get("history_window", "") or "").strip()
        tracker = payload.get("clarification_tracking", {}) or {}

        if not user or not history:
            return False

        clarification_asked = bool(
            tracker.get("clarification_asked_in_last_ai_turn", False)
            or tracker.get("clarification_asked_in_history", False)
        )

        if clarification_asked and self._looks_like_fragment(user):
            return True

        if self._re_pronoun.search(user):
            return True

        if len(user.split()) <= 8 and history:
            return True

        return False

    def _fallback_refinement(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user = str(payload.get("original_question", "") or "").strip()
        history = str(payload.get("history_window", "") or "").strip()
        request_type = str(payload.get("request_type", "unknown") or "unknown").strip()
        resolved_slots = payload.get("resolved_slots", {}) or {}
        required_general = payload.get("required_general_slots", []) or []
        required_critical = payload.get("required_critical_slots", []) or []
        tracker = payload.get("clarification_tracking", {}) or {}

        clarification_asked = bool(
            tracker.get("clarification_asked_in_last_ai_turn", False)
            or tracker.get("clarification_asked_in_history", False)
        )
        previous_user_turn = self._extract_last_user_before_last_ai(history)

        def val(k: str) -> str:
            return str((resolved_slots.get(k, {}) or {}).get("value", "")).strip()

        program = val("program")
        degree = val("degree_level")
        intake = val("intake_year_or_term")
        fee_type = val("fee_type")
        scholarship_type = val("scholarship_type")

        # ignore noisy placeholder values
        if intake.lower() in {"current", "now", "present", "latest"}:
            intake = ""

        used_slots = [
            s for s in self._dedupe(list(required_general) + list(required_critical))
            if s in resolved_slots
        ]

        program_phrase = self._compose_program_phrase(program, degree)

        # clarification reply case
        if clarification_asked and previous_user_turn and self._looks_like_fragment(user):
            if request_type == "program_info" and program_phrase:
                rq = self._ensure_question(f"Is {program_phrase} available at AIT")
                return {
                    "used": True,
                    "refined_query": rq,
                    "refinement_type": "clarification_reply",
                    "used_slots": used_slots,
                }

            if request_type == "deadlines":
                if program_phrase and intake:
                    rq = self._ensure_question(
                        f"When is the application deadline for {program_phrase} at AIT for the {intake} intake"
                    )
                elif program_phrase:
                    rq = self._ensure_question(
                        f"When is the application deadline for {program_phrase} at AIT"
                    )
                else:
                    rq = self._ensure_question(previous_user_turn)

                return {
                    "used": True,
                    "refined_query": rq,
                    "refinement_type": "clarification_reply",
                    "used_slots": used_slots,
                }

            if request_type == "fees":
                ft = fee_type or "tuition fee"
                if program_phrase:
                    rq = self._ensure_question(f"What is the {ft} for {program_phrase} at AIT")
                else:
                    rq = self._ensure_question(previous_user_turn)

                return {
                    "used": True,
                    "refined_query": rq,
                    "refinement_type": "clarification_reply",
                    "used_slots": used_slots,
                }

            if request_type == "scholarship":
                if scholarship_type and degree:
                    rq = self._ensure_question(
                        f"What are the requirements for the {scholarship_type} scholarship for {degree} programs at AIT"
                    )
                elif degree:
                    rq = self._ensure_question(
                        f"What scholarship opportunities are available for {degree} programs at AIT"
                    )
                else:
                    rq = self._ensure_question(previous_user_turn)

                return {
                    "used": True,
                    "refined_query": rq,
                    "refinement_type": "clarification_reply",
                    "used_slots": used_slots,
                }

        # history/pronoun/follow-up case
        if self._re_pronoun.search(user) or (len(user.split()) <= 8 and history):
            if request_type == "deadlines":
                if program_phrase and intake:
                    rq = self._ensure_question(
                        f"When is the application deadline for {program_phrase} at AIT for the {intake} intake"
                    )
                elif program_phrase:
                    rq = self._ensure_question(
                        f"When is the application deadline for {program_phrase} at AIT"
                    )
                else:
                    rq = self._ensure_question(user)
                return {
                    "used": True,
                    "refined_query": rq,
                    "refinement_type": "pronoun_resolution" if self._re_pronoun.search(user) else "history_followup",
                    "used_slots": used_slots,
                }

            if request_type == "fees":
                ft = fee_type or "tuition fee"
                if program_phrase:
                    rq = self._ensure_question(f"What is the {ft} for {program_phrase} at AIT")
                    return {
                        "used": True,
                        "refined_query": rq,
                        "refinement_type": "pronoun_resolution" if self._re_pronoun.search(user) else "history_followup",
                        "used_slots": used_slots,
                    }

            if request_type == "program_info" and program_phrase:
                rq = self._ensure_question(f"Is {program_phrase} available at AIT")
                return {
                    "used": True,
                    "refined_query": rq,
                    "refinement_type": "pronoun_resolution" if self._re_pronoun.search(user) else "history_followup",
                    "used_slots": used_slots,
                }

            if request_type == "admission_requirements" and program_phrase:
                rq = self._ensure_question(
                    f"What are the admission requirements for {program_phrase} at AIT"
                )
                return {
                    "used": True,
                    "refined_query": rq,
                    "refinement_type": "pronoun_resolution" if self._re_pronoun.search(user) else "history_followup",
                    "used_slots": used_slots,
                }

        return {
            "used": False,
            "refined_query": "",
            "refinement_type": "none",
            "used_slots": [],
        }

    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        base = (self.base_url or "").rstrip("/")
        url = f"{base}/api/chat"

        messages = []
        sys_text = (self.system_prompt or "").strip()
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                **(self.extra_options or {}),
            },
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

    def build_output(self) -> Message:
        raw_text = self._extract_raw_text(self.input_message)
        payload = self._parse_payload(raw_text)

        if not isinstance(payload, dict) or not payload:
            error_result = {
                "_error": "Could not parse input_message into a dictionary payload.",
                "_raw_input": raw_text,
            }
            error_text = json.dumps(error_result, ensure_ascii=False, indent=2)
            self.status = error_text
            return Message(text=error_text, data=error_result)

        if not bool(self.enable_refinement):
            result = {
                "enabled": False,
                "used": False,
                "refined_query": "",
                "refinement_type": "disabled",
                "used_slots": [],
                "llm_metrics": {},
            }
            output_text = json.dumps(result, ensure_ascii=False, indent=2)
            self.status = "disabled"
            return Message(text=output_text, data=result)

        # only the key fields need to be sent to LLM
        llm_payload = {
            "original_question": str(payload.get("original_question", "") or "").strip(),
            "history_window": self._trim_history(str(payload.get("history_window", "") or "").strip()),
            "request_type": str(payload.get("request_type", "unknown") or "unknown").strip(),
            "resolved_slots": payload.get("resolved_slots", {}) or {},
            "required_general_slots": payload.get("required_general_slots", []) or [],
            "required_critical_slots": payload.get("required_critical_slots", []) or [],
            "clarification_tracking": payload.get("clarification_tracking", {}) or {},
        }

        prompt = (self.user_prompt_template or "").replace(
            "{{MERGED_PAYLOAD_JSON}}",
            json.dumps(llm_payload, ensure_ascii=False, indent=2),
        )

        env = self._call_ollama(prompt)
        parsed = self._extract_json(env.get("text", "")) or {}
        metrics = env.get("metrics", {})

        used = bool(parsed.get("used", False))
        refined_query = self._ensure_question(str(parsed.get("refined_query", "") or ""))
        refinement_type = str(parsed.get("refinement_type", "none") or "none").strip()
        used_slots = self._dedupe([str(x).strip() for x in parsed.get("used_slots", []) if str(x).strip()])

        allowed_types = {
            "none",
            "history_followup",
            "pronoun_resolution",
            "clarification_reply",
            "ellipsis",
            "other",
        }
        if refinement_type not in allowed_types:
            refinement_type = "other" if used else "none"

        # rescue weak-model misses
        if (not used) and self._needs_force_refinement(payload):
            fallback = self._fallback_refinement(payload)
            used = bool(fallback.get("used", False))
            refined_query = str(fallback.get("refined_query", "") or "").strip()
            refinement_type = str(fallback.get("refinement_type", "none") or "none").strip()
            used_slots = self._dedupe([str(x).strip() for x in fallback.get("used_slots", []) if str(x).strip()])

        if used and not refined_query:
            fallback = self._fallback_refinement(payload)
            used = bool(fallback.get("used", False))
            refined_query = str(fallback.get("refined_query", "") or "").strip()
            refinement_type = str(fallback.get("refinement_type", "none") or "none").strip()
            used_slots = self._dedupe([str(x).strip() for x in fallback.get("used_slots", []) if str(x).strip()])

        if not used:
            refined_query = ""
            refinement_type = "none"
            used_slots = []

        result = {
            "enabled": True,
            "used": used,
            "refined_query": refined_query,
            "refinement_type": refinement_type,
            "used_slots": used_slots,
            "llm_metrics": metrics,
        }

        output_text = json.dumps(result, ensure_ascii=False, indent=2)
        self.status = (
            f"used={int(used)} "
            f"type={refinement_type} "
            f"in={metrics.get('prompt_eval_count')} "
            f"out={metrics.get('eval_count')}"
        )
        return Message(text=output_text, data=result)