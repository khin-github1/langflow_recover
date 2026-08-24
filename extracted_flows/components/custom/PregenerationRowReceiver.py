# Recovered Langflow component
# type: PregenerationRowReceiver
# class: PregenerationRowReceiver
# used in 1 flow(s): Cognitive RAG V1.2.0 GPT
# json path: node.data.node.template.code.value

import json
import hashlib
from typing import Any, Dict, Optional

from lfx.custom import Component
from lfx.io import MessageTextInput, Output
from lfx.schema import Data
from lfx.schema.message import Message


class PregenerationRowReceiver(Component):
    display_name = "Pregeneration Row Receiver"
    description = "Receives one pregeneration JSONL row from API input and normalizes it into joined_record format."
    icon = "inbox"
    name = "PregenerationRowReceiver"

    inputs = [
        MessageTextInput(
            name="input_payload",
            display_name="Input Payload",
            info="One JSONL row sent from the external Python loop. Usually comes from Chat/Input value.",
            required=True,
        )
    ]

    outputs = [
        Output(
            display_name="Joined Record",
            name="joined_record",
            method="build_joined_record",
        ),
        Output(
            display_name="Generation Mode",
            name="generation_mode_message",
            method="build_generation_mode_message",
        ),
        Output(
            display_name="Prompt Message",
            name="prompt_message",
            method="build_prompt_message",
        ),
        Output(
            display_name="Analyzer Data",
            name="analyzer_data",
            method="build_analyzer_data",
        ),
    ]

    def _parse_input(self) -> Dict[str, Any]:
        raw = self.input_payload

        if raw is None:
            raise ValueError("input_payload is empty.")

        if isinstance(raw, dict):
            return raw

        if hasattr(raw, "data") and isinstance(raw.data, dict):
            return raw.data

        if hasattr(raw, "text"):
            raw = raw.text

        text = str(raw or "").strip()
        if not text:
            raise ValueError("input_payload text is empty.")

        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"input_payload is not valid JSON: {e}") from e

        if not isinstance(obj, dict):
            raise ValueError("input_payload JSON must be an object/dict.")

        return obj

    def _normalize_mode(self, value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("generation_mode", "")
        value = str(value or "").strip().lower()
        return "reason" if value == "reason" else "normal"

    def _extract_prompt_text(self, row: Dict[str, Any]) -> str:
        gp = row.get("generator_prompt", {})

        if isinstance(gp, dict):
            for key in ["raw_prompt_text", "prompt_text", "prompt", "text"]:
                val = gp.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()

        for key in ["raw_prompt_text", "prompt_text", "prompt", "text"]:
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        return ""

    def _extract_user_query(self, row: Dict[str, Any]) -> str:
        gp = row.get("generator_prompt", {})
        if isinstance(gp, dict):
            val = gp.get("user_query")
            if isinstance(val, str) and val.strip():
                return val.strip()

        analyzer = row.get("analyzer", {})
        if isinstance(analyzer, dict):
            val = analyzer.get("original_question")
            if isinstance(val, str) and val.strip():
                return val.strip()

        return ""

    def _extract_analyzer(self, row: Dict[str, Any]) -> Dict[str, Any]:
        analyzer = row.get("analyzer", {})
        if isinstance(analyzer, dict):
            return analyzer

        # Some early-stage rows may not have analyzer.
        # Return minimal analyzer so downstream verifier does not crash immediately.
        user_query = self._extract_user_query(row)
        return {
            "original_question": user_query,
            "expected_general_slots": [],
            "expected_critical_slots": [],
        }

    def _extract_custom_id(self, row: Dict[str, Any], mode: str, prompt_text: str) -> str:
        for key in ["custom_id", "_expected_custom_id", "id", "eval_id", "sample_id"]:
            val = row.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()

        # Stable fallback custom_id from prompt content.
        h = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:12]
        return f"{mode}_{h}"

    def _extract_answer_text_if_any(self, row: Dict[str, Any]) -> str:
        # For pre-GPT rows this will be empty.
        # For replay/verification rows, Python may attach one of these.
        for key in [
            "batch_answer_text",
            "generated_answer",
            "assistant_answer",
            "gpt_answer",
            "answer_text",
            "answer",
        ]:
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # OpenAI batch output shape fallback
        response = row.get("response")
        if isinstance(response, dict):
            body = response.get("body")
            if isinstance(body, dict):
                choices = body.get("choices")
                if isinstance(choices, list) and choices:
                    msg = (choices[0] or {}).get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()

        return ""

    def _build_joined(self) -> Dict[str, Any]:
        row = self._parse_input()

        mode = self._normalize_mode(row.get("generation_mode", "normal"))
        prompt_text = self._extract_prompt_text(row)
        analyzer_data = self._extract_analyzer(row)
        custom_id = self._extract_custom_id(row, mode, prompt_text)
        answer_text = self._extract_answer_text_if_any(row)

        if not prompt_text:
            raise ValueError(
                "Could not extract generator prompt text. Expected generator_prompt.raw_prompt_text."
            )

        joined = {
            "custom_id": custom_id,
            "generation_mode": mode,
            "generator_prompt_text": prompt_text,
            "analyzer_data": analyzer_data,
            "batch_answer_text": answer_text,
            "pregeneration_record": row,
            "batch_output_record": row if answer_text else {},
        }

        self.status = (
            f"received custom_id={custom_id} | "
            f"mode={mode} | "
            f"has_answer={bool(answer_text)}"
        )

        return joined

    def build_joined_record(self) -> Data:
        joined = self._build_joined()
        return Data(
            data=joined,
            text_key="custom_id",
            default_value=joined.get("custom_id", ""),
        )

    def build_generation_mode_message(self) -> Message:
        joined = self._build_joined()
        return Message(
            text=json.dumps(
                {"generation_mode": joined.get("generation_mode", "normal")},
                ensure_ascii=False,
            )
        )

    def build_prompt_message(self) -> Message:
        joined = self._build_joined()
        return Message(text=joined.get("generator_prompt_text", ""))

    def build_analyzer_data(self) -> Data:
        joined = self._build_joined()
        return Data(
            data=joined.get("analyzer_data", {}),
            text_key="original_question",
            default_value="",
        )