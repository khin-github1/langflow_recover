# Recovered Langflow component
# type: GPTPairReceiver
# class: GPTPairReceiver
# used in 4 flow(s): Cognitive RAG V1.3.0 GPT, Cognitive RAG V1.3.0 GPT Loose AIT , Cognitive RAG V1.3.0 GPT Strict, Cognitive RAG V1.3.0 GPT Strict AIT
# json path: node.data.node.template.code.value

import json
from typing import Any, Dict, Optional

from lfx.custom import Component
from lfx.io import MessageTextInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message


class GPTPairReceiver(Component):
    display_name = "GPT Pair Receiver"
    description = "Receives one pregeneration row and one matching GPT batch-output row, then emits joined_record."
    icon = "merge"
    name = "GPTPairReceiver"

    inputs = [
        MessageTextInput(
            name="input_payload",
            display_name="Input Payload",
            info=(
                "JSON object containing one pregeneration_payload row and one batch_output row. "
                "Expected keys: pregeneration_payload, batch_output."
            ),
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
            display_name="Generation Mode Message",
            name="generation_mode_message",
            method="build_generation_mode_message",
        ),
        Output(
            display_name="Analyzer Data",
            name="analyzer_data",
            method="build_analyzer_data",
        ),
        Output(
            display_name="Prompt Message",
            name="prompt_message",
            method="build_prompt_message",
        ),
        Output(
            display_name="Generated Answer Message",
            name="generated_answer_message",
            method="build_generated_answer_message",
        ),
    ]

    # -----------------------------
    # Basic parsing helpers
    # -----------------------------
    def _raw_to_dict(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, dict):
            return value

        if hasattr(value, "data") and isinstance(value.data, dict):
            return value.data

        if hasattr(value, "text"):
            value = value.text

        text = str(value or "").strip()
        if not text:
            return {}

        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Input is not valid JSON: {e}") from e

        if not isinstance(obj, dict):
            raise ValueError("Input JSON must be an object/dict.")

        return obj

    def _parse_pair(self) -> Dict[str, Any]:
        wrapper = self._raw_to_dict(self.input_payload)

        pregen = (
            wrapper.get("pregeneration_payload")
            or wrapper.get("pregeneration_record")
            or wrapper.get("payload")
            or wrapper.get("pregen")
        )

        batch = (
            wrapper.get("batch_output")
            or wrapper.get("batch_output_record")
            or wrapper.get("gpt_output")
            or wrapper.get("gpt_result")
        )

        if not isinstance(pregen, dict):
            raise ValueError(
                "Missing or invalid pregeneration payload. "
                "Expected wrapper['pregeneration_payload'] to be a dict."
            )

        if not isinstance(batch, dict):
            raise ValueError(
                "Missing or invalid GPT batch output. "
                "Expected wrapper['batch_output'] to be a dict."
            )

        return {
            "pregeneration_payload": pregen,
            "batch_output": batch,
        }

    # -----------------------------
    # Extraction helpers
    # -----------------------------
    def _extract_generation_mode(self, pregen: Dict[str, Any]) -> str:
        mode = pregen.get("generation_mode", "normal")

        if isinstance(mode, dict):
            mode = mode.get("generation_mode", "normal")

        mode = str(mode or "normal").strip().lower()
        return "reason" if mode == "reason" else "normal"

    def _extract_prompt_text(self, pregen: Dict[str, Any]) -> str:
        generator_prompt = pregen.get("generator_prompt", {})

        if isinstance(generator_prompt, dict):
            for key in ["raw_prompt_text", "prompt_text", "prompt", "text"]:
                value = generator_prompt.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        for key in ["raw_prompt_text", "prompt_text", "prompt", "text"]:
            value = pregen.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    def _extract_user_query(self, pregen: Dict[str, Any]) -> str:
        generator_prompt = pregen.get("generator_prompt", {})
        if isinstance(generator_prompt, dict):
            value = generator_prompt.get("user_query")
            if isinstance(value, str) and value.strip():
                return value.strip()

        analyzer = pregen.get("analyzer", {})
        if isinstance(analyzer, dict):
            value = analyzer.get("original_question")
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    def _extract_analyzer_data(self, pregen: Dict[str, Any]) -> Dict[str, Any]:
        analyzer = pregen.get("analyzer")

        if isinstance(analyzer, dict):
            return analyzer

        # Minimal fallback, useful for generation_prompt_stage_writer rows.
        return {
            "original_question": self._extract_user_query(pregen),
            "expected_general_slots": [],
            "expected_critical_slots": [],
        }

    def _extract_custom_id(self, pregen: Dict[str, Any], batch: Dict[str, Any], mode: str) -> str:
        # Prefer GPT batch custom_id because OpenAI batch output is keyed by it.
        for source in [batch, pregen]:
            for key in ["custom_id", "_expected_custom_id", "id", "eval_id", "sample_id"]:
                value = source.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()

        return f"{mode}_unknown"

    def _extract_batch_answer_text(self, batch: Dict[str, Any]) -> str:
        # OpenAI Batch output shape:
        # {
        #   "custom_id": "...",
        #   "response": {
        #      "body": {
        #         "choices": [
        #            {"message": {"content": "..."}}
        #         ]
        #      }
        #   }
        # }

        for key in [
            "batch_answer_text",
            "generated_answer",
            "assistant_answer",
            "gpt_answer",
            "answer_text",
            "answer",
        ]:
            value = batch.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        response = batch.get("response")
        if isinstance(response, dict):
            body = response.get("body")
            if isinstance(body, dict):
                choices = body.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0] or {}
                    if isinstance(first, dict):
                        message = first.get("message", {})
                        if isinstance(message, dict):
                            content = message.get("content")
                            if isinstance(content, str) and content.strip():
                                return content.strip()

        return ""

    def _extract_batch_status_code(self, batch: Dict[str, Any]) -> Optional[int]:
        response = batch.get("response")
        if isinstance(response, dict):
            status_code = response.get("status_code")
            try:
                return int(status_code)
            except Exception:
                return None
        return None

    def _extract_batch_metrics(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        response = batch.get("response")
        if not isinstance(response, dict):
            return {}

        body = response.get("body")
        if not isinstance(body, dict):
            return {}

        return {
            "model": body.get("model"),
            "created": body.get("created"),
            "usage": body.get("usage"),
            "service_tier": body.get("service_tier"),
            "system_fingerprint": body.get("system_fingerprint"),
            "openai_response_id": body.get("id"),
            "openai_object": body.get("object"),
        }

    # -----------------------------
    # Main normalized output
    # -----------------------------
    def _build_joined_dict(self) -> Dict[str, Any]:
        pair = self._parse_pair()
        pregen = pair["pregeneration_payload"]
        batch = pair["batch_output"]

        mode = self._extract_generation_mode(pregen)
        custom_id = self._extract_custom_id(pregen, batch, mode)
        prompt_text = self._extract_prompt_text(pregen)
        analyzer_data = self._extract_analyzer_data(pregen)
        answer_text = self._extract_batch_answer_text(batch)
        status_code = self._extract_batch_status_code(batch)
        batch_metrics = self._extract_batch_metrics(batch)

        if not prompt_text:
            raise ValueError(
                "Could not extract prompt text from pregeneration payload. "
                "Expected generator_prompt.raw_prompt_text."
            )

        if not answer_text:
            raise ValueError(
                "Could not extract GPT answer text from batch_output. "
                "Expected response.body.choices[0].message.content."
            )

        joined = {
            "custom_id": custom_id,
            "generation_mode": mode,

            # Existing downstream extractors should use these:
            "generator_prompt_text": prompt_text,
            "batch_answer_text": answer_text,
            "analyzer_data": analyzer_data,

            # Useful extras:
            "user_query": self._extract_user_query(pregen),
            "batch_status_code": status_code,
            "batch_metrics": batch_metrics,
            "batch_error": batch.get("error"),

            # Raw records for writers/debugging:
            "pregeneration_record": pregen,
            "batch_output_record": batch,
        }

        self.status = (
            f"custom_id={custom_id} | "
            f"mode={mode} | "
            f"status={status_code} | "
            f"answer_chars={len(answer_text)}"
        )

        return joined

    def build_joined_record(self) -> Data:
        joined = self._build_joined_dict()
        return Data(
            data=joined,
            text_key="custom_id",
            default_value=joined.get("custom_id", ""),
        )

    def build_generation_mode_message(self) -> Message:
        joined = self._build_joined_dict()
        return Message(
            text=json.dumps(
                {"generation_mode": joined.get("generation_mode", "normal")},
                ensure_ascii=False,
            )
        )

    def build_analyzer_data(self) -> Data:
        joined = self._build_joined_dict()
        return Data(
            data=joined.get("analyzer_data", {}),
            text_key="original_question",
            default_value="",
        )

    def build_prompt_message(self) -> Message:
        joined = self._build_joined_dict()
        return Message(text=joined.get("generator_prompt_text", ""))

    def build_generated_answer_message(self) -> Message:
        joined = self._build_joined_dict()
        return Message(text=joined.get("batch_answer_text", ""))