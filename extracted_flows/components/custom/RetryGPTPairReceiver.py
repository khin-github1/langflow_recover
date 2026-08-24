# Recovered Langflow component
# type: RetryGPTPairReceiver
# class: RetryGPTPairReceiver
# used in 4 flow(s): Retry RAG V1.3.0 GPT, Retry RAG V1.3.0 GPT Loose AIT, Retry RAG V1.3.0 GPT Strict , Retry RAG V1.3.0 GPT Strict AIT 
# json path: node.data.node.template.code.value

import json
import re
from typing import Any, Dict, Optional

from lfx.custom import Component
from lfx.io import MessageTextInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message


class RetryGPTPairReceiver(Component):
    display_name = "Retry GPT Pair Receiver"
    description = (
        "Receives one retry-stage payload and one matching GPT retry batch-output row, "
        "then emits joined_record, analyzer_data, prompt_message, generated_answer, and generation_mode."
    )
    icon = "merge"
    name = "RetryGPTPairReceiver"

    inputs = [
        MessageTextInput(
            name="input_payload",
            display_name="Input Payload",
            info=(
                "JSON object from external runner. Expected keys: stage_payload and batch_output."
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

    def _parse_wrapper(self) -> Dict[str, Any]:
        wrapper = self._raw_to_dict(self.input_payload)

        stage_payload = (
            wrapper.get("stage_payload")
            or wrapper.get("retry_payload")
            or wrapper.get("pregeneration_payload")
            or wrapper.get("payload")
            or wrapper.get("pregen")
        )

        batch_output = (
            wrapper.get("batch_output")
            or wrapper.get("batch_output_record")
            or wrapper.get("gpt_output")
            or wrapper.get("gpt_result")
        )

        if not isinstance(stage_payload, dict):
            raise ValueError(
                "Missing or invalid stage_payload. "
                "Expected wrapper['stage_payload'] to be a dict."
            )

        if not isinstance(batch_output, dict):
            raise ValueError(
                "Missing or invalid batch_output. "
                "Expected wrapper['batch_output'] to be a dict."
            )

        return {
            "wrapper": wrapper,
            "stage_payload": stage_payload,
            "batch_output": batch_output,
        }

    # -----------------------------
    # Extraction helpers
    # -----------------------------

    def _extract_batch_custom_id(self, batch: Dict[str, Any]) -> str:
        return str(batch.get("custom_id", "") or "").strip()

    def _extract_original_custom_id_from_batch(self, batch_custom_id: str) -> str:
        """
        reason_retry_normal_000374 -> normal_000374
        reason_retry_reason_000123 -> reason_000123
        """
        batch_custom_id = str(batch_custom_id or "").strip()

        m = re.search(r"(normal|reason)_\d{6}$", batch_custom_id)
        if m:
            return m.group(0)

        return batch_custom_id

    def _extract_generation_mode(self, stage_payload: Dict[str, Any], batch: Dict[str, Any]) -> str:
        """
        Retry GPT rows should be treated as reason mode because they were generated
        by the retry/reason batch.
        """
        batch_custom_id = self._extract_batch_custom_id(batch).lower()

        if batch_custom_id.startswith("reason_retry"):
            return "reason"

        # Fallback to payload first attempt mode
        try:
            mode = stage_payload["first_attempt"]["joined"]["generation_mode"]
            mode = str(mode or "normal").strip().lower()
            if mode in {"normal", "reason"}:
                return mode
        except Exception:
            pass

        return "normal"

    def _extract_analyzer_data(self, stage_payload: Dict[str, Any]) -> Dict[str, Any]:
        analyzer = stage_payload.get("analyzer")
        if isinstance(analyzer, dict):
            return analyzer

        return {
            "original_question": "",
            "expected_general_slots": [],
            "expected_critical_slots": [],
        }

    def _extract_original_question(self, stage_payload: Dict[str, Any]) -> str:
        analyzer = self._extract_analyzer_data(stage_payload)
        value = analyzer.get("original_question")
        if isinstance(value, str):
            return value.strip()
        return ""

    def _extract_prompt_text(self, stage_payload: Dict[str, Any]) -> str:
        """
        For retry verification, use the original generator prompt/context from first attempt.
        This is what verifier should use as RAG context.
        """
        try:
            text = stage_payload["first_attempt"]["joined"]["generator_prompt_text"]
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            pass

        # Fallbacks
        first_attempt = stage_payload.get("first_attempt", {})
        if isinstance(first_attempt, dict):
            joined = first_attempt.get("joined", {})
            if isinstance(joined, dict):
                for key in ["generator_prompt_text", "raw_prompt_text", "prompt_text", "prompt", "text"]:
                    value = joined.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

        for key in ["generator_prompt_text", "raw_prompt_text", "prompt_text", "prompt", "text"]:
            value = stage_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        final_response = stage_payload.get("final_response")
        if isinstance(final_response, dict):
            value = final_response.get("text")
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    def _extract_batch_answer_text(self, batch: Dict[str, Any]) -> str:
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

        usage = body.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}

        completion_details = usage.get("completion_tokens_details", {})
        if not isinstance(completion_details, dict):
            completion_details = {}

        return {
            "model": body.get("model"),
            "created": body.get("created"),
            "usage": usage,
            "service_tier": body.get("service_tier"),
            "system_fingerprint": body.get("system_fingerprint"),
            "openai_response_id": body.get("id"),
            "openai_object": body.get("object"),
            "reasoning_tokens": int(completion_details.get("reasoning_tokens", 0) or 0),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    def _extract_first_attempt_summary(self, stage_payload: Dict[str, Any]) -> Dict[str, Any]:
        first_attempt = stage_payload.get("first_attempt")
        if not isinstance(first_attempt, dict):
            return {}

        generation = first_attempt.get("generation", {})
        support_gate = first_attempt.get("support_gate", {})
        verification = first_attempt.get("verification", {})

        return {
            "generation": generation if isinstance(generation, dict) else {},
            "support_gate": support_gate if isinstance(support_gate, dict) else {},
            "verification": verification if isinstance(verification, dict) else {},
        }

    # -----------------------------
    # Main normalized output
    # -----------------------------

    def _build_joined_dict(self) -> Dict[str, Any]:
        parsed = self._parse_wrapper()
        wrapper = parsed["wrapper"]
        stage_payload = parsed["stage_payload"]
        batch = parsed["batch_output"]

        batch_custom_id = self._extract_batch_custom_id(batch)
        original_custom_id = self._extract_original_custom_id_from_batch(batch_custom_id)

        mode = self._extract_generation_mode(stage_payload, batch)
        prompt_text = self._extract_prompt_text(stage_payload)
        analyzer_data = self._extract_analyzer_data(stage_payload)
        answer_text = self._extract_batch_answer_text(batch)
        status_code = self._extract_batch_status_code(batch)
        batch_metrics = self._extract_batch_metrics(batch)

        if not prompt_text:
            raise ValueError(
                "Could not extract prompt/context text from stage_payload. "
                "Expected first_attempt.joined.generator_prompt_text."
            )

        if not answer_text:
            raise ValueError(
                "Could not extract GPT retry answer text from batch_output. "
                "Expected response.body.choices[0].message.content."
            )

        joined = {
            "custom_id": batch_custom_id,
            "original_custom_id": original_custom_id,
            "generation_mode": mode,

            # Existing downstream extractors can use these:
            "generator_prompt_text": prompt_text,
            "batch_answer_text": answer_text,
            "analyzer_data": analyzer_data,

            # Useful extras:
            "user_query": self._extract_original_question(stage_payload),
            "batch_status_code": status_code,
            "batch_metrics": batch_metrics,
            "batch_error": batch.get("error"),

            # Retry-specific traceability:
            "is_retry_output": True,
            "first_attempt_summary": self._extract_first_attempt_summary(stage_payload),

            # Raw records for writers/debugging:
            "stage_payload_record": stage_payload,
            "pregeneration_record": stage_payload,
            "batch_output_record": batch,
            "wrapper_record": wrapper,
        }

        self.status = (
            f"custom_id={batch_custom_id} | "
            f"original={original_custom_id} | "
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
        payload = {"generation_mode": joined.get("generation_mode", "normal")}
        return Message(
            text=json.dumps(payload, ensure_ascii=False),
            data=payload,
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
        text = joined.get("generator_prompt_text", "")
        return Message(text=text, data={"text": text})

    def build_generated_answer_message(self) -> Message:
        joined = self._build_joined_dict()
        text = joined.get("batch_answer_text", "")
        return Message(text=text, data={"text": text})