# Recovered Langflow component
# type: UnifiedFirstPassWriter
# class: UnifiedFirstPassWriter
# used in 1 flow(s): Cognitive RAG V1.2.5 GPT
# json path: node.data.node.template.code.value

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

from lfx.custom.custom_component.component import Component
from lfx.io import BoolInput, DataInput, MessageTextInput, Output
from lfx.schema.message import Message


class UnifiedFirstPassWriter(Component):
    display_name = "Unified First Pass Writer"
    description = (
        "Loop-safe first-pass writer. Writes accepted/retry/rejected internally "
        "and always returns one Message to continue the Loop."
    )
    icon = "file-text"
    name = "UnifiedFirstPassWriter"

    inputs = [
        DataInput(
            name="joined_record",
            display_name="Joined Record",
            required=True,
        ),
        DataInput(
            name="verification_output",
            display_name="Verification Output",
            required=True,
        ),
        MessageTextInput(
            name="support_gate_output",
            display_name="Support Gate Output",
            required=True,
        ),
        MessageTextInput(
            name="accepted_file",
            display_name="Accepted JSONL File",
            value="/mnt/data/1stGPTAcceptData.jsonl",
            required=True,
        ),
        MessageTextInput(
            name="retry_file",
            display_name="Retry JSONL File",
            value="/mnt/data/1stGPTRetryInput.jsonl",
            required=True,
        ),
        MessageTextInput(
            name="rejected_file",
            display_name="Rejected JSONL File",
            value="/mnt/data/1stGPTRejectData.jsonl",
            required=True,
        ),
        BoolInput(
            name="append_mode",
            display_name="Append Mode",
            value=True,
            required=False,
        ),
    ]

    outputs = [
        Output(
            display_name="Message",
            name="message",
            method="write_record",
        )
    ]

    def _dict_from_data(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if hasattr(value, "data") and isinstance(value.data, dict):
            return value.data

        if isinstance(value, dict):
            return value

        if hasattr(value, "text") and value.text:
            text = str(value.text).strip()
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return {"text": text}

        return {}

    def _dict_from_text(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, dict):
            return value

        if hasattr(value, "text") and value.text is not None:
            text = str(value.text).strip()
        else:
            text = str(value or "").strip()

        if not text:
            return {}

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        return {"text": text}

    def _append_jsonl(self, file_path: str, record: Dict[str, Any]) -> None:
        file_path = str(file_path or "").strip()
        if not file_path:
            raise ValueError("Output file path is empty.")

        folder = os.path.dirname(file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        mode = "a" if bool(self.append_mode) else "w"

        with open(file_path, mode, encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _extract_answer_text(self, joined: Dict[str, Any]) -> str:
        for key in ["batch_answer_text", "generated_answer", "answer_text"]:
            val = joined.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        try:
            return (
                joined.get("batch_output_record", {})
                .get("response", {})
                .get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except Exception:
            return ""

    def _normalize_decision(self, gate: Dict[str, Any]) -> str:
        decision = str(gate.get("decision", "") or "").strip().lower()

        if decision in {"accept", "accepted"}:
            return "accepted"

        if decision in {"retry", "needs_retry"}:
            return "retry"

        if decision in {"reject", "rejected", "abstain", "abstained"}:
            return "rejected"

        # Safe fallback if decision is missing.
        try:
            score = float(gate.get("final_support_score"))
            threshold = float(gate.get("final_support_threshold", 0.75))
            mode = str(gate.get("generation_mode_in", "normal")).strip().lower()

            if score >= threshold:
                return "accepted"
            if mode == "normal":
                return "retry"
            return "rejected"
        except Exception:
            return "rejected"

    def _build_retry_batch_record(
        self,
        joined: Dict[str, Any],
        verification: Dict[str, Any],
        gate: Dict[str, Any],
    ) -> Dict[str, Any]:
        first_custom_id = str(joined.get("custom_id", "") or "").strip()

        retry_custom_id = first_custom_id.replace("normal_", "retry_", 1)
        if retry_custom_id == first_custom_id:
            retry_custom_id = f"retry_{first_custom_id}"

        original_prompt = str(joined.get("generator_prompt_text", "") or "").strip()

        retry_prompt = (
            "The previous answer did not pass evidence verification.\n"
            "Regenerate the answer using ONLY the provided context.\n"
            "Do not add unsupported details. If the context is insufficient, say so.\n\n"
            "ORIGINAL GENERATOR PROMPT:\n"
            f"{original_prompt}\n\n"
            "FIRST SUPPORT GATE OUTPUT:\n"
            f"{json.dumps(gate, ensure_ascii=False, indent=2)}"
        )

        # This is NOT OpenAI batch API format yet.
        # It is a clean retry staging record.
        # Convert it to real GPT batch input later.
        return {
            "record_type": "first_pass_retry_staging_record",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "first_custom_id": first_custom_id,
            "custom_id": retry_custom_id,
            "next_generation_mode": gate.get("next_generation_mode", "reason"),
            "retry_prompt": retry_prompt,
            "first_joined_record": joined,
            "first_verification_output": verification,
            "first_support_gate_output": gate,
        }

    def write_record(self) -> Message:
        joined = self._dict_from_data(self.joined_record)
        verification = self._dict_from_data(self.verification_output)
        gate = self._dict_from_text(self.support_gate_output)

        route = self._normalize_decision(gate)
        answer_text = self._extract_answer_text(joined)

        base_record = {
            "record_type": "first_pass_result",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "route": route,
            "record_index": joined.get("record_index"),
            "custom_id": joined.get("custom_id"),
            "generation_mode": joined.get("generation_mode"),
            "answer_text": answer_text,
            "joined_record": joined,
            "verification_output": verification,
            "support_gate_output": gate,
        }

        if route == "accepted":
            output_file = self.accepted_file
            self._append_jsonl(output_file, base_record)

        elif route == "retry":
            output_file = self.retry_file
            retry_record = self._build_retry_batch_record(joined, verification, gate)
            self._append_jsonl(output_file, retry_record)

        else:
            output_file = self.rejected_file
            self._append_jsonl(output_file, base_record)

        result = {
            "status": "written",
            "route": route,
            "custom_id": joined.get("custom_id"),
            "output_file": output_file,
        }

        self.status = json.dumps(result, ensure_ascii=False)

        return Message(text=json.dumps(result, ensure_ascii=False))