# Recovered Langflow component
# type: JoinedRowToGPTEnvelope
# class: JoinedRowToGPTEnvelope
# used in 10 flow(s): Cognitive RAG V1.2.0 GPT, Cognitive RAG V1.2.5 GPT, Cognitive RAG V1.3.0 GPT, Cognitive RAG V1.3.0 GPT Loose AIT , Cognitive RAG V1.3.0 GPT Strict, Cognitive RAG V1.3.0 GPT Strict AIT, Retry RAG V1.3.0 GPT, Retry RAG V1.3.0 GPT Loose AIT ...
# json path: node.data.node.template.code.value

import json
from typing import Any, Dict

from lfx.custom import Component
from lfx.io import DataInput, Output
from lfx.schema import Data


class JoinedRowToGPTEnvelope(Component):
    display_name = "Joined Row → GPT Envelope"
    description = "Builds a generator-like envelope Data from a joined GPT batch row."
    icon = "package"
    name = "JoinedRowToGPTEnvelope"

    inputs = [
        DataInput(
            name="joined_record",
            display_name="Joined Record",
            required=True,
        )
    ]

    outputs = [
        Output(
            display_name="Envelope",
            name="envelope",
            method="build_envelope",
        )
    ]

    def _payload(self) -> Dict[str, Any]:
        if isinstance(self.joined_record, Data):
            return self.joined_record.data or {}
        if hasattr(self.joined_record, "data") and isinstance(self.joined_record.data, dict):
            return self.joined_record.data
        if isinstance(self.joined_record, dict):
            return self.joined_record
        return {}

    def build_envelope(self) -> Data:
        joined = self._payload()
        batch_record = joined.get("batch_output_record", {}) or {}
        generation_mode = str(joined.get("generation_mode", "normal") or "normal").strip().lower()
        answer_text = str(joined.get("batch_answer_text", "") or "").strip()
        prompt_text = str(joined.get("generator_prompt_text", "") or "").strip()

        response = batch_record.get("response", {}) if isinstance(batch_record, dict) else {}
        body = response.get("body", {}) if isinstance(response, dict) else {}
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        completion_details = usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}

        env = {
            "text": answer_text,
            "thinking": "",
            "metrics": {
                "model": body.get("model"),
                "created_at": body.get("created"),
                "done": response.get("status_code") == 200,
                "done_reason": (
                    body.get("choices", [{}])[0].get("finish_reason")
                    if isinstance(body.get("choices"), list) and body.get("choices")
                    else None
                ),
                "prompt_eval_count": int(usage.get("prompt_tokens", 0) or 0),
                "eval_count": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                "reasoning_tokens": int(completion_details.get("reasoning_tokens", 0) or 0),
                "status_code": response.get("status_code"),
                "request_id": response.get("request_id"),
                "requested_think": generation_mode == "reason",
                "thinking_used": generation_mode == "reason",
                "observed_has_thinking_field": int(completion_details.get("reasoning_tokens", 0) or 0) > 0,
            },
            "raw": batch_record,
            "generation_mode": generation_mode,
            "thinking_used": generation_mode == "reason",
            "used_reasoning": generation_mode == "reason",
            "effective_think": generation_mode == "reason",
            "has_thinking": int(completion_details.get("reasoning_tokens", 0) or 0) > 0,
            "used_prompt": prompt_text,
            "source": "openai_batch_output",
        }

        return Data(text_key="text", data=env, default_value="")