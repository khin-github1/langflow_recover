# Recovered Langflow component
# type: GPTBatchJoinerSingle
# class: GPTBatchJoinerSingle
# used in 6 flow(s): Cognitive RAG V1.2.0 GPT, Cognitive RAG V1.2.5 GPT, Cognitive RAG V1.3.0 GPT, Cognitive RAG V1.3.0 GPT Loose AIT , Cognitive RAG V1.3.0 GPT Strict, Cognitive RAG V1.3.0 GPT Strict AIT
# json path: node.data.node.template.code.value

import json
from typing import Any, Dict, List, Optional

from lfx.custom import Component
from lfx.io import DataInput, IntInput, BoolInput, Output
from lfx.schema import Data


class GPTBatchJoinerSingle(Component):
    display_name = "GPT Batch Joiner (Single Output)"
    description = "Matches one batch_output row with one pregeneration row and returns one joined Data object."
    icon = "link"
    name = "GPTBatchJoinerSingle"

    inputs = [
        DataInput(
            name="pregeneration_payload",
            display_name="Pregeneration Payload",
            required=True,
        ),
        DataInput(
            name="batch_output_payload",
            display_name="Batch Output Payload",
            required=True,
        ),
        IntInput(
            name="record_index",
            display_name="Record Index",
            value=0,
            required=False,
        ),
        BoolInput(
            name="match_by_custom_id",
            display_name="Match By custom_id",
            value=True,
            required=False,
        ),
    ]

    outputs = [
        Output(
            display_name="Joined Record",
            name="joined_record",
            method="build_joined_record",
        )
    ]

    def _as_dict(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, Data) and isinstance(value.data, dict):
            return value.data

        if hasattr(value, "data") and isinstance(value.data, dict):
            return value.data

        if hasattr(value, "text") and value.text:
            try:
                obj = json.loads(str(value.text).strip())
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        if isinstance(value, dict):
            return value

        return {}

    def _extract_records(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ["records", "rows", "items", "data"]:
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]

        if isinstance(payload, dict) and all(not isinstance(v, list) for v in payload.values()):
            return [payload]

        return []

    def _normalize_mode(self, value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("generation_mode", "")
        value = str(value or "").strip().lower()
        return "reason" if value == "reason" else "normal"

    def _extract_prompt_text(self, pregen_record: Dict[str, Any]) -> str:
        gp = pregen_record.get("generator_prompt", {})
        if isinstance(gp, dict):
            raw = str(gp.get("raw_prompt_text", "") or "").strip()
            if raw:
                return raw

        for key in ["raw_prompt_text", "prompt_text", "prompt", "text"]:
            val = pregen_record.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        return ""

    def _batch_ready_pregen_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in records:
            if not isinstance(r, dict):
                continue
            if "analyzer" not in r:
                continue
            if "generator_prompt" not in r:
                continue
            if "generation_mode" not in r:
                continue

            gp = r.get("generator_prompt", {})
            has_prompt = False
            if isinstance(gp, dict) and str(gp.get("raw_prompt_text", "") or "").strip():
                has_prompt = True

            if not has_prompt:
                continue

            out.append(r)
        return out

    def _attach_expected_custom_ids(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for idx, r in enumerate(records, start=1):
            mode = self._normalize_mode(r.get("generation_mode"))
            rr = dict(r)
            rr["_expected_custom_id"] = f"{mode}_{idx:06d}"
            rr["_expected_generation_mode"] = mode
            out.append(rr)
        return out

    def _extract_answer_text(self, batch_record: Dict[str, Any]) -> str:
        response = batch_record.get("response", {})
        if not isinstance(response, dict):
            return ""

        body = response.get("body", {})
        if not isinstance(body, dict):
            return ""

        choices = body.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""

        first = choices[0]
        if not isinstance(first, dict):
            return ""

        msg = first.get("message", {})
        if not isinstance(msg, dict):
            return ""

        return str(msg.get("content", "") or "").strip()

    def build_joined_record(self) -> Data:
        pregen_payload = self._as_dict(self.pregeneration_payload)
        batch_payload = self._as_dict(self.batch_output_payload)

        pregen_records_raw = self._extract_records(pregen_payload)
        batch_records = self._extract_records(batch_payload)

        pregen_records = self._attach_expected_custom_ids(
            self._batch_ready_pregen_records(pregen_records_raw)
        )

        if not pregen_records:
            raise ValueError("No batch-ready pregeneration records found.")
        if not batch_records:
            raise ValueError("No batch output records found.")

        idx = int(self.record_index or 0)
        if idx < 0 or idx >= len(batch_records):
            raise ValueError(f"record_index {idx} out of range for batch records ({len(batch_records)}).")

        batch_record = batch_records[idx]

        if bool(self.match_by_custom_id):
            wanted = str(batch_record.get("custom_id", "") or "").strip()
            matched = None
            for r in pregen_records:
                if str(r.get("_expected_custom_id", "") or "").strip() == wanted:
                    matched = r
                    break
            if matched is None:
                raise ValueError(f"Could not match pregeneration row for batch custom_id={wanted}")
            pregen_record = matched
        else:
            if idx >= len(pregen_records):
                raise ValueError(f"record_index {idx} out of range for pregeneration records ({len(pregen_records)}).")
            pregen_record = pregen_records[idx]

        answer_text = self._extract_answer_text(batch_record)
        if not answer_text:
            raise ValueError("Matched batch row has no assistant answer text.")

        joined = {
            "record_index": idx,
            "custom_id": str(batch_record.get("custom_id", "") or "").strip(),
            "generation_mode": pregen_record.get("_expected_generation_mode", "normal"),
            "generator_prompt_text": self._extract_prompt_text(pregen_record),
            "analyzer_data": pregen_record.get("analyzer", {}),
            "batch_answer_text": answer_text,
            "pregeneration_record": pregen_record,
            "batch_output_record": batch_record,
        }

        self.status = f"joined custom_id={joined['custom_id']}"
        return Data(text_key="custom_id", data=joined, default_value="")