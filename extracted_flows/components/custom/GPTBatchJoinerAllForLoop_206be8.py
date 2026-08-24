# Recovered Langflow component
# type: GPTBatchJoinerAllForLoop
# class: GPTBatchJoinerAllForLoop
# used in 1 flow(s): Cognitive RAG V1.2.5 GPT
# json path: node.data.node.template.code.value

import json
from typing import Any, Dict, List, Optional

from lfx.custom.custom_component.component import Component
from lfx.io import BoolInput, DataInput, IntInput, Output
from lfx.schema.data import Data


class GPTBatchJoinerAllForLoop(Component):
    display_name = "GPT Batch Joiner All For Loop"
    description = "Joins all GPT batch output rows with pregeneration rows and returns loop-ready records."
    icon = "repeat"
    name = "GPTBatchJoinerAllForLoop"

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
        BoolInput(
            name="match_by_custom_id",
            display_name="Match By custom_id",
            value=True,
            required=False,
        ),
        IntInput(
            name="max_records",
            display_name="Max Records",
            value=0,
            required=False,
            info="0 means process all records.",
        ),
    ]

    outputs = [
        Output(
            display_name="Loop Input",
            name="loop_input",
            method="build_loop_input",
        )
    ]

    def _as_dict(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, Data) and isinstance(value.data, dict):
            return value.data

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
                if isinstance(obj, list):
                    return {"records": obj}
            except Exception:
                pass

        if isinstance(value, list):
            return {"records": value}

        return {}

    def _extract_records(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []

        for key in ["records", "rows", "items", "data", "text"]:
            val = payload.get(key)

            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]

            if isinstance(val, str) and val.strip():
                try:
                    obj = json.loads(val)
                    if isinstance(obj, list):
                        return [x for x in obj if isinstance(x, dict)]
                    if isinstance(obj, dict):
                        return [obj]
                except Exception:
                    pass

        if payload:
            return [payload]

        return []

    def _normalize_mode(self, value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("generation_mode", "")

        value = str(value or "").strip().lower()
        return "reason" if value == "reason" else "normal"

    def _extract_prompt_text(self, pregen_record: Dict[str, Any]) -> str:
        generator_prompt = pregen_record.get("generator_prompt", {})

        if isinstance(generator_prompt, dict):
            for key in ["raw_prompt_text", "prompt_text", "prompt", "text"]:
                val = generator_prompt.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()

        for key in ["raw_prompt_text", "prompt_text", "prompt", "text"]:
            val = pregen_record.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        return ""

    def _batch_ready_pregen_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ready = []

        for record in records:
            if not isinstance(record, dict):
                continue

            if "analyzer" not in record:
                continue

            if "generation_mode" not in record:
                continue

            if not self._extract_prompt_text(record):
                continue

            ready.append(record)

        return ready

    def _attach_expected_custom_ids(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []

        for idx, record in enumerate(records, start=1):
            mode = self._normalize_mode(record.get("generation_mode"))
            row = dict(record)
            row["_expected_custom_id"] = f"{mode}_{idx:06d}"
            row["_expected_generation_mode"] = mode
            out.append(row)

        return out

    def _extract_answer_text(self, batch_record: Dict[str, Any]) -> str:
        try:
            answer = (
                batch_record.get("response", {})
                .get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
        except Exception:
            pass

        for key in ["batch_answer_text", "answer_text", "generated_answer", "content", "text"]:
            val = batch_record.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        return ""

    def _match_pregen_by_custom_id(
        self,
        pregen_records: List[Dict[str, Any]],
        custom_id: str,
    ) -> Optional[Dict[str, Any]]:
        wanted = str(custom_id or "").strip()

        for record in pregen_records:
            if str(record.get("_expected_custom_id", "") or "").strip() == wanted:
                return record

        return None

    def build_loop_input(self) -> Data:
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
            raise ValueError("No GPT batch output records found.")

        max_records = int(self.max_records or 0)
        if max_records > 0:
            batch_records = batch_records[:max_records]

        joined_records: List[Dict[str, Any]] = []
        skipped_records: List[Dict[str, Any]] = []

        for idx, batch_record in enumerate(batch_records):
            custom_id = str(batch_record.get("custom_id", "") or "").strip()

            if bool(self.match_by_custom_id):
                pregen_record = self._match_pregen_by_custom_id(pregen_records, custom_id)
                if pregen_record is None:
                    skipped_records.append(
                        {
                            "record_index": idx,
                            "custom_id": custom_id,
                            "reason": "no_matching_pregeneration_record",
                        }
                    )
                    continue
            else:
                if idx >= len(pregen_records):
                    skipped_records.append(
                        {
                            "record_index": idx,
                            "custom_id": custom_id,
                            "reason": "pregeneration_index_out_of_range",
                        }
                    )
                    continue
                pregen_record = pregen_records[idx]

            answer_text = self._extract_answer_text(batch_record)

            if not answer_text:
                skipped_records.append(
                    {
                        "record_index": idx,
                        "custom_id": custom_id,
                        "reason": "empty_answer_text",
                    }
                )
                continue

            joined_records.append(
                {
                    "record_index": idx,
                    "custom_id": custom_id,
                    "generation_mode": pregen_record.get("_expected_generation_mode", "normal"),
                    "generator_prompt_text": self._extract_prompt_text(pregen_record),
                    "analyzer_data": pregen_record.get("analyzer", {}),
                    "batch_answer_text": answer_text,
                    "pregeneration_record": pregen_record,
                    "batch_output_record": batch_record,
                }
            )

        if not joined_records:
            raise ValueError(
                "No joined records created. "
                f"Skipped records: {json.dumps(skipped_records, ensure_ascii=False)}"
            )

        payload = {
            "records": joined_records,
            "items": joined_records,
            "record_count": len(joined_records),
            "skipped_count": len(skipped_records),
            "skipped_records": skipped_records,
            "custom_ids": [r.get("custom_id") for r in joined_records],
        }

        self.status = (
            f"joined={len(joined_records)}, skipped={len(skipped_records)}, "
            f"custom_ids={payload['custom_ids']}"
        )

        return Data(
            data=payload,
            text_key="records",
            default_value="",
        )