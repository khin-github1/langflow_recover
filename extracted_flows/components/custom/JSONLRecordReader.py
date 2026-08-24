# Recovered Langflow component
# type: JSONLRecordReader
# class: JSONLRecordReader
# used in 3 flow(s): Cognitive RAG V1.2.0 GPT, Cognitive RAG V1.2.5 GPT, Cognitive RAG V1.3.0 GPT Ingest
# json path: node.data.node.template.code.value

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lfx.custom import Component
from lfx.io import BoolInput, FileInput, IntInput, Output, StrInput
from lfx.schema import Data, Message


class JSONLRecordReader(Component):
    display_name = "JSONL Record Reader"
    description = (
        "Reads a .jsonl file and returns either one record by index or all records. "
        "Useful for replaying saved GPT pre-generation payloads."
    )
    icon = "file-text"
    name = "JSONLRecordReader"

    inputs = [
        FileInput(
            name="path",
            display_name="JSONL File",
            file_types=["jsonl", "txt"],
            required=False,
            info="Upload a .jsonl file here.",
        ),
        StrInput(
            name="file_path_str",
            display_name="File Path",
            required=False,
            advanced=True,
            tool_mode=True,
            info="Optional direct file path if you want to load from an existing saved path.",
        ),
        IntInput(
            name="record_index",
            display_name="Record Index",
            value=0,
            required=False,
            info="Which record to return. Default is 0.",
        ),
        BoolInput(
            name="return_all_records",
            display_name="Return All Records",
            value=False,
            required=False,
            info="If true, returns all parsed records in one Data output.",
        ),
        BoolInput(
            name="strict_json",
            display_name="Strict JSON",
            value=True,
            required=False,
            advanced=True,
            info="If true, invalid JSONL lines raise an error. If false, invalid lines are skipped.",
        ),
    ]

    outputs = [
        Output(
            display_name="Current Record Data",
            name="current_record_data",
            method="build_current_record_data",
        ),
        Output(
            display_name="Current Record Message",
            name="current_record_message",
            method="build_current_record_message",
        ),
        Output(
            display_name="All Records Data",
            name="all_records_data",
            method="build_all_records_data",
        ),
        Output(
            display_name="Metadata Message",
            name="metadata_message",
            method="build_metadata_message",
        ),
    ]

    # -------------------------
    # Helpers
    # -------------------------
    def _resolve_input_path(self) -> Path:
        # 1) explicit file path string
        raw_path = str(getattr(self, "file_path_str", "") or "").strip()
        if raw_path:
            p = Path(raw_path).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            return p

        # 2) uploaded file from FileInput
        uploaded = getattr(self, "path", None)

        if not uploaded:
            raise ValueError("No JSONL input provided. Upload a file or provide file_path_str.")

        if isinstance(uploaded, list):
            if not uploaded:
                raise ValueError("Uploaded file list is empty.")
            uploaded = uploaded[0]

        p = Path(str(uploaded)).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Uploaded file not found: {p}")
        return p

    def _read_jsonl(self) -> Dict[str, Any]:
        file_path = self._resolve_input_path()
        strict = bool(getattr(self, "strict_json", True))

        records: List[Dict[str, Any]] = []
        skipped_lines: List[int] = []

        with file_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                    else:
                        if strict:
                            raise ValueError(f"Line {lineno} is valid JSON but not an object.")
                        skipped_lines.append(lineno)
                except Exception:
                    if strict:
                        raise ValueError(f"Invalid JSONL at line {lineno}")
                    skipped_lines.append(lineno)

        return {
            "file_path": str(file_path),
            "record_count": len(records),
            "records": records,
            "skipped_lines": skipped_lines,
        }

    def _get_current_record(self) -> Dict[str, Any]:
        parsed = self._read_jsonl()
        records = parsed["records"]
        idx = int(getattr(self, "record_index", 0) or 0)

        if not records:
            return {
                "_error": "No valid records found in JSONL file.",
                "file_path": parsed["file_path"],
                "record_count": 0,
                "requested_index": idx,
                "skipped_lines": parsed["skipped_lines"],
            }

        if idx < 0 or idx >= len(records):
            return {
                "_error": f"record_index out of range: {idx}",
                "file_path": parsed["file_path"],
                "record_count": len(records),
                "requested_index": idx,
                "skipped_lines": parsed["skipped_lines"],
            }

        return records[idx]

    # -------------------------
    # Outputs
    # -------------------------
    def build_current_record_data(self) -> Data:
        record = self._get_current_record()
        self.status = f"Loaded record_index={int(getattr(self, 'record_index', 0) or 0)}"
        return Data(data=record, text_key="text", default_value="")

    def build_current_record_message(self) -> Message:
        record = self._get_current_record()
        text = json.dumps(record, ensure_ascii=False, indent=2)
        self.status = f"Loaded record_index={int(getattr(self, 'record_index', 0) or 0)}"
        return Message(text=text, data=record)

    def build_all_records_data(self) -> Data:
        parsed = self._read_jsonl()

        if not bool(getattr(self, "return_all_records", False)):
            payload = {
                "message": "return_all_records is false. Enable it to emit the full record list.",
                "file_path": parsed["file_path"],
                "record_count": parsed["record_count"],
                "skipped_lines": parsed["skipped_lines"],
            }
            return Data(data=payload, text_key="text", default_value="")

        payload = {
            "file_path": parsed["file_path"],
            "record_count": parsed["record_count"],
            "records": parsed["records"],
            "skipped_lines": parsed["skipped_lines"],
        }
        return Data(data=payload, text_key="text", default_value="")

    def build_metadata_message(self) -> Message:
        parsed = self._read_jsonl()
        payload = {
            "file_path": parsed["file_path"],
            "record_count": parsed["record_count"],
            "requested_index": int(getattr(self, "record_index", 0) or 0),
            "return_all_records": bool(getattr(self, "return_all_records", False)),
            "skipped_lines": parsed["skipped_lines"],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return Message(text=text, data=payload)