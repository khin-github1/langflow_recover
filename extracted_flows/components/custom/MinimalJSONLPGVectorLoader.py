# Recovered Langflow component
# type: MinimalJSONLPGVectorLoader
# class: MinimalJSONLPGVectorLoader
# used in 1 flow(s): Data Ingestion
# json path: node.data.node.template.code.value

import json
from pathlib import Path
from typing import Any, List

from lfx.custom import Component
from lfx.io import BoolInput, FileInput, IntInput, Output, StrInput
from lfx.schema import Data, Message


class MinimalJSONLPGVectorLoader(Component):
    display_name = "Minimal JSONL PGVector Loader"
    description = "Reads minimal JSONL rows with doc_id, source_id, text and emits one Data object per row for PGVector ingestion."
    icon = "file-json"
    name = "MinimalJSONLPGVectorLoader"

    inputs = [
        FileInput(
            name="path",
            display_name="JSONL File",
            file_types=["jsonl", "txt"],
            required=False,
            info="Upload squad_ingest.jsonl or ragtruth_ingest.jsonl.",
        ),
        StrInput(
            name="file_path_str",
            display_name="File Path",
            required=False,
            advanced=True,
            tool_mode=True,
            info="Optional direct file path inside the Langflow environment.",
        ),
        StrInput(
            name="text_key",
            display_name="Text Key",
            value="text",
            required=False,
        ),
        StrInput(
            name="doc_id_key",
            display_name="Document ID Key",
            value="doc_id",
            required=False,
        ),
        StrInput(
            name="source_id_key",
            display_name="Source ID Key",
            value="source_id",
            required=False,
        ),
        BoolInput(
            name="deduplicate_doc_id",
            display_name="Deduplicate by doc_id",
            value=True,
            required=False,
        ),
        IntInput(
            name="limit",
            display_name="Limit Rows",
            value=0,
            required=False,
            info="0 means load all rows. Use 10 for testing first.",
        ),
    ]

    outputs = [
        Output(
            display_name="Ingest Data",
            name="ingest_data",
            method="build_ingest_data",
        ),
        Output(
            display_name="Metadata Message",
            name="metadata_message",
            method="build_metadata_message",
        ),
    ]

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()

    def _resolve_input_path(self) -> Path:
        raw_path = str(getattr(self, "file_path_str", "") or "").strip()

        if raw_path:
            file_path = Path(raw_path).expanduser()
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            return file_path

        uploaded = getattr(self, "path", None)

        if not uploaded:
            raise ValueError("No JSONL file provided. Upload a file or provide file_path_str.")

        if isinstance(uploaded, list):
            if not uploaded:
                raise ValueError("Uploaded file list is empty.")
            uploaded = uploaded[0]

        file_path = Path(str(uploaded)).expanduser()

        if not file_path.exists():
            raise FileNotFoundError(f"Uploaded file not found: {file_path}")

        return file_path

    def _read_rows(self) -> dict:
        file_path = self._resolve_input_path()

        text_key = self.text_key or "text"
        doc_id_key = self.doc_id_key or "doc_id"
        source_id_key = self.source_id_key or "source_id"

        limit = int(self.limit or 0)

        rows: List[Data] = []
        seen_doc_ids = set()

        total_lines = 0
        skipped_empty_text = 0
        skipped_duplicate = 0

        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                total_lines += 1

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e

                if not isinstance(obj, dict):
                    raise ValueError(f"Line {line_no} is valid JSON but not an object.")

                doc_id = self._clean_text(obj.get(doc_id_key, ""))
                source_id = self._clean_text(obj.get(source_id_key, ""))
                text = self._clean_text(obj.get(text_key, ""))

                if not doc_id:
                    raise ValueError(f"Missing doc_id at line {line_no}")

                if not source_id:
                    raise ValueError(f"Missing source_id at line {line_no}")

                if not text:
                    skipped_empty_text += 1
                    continue

                if self.deduplicate_doc_id and doc_id in seen_doc_ids:
                    skipped_duplicate += 1
                    continue

                seen_doc_ids.add(doc_id)

                rows.append(
                    Data(
                        data={
                            "text": text,
                            "doc_id": doc_id,
                            "source_id": source_id,
                        },
                        text_key="text",
                    )
                )

                if limit > 0 and len(rows) >= limit:
                    break

        return {
            "file_path": str(file_path),
            "total_lines": total_lines,
            "loaded_rows": len(rows),
            "skipped_empty_text": skipped_empty_text,
            "skipped_duplicate": skipped_duplicate,
            "rows": rows,
        }

    def build_ingest_data(self) -> List[Data]:
        parsed = self._read_rows()

        self.status = (
            f"Loaded {parsed['loaded_rows']} rows from "
            f"{Path(parsed['file_path']).name}"
        )

        return parsed["rows"]

    def build_metadata_message(self) -> Message:
        parsed = self._read_rows()

        payload = {
            "file_path": parsed["file_path"],
            "total_lines": parsed["total_lines"],
            "loaded_rows": parsed["loaded_rows"],
            "skipped_empty_text": parsed["skipped_empty_text"],
            "skipped_duplicate": parsed["skipped_duplicate"],
        }

        return Message(
            text=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
        )