# Recovered Langflow component
# type: MinimalJSONLIngestLoader
# class: MinimalJSONLIngestLoader
# used in 1 flow(s): Data Ingestion
# json path: node.data.node.template.code.value

import json
from pathlib import Path
from typing import Any

from lfx.custom import Component
from lfx.io import BoolInput, IntInput, Output, StrInput
from lfx.schema import Data


class MinimalJSONLIngestLoader(Component):
    display_name = "Minimal JSONL Ingest Loader"
    description = "Loads minimal JSONL rows with doc_id, source_id, and text for PGVector ingestion."
    icon = "file-json"
    name = "MinimalJSONLIngestLoader"

    inputs = [
        StrInput(
            name="file_path",
            display_name="JSONL File Path",
            info="Path to squad_ingest.jsonl or ragtruth_ingest.jsonl inside the Langflow environment.",
            required=True,
        ),
        StrInput(
            name="text_key",
            display_name="Text Key",
            value="text",
            info="Column/key containing the text to embed.",
        ),
        StrInput(
            name="doc_id_key",
            display_name="Document ID Key",
            value="doc_id",
            info="Column/key containing the unique chunk ID.",
        ),
        StrInput(
            name="source_id_key",
            display_name="Source ID Key",
            value="source_id",
            info="Column/key containing the original source/document ID.",
        ),
        BoolInput(
            name="deduplicate_doc_id",
            display_name="Deduplicate by doc_id",
            value=True,
            info="Skip duplicate doc_id rows if any exist.",
        ),
        IntInput(
            name="limit",
            display_name="Limit Rows",
            value=0,
            info="0 means load all rows. Use a small number like 10 for testing.",
        ),
    ]

    outputs = [
        Output(
            display_name="Ingest Data",
            name="ingest_data",
            method="load_jsonl",
        ),
    ]

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()

    def load_jsonl(self) -> list[Data]:
        file_path = Path(str(self.file_path)).expanduser()

        if not file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        rows: list[Data] = []
        seen_doc_ids: set[str] = set()

        text_key = self.text_key or "text"
        doc_id_key = self.doc_id_key or "doc_id"
        source_id_key = self.source_id_key or "source_id"

        limit = int(self.limit or 0)

        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e

                doc_id = self._clean_text(obj.get(doc_id_key, ""))
                source_id = self._clean_text(obj.get(source_id_key, ""))
                text = self._clean_text(obj.get(text_key, ""))

                if not doc_id:
                    raise ValueError(f"Missing doc_id at line {line_no}")

                if not source_id:
                    raise ValueError(f"Missing source_id at line {line_no}")

                if not text:
                    continue

                if self.deduplicate_doc_id and doc_id in seen_doc_ids:
                    continue

                seen_doc_ids.add(doc_id)

                # Langflow/PGVector should embed the text field.
                # doc_id/source_id are kept only for traceability.
                rows.append(
                    Data(
                        data={
                            "text": text,
                            "doc_id": doc_id,
                            "source_id": source_id,
                        }
                    )
                )

                if limit > 0 and len(rows) >= limit:
                    break

        self.status = f"Loaded {len(rows)} rows from {file_path.name}"
        return rows