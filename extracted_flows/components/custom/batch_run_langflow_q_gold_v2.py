# Recovered Langflow component
# type: batch_run_langflow_q_gold_v2
# class: BatchRunLangFlow
# used in 1 flow(s): CRCV Evaluation Runner 0.0.01
# json path: node.data.node.template.code.value

from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Iterable, Tuple

import requests

from langflow.custom.custom_component.component import Component
from langflow.io import (
    BoolInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    Output,
    SecretStrInput,
)
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame
from langflow.schema.message import Message


class BatchRunLangFlow(Component):
    display_name = "Batch Runner (Q + Gold Answer + Metrics)"
    description = (
        "Reads a CSV/DataFrame, uses Question column to call a LangFlow flow for each row, "
        "stores generated_answer + gold_answer + metrics into a results table and optional CSV."
    )
    icon = "repeat"
    name = "batch_run_langflow_q_gold_v2"

    inputs = [
        HandleInput(
            name="table",
            display_name="Input (Message CSV / Data CSV / DataFrame)",
            input_types=["Message", "DataFrame", "Data"],
            required=True,
        ),
        MessageTextInput(
            name="flow_run_url",
            display_name="Flow Run URL",
            info="Example: https://.../api/v1/run/<FLOW_ID>",
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="x-api-key",
            required=True,
        ),
        MessageTextInput(
            name="chat_input_node_id",
            display_name="ChatInput Node ID",
            info="Node ID in the TARGET (generation) flow. Example: ChatInput-4dvLW",
            required=True,
        ),
        MessageTextInput(
            name="payload_output_node_id",
            display_name="Payload Output Node ID",
            info="Node ID in the TARGET (generation) flow. Example: CustomComponent-YcKiJ",
            required=True,
        ),
        MessageTextInput(
            name="question_col",
            display_name="Question Column (RUN THIS)",
            value="Questions",
            required=True,
        ),
        MessageTextInput(
            name="gold_answer_col",
            display_name="Gold Answer Column (STORE THIS)",
            value="Answers",
            required=True,
        ),
        MessageTextInput(
            name="qid_col",
            display_name="QID Column (optional)",
            value="No",
            advanced=True,
        ),
        MessageTextInput(
            name="copy_cols",
            display_name="Pass-through Columns (comma-separated, * = all)",
            info="Extra columns to copy into output rows. Example: Source, Category. Use * to copy all.",
            value="Source, Category",
            advanced=True,
        ),
        IntInput(
            name="max_rows",
            display_name="Max Rows (0 = all)",
            value=0,
            advanced=True,
        ),
        IntInput(
            name="timeout_s",
            display_name="HTTP Timeout (seconds)",
            value=180,
            advanced=True,
        ),
        BoolInput(
            name="include_retrieval_chunks",
            display_name="Include Retrieval Chunks",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="include_generation_raw",
            display_name="Include Generation Raw",
            value=False,
            advanced=True,
        ),
        BoolInput(
            name="write_csv",
            display_name="Write CSV file",
            value=True,
            advanced=True,
        ),
        MessageTextInput(
            name="csv_filename",
            display_name="CSV filename",
            value="batch_results.csv",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Results DataFrame", name="results_df", method="run_batch"),
        Output(display_name="CSV File (Data)", name="csv_file", method="get_csv_file"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._csv_path: Optional[str] = None

    # -------------------------
    # Small helpers
    # -------------------------
    @staticmethod
    def _norm_key(s: str) -> str:
        return (s or "").strip().lower()

    @staticmethod
    def _safe_out_key(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_]+", "", s)
        return s or "col"

    def _get_cell(self, row: Dict[str, Any], col_name: str) -> Any:
        """Case-insensitive column access."""
        if not isinstance(row, dict):
            return None
        if col_name in row:
            return row.get(col_name)
        target = self._norm_key(col_name)
        for k, v in row.items():
            if self._norm_key(str(k)) == target:
                return v
        return None

    def _parse_copy_cols(self) -> List[str]:
        raw = (self.copy_cols or "").strip()
        if not raw:
            return []
        if raw == "*":
            return ["*"]
        cols = [c.strip() for c in raw.split(",")]
        return [c for c in cols if c]

    @staticmethod
    def _safe_json_dumps(x: Any) -> str:
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    @staticmethod
    def _get_api_key_value(x: Any) -> str:
        """
        SecretStrInput may return a SecretStr-like object.
        Make sure we always end up with a plain string.
        """
        if isinstance(x, str):
            return x
        # Pydantic SecretStr style
        if hasattr(x, "get_secret_value"):
            try:
                v = x.get_secret_value()
                return v if isinstance(v, str) else str(v)
            except Exception:
                pass
        return str(x)

    # -------------------------
    # CSV/DataFrame input parsing
    # -------------------------
    @staticmethod
    def _read_csv_text(csv_text: str) -> List[Dict[str, Any]]:
        f = io.StringIO(csv_text)
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]

    def _rows_from_input(self) -> List[Dict[str, Any]]:
        t = self.table

        if isinstance(t, DataFrame):
            return [row.to_dict() for _, row in t.iterrows()]

        if isinstance(t, Message):
            txt = (t.text or "").strip()
            return self._read_csv_text(txt) if txt else []

        if isinstance(t, Data) and isinstance(t.data, dict):
            txt = t.data.get("text")
            if isinstance(txt, str) and txt.strip():
                return self._read_csv_text(txt)

            fp = t.data.get("file_path")
            if isinstance(fp, str) and fp.strip() and os.path.exists(fp):
                # Use tolerant decode since Mac CSVs can contain smart quotes etc.
                # Try utf-8-sig first, then latin-1 fallback.
                try:
                    with open(fp, "r", encoding="utf-8-sig", errors="strict") as f:
                        return self._read_csv_text(f.read())
                except Exception:
                    with open(fp, "r", encoding="latin-1", errors="ignore") as f:
                        return self._read_csv_text(f.read())

        return []

    # -------------------------
    # LangFlow response parsing (robust)
    # -------------------------
    @staticmethod
    def _is_payload(d: Any) -> bool:
        return (
            isinstance(d, dict)
            and "timestamp_utc" in d
            and "user_text" in d
            and "assistant_text" in d
            and "metrics" in d
        )

    @staticmethod
    def _try_parse_json_text(s: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(s, str):
            return None
        ss = s.strip()
        if not (ss.startswith("{") and ss.endswith("}")):
            return None
        try:
            obj = json.loads(ss)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _deep_find_payload(self, obj: Any) -> Optional[Dict[str, Any]]:
        """
        Generic deep search:
        - returns dict matching payload schema
        - also accepts payload nested in {"data": {...payload...}}
        - also accepts JSON string that parses into payload dict
        """
        if self._is_payload(obj):
            return obj

        # JSON-in-text case
        parsed = self._try_parse_json_text(obj)
        if parsed and self._is_payload(parsed):
            return parsed

        if isinstance(obj, dict):
            # data wrapper case
            d = obj.get("data")
            if isinstance(d, dict) and self._is_payload(d):
                return d

            for v in obj.values():
                found = self._deep_find_payload(v)
                if found is not None:
                    return found

        if isinstance(obj, list):
            for it in obj:
                found = self._deep_find_payload(it)
                if found is not None:
                    return found

        return None

    @staticmethod
    def _iter_langflow_node_outputs(resp_json: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        """
        Typical Langflow run response:
          {"session_id": "...", "outputs": [ {"inputs":..., "outputs": [ {node_result...}, ... ] } ]}
        We yield each node output dict we can find.
        """
        outs = resp_json.get("outputs")
        if not isinstance(outs, list):
            return
        for block in outs:
            if not isinstance(block, dict):
                continue
            node_outs = block.get("outputs")
            if not isinstance(node_outs, list):
                continue
            for node in node_outs:
                if isinstance(node, dict):
                    yield node

    @staticmethod
    def _node_matches_id(node: Dict[str, Any], want_id: str) -> bool:
        if not want_id:
            return False
        want = want_id.strip()
        for k in ("component_id", "id", "vertex_id", "node_id", "name"):
            v = node.get(k)
            if isinstance(v, str) and v.strip() == want:
                return True
        return False

    def _extract_payload_from_run_response(self, resp_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        First try: locate the specific payload node by id in the Langflow response and search inside its results.
        Fallback: deep search anywhere.
        """
        want_id = (self.payload_output_node_id or "").strip()

        # Try targeting the payload node first
        for node in self._iter_langflow_node_outputs(resp_json):
            if self._node_matches_id(node, want_id):
                # common containers
                for key in ("results", "outputs", "result", "data"):
                    found = self._deep_find_payload(node.get(key))
                    if found is not None:
                        return found
                # last resort: deep search this node
                found = self._deep_find_payload(node)
                if found is not None:
                    return found

        # Fallback: deep search whole response
        return self._deep_find_payload(resp_json)

    # -------------------------
    # CSV writer
    # -------------------------
    def _write_csv(self, rows: List[Dict[str, Any]]) -> Optional[str]:
        if not rows:
            return None

        filename = (self.csv_filename or "batch_results.csv").strip() or "batch_results.csv"
        out_dir = "/mnt/data" if os.path.isdir("/mnt/data") else None
        path = os.path.join(out_dir, filename) if out_dir else tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name

        keys = set()
        for r in rows:
            keys.update(r.keys())
        fieldnames = sorted(keys)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

        return path

    # -------------------------
    # Main
    # -------------------------
    def run_batch(self) -> DataFrame:
        self._csv_path = None

        rows = self._rows_from_input()
        if not rows:
            return DataFrame([])

        max_rows = int(self.max_rows or 0)
        if max_rows > 0:
            rows = rows[:max_rows]

        api_key_val = self._get_api_key_value(self.api_key)
        headers = {"x-api-key": api_key_val}

        results: List[Dict[str, Any]] = []
        copy_cols = self._parse_copy_cols()
        copy_all = (len(copy_cols) == 1 and copy_cols[0] == "*")

        for idx, row in enumerate(rows):
            question = str(self._get_cell(row, self.question_col) or "").strip()
            gold_answer = self._get_cell(row, self.gold_answer_col)

            qid = None
            if (self.qid_col or "").strip():
                qid = self._get_cell(row, self.qid_col)

            session_id = str(uuid.uuid4())

            # IMPORTANT: output_type="any" so the API returns non-text outputs too
            req_payload = {
                "output_type": "any",
                "input_type": "chat",
                "session_id": session_id,
                "tweaks": {
                    self.chat_input_node_id: {
                        "input_value": question,
                        "session_id": session_id,
                    },
                    self.payload_output_node_id: {
                        "include_retrieval_chunks": bool(self.include_retrieval_chunks),
                        "include_generation_raw": bool(self.include_generation_raw),
                    },
                },
            }

            rec: Dict[str, Any] = {
                "row_index": idx,
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "generated_answer": "",
                "session_id": session_id,
                "ok": False,
                "error": None,
            }

            # pass-through columns
            if copy_all:
                for k, v in (row or {}).items():
                    outk = "in_" + self._safe_out_key(str(k))
                    if outk not in rec:
                        rec[outk] = v
            else:
                for c in copy_cols:
                    v = self._get_cell(row, c)
                    outk = "in_" + self._safe_out_key(c)
                    if outk not in rec:
                        rec[outk] = v

            if not question:
                rec["error"] = f"Empty question for row {idx} (question_col='{self.question_col}')"
                results.append(rec)
                continue

            try:
                resp = requests.post(
                    self.flow_run_url,
                    json=req_payload,
                    headers=headers,
                    timeout=int(self.timeout_s or 180),
                )
                resp.raise_for_status()
                j = resp.json()

                payload = self._extract_payload_from_run_response(j)

                if payload is None:
                    rec["error"] = (
                        "Payload not found in response. "
                        "Most common causes: (1) wrong flow_run_url, (2) payload node not executed (not wired to output path), "
                        "(3) wrong payload_output_node_id. "
                        "See raw_response_json."
                    )
                    rec["raw_response_json"] = self._safe_json_dumps(j)
                else:
                    rec["ok"] = True
                    rec["timestamp_utc"] = payload.get("timestamp_utc")
                    rec["generated_answer"] = payload.get("assistant_text", "")
                    rec["payload_json"] = self._safe_json_dumps(payload)

                    gen = payload.get("metrics", {}).get("generation", {})
                    if isinstance(gen, dict):
                        for k, v in gen.items():
                            rec[f"gen_{k}"] = v

                    ret = payload.get("metrics", {}).get("retrieval", {})
                    if isinstance(ret, dict):
                        for k, v in ret.items():
                            rec[f"ret_{k}"] = self._safe_json_dumps(v) if isinstance(v, (list, dict)) else v

                    rchunks = payload.get("retrieval_chunks")
                    if isinstance(rchunks, list):
                        rec["retrieval_chunks_json"] = self._safe_json_dumps(rchunks)

            except Exception as e:
                rec["error"] = str(e)

            results.append(rec)

        if bool(self.write_csv):
            self._csv_path = self._write_csv(results)

        return DataFrame([Data(data=x) for x in results])

    def get_csv_file(self) -> Data:
        return Data(
            data={
                "file_path": self._csv_path,
                "file_name": (self.csv_filename or "batch_results.csv"),
                "mime_type": "text/csv",
            }
        )
