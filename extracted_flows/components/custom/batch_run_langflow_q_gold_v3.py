# Recovered Langflow component
# type: batch_run_langflow_q_gold_v3
# class: BatchRunLangFlow
# used in 3 flow(s): CRCV Evaluation Runner 0.0.01, CRCV Evaluation Runner 1.0.0, CRCV Evaluation Runner 1.0.0 (backup)
# json path: node.data.node.template.code.value

from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import requests

from langflow.custom.custom_component.component import Component
from langflow.io import (
    BoolInput,
    DropdownInput,
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
    display_name = "Batch Runner (Q + Gold Answer + Metrics) v3"
    description = (
        "Reads CSV/DataFrame, calls a LangFlow run endpoint per row, "
        "extracts payload from your output_component, and writes results + optional CSV."
    )
    icon = "repeat"
    name = "batch_run_langflow_q_gold_v3"

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
            info="Node ID in the TARGET flow. Example: CustomComponent-YcKiJ",
            required=True,
        ),
        # ✅ CRITICAL FIX
        MessageTextInput(
            name="output_component_id",
            display_name="output_component (return this node)",
            info="LangFlow will return outputs for this node. Usually same as Payload Output Node ID.",
            value="",
            advanced=True,
        ),

        # Column selectors
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
            info="Extra columns copied into output with prefix in_. Example: Source, Category. Use * to copy all.",
            value="Source, Category",
            advanced=True,
        ),

        # ✅ Session control (what you asked)
        DropdownInput(
            name="session_mode",
            display_name="Session Mode",
            options=["per_row_new_session", "single_batch_session"],
            value="per_row_new_session",
            info=(
                "per_row_new_session: each question uses a new session_id (best for fair evaluation).\n"
                "single_batch_session: all questions share one session_id (reduces UI session clutter, but chat history will accumulate)."
            ),
            real_time_refresh=False,
        ),
        MessageTextInput(
            name="batch_session_id",
            display_name="Batch Session ID (used only if single_batch_session)",
            info="Leave empty to auto-generate one session id for the whole batch.",
            value="",
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
    # Helpers
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
        if isinstance(x, str):
            return x
        if hasattr(x, "get_secret_value"):
            try:
                v = x.get_secret_value()
                return v if isinstance(v, str) else str(v)
            except Exception:
                pass
        return str(x)

    # -------------------------
    # Input parsing
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
                # tolerant decode for Mac CSV quirks
                try:
                    with open(fp, "r", encoding="utf-8-sig", errors="strict") as f:
                        return self._read_csv_text(f.read())
                except Exception:
                    with open(fp, "r", encoding="latin-1", errors="ignore") as f:
                        return self._read_csv_text(f.read())

        return []

    # -------------------------
    # Payload detection
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

    def _deep_find_payload(self, obj: Any) -> Optional[Dict[str, Any]]:
        if self._is_payload(obj):
            return obj
        if isinstance(obj, dict):
            # common wrappers
            for k in ("payload", "data", "result", "results", "output"):
                v = obj.get(k)
                if self._is_payload(v):
                    return v
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

        copy_cols = self._parse_copy_cols()
        copy_all = (len(copy_cols) == 1 and copy_cols[0] == "*")

        # ✅ session policy
        if self.session_mode == "single_batch_session":
            batch_sid = (self.batch_session_id or "").strip()
            batch_sid = batch_sid if batch_sid else str(uuid.uuid4())
        else:
            batch_sid = None

        # ✅ output_component fix
        out_comp = (self.output_component_id or "").strip()
        if not out_comp:
            out_comp = (self.payload_output_node_id or "").strip()

        results: List[Dict[str, Any]] = []

        for idx, row in enumerate(rows):
            question = str(self._get_cell(row, self.question_col) or "").strip()
            gold_answer = self._get_cell(row, self.gold_answer_col)

            qid = None
            if (self.qid_col or "").strip():
                qid = self._get_cell(row, self.qid_col)

            session_id = batch_sid or str(uuid.uuid4())

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
                    outk2 = "in_" + self._safe_out_key(str(k))
                    if outk2 not in rec:
                        rec[outk2] = v
            else:
                for c in copy_cols:
                    v = self._get_cell(row, c)
                    outk2 = "in_" + self._safe_out_key(c)
                    if outk2 not in rec:
                        rec[outk2] = v

            if not question:
                rec["error"] = f"Empty question for row {idx} (question_col='{self.question_col}')"
                results.append(rec)
                continue

            # ✅ match your working python request format
            req_payload = {
                "output_type": "any",
                "input_type": "chat",
                "output_component": out_comp,
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

            try:
                resp = requests.post(
                    self.flow_run_url,
                    json=req_payload,
                    headers=headers,
                    timeout=int(self.timeout_s or 180),
                )
                resp.raise_for_status()
                j = resp.json()

                payload = self._deep_find_payload(j)

                if payload is None:
                    rec["error"] = "Payload not found in response."
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
