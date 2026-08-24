# Recovered Langflow component
# type: ground_truth_evaluator_minimal
# class: GroundTruthEvaluatorMinimal
# used in 1 flow(s): CRCV Evaluation Runner 0.0.01
# json path: node.data.node.template.code.value

from __future__ import annotations

import csv
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Sequence

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, HandleInput, MessageTextInput, Output
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame


class GroundTruthEvaluatorMinimal(Component):
    display_name = "Evaluator (EM/F1 → Minimal CSV)"
    description = (
        "Computes EM + token F1 between generated answer and gold answer. "
        "Outputs ONLY: question, generated_answer, gold_answer, em, token_f1. "
        "Auto-detects columns if needed and can write CSV."
    )
    icon = "check-circle"
    name = "ground_truth_evaluator_minimal"

    inputs = [
        HandleInput(
            name="results_df",
            display_name="Runner Results (DataFrame)",
            input_types=["DataFrame", "Data"],
            required=True,
        ),
        MessageTextInput(
            name="question_col",
            display_name="Question Column",
            value="question",
            advanced=True,
        ),
        MessageTextInput(
            name="generated_col",
            display_name="Generated Answer Column",
            value="generated_answer",  # IMPORTANT: default matches your CSV
            advanced=True,
        ),
        MessageTextInput(
            name="gold_col",
            display_name="Gold Answer Column",
            value="gold_answer",
            advanced=True,
        ),
        BoolInput(
            name="fail_on_missing_cols",
            display_name="Fail if required columns not found",
            value=True,
            advanced=True,
            info="If true, raises an error when question/generated/gold cannot be resolved.",
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
            value="eval_em_f1_minimal.csv",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Eval DataFrame", name="eval_df", method="evaluate"),
        Output(display_name="CSV File (Data)", name="csv_file", method="get_csv_file"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._csv_path: Optional[str] = None

    # -------------------------
    # Normalization + metrics
    # -------------------------
    _re_punc = re.compile(r"[^\w\s]", re.UNICODE)

    @classmethod
    def _normalize(cls, s: str) -> str:
        s = (s or "").lower().strip()
        s = cls._re_punc.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def _tokens(cls, s: str) -> List[str]:
        ns = cls._normalize(s)
        return ns.split() if ns else []

    @classmethod
    def _exact_match(cls, pred: str, gold: str) -> int:
        ng = cls._normalize(gold)
        return 1 if cls._normalize(pred) == ng and ng != "" else 0

    @classmethod
    def _f1(cls, pred: str, gold: str) -> float:
        pt = cls._tokens(pred)
        gt = cls._tokens(gold)
        if not pt and not gt:
            return 1.0
        if not pt or not gt:
            return 0.0

        from collections import Counter

        pc = Counter(pt)
        gc = Counter(gt)
        common = pc & gc
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pt)
        recall = num_same / len(gt)
        return (2 * precision * recall) / (precision + recall)

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _coerce_text(v: Any) -> str:
        """Best-effort extract text from common Langflow-ish values."""
        if v is None:
            return ""
        # Data object
        if isinstance(v, Data):
            d = v.data
            if isinstance(d, str):
                return d
            if isinstance(d, dict):
                for k in ("text", "content", "message", "value"):
                    if k in d and isinstance(d[k], str):
                        return d[k]
                return str(d)
            return str(d)
        # dict
        if isinstance(v, dict):
            for k in ("text", "content", "message", "value"):
                if k in v and isinstance(v[k], str):
                    return v[k]
            return str(v)
        return str(v)

    @staticmethod
    def _lower_keymap(keys: Sequence[str]) -> Dict[str, str]:
        return {k.lower(): k for k in keys}

    def _resolve_col(self, requested: str, keys: Sequence[str], fallbacks: Sequence[str]) -> Optional[str]:
        """Resolve requested column name against actual keys (case-insensitive), else try fallbacks."""
        requested = (requested or "").strip()
        km = self._lower_keymap(keys)

        if requested and requested in keys:
            return requested
        if requested and requested.lower() in km:
            return km[requested.lower()]

        for f in fallbacks:
            f = (f or "").strip()
            if not f:
                continue
            if f in keys:
                return f
            if f.lower() in km:
                return km[f.lower()]

        return None

    def _iter_rows(self) -> List[Dict[str, Any]]:
        t = self.results_df
        if isinstance(t, DataFrame):
            out = []
            for _, row in t.iterrows():
                out.append(row.to_dict())
            return out
        if isinstance(t, Data) and isinstance(t.data, dict):
            return [dict(t.data)]
        return []

    def _write_csv(self, rows: List[Dict[str, Any]]) -> Optional[str]:
        if not rows:
            return None

        filename = (self.csv_filename or "eval_em_f1_minimal.csv").strip() or "eval_em_f1_minimal.csv"
        out_dir = "/mnt/data" if os.path.isdir("/mnt/data") else None

        if out_dir:
            path = os.path.join(out_dir, filename)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            path = tmp.name
            tmp.close()

        fieldnames = ["question", "generated_answer", "gold_answer", "em", "token_f1"]

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

        return path

    # -------------------------
    # Main
    # -------------------------
    def evaluate(self) -> DataFrame:
        self._csv_path = None
        rows = self._iter_rows()
        if not rows:
            self.status = "No rows received."
            return DataFrame([])

        keys = list(rows[0].keys())

        # fallbacks cover common runner schemas
        qcol = self._resolve_col(self.question_col, keys, fallbacks=["question", "query", "user_question"])
        pcol = self._resolve_col(
            self.generated_col,
            keys,
            fallbacks=["generated_answer", "assistant_text", "assistant", "answer", "response", "output"],
        )
        gcol = self._resolve_col(self.gold_col, keys, fallbacks=["gold_answer", "ground_truth", "reference", "expected_answer"])

        if self.fail_on_missing_cols and (qcol is None or pcol is None or gcol is None):
            raise ValueError(
                "Could not resolve required columns.\n"
                f"Resolved: question={qcol}, generated={pcol}, gold={gcol}\n"
                f"Available columns: {keys}\n"
                "Fix by setting question_col/generated_col/gold_col to the correct names."
            )

        # if not failing, treat missing columns as empty
        out_rows: List[Dict[str, Any]] = []

        for r in rows:
            q = self._coerce_text(r.get(qcol, "")) if qcol else ""
            pred = self._coerce_text(r.get(pcol, "")) if pcol else ""
            gold = self._coerce_text(r.get(gcol, "")) if gcol else ""

            em = self._exact_match(pred, gold)
            f1 = self._f1(pred, gold)

            out_rows.append(
                {
                    "question": q,
                    "generated_answer": pred,
                    "gold_answer": gold,
                    "em": em,
                    "token_f1": f1,
                }
            )

        if bool(self.write_csv):
            self._csv_path = self._write_csv(out_rows)

        # helpful status line for debugging in Langflow UI
        self.status = f"Cols: q={qcol}, gen={pcol}, gold={gcol} | rows={len(out_rows)} | csv={self._csv_path or 'none'}"

        return DataFrame([Data(data=x) for x in out_rows])

    def get_csv_file(self) -> Data:
        return Data(
            data={
                "file_path": self._csv_path,
                "file_name": (self.csv_filename or "eval_em_f1_minimal.csv"),
            }
        )
