# Recovered Langflow component
# type: ground_truth_evaluator
# class: GroundTruthEvaluator
# used in 1 flow(s): CRCV Evaluation Runner 0.0.01
# json path: node.data.node.template.code.value

from __future__ import annotations

import csv
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, HandleInput, MessageTextInput, Output
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame


class GroundTruthEvaluator(Component):
    display_name = "Evaluator (Gold Answer: EM/F1)"
    description = "Computes EM + token F1 between assistant_text and gold_answer. Outputs DataFrame + optional CSV."
    icon = "check-circle"
    name = "ground_truth_evaluator"

    inputs = [
        HandleInput(
            name="results_df",
            display_name="Runner Results (DataFrame)",
            input_types=["DataFrame", "Data"],
            required=True,
        ),
        MessageTextInput(
            name="assistant_col",
            display_name="Assistant Column",
            value="assistant_text",
            advanced=True,
        ),
        MessageTextInput(
            name="gold_col",
            display_name="Gold Answer Column",
            value="gold_answer",
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
            value="eval_ground_truth.csv",
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
        return 1 if cls._normalize(pred) == cls._normalize(gold) and cls._normalize(gold) != "" else 0

    @classmethod
    def _f1(cls, pred: str, gold: str) -> float:
        pt = cls._tokens(pred)
        gt = cls._tokens(gold)
        if not pt and not gt:
            return 1.0
        if not pt or not gt:
            return 0.0
        # multiset overlap
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
    # Row iteration
    # -------------------------
    def _iter_rows(self) -> List[Dict[str, Any]]:
        t = self.results_df
        if isinstance(t, DataFrame):
            out = []
            for _, row in t.iterrows():
                out.append(row.to_dict())
            return out
        if isinstance(t, Data) and isinstance(t.data, dict):
            # if someone passes one row
            return [dict(t.data)]
        return []

    def _write_csv(self, rows: List[Dict[str, Any]]) -> Optional[str]:
        if not rows:
            return None

        filename = (self.csv_filename or "eval_ground_truth.csv").strip() or "eval_ground_truth.csv"
        out_dir = "/mnt/data" if os.path.isdir("/mnt/data") else None

        if out_dir:
            path = os.path.join(out_dir, filename)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            path = tmp.name
            tmp.close()

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
    def evaluate(self) -> DataFrame:
        self._csv_path = None

        rows = self._iter_rows()
        out_rows: List[Dict[str, Any]] = []

        for r in rows:
            pred = str(r.get(self.assistant_col, "") or "")
            gold = str(r.get(self.gold_col, "") or "")
            em = self._exact_match(pred, gold)
            f1 = self._f1(pred, gold)

            rr = dict(r)
            rr["gt_em"] = em
            rr["gt_f1"] = f1
            out_rows.append(rr)

        if bool(self.write_csv):
            self._csv_path = self._write_csv(out_rows)

        return DataFrame([Data(data=x) for x in out_rows])

    def get_csv_file(self) -> Data:
        return Data(
            data={
                "file_path": self._csv_path,
                "file_name": (self.csv_filename or "eval_ground_truth.csv"),
            }
        )
