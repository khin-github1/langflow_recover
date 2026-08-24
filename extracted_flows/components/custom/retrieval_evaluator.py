# Recovered Langflow component
# type: retrieval_evaluator
# class: RetrievalEvaluator
# used in 1 flow(s): CRCV Evaluation Runner 0.0.01
# json path: node.data.node.template.code.value

from __future__ import annotations

import csv
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, HandleInput, MessageTextInput, Output, IntInput
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame


class RetrievalEvaluator(Component):
    display_name = "Evaluator (Retrieval: Recall@k / Precision@k / MRR)"
    description = "Uses retrieved_chunk_ids_ranked vs gold_chunk_ids to compute Recall@k, Precision@k, MRR."
    icon = "target"
    name = "retrieval_evaluator"

    inputs = [
        HandleInput(
            name="results_df",
            display_name="Runner Results (DataFrame)",
            input_types=["DataFrame", "Data"],
            required=True,
        ),
        MessageTextInput(
            name="retrieved_ids_col",
            display_name="Retrieved IDs Column",
            value="ret_retrieved_chunk_ids_ranked",
            advanced=True,
        ),
        MessageTextInput(
            name="gold_ids_col",
            display_name="Gold Chunk IDs Column",
            value="gold_chunk_ids",
            advanced=True,
        ),
        IntInput(
            name="k",
            display_name="k (for Recall@k / Precision@k)",
            value=5,
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
            value="eval_retrieval.csv",
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

    @staticmethod
    def _parse_ids(v: Any) -> List[str]:
        """
        Accept:
          - list[str]
          - JSON string '["a","b"]'
          - pipe-separated 'a|b|c'
          - comma-separated 'a,b,c'
        """
        if v is None:
            return []

        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]

        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            # try JSON
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    return [str(x).strip() for x in j if str(x).strip()]
            except Exception:
                pass

            # splitters
            if "|" in s:
                return [x.strip() for x in s.split("|") if x.strip()]
            if "," in s:
                return [x.strip() for x in s.split(",") if x.strip()]

            # single id
            return [s]

        # fallback
        return [str(v).strip()] if str(v).strip() else []

    @staticmethod
    def _metrics(retrieved: List[str], gold: Set[str], k: int) -> Tuple[Optional[float], Optional[float], float]:
        """
        recall@k = hits / |gold|
        precision@k = hits / k
        mrr = 1/rank of first relevant else 0
        """
        if k <= 0:
            k = len(retrieved)

        topk = retrieved[:k]
        hits = sum(1 for x in topk if x in gold)

        recall = None if len(gold) == 0 else (hits / len(gold))
        precision = None if k == 0 else (hits / k)

        mrr = 0.0
        for i, x in enumerate(retrieved, start=1):
            if x in gold:
                mrr = 1.0 / i
                break

        return recall, precision, mrr

    def _write_csv(self, rows: List[Dict[str, Any]]) -> Optional[str]:
        if not rows:
            return None

        filename = (self.csv_filename or "eval_retrieval.csv").strip() or "eval_retrieval.csv"
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

    def evaluate(self) -> DataFrame:
        self._csv_path = None

        rows = self._iter_rows()
        out_rows: List[Dict[str, Any]] = []
        k = int(self.k or 5)

        for r in rows:
            retrieved = self._parse_ids(r.get(self.retrieved_ids_col))
            gold_list = self._parse_ids(r.get(self.gold_ids_col))
            gold = set(gold_list)

            recall_k, prec_k, mrr = self._metrics(retrieved, gold, k)

            rr = dict(r)
            rr["ret_eval_k"] = k
            rr["ret_recall_at_k"] = recall_k
            rr["ret_precision_at_k"] = prec_k
            rr["ret_mrr"] = mrr
            out_rows.append(rr)

        if bool(self.write_csv):
            self._csv_path = self._write_csv(out_rows)

        return DataFrame([Data(data=x) for x in out_rows])

    def get_csv_file(self) -> Data:
        return Data(
            data={
                "file_path": self._csv_path,
                "file_name": (self.csv_filename or "eval_retrieval.csv"),
            }
        )
