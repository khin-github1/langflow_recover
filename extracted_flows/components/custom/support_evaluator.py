# Recovered Langflow component
# type: support_evaluator
# class: SupportEvaluator
# used in 1 flow(s): CRCV Evaluation Runner 0.0.01
# json path: node.data.node.template.code.value

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, HandleInput, MessageTextInput, Output, IntInput
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame


class SupportEvaluator(Component):
    display_name = "Evaluator (Support: Supported / Hallucination Rate)"
    description = (
        "Uses an LLM judge to classify the generated answer as entailed/contradicted/not_enough_info "
        "with respect to retrieved evidence, and computes hallucination rate (unsupportedness)."
    )
    icon = "check-circle"
    name = "support_evaluator"

    inputs = [
        HandleInput(
            name="results_df",
            display_name="Runner Results (DataFrame)",
            input_types=["DataFrame", "Data"],
            required=True,
        ),
        HandleInput(
            name="llm",
            display_name="Judge LLM",
            input_types=["LanguageModel", "ChatModel", "LLM"],
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
            value="generated_answer",
            advanced=True,
        ),
        MessageTextInput(
            name="evidence_col",
            display_name="Evidence/Context Column",
            value="retrieved_context",
            advanced=True,
        ),
        IntInput(
            name="max_evidence_chars",
            display_name="Max evidence chars (truncate)",
            value=6000,
            advanced=True,
        ),
        MessageTextInput(
            name="support_conf_threshold",
            display_name="Support confidence threshold",
            value="0.5",
            advanced=True,
        ),
        BoolInput(
            name="strict",
            display_name="Strict support (no extra claims)",
            value=True,
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
            value="eval_support.csv",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Eval DataFrame", name="eval_df", method="evaluate"),
        Output(display_name="CSV File (Data)", name="csv_file", method="get_csv_file"),
        Output(display_name="Summary (Data)", name="summary", method="get_summary"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._csv_path: Optional[str] = None
        self._summary: Dict[str, Any] = {}

    # -------------------------
    # Iteration / coercion
    # -------------------------
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
    def _coerce_text(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, Data):
            d = v.data
            if isinstance(d, str):
                return d
            if isinstance(d, dict):
                for k in ("text", "content", "message", "value"):
                    if k in d and isinstance(d[k], str):
                        return d[k]
                return json.dumps(d, ensure_ascii=False)
            return str(d)
        if isinstance(v, dict):
            for k in ("text", "content", "message", "value"):
                if k in v and isinstance(v[k], str):
                    return v[k]
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    @staticmethod
    def _try_float(x: Any, default: float = 0.5) -> float:
        try:
            return float(x)
        except Exception:
            return default

    # -------------------------
    # LLM call
    # -------------------------
    def _call_llm(self, prompt: str) -> str:
        llm = self.llm
        if llm is None:
            raise ValueError("No LLM connected to SupportEvaluator.llm")

        if hasattr(llm, "invoke"):
            resp = llm.invoke(prompt)
        elif hasattr(llm, "predict"):
            resp = llm.predict(prompt)
        elif hasattr(llm, "generate"):
            resp = llm.generate(prompt)
        else:
            raise TypeError("Judge LLM does not expose invoke/predict/generate")

        if isinstance(resp, str):
            return resp
        if hasattr(resp, "content") and isinstance(resp.content, str):
            return resp.content
        if isinstance(resp, dict) and "content" in resp and isinstance(resp["content"], str):
            return resp["content"]
        return str(resp)

    # -------------------------
    # Prompt + parse
    # -------------------------
    _re_json_obj = re.compile(r"\{.*\}", re.DOTALL)

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = (label or "").strip().lower()
        if label in ("entailment", "entails", "entailed", "supported"):
            return "entails"
        if label in ("contradiction", "contradicts", "contradicted"):
            return "contradicts"
        return "not_enough_info"

    def _build_prompt(self, question: str, answer: str, evidence: str) -> str:
        mode = "STRICT" if bool(self.strict) else "LENIENT"

        # NOTE: JSON braces are escaped with {{ }}
        template = """You are an NLI judge for RAG grounding.

Mode: {mode}

Return ONLY valid JSON with keys: label, confidence, rationale.

Labels:
- "entails": Evidence supports ALL claims in Answer.
- "contradicts": Evidence directly contradicts Answer.
- "not_enough_info": Evidence does not fully support Answer (missing or partial support).

Rules:
- In STRICT mode, if the Answer adds any detail not in Evidence => "not_enough_info".
- Confidence is a number from 0.0 to 1.0.
- Keep rationale brief (<= 1-2 sentences).

Question:
{question}

Evidence:
{evidence}

Answer:
{answer}

Return JSON like:
{{"label":"entails","confidence":0.83,"rationale":"..."}}
"""
        return template.format(mode=mode, question=question, evidence=evidence, answer=answer)

    def _parse_judge(self, text: str) -> Tuple[str, float, str]:
        raw = (text or "").strip()

        # strip code fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        m = self._re_json_obj.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                label = self._normalize_label(str(obj.get("label", "")))
                conf = self._try_float(obj.get("confidence", 0.0), 0.0)
                conf = max(0.0, min(1.0, conf))
                rat = str(obj.get("rationale", "")).strip()
                return label, conf, rat
            except Exception:
                pass

        # fallback
        low = raw.lower()
        if "entail" in low or "supported" in low:
            return "entails", 0.5, ""
        if "contradict" in low:
            return "contradicts", 0.5, ""
        return "not_enough_info", 0.5, ""

    # -------------------------
    # CSV writing
    # -------------------------
    def _write_csv(self, rows: List[Dict[str, Any]]) -> Optional[str]:
        if not rows:
            return None

        filename = (self.csv_filename or "eval_support.csv").strip() or "eval_support.csv"
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
        self._summary = {}

        rows = self._iter_rows()
        out_rows: List[Dict[str, Any]] = []

        qcol = (self.question_col or "question").strip() or "question"
        acol = (self.generated_col or "generated_answer").strip() or "generated_answer"
        ecol = (self.evidence_col or "retrieved_context").strip() or "retrieved_context"

        conf_th = self._try_float(self.support_conf_threshold, 0.5)
        max_chars = int(self.max_evidence_chars or 6000)

        n_total = 0
        n_supported = 0
        n_hallu = 0
        n_entails = 0
        n_contra = 0
        n_nei = 0

        for r in rows:
            question = self._coerce_text(r.get(qcol, ""))
            answer = self._coerce_text(r.get(acol, ""))
            evidence = self._coerce_text(r.get(ecol, ""))

            if max_chars > 0 and len(evidence) > max_chars:
                evidence = evidence[:max_chars] + "\n[TRUNCATED]"

            if not answer.strip():
                label, conf, rat = "not_enough_info", 1.0, "empty answer"
            elif not evidence.strip():
                label, conf, rat = "not_enough_info", 1.0, "no evidence provided"
            else:
                prompt = self._build_prompt(question, answer, evidence)
                raw = self._call_llm(prompt)
                label, conf, rat = self._parse_judge(raw)

            supported = 1 if (label == "entails" and conf >= conf_th) else 0

            # Hallucination as "unsupportedness" for non-empty answers
            hallucinated = 1 if (answer.strip() and supported == 0) else 0

            rr = dict(r)
            rr["sup_label"] = label
            rr["sup_confidence"] = conf
            rr["sup_rationale"] = rat
            rr["sup_supported"] = supported
            rr["sup_hallucinated"] = hallucinated
            rr["sup_conf_threshold"] = conf_th
            out_rows.append(rr)

            n_total += 1
            n_supported += supported
            n_hallu += hallucinated
            if label == "entails":
                n_entails += 1
            elif label == "contradicts":
                n_contra += 1
            else:
                n_nei += 1

        support_rate = (n_supported / n_total) if n_total else 0.0
        halluc_rate = (n_hallu / n_total) if n_total else 0.0

        self._summary = {
            "rows": n_total,
            "support_rate": support_rate,
            "hallucination_rate": halluc_rate,
            "label_entails": n_entails,
            "label_contradicts": n_contra,
            "label_not_enough_info": n_nei,
            "conf_threshold": conf_th,
            "question_col": qcol,
            "generated_col": acol,
            "evidence_col": ecol,
        }

        self.status = (
            f"Rows={n_total} | support_rate={support_rate:.3f} | halluc_rate={halluc_rate:.3f} "
            f"| entails={n_entails} contra={n_contra} nei={n_nei}"
        )

        if bool(self.write_csv):
            self._csv_path = self._write_csv(out_rows)

        return DataFrame([Data(data=x) for x in out_rows])

    def get_csv_file(self) -> Data:
        return Data(
            data={
                "file_path": self._csv_path,
                "file_name": (self.csv_filename or "eval_support.csv"),
            }
        )

    def get_summary(self) -> Data:
        return Data(data=self._summary)
