# Recovered Langflow component
# type: nli_evaluator_dual
# class: NliEvaluatorDual
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
from langflow.io import BoolInput, HandleInput, IntInput, MessageTextInput, Output
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame


class NliEvaluatorDual(Component):
    display_name = "Evaluator (NLI: Evidence Support + Gold Accuracy)"
    description = (
        "Runs an LLM NLI judge in two ways:\n"
        "1) Evidence Support: (premise=retrieved evidence, hypothesis=generated answer) -> supported/hallucinated\n"
        "2) Gold Accuracy: bidirectional NLI between generated answer and gold answer -> paraphrase-robust semantic correctness\n"
        "Outputs a minimal DataFrame + optional CSV, and a summary."
    )
    icon = "check-circle"
    name = "nli_evaluator_dual"

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
        # ---- columns
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
            name="gold_col",
            display_name="Gold Answer Column",
            value="gold_answer",
            advanced=True,
        ),
        MessageTextInput(
            name="evidence_col",
            display_name="Evidence Column (retrieved_context or chunks)",
            value="retrieved_context",
            advanced=True,
        ),
        MessageTextInput(
            name="chunk_ids_col",
            display_name="Chunk IDs Column (optional)",
            value="retrieved_chunk_ids_ranked",
            advanced=True,
            info="Optional. If present, will be included in output.",
        ),
        # ---- toggles
        BoolInput(
            name="eval_support",
            display_name="Evaluate Evidence Support (gen vs evidence)",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="eval_gold",
            display_name="Evaluate Gold Accuracy (gen vs gold, bidirectional)",
            value=True,
            advanced=True,
        ),
        # ---- truncation / limits
        IntInput(
            name="max_evidence_chars",
            display_name="Max evidence chars (truncate)",
            value=6000,
            advanced=True,
        ),
        IntInput(
            name="max_gold_chars",
            display_name="Max gold chars (truncate)",
            value=4000,
            advanced=True,
        ),
        IntInput(
            name="max_gen_chars",
            display_name="Max generated chars (truncate)",
            value=4000,
            advanced=True,
        ),
        # ---- thresholds / modes
        MessageTextInput(
            name="conf_threshold",
            display_name="Confidence threshold (0-1)",
            value="0.5",
            advanced=True,
        ),
        BoolInput(
            name="support_strict",
            display_name="Support STRICT mode (any extra detail => NEI)",
            value=True,
            advanced=True,
        ),
        # ---- exposed judge prompt
        MessageTextInput(
            name="judge_system",
            display_name="Judge System Message",
            value=(
                "You are a careful NLI judge for evaluating factual entailment.\n"
                "Follow the user instructions exactly. Return ONLY valid JSON.\n"
                "Do not include code fences."
            ),
            advanced=True,
        ),
        MessageTextInput(
            name="judge_user_template",
            display_name="Judge User Template (use {{MODE}}, {{PREMISE}}, {{HYPOTHESIS}})",
            value=(
                "Task: Natural Language Inference (NLI) / entailment judgment.\n\n"
                "Return ONLY valid JSON on one line with keys: label, confidence, rationale.\n\n"
                "Labels:\n"
                "- entails\n"
                "- contradicts\n"
                "- not_enough_info\n\n"
                "Mode: {{MODE}}\n"
                "Definitions:\n"
                "- entails: Premise supports ALL claims in Hypothesis.\n"
                "- contradicts: Premise directly contradicts Hypothesis.\n"
                "- not_enough_info: Premise does not fully support Hypothesis.\n\n"
                "Premise:\n"
                "{{PREMISE}}\n\n"
                "Hypothesis:\n"
                "{{HYPOTHESIS}}\n\n"
                "Output JSON example:\n"
                "{\"label\":\"entails\",\"confidence\":0.83,\"rationale\":\"brief\"}"
            ),
            advanced=True,
        ),
        # ---- csv
        BoolInput(
            name="write_csv",
            display_name="Write CSV file",
            value=True,
            advanced=True,
        ),
        MessageTextInput(
            name="csv_filename",
            display_name="CSV filename",
            value="eval_nli_support_and_gold.csv",
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
    # Utils
    # -------------------------
    @staticmethod
    def _try_float(x: Any, default: float = 0.5) -> float:
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _clip(s: str, max_chars: int) -> str:
        if max_chars <= 0:
            return s
        if len(s) <= max_chars:
            return s
        return s[:max_chars] + "\n[TRUNCATED]"

    @staticmethod
    def _coerce_text(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, Data):
            v = v.data
        # list of chunks / strings / dicts
        if isinstance(v, list):
            parts = []
            for item in v:
                if isinstance(item, dict):
                    t = item.get("text") or item.get("content") or ""
                    parts.append(str(t))
                else:
                    parts.append(str(item))
            return "\n\n---\n\n".join([p for p in parts if p.strip()])
        if isinstance(v, dict):
            for k in ("text", "content", "message", "value"):
                if k in v and isinstance(v[k], str):
                    return v[k]
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    @staticmethod
    def _coerce_ids(v: Any) -> str:
        """Return a compact string for chunk ids (list -> JSON)."""
        if v is None:
            return ""
        if isinstance(v, Data):
            v = v.data
        if isinstance(v, list):
            try:
                return json.dumps([str(x) for x in v], ensure_ascii=False)
            except Exception:
                return ",".join(str(x) for x in v)
        return str(v)

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

    # -------------------------
    # LLM judge call
    # -------------------------
    def _call_llm(self, system: str, user: str) -> str:
        llm = self.llm
        if llm is None:
            raise ValueError("No LLM connected to NliEvaluatorDual.llm")

        prompt = f"SYSTEM:\n{system}\n\nUSER:\n{user}".strip()

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
    # Prompt build + parse
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

    def _fill_template(self, template: str, *, mode: str, premise: str, hypothesis: str) -> str:
        t = template or ""
        # simple placeholder replacement (avoids .format brace escaping issues)
        t = t.replace("{{MODE}}", mode)
        t = t.replace("{{PREMISE}}", premise)
        t = t.replace("{{HYPOTHESIS}}", hypothesis)
        return t

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

        # fallback heuristics
        low = raw.lower()
        if "entail" in low or "supported" in low:
            return "entails", 0.5, ""
        if "contradict" in low:
            return "contradicts", 0.5, ""
        return "not_enough_info", 0.5, ""

    def _judge(self, *, mode: str, premise: str, hypothesis: str) -> Tuple[str, float, str]:
        system = (self.judge_system or "").strip()
        user_template = self.judge_user_template or ""
        user = self._fill_template(user_template, mode=mode, premise=premise, hypothesis=hypothesis)
        raw = self._call_llm(system, user)
        return self._parse_judge(raw)

    # -------------------------
    # CSV
    # -------------------------
    def _write_csv(self, rows: List[Dict[str, Any]]) -> Optional[str]:
        if not rows:
            return None

        filename = (self.csv_filename or "eval_nli_support_and_gold.csv").strip() or "eval_nli_support_and_gold.csv"
        out_dir = "/mnt/data" if os.path.isdir("/mnt/data") else None

        if out_dir:
            path = os.path.join(out_dir, filename)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            path = tmp.name
            tmp.close()

        fieldnames = [
            "question",
            "generated_answer",
            "gold_answer",
            "evidence",
            "chunk_ids",
            # evidence support NLI
            "sup_label",
            "sup_confidence",
            "sup_supported",
            "sup_hallucinated",
            # gold semantic NLI (bidirectional)
            "gold_label_gold_to_gen",
            "gold_conf_gold_to_gen",
            "gold_label_gen_to_gold",
            "gold_conf_gen_to_gold",
            "gold_correct_bidirectional",
            "gold_correct_any",
            "gold_contradiction_any",
        ]

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
        if not rows:
            self.status = "No rows received."
            return DataFrame([])

        qcol = (self.question_col or "question").strip() or "question"
        gencol = (self.generated_col or "generated_answer").strip() or "generated_answer"
        goldcol = (self.gold_col or "gold_answer").strip() or "gold_answer"
        evcol = (self.evidence_col or "retrieved_context").strip() or "retrieved_context"
        idcol = (self.chunk_ids_col or "").strip()

        conf_th = max(0.0, min(1.0, self._try_float(self.conf_threshold, 0.5)))
        strict_mode = "STRICT" if bool(self.support_strict) else "LENIENT"

        max_evidence = int(self.max_evidence_chars or 6000)
        max_gold = int(self.max_gold_chars or 4000)
        max_gen = int(self.max_gen_chars or 4000)

        # summaries
        n = 0
        sup_supported_cnt = 0
        sup_hallu_cnt = 0
        sup_entails = 0
        sup_contra = 0
        sup_nei = 0

        gold_bidir_correct_cnt = 0
        gold_any_correct_cnt = 0
        gold_any_contra_cnt = 0

        out_rows: List[Dict[str, Any]] = []

        for r in rows:
            question = self._coerce_text(r.get(qcol, ""))
            gen = self._clip(self._coerce_text(r.get(gencol, "")), max_gen)
            gold = self._clip(self._coerce_text(r.get(goldcol, "")), max_gold)
            evidence = self._clip(self._coerce_text(r.get(evcol, "")), max_evidence)
            chunk_ids = self._coerce_ids(r.get(idcol)) if idcol else ""

            # default outputs
            sup_label = ""
            sup_conf = ""
            sup_supported = ""
            sup_hallu = ""

            gg_label = ""  # gold -> gen
            gg_conf = ""
            gG_label = ""  # gen -> gold
            gG_conf = ""
            gold_bidir = ""
            gold_any = ""
            gold_contra_any = ""

            # ---- Evidence support NLI
            if bool(self.eval_support):
                if gen.strip() and evidence.strip():
                    sup_label_v, sup_conf_v, _ = self._judge(
                        mode=strict_mode,
                        premise=evidence,
                        hypothesis=gen,
                    )
                else:
                    sup_label_v, sup_conf_v = "not_enough_info", 1.0

                sup_label = sup_label_v
                sup_conf = sup_conf_v

                is_supported = 1 if (sup_label_v == "entails" and float(sup_conf_v) >= conf_th) else 0
                is_hallu = 1 if (gen.strip() and is_supported == 0) else 0

                sup_supported = is_supported
                sup_hallu = is_hallu

                sup_supported_cnt += is_supported
                sup_hallu_cnt += is_hallu
                if sup_label_v == "entails":
                    sup_entails += 1
                elif sup_label_v == "contradicts":
                    sup_contra += 1
                else:
                    sup_nei += 1

            # ---- Gold semantic NLI (bidirectional)
            if bool(self.eval_gold):
                if gen.strip() and gold.strip():
                    # gold => gen (checks "no extra claims" relative to gold)
                    gg_label_v, gg_conf_v, _ = self._judge(
                        mode="GOLD",  # you can interpret this any way in your template; it's just a tag
                        premise=gold,
                        hypothesis=gen,
                    )
                    # gen => gold (checks "did you cover what gold says")
                    gG_label_v, gG_conf_v, _ = self._judge(
                        mode="GOLD",
                        premise=gen,
                        hypothesis=gold,
                    )
                else:
                    gg_label_v, gg_conf_v = "not_enough_info", 1.0
                    gG_label_v, gG_conf_v = "not_enough_info", 1.0

                gg_label = gg_label_v
                gg_conf = gg_conf_v
                gG_label = gG_label_v
                gG_conf = gG_conf_v

                gg_ok = (gg_label_v == "entails" and float(gg_conf_v) >= conf_th)
                gG_ok = (gG_label_v == "entails" and float(gG_conf_v) >= conf_th)

                bidir_ok = 1 if (gg_ok and gG_ok) else 0
                any_ok = 1 if (gg_ok or gG_ok) else 0

                contra_any = 1 if (gg_label_v == "contradicts" or gG_label_v == "contradicts") else 0

                gold_bidir = bidir_ok
                gold_any = any_ok
                gold_contra_any = contra_any

                gold_bidir_correct_cnt += bidir_ok
                gold_any_correct_cnt += any_ok
                gold_any_contra_cnt += contra_any

            out_rows.append(
                {
                    "question": question,
                    "generated_answer": gen,
                    "gold_answer": gold,
                    "evidence": evidence,
                    "chunk_ids": chunk_ids,
                    "sup_label": sup_label,
                    "sup_confidence": sup_conf,
                    "sup_supported": sup_supported,
                    "sup_hallucinated": sup_hallu,
                    "gold_label_gold_to_gen": gg_label,
                    "gold_conf_gold_to_gen": gg_conf,
                    "gold_label_gen_to_gold": gG_label,
                    "gold_conf_gen_to_gold": gG_conf,
                    "gold_correct_bidirectional": gold_bidir,
                    "gold_correct_any": gold_any,
                    "gold_contradiction_any": gold_contra_any,
                }
            )

            n += 1

        # summary
        self._summary = {
            "rows": n,
            "conf_threshold": conf_th,
            "support_mode": strict_mode if bool(self.eval_support) else None,
            "support_rate": (sup_supported_cnt / n) if (n and bool(self.eval_support)) else None,
            "hallucination_rate": (sup_hallu_cnt / n) if (n and bool(self.eval_support)) else None,
            "support_label_entails": sup_entails if bool(self.eval_support) else None,
            "support_label_contradicts": sup_contra if bool(self.eval_support) else None,
            "support_label_not_enough_info": sup_nei if bool(self.eval_support) else None,
            "gold_bidirectional_accuracy": (gold_bidir_correct_cnt / n) if (n and bool(self.eval_gold)) else None,
            "gold_any_direction_accuracy": (gold_any_correct_cnt / n) if (n and bool(self.eval_gold)) else None,
            "gold_contradiction_rate_any_direction": (gold_any_contra_cnt / n) if (n and bool(self.eval_gold)) else None,
            "cols": {"q": qcol, "gen": gencol, "gold": goldcol, "evidence": evcol, "chunk_ids": idcol or None},
        }

        # status string
        parts = [f"Rows={n}", f"th={conf_th:.2f}"]
        if bool(self.eval_support):
            parts.append(f"support={sup_supported_cnt/n:.3f}")
            parts.append(f"hallu={sup_hallu_cnt/n:.3f}")
        if bool(self.eval_gold):
            parts.append(f"gold_bidir={gold_bidir_correct_cnt/n:.3f}")
            parts.append(f"gold_any={gold_any_correct_cnt/n:.3f}")
        self.status = " | ".join(parts)

        if bool(self.write_csv):
            self._csv_path = self._write_csv(out_rows)

        return DataFrame([Data(data=x) for x in out_rows])

    def get_csv_file(self) -> Data:
        return Data(
            data={
                "file_path": self._csv_path,
                "file_name": (self.csv_filename or "eval_nli_support_and_gold.csv"),
            }
        )

    def get_summary(self) -> Data:
        return Data(data=self._summary)
