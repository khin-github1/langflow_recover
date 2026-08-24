# Recovered Langflow component
# type: nli_evaluator_with_chunk_fix
# class: NliEvaluatorWithChunkFix
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


class NliEvaluatorWithChunkFix(Component):
    display_name = "Evaluator (NLI: Gold + Support) + Chunk Fix (Minimal CSV)"
    description = (
        "Parses retrieval_chunks_json to build evidence text, then runs NLI using a judge LLM.\n"
        "Support NLI: generated_answer vs evidence (hallucination/grounding).\n"
        "Gold NLI: generated_answer vs gold_answer (semantic accuracy).\n"
        "Outputs minimal CSV."
    )
    icon = "check-circle"
    name = "nli_evaluator_with_chunk_fix"

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
        # Column names
        MessageTextInput(name="question_col", display_name="Question Column", value="question", advanced=True),
        MessageTextInput(name="generated_col", display_name="Generated Answer Column", value="generated_answer", advanced=True),
        MessageTextInput(name="gold_col", display_name="Gold Answer Column", value="gold_answer", advanced=True),

        # Evidence inputs (choose one)
        MessageTextInput(
            name="evidence_col",
            display_name="Evidence Column (plain text, optional)",
            value="retrieved_context",
            advanced=True,
            info="If present, use this as evidence text. If empty/missing, will try retrieval_chunks_json.",
        ),
        MessageTextInput(
            name="chunks_json_col",
            display_name="Chunks JSON Column",
            value="retrieval_chunks_json",
            advanced=True,
            info="Column containing JSON list of chunks with fields like {rank, chunk_id, text, score}.",
        ),
        IntInput(
            name="evidence_top_k",
            display_name="Evidence top-k chunks",
            value=5,
            advanced=True,
        ),
        IntInput(
            name="max_evidence_chars",
            display_name="Max evidence chars (truncate)",
            value=6000,
            advanced=True,
        ),

        # What to run
        BoolInput(name="run_support_nli", display_name="Run Support NLI (gen vs evidence)", value=True, advanced=True),
        BoolInput(name="run_gold_nli", display_name="Run Gold NLI (gen vs gold)", value=True, advanced=True),

        # Judge prompt controls (editable in UI)
        BoolInput(name="support_strict", display_name="Support NLI strict mode", value=True, advanced=True),
        MessageTextInput(
            name="support_judge_instructions",
            display_name="Support Judge Instructions (editable)",
            advanced=True,
            value=(
                "You are an NLI judge for RAG grounding.\n"
                "Task: decide whether the Evidence supports ALL factual claims in the Answer.\n"
                'Return ONLY valid JSON with keys: label, confidence, rationale.\n'
                'Labels: "entails", "contradicts", "not_enough_info".\n'
                "Strict rule: if Answer adds any specific factual detail not supported by Evidence -> not_enough_info.\n"
                "Confidence must be 0.0 to 1.0. Keep rationale <= 2 sentences."
            ),
        ),
        MessageTextInput(
            name="gold_judge_instructions",
            display_name="Gold Judge Instructions (editable)",
            advanced=True,
            value=(
                "You are an NLI judge for semantic answer accuracy.\n"
                "Task: compare Answer against Gold Reference.\n"
                'Return ONLY valid JSON with keys: label, confidence, rationale.\n'
                'Labels: "entails", "contradicts", "not_enough_info".\n'
                "Guidance: If Answer is a correct paraphrase of Gold, label entails.\n"
                "If Answer conflicts with Gold, label contradicts.\n"
                "If Answer is incomplete/too vague relative to Gold, label not_enough_info.\n"
                "Confidence 0.0 to 1.0. Keep rationale <= 2 sentences."
            ),
        ),

        # Thresholding
        MessageTextInput(
            name="conf_threshold",
            display_name="Confidence threshold (entails => supported/accurate)",
            value="0.5",
            advanced=True,
        ),

        # Output controls
        BoolInput(name="include_rationale", display_name="Include rationale in output", value=False, advanced=True),
        BoolInput(name="include_evidence_text", display_name="Include evidence_text in output", value=False, advanced=True),
        BoolInput(name="write_csv", display_name="Write CSV file", value=True, advanced=True),
        MessageTextInput(name="csv_filename", display_name="CSV filename", value="eval_nli_minimal.csv", advanced=True),

        # Hygiene
        BoolInput(
            name="skip_bad_rows",
            display_name="Skip rows where ok==False or error present (if columns exist)",
            value=True,
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
    # Utilities
    # -------------------------
    _re_json_obj = re.compile(r"\{.*\}", re.DOTALL)

    @staticmethod
    def _try_float(x: Any, default: float = 0.5) -> float:
        try:
            return float(x)
        except Exception:
            return default

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

    def _iter_rows(self) -> List[Dict[str, Any]]:
        t = self.results_df
        if isinstance(t, DataFrame):
            return [row.to_dict() for _, row in t.iterrows()]
        if isinstance(t, Data) and isinstance(t.data, dict):
            return [dict(t.data)]
        return []

    @staticmethod
    def _normalize_label(label: str) -> str:
        lab = (label or "").strip().lower()
        if lab in ("entailment", "entailed", "entails", "supported", "support"):
            return "entails"
        if lab in ("contradiction", "contradicted", "contradicts"):
            return "contradicts"
        return "not_enough_info"

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

        low = raw.lower()
        if "contradict" in low:
            return "contradicts", 0.5, ""
        if "entail" in low or "supported" in low:
            return "entails", 0.5, ""
        return "not_enough_info", 0.5, ""

    def _call_llm(self, prompt: str) -> str:
        llm = self.llm
        if llm is None:
            raise ValueError("No judge LLM connected.")

        if hasattr(llm, "invoke"):
            resp = llm.invoke(prompt)
        elif hasattr(llm, "predict"):
            resp = llm.predict(prompt)
        elif hasattr(llm, "generate"):
            resp = llm.generate(prompt)
        else:
            raise TypeError("Judge LLM must expose invoke/predict/generate.")

        if isinstance(resp, str):
            return resp
        if hasattr(resp, "content") and isinstance(resp.content, str):
            return resp.content
        if isinstance(resp, dict) and isinstance(resp.get("content"), str):
            return resp["content"]
        return str(resp)

    # -------------------------
    # Chunk fix: build evidence_text from retrieval_chunks_json
    # -------------------------
    @staticmethod
    def _parse_chunks_json(v: Any) -> List[Dict[str, Any]]:
        if v is None:
            return []
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    return [x for x in j if isinstance(x, dict)]
            except Exception:
                return []
        if isinstance(v, dict):
            # sometimes a single chunk dict
            return [v]
        return []

    def _build_evidence_from_chunks(self, row: Dict[str, Any], top_k: int) -> Tuple[str, List[str]]:
        chunks = self._parse_chunks_json(row.get(self.chunks_json_col))
        if not chunks:
            return "", []

        # sort by rank if present
        def _rank_key(c: Dict[str, Any]) -> int:
            try:
                return int(c.get("rank", 10**9))
            except Exception:
                return 10**9

        chunks = sorted(chunks, key=_rank_key)
        if top_k > 0:
            chunks = chunks[:top_k]

        ids: List[str] = []
        parts: List[str] = []
        for c in chunks:
            cid = str(c.get("chunk_id", "") or "").strip()
            txt = str(c.get("text", "") or "").strip()
            rnk = c.get("rank", None)

            if cid:
                ids.append(cid)

            if txt:
                prefix = f"[rank {rnk}] " if rnk is not None else ""
                parts.append(prefix + txt)

        evidence = "\n\n".join(parts).strip()
        return evidence, ids

    # -------------------------
    # Prompt builders
    # -------------------------
    def _prompt_support(self, question: str, answer: str, evidence: str) -> str:
        strict_line = "Mode: STRICT" if bool(self.support_strict) else "Mode: LENIENT"
        instr = (self.support_judge_instructions or "").strip()
        return (
            f"{instr}\n\n"
            f"{strict_line}\n\n"
            "Return JSON only.\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Answer:\n{answer}\n\n"
            'JSON: {"label":"entails","confidence":0.83,"rationale":"..."}'
        )

    def _prompt_gold(self, question: str, answer: str, gold: str) -> str:
        instr = (self.gold_judge_instructions or "").strip()
        return (
            f"{instr}\n\n"
            "Return JSON only.\n\n"
            f"Question:\n{question}\n\n"
            f"Gold Reference:\n{gold}\n\n"
            f"Answer:\n{answer}\n\n"
            'JSON: {"label":"entails","confidence":0.83,"rationale":"..."}'
        )

    # -------------------------
    # CSV
    # -------------------------
    def _write_csv(self, rows: List[Dict[str, Any]]) -> Optional[str]:
        if not rows:
            return None

        filename = (self.csv_filename or "eval_nli_minimal.csv").strip() or "eval_nli_minimal.csv"
        out_dir = "/mnt/data" if os.path.isdir("/mnt/data") else None
        if out_dir:
            path = os.path.join(out_dir, filename)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            path = tmp.name
            tmp.close()

        # fixed minimal schema
        fieldnames = [
            "question",
            "generated_answer",
            "gold_answer",
            "evidence_chunk_ids",
            "support_label",
            "support_confidence",
            "support_supported",
            "support_hallucinated",
            "gold_label",
            "gold_confidence",
            "gold_correct",
        ]
        if bool(self.include_rationale):
            fieldnames += ["support_rationale", "gold_rationale"]
        if bool(self.include_evidence_text):
            fieldnames += ["evidence_text"]

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
        gcol = (self.gold_col or "gold_answer").strip() or "gold_answer"
        ecol = (self.evidence_col or "retrieved_context").strip() or "retrieved_context"

        conf_th = self._try_float(self.conf_threshold, 0.5)
        top_k = int(self.evidence_top_k or 5)
        max_chars = int(self.max_evidence_chars or 6000)

        n = 0
        n_support = 0
        n_hallu = 0
        n_gold_ok = 0

        for r in rows:
            # optional hygiene: skip bad rows
            if bool(self.skip_bad_rows):
                ok_val = r.get("ok", None)
                err_val = r.get("error", None)
                if ok_val is False:
                    continue
                if isinstance(err_val, str) and err_val.strip():
                    continue

            question = self._coerce_text(r.get(qcol, ""))
            answer = self._coerce_text(r.get(acol, ""))
            gold = self._coerce_text(r.get(gcol, ""))

            # skip totally empty padding rows
            if not question.strip() and not answer.strip() and not gold.strip():
                continue

            # evidence: prefer plain text col if available, else build from chunks_json
            evidence = self._coerce_text(r.get(ecol, "")) if ecol else ""
            chunk_ids: List[str] = []
            if not evidence.strip():
                evidence, chunk_ids = self._build_evidence_from_chunks(r, top_k=top_k)

            if max_chars > 0 and len(evidence) > max_chars:
                evidence = evidence[:max_chars] + "\n[TRUNCATED]"

            # --- Support NLI
            support_label, support_conf, support_rat = "not_enough_info", 0.0, ""
            support_supported = 0
            support_hallucinated = 0

            if bool(self.run_support_nli):
                if answer.strip() and evidence.strip():
                    raw = self._call_llm(self._prompt_support(question, answer, evidence))
                    support_label, support_conf, support_rat = self._parse_judge(raw)
                elif answer.strip() and not evidence.strip():
                    support_label, support_conf, support_rat = "not_enough_info", 1.0, "no evidence provided"
                else:
                    support_label, support_conf, support_rat = "not_enough_info", 1.0, "empty answer"

                support_supported = 1 if (support_label == "entails" and support_conf >= conf_th) else 0
                support_hallucinated = 1 if (answer.strip() and support_supported == 0) else 0

            # --- Gold NLI
            gold_label, gold_conf, gold_rat = "not_enough_info", 0.0, ""
            gold_correct = 0
            if bool(self.run_gold_nli):
                if answer.strip() and gold.strip():
                    raw = self._call_llm(self._prompt_gold(question, answer, gold))
                    gold_label, gold_conf, gold_rat = self._parse_judge(raw)
                elif answer.strip() and not gold.strip():
                    gold_label, gold_conf, gold_rat = "not_enough_info", 1.0, "no gold provided"
                else:
                    gold_label, gold_conf, gold_rat = "not_enough_info", 1.0, "empty answer"

                gold_correct = 1 if (gold_label == "entails" and gold_conf >= conf_th) else 0

            row_out: Dict[str, Any] = {
                "question": question,
                "generated_answer": answer,
                "gold_answer": gold,
                "evidence_chunk_ids": json.dumps(chunk_ids, ensure_ascii=False),
                "support_label": support_label,
                "support_confidence": support_conf,
                "support_supported": support_supported,
                "support_hallucinated": support_hallucinated,
                "gold_label": gold_label,
                "gold_confidence": gold_conf,
                "gold_correct": gold_correct,
            }
            if bool(self.include_rationale):
                row_out["support_rationale"] = support_rat
                row_out["gold_rationale"] = gold_rat
            if bool(self.include_evidence_text):
                row_out["evidence_text"] = evidence

            out_rows.append(row_out)

            n += 1
            n_support += int(support_supported) if bool(self.run_support_nli) else 0
            n_hallu += int(support_hallucinated) if bool(self.run_support_nli) else 0
            n_gold_ok += int(gold_correct) if bool(self.run_gold_nli) else 0

        support_rate = (n_support / n) if (n and bool(self.run_support_nli)) else None
        halluc_rate = (n_hallu / n) if (n and bool(self.run_support_nli)) else None
        gold_acc = (n_gold_ok / n) if (n and bool(self.run_gold_nli)) else None

        self._summary = {
            "rows": n,
            "conf_threshold": conf_th,
            "support_rate": support_rate,
            "hallucination_rate": halluc_rate,
            "gold_accuracy_rate": gold_acc,
            "used_chunks_json_col": self.chunks_json_col,
            "used_evidence_col": ecol,
            "evidence_top_k": top_k,
        }

        self.status = (
            f"rows={n} | "
            f"support_rate={support_rate if support_rate is not None else 'NA'} | "
            f"halluc_rate={halluc_rate if halluc_rate is not None else 'NA'} | "
            f"gold_acc={gold_acc if gold_acc is not None else 'NA'}"
        )

        if bool(self.write_csv):
            self._csv_path = self._write_csv(out_rows)

        return DataFrame([Data(data=x) for x in out_rows])

    def get_csv_file(self) -> Data:
        return Data(
            data={
                "file_path": self._csv_path,
                "file_name": (self.csv_filename or "eval_nli_minimal.csv"),
            }
        )

    def get_summary(self) -> Data:
        return Data(data=self._summary)
