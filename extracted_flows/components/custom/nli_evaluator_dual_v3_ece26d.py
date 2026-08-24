# Recovered Langflow component
# type: nli_evaluator_dual_v3
# class: NliEvaluatorDualV3
# used in 2 flow(s): CRCV Evaluation Runner 0.0.01, CRCV Evaluation Runner 1.0.0
# json path: node.data.node.template.code.value

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, HandleInput, IntInput, MessageTextInput, Output
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame


class NliEvaluatorDualV3(Component):
    display_name = "Evaluator (NLI: Support + Gold, auto-chunk-ids) [V3 no-CSV]"
    description = (
        "LLM-judge NLI in two ways:\n"
        "1) Support NLI: Evidence -> Generated (grounding / hallucination)\n"
        "2) Gold NLI: bidirectional Generated <-> Gold (paraphrase-robust accuracy)\n\n"
        "Evidence column can be plain text OR retrieval_chunks_json (list/JSON). "
        "Chunk IDs are auto-extracted from the evidence chunks when available.\n"
        "Outputs: DataFrame + summary (no CSV writing)."
    )
    icon = "check-circle"
    name = "nli_evaluator_dual_v3"

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

        # Column mapping
        MessageTextInput(name="question_col", display_name="Question Column", value="question", advanced=True),
        MessageTextInput(name="generated_col", display_name="Generated Answer Column", value="generated_answer", advanced=True),
        MessageTextInput(name="gold_col", display_name="Gold Answer Column", value="gold_answer", advanced=True),

        MessageTextInput(
            name="evidence_col",
            display_name="Evidence Column (text OR retrieval_chunks_json)",
            value="retrieval_chunks_json",  # <-- set default to your real column
            advanced=True,
        ),

        # Toggles
        BoolInput(name="eval_support", display_name="Run Support NLI (gen vs evidence)", value=True, advanced=True),
        BoolInput(name="eval_gold", display_name="Run Gold NLI (gen vs gold, bidirectional)", value=True, advanced=True),

        # Truncation / limits
        IntInput(name="max_evidence_chars", display_name="Max evidence chars", value=6000, advanced=True),
        IntInput(name="max_gold_chars", display_name="Max gold chars", value=4000, advanced=True),
        IntInput(name="max_gen_chars", display_name="Max generated chars", value=4000, advanced=True),

        # Threshold / strictness
        MessageTextInput(name="conf_threshold", display_name="Confidence threshold (0-1)", value="0.5", advanced=True),
        BoolInput(name="support_strict", display_name="Support STRICT mode", value=True, advanced=True),

        # Judge prompts (editable)
        MessageTextInput(
            name="judge_system",
            display_name="Judge System Message",
            value="You are a careful NLI judge. Return ONLY valid JSON. No code fences.",
            advanced=True,
        ),
        MessageTextInput(
            name="judge_user_template",
            display_name="Judge User Template ({{MODE}}, {{PREMISE}}, {{HYPOTHESIS}})",
            value=(
                "Task: Natural Language Inference (NLI).\n"
                "Return ONLY valid JSON on one line with keys: label, confidence, rationale.\n\n"
                "Labels: entails | contradicts | not_enough_info\n\n"
                "Mode: {{MODE}}\n"
                "- entails: Premise supports ALL claims in Hypothesis.\n"
                "- contradicts: Premise directly contradicts Hypothesis.\n"
                "- not_enough_info: Premise does not fully support Hypothesis.\n\n"
                "Premise:\n{{PREMISE}}\n\n"
                "Hypothesis:\n{{HYPOTHESIS}}\n\n"
                'JSON: {"label":"entails","confidence":0.83,"rationale":"brief"}'
            ),
            advanced=True,
        ),

        # Output shaping (optional)
        BoolInput(name="include_evidence_text", display_name="Include evidence text in output", value=False, advanced=True),
        BoolInput(name="include_rationale", display_name="Include judge rationale in output", value=False, advanced=True),
    ]

    outputs = [
        Output(display_name="Eval DataFrame", name="eval_df", method="evaluate"),
        Output(display_name="Summary (Data)", name="summary", method="get_summary"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._summary: Dict[str, Any] = {}

    # -------------------------
    # Helpers
    # -------------------------
    _re_json_obj = re.compile(r"\{.*\}", re.DOTALL)

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
        return s if len(s) <= max_chars else (s[:max_chars] + "\n[TRUNCATED]")

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

    @classmethod
    def _parse_judge(cls, text: str) -> Tuple[str, float, str]:
        raw = (text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        m = cls._re_json_obj.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                label = cls._normalize_label(str(obj.get("label", "")))
                conf = cls._try_float(obj.get("confidence", 0.0), 0.0)
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

    def _fill_template(self, template: str, *, mode: str, premise: str, hypothesis: str) -> str:
        t = template or ""
        t = t.replace("{{MODE}}", mode)
        t = t.replace("{{PREMISE}}", premise)
        t = t.replace("{{HYPOTHESIS}}", hypothesis)
        return t

    def _judge(self, *, mode: str, premise: str, hypothesis: str) -> Tuple[str, float, str]:
        system = (self.judge_system or "").strip()
        user = self._fill_template(self.judge_user_template or "", mode=mode, premise=premise, hypothesis=hypothesis)
        prompt = (system + "\n\n" + user).strip()
        raw = self._call_llm(prompt)
        return self._parse_judge(raw)

    # -------------------------
    # Evidence + chunk id extraction
    # -------------------------
    @staticmethod
    def _as_python(v: Any) -> Any:
        if isinstance(v, Data):
            v = v.data
        if isinstance(v, str):
            s = v.strip()
            # try JSON parse if it looks like JSON
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                try:
                    return json.loads(s)
                except Exception:
                    return v
        return v

    def _evidence_text_and_ids(self, v: Any) -> Tuple[str, List[str]]:
        v = self._as_python(v)

        chunk_ids: List[str] = []
        parts: List[str] = []

        # list of chunk dicts
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    cid = (item.get("chunk_id") or item.get("id") or "").strip()
                    if cid:
                        chunk_ids.append(str(cid))

                    txt = item.get("text") or item.get("content") or ""
                    rnk = item.get("rank", None)
                    if str(txt).strip():
                        prefix = f"[rank {rnk}] " if rnk is not None else ""
                        parts.append(prefix + str(txt).strip())
                else:
                    if str(item).strip():
                        parts.append(str(item).strip())
            return ("\n\n---\n\n".join(parts).strip(), chunk_ids)

        # dict maybe holding chunks under a key (just in case)
        if isinstance(v, dict):
            for key in ("retrieval_chunks", "chunks", "retrieval_chunks_json"):
                if key in v and isinstance(v[key], list):
                    return self._evidence_text_and_ids(v[key])
            # otherwise treat as text-ish
            return (json.dumps(v, ensure_ascii=False), chunk_ids)

        # plain text evidence
        if isinstance(v, str):
            return (v, chunk_ids)

        return (str(v), chunk_ids)

    # -------------------------
    # Main
    # -------------------------
    def evaluate(self) -> DataFrame:
        rows = self._iter_rows()
        if not rows:
            self.status = "No rows received."
            return DataFrame([])

        qcol = (self.question_col or "question").strip()
        gencol = (self.generated_col or "generated_answer").strip()
        goldcol = (self.gold_col or "gold_answer").strip()
        evcol = (self.evidence_col or "retrieval_chunks_json").strip()

        conf_th = max(0.0, min(1.0, self._try_float(self.conf_threshold, 0.5)))
        strict_mode = "STRICT" if bool(self.support_strict) else "LENIENT"

        max_evidence = int(self.max_evidence_chars or 6000)
        max_gold = int(self.max_gold_chars or 4000)
        max_gen = int(self.max_gen_chars or 4000)

        n = 0
        sup_supported_cnt = 0
        sup_hallu_cnt = 0
        gold_bidir_correct_cnt = 0
        gold_any_correct_cnt = 0
        gold_any_contra_cnt = 0

        out_rows: List[Dict[str, Any]] = []

        for r in rows:
            question = str(r.get(qcol, "") or "")
            gen = self._clip(str(r.get(gencol, "") or ""), max_gen)
            gold = self._clip(str(r.get(goldcol, "") or ""), max_gold)

            evidence_text, chunk_ids = self._evidence_text_and_ids(r.get(evcol, ""))
            evidence_text = self._clip(evidence_text, max_evidence)

            # Support
            sup_label, sup_conf, sup_rat = "", "", ""
            sup_supported, sup_hallu = "", ""
            if bool(self.eval_support):
                if gen.strip() and evidence_text.strip():
                    sup_label_v, sup_conf_v, sup_rat_v = self._judge(
                        mode=strict_mode, premise=evidence_text, hypothesis=gen
                    )
                else:
                    sup_label_v, sup_conf_v, sup_rat_v = "not_enough_info", 1.0, "missing gen/evidence"

                sup_supported_int = 1 if (sup_label_v == "entails" and float(sup_conf_v) >= conf_th) else 0
                sup_hallu_int = 1 if (gen.strip() and sup_supported_int == 0) else 0

                sup_label, sup_conf, sup_rat = sup_label_v, sup_conf_v, sup_rat_v
                sup_supported, sup_hallu = sup_supported_int, sup_hallu_int

                sup_supported_cnt += sup_supported_int
                sup_hallu_cnt += sup_hallu_int

            # Gold (bidirectional)
            gg_label, gg_conf, gg_rat = "", "", ""
            gG_label, gG_conf, gG_rat = "", "", ""
            gold_bidir, gold_any, contra_any = "", "", ""
            if bool(self.eval_gold):
                if gen.strip() and gold.strip():
                    gg_label_v, gg_conf_v, gg_rat_v = self._judge(mode="GOLD", premise=gold, hypothesis=gen)
                    gG_label_v, gG_conf_v, gG_rat_v = self._judge(mode="GOLD", premise=gen, hypothesis=gold)
                else:
                    gg_label_v, gg_conf_v, gg_rat_v = "not_enough_info", 1.0, "missing gen/gold"
                    gG_label_v, gG_conf_v, gG_rat_v = "not_enough_info", 1.0, "missing gen/gold"

                gg_ok = (gg_label_v == "entails" and float(gg_conf_v) >= conf_th)
                gG_ok = (gG_label_v == "entails" and float(gG_conf_v) >= conf_th)

                gold_bidir_int = 1 if (gg_ok and gG_ok) else 0
                gold_any_int = 1 if (gg_ok or gG_ok) else 0
                contra_any_int = 1 if (gg_label_v == "contradicts" or gG_label_v == "contradicts") else 0

                gg_label, gg_conf, gg_rat = gg_label_v, gg_conf_v, gg_rat_v
                gG_label, gG_conf, gG_rat = gG_label_v, gG_conf_v, gG_rat_v
                gold_bidir, gold_any, contra_any = gold_bidir_int, gold_any_int, contra_any_int

                gold_bidir_correct_cnt += gold_bidir_int
                gold_any_correct_cnt += gold_any_int
                gold_any_contra_cnt += contra_any_int

            row_out: Dict[str, Any] = {
                "question": question,
                "generated_answer": gen,
                "gold_answer": gold,
                "chunk_ids": json.dumps(chunk_ids, ensure_ascii=False),
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
                "gold_contradiction_any": contra_any,
            }

            if bool(self.include_evidence_text):
                row_out["evidence"] = evidence_text
            if bool(self.include_rationale):
                row_out["sup_rationale"] = sup_rat
                row_out["gold_rationale_gold_to_gen"] = gg_rat
                row_out["gold_rationale_gen_to_gold"] = gG_rat

            out_rows.append(row_out)
            n += 1

        self._summary = {
            "rows": n,
            "conf_threshold": conf_th,
            "support_mode": strict_mode if bool(self.eval_support) else None,
            "support_rate": (sup_supported_cnt / n) if (n and bool(self.eval_support)) else None,
            "hallucination_rate": (sup_hallu_cnt / n) if (n and bool(self.eval_support)) else None,
            "gold_bidirectional_accuracy": (gold_bidir_correct_cnt / n) if (n and bool(self.eval_gold)) else None,
            "gold_any_direction_accuracy": (gold_any_correct_cnt / n) if (n and bool(self.eval_gold)) else None,
            "gold_contradiction_rate_any_direction": (gold_any_contra_cnt / n) if (n and bool(self.eval_gold)) else None,
            "cols": {"q": qcol, "gen": gencol, "gold": goldcol, "evidence": evcol},
        }

        parts = [f"Rows={n}", f"th={conf_th:.2f}"]
        if bool(self.eval_support) and n:
            parts.append(f"support={sup_supported_cnt/n:.3f}")
            parts.append(f"hallu={sup_hallu_cnt/n:.3f}")
        if bool(self.eval_gold) and n:
            parts.append(f"gold_bidir={gold_bidir_correct_cnt/n:.3f}")
            parts.append(f"gold_any={gold_any_correct_cnt/n:.3f}")
        self.status = " | ".join(parts)

        return DataFrame([Data(data=x) for x in out_rows])

    def get_summary(self) -> Data:
        return Data(data=self._summary)
