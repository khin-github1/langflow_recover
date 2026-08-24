# Recovered Langflow component
# type: nli_evaluator_support_gold_fast_v1
# class: NliEvaluatorSupportGoldFastV1
# used in 1 flow(s): CRCV Evaluation Runner 1.0.0 (backup)
# json path: node.data.node.template.code.value

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, HandleInput, IntInput, MessageTextInput, Output
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame


class NliEvaluatorSupportGoldFastV1(Component):
    display_name = "Evaluator (Support SR + Gen→Gold NLI) [Fast V1]"
    description = (
        "Lean NLI evaluator for:\n"
        "1) Support claim-level SR: evidence -> generated claims (hallucination / grounding)\n"
        "2) Generated -> Gold NLI: generated -> gold (semantic accuracy)\n\n"
        "Optimized for fewer LLM calls:\n"
        "- All support claims for one answer are judged in ONE batch call\n"
        "- Gold NLI uses ONE call per row\n"
        "- Optional caching and top-k evidence chunk limiting\n"
    )
    icon = "check-circle"
    name = "nli_evaluator_support_gold_fast_v1"

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
            value="retrieval_chunks_json",
            advanced=True,
        ),

        # Main thresholds
        MessageTextInput(
            name="conf_threshold",
            display_name="Confidence threshold θ (0-1)",
            value="0.5",
            advanced=True,
        ),
        MessageTextInput(
            name="sr_threshold",
            display_name="SR threshold τ_support (0-1)",
            value="0.8",
            advanced=True,
        ),

        # Limits
        IntInput(name="max_claims", display_name="Max claims per answer", value=5, advanced=True),
        IntInput(name="top_k_chunks", display_name="Top-k evidence chunks to use", value=4, advanced=True),
        IntInput(name="max_evidence_chars", display_name="Max evidence chars", value=4000, advanced=True),
        IntInput(name="max_gold_chars", display_name="Max gold chars", value=2500, advanced=True),
        IntInput(name="max_gen_chars", display_name="Max generated chars", value=2500, advanced=True),

        # Toggles
        BoolInput(name="run_support", display_name="Run Support SR", value=True, advanced=True),
        BoolInput(name="run_gold", display_name="Run Generated→Gold NLI", value=True, advanced=True),
        BoolInput(name="use_cache", display_name="Cache judge results in this run", value=True, advanced=True),
        BoolInput(name="include_evidence_text", display_name="Include evidence text in output", value=False, advanced=True),

        # Judge prompts
        MessageTextInput(
            name="judge_system",
            display_name="Judge System Message",
            value="You are a careful NLI judge. Return ONLY valid JSON. No code fences. No extra text.",
            advanced=True,
        ),
        MessageTextInput(
            name="support_batch_template",
            display_name="Support Batch Template",
            value=(
                "Task: Natural Language Inference (NLI) for grounding support.\n"
                "Judge each claim against the evidence.\n\n"
                "Return ONLY valid JSON in this exact format:\n"
                '{"claims":[{"id":1,"label":"entails","confidence":0.91},{"id":2,"label":"not_enough_info","confidence":0.22}]}\n\n'
                "Allowed labels: entails | contradicts | not_enough_info\n\n"
                "Rules:\n"
                "- entails: evidence supports the full claim\n"
                "- contradicts: evidence directly contradicts the claim\n"
                "- not_enough_info: evidence does not fully support the claim\n"
                "- Be strict. If support is partial or uncertain, use not_enough_info.\n\n"
                "Evidence:\n{{EVIDENCE}}\n\n"
                "Claims JSON:\n{{CLAIMS_JSON}}\n"
            ),
            advanced=True,
        ),
        MessageTextInput(
            name="gold_template",
            display_name="Generated→Gold Template",
            value=(
                "Task: Natural Language Inference (NLI) for semantic accuracy.\n"
                "Judge whether the GENERATED answer semantically entails the GOLD answer.\n\n"
                "Return ONLY valid JSON in this exact format:\n"
                '{"label":"entails","confidence":0.84}\n\n'
                "Allowed labels: entails | contradicts | not_enough_info\n\n"
                "Interpretation:\n"
                "- entails: generated answer covers the core meaning required by the gold answer\n"
                "- contradicts: generated answer conflicts with the gold answer\n"
                "- not_enough_info: generated answer misses key gold content or is only partially correct\n\n"
                "Premise (GENERATED):\n{{GENERATED}}\n\n"
                "Hypothesis (GOLD):\n{{GOLD}}\n"
            ),
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Eval DataFrame", name="eval_df", method="evaluate"),
        Output(display_name="Summary (Data)", name="summary", method="get_summary"),
    ]

    _re_json_obj = re.compile(r"\{.*\}", re.DOTALL)
    _re_json_arr = re.compile(r"\[.*\]", re.DOTALL)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._summary: Dict[str, Any] = {}
        self._judge_cache: Dict[str, Any] = {}

    # -------------------------
    # Basic helpers
    # -------------------------
    @staticmethod
    def _try_float(x: Any, default: float = 0.5) -> float:
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _clip(s: str, max_chars: int) -> str:
        s = str(s or "")
        if max_chars <= 0:
            return s
        return s if len(s) <= max_chars else (s[:max_chars] + "\n[TRUNCATED]")

    @staticmethod
    def _normalize_label(label: str) -> str:
        lab = (label or "").strip().lower()
        if lab in ("entailment", "entailed", "entails", "supported", "support"):
            return "entails"
        if lab in ("contradiction", "contradicted", "contradicts"):
            return "contradicts"
        return "not_enough_info"

    def _iter_rows(self) -> List[Dict[str, Any]]:
        t = self.results_df
        if isinstance(t, DataFrame):
            return [row.to_dict() for _, row in t.iterrows()]
        if isinstance(t, Data) and isinstance(t.data, dict):
            return [dict(t.data)]
        return []

    # -------------------------
    # JSON parsing
    # -------------------------
    @classmethod
    def _extract_json(cls, text: str) -> Any:
        raw = (text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        # whole string
        try:
            return json.loads(raw)
        except Exception:
            pass

        # first object
        m = cls._re_json_obj.search(raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        # first array
        m = cls._re_json_arr.search(raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        return None

    # -------------------------
    # LLM calling
    # -------------------------
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

    def _cached_call(self, prompt: str) -> str:
        key = prompt
        if bool(self.use_cache) and key in self._judge_cache:
            return self._judge_cache[key]

        raw = self._call_llm(prompt)

        if bool(self.use_cache):
            self._judge_cache[key] = raw
        return raw

    # -------------------------
    # Claim splitting
    # -------------------------
    @staticmethod
    def _split_claims(text: str, max_claims: int) -> List[str]:
        t = (text or "").strip()
        if not t:
            return []

        t = re.sub(r"\r\n?", "\n", t)

        lines: List[str] = []
        for line in t.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^(\-|\*|\u2022|\d+[\.\)])\s*", "", line).strip()
            if line:
                lines.append(line)

        claims: List[str] = []
        for ln in lines:
            sents = re.split(r"(?<=[\.\?\!])\s+", ln)
            for s in sents:
                s = s.strip()
                if len(s) >= 6:
                    claims.append(s)
                if len(claims) >= max_claims:
                    return claims[:max_claims]

        return claims[:max_claims]

    # -------------------------
    # Evidence parsing
    # -------------------------
    @staticmethod
    def _as_python(v: Any) -> Any:
        if isinstance(v, Data):
            v = v.data
        if isinstance(v, str):
            s = v.strip()
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                try:
                    return json.loads(s)
                except Exception:
                    return v
        return v

    def _evidence_text_and_ids(self, v: Any, top_k_chunks: int) -> Tuple[str, List[str]]:
        v = self._as_python(v)

        chunk_ids: List[str] = []
        parts: List[str] = []

        if isinstance(v, list):
            used = 0
            for item in v:
                if top_k_chunks > 0 and used >= top_k_chunks:
                    break

                if isinstance(item, dict):
                    cid = (item.get("chunk_id") or item.get("id") or "").strip()
                    if cid:
                        chunk_ids.append(str(cid))

                    txt = item.get("text") or item.get("content") or ""
                    rnk = item.get("rank", None)

                    if str(txt).strip():
                        prefix = f"[rank {rnk}] " if rnk is not None else ""
                        parts.append(prefix + str(txt).strip())
                        used += 1
                else:
                    if str(item).strip():
                        parts.append(str(item).strip())
                        used += 1

            return ("\n\n---\n\n".join(parts).strip(), chunk_ids)

        if isinstance(v, dict):
            for key in ("retrieval_chunks", "chunks", "retrieval_chunks_json"):
                if key in v and isinstance(v[key], list):
                    return self._evidence_text_and_ids(v[key], top_k_chunks=top_k_chunks)
            return (json.dumps(v, ensure_ascii=False), chunk_ids)

        if isinstance(v, str):
            return (v, chunk_ids)

        return (str(v), chunk_ids)

    # -------------------------
    # Support batch judge
    # -------------------------
    def _judge_support_batch(self, evidence: str, claims: List[str]) -> List[Dict[str, Any]]:
        if not claims:
            return []

        claims_payload = [{"id": i + 1, "claim": c} for i, c in enumerate(claims)]

        prompt = (
            (self.judge_system or "").strip()
            + "\n\n"
            + (self.support_batch_template or "")
                .replace("{{EVIDENCE}}", evidence)
                .replace("{{CLAIMS_JSON}}", json.dumps(claims_payload, ensure_ascii=False))
        ).strip()

        raw = self._cached_call(prompt)
        obj = self._extract_json(raw)

        out: List[Dict[str, Any]] = []
        default_map = {i + 1: {"id": i + 1, "label": "not_enough_info", "confidence": 0.0} for i in range(len(claims))}

        if isinstance(obj, dict) and isinstance(obj.get("claims"), list):
            items = obj["claims"]
        elif isinstance(obj, list):
            items = obj
        else:
            items = []

        for it in items:
            if not isinstance(it, dict):
                continue
            cid = it.get("id")
            try:
                cid = int(cid)
            except Exception:
                continue
            if cid < 1 or cid > len(claims):
                continue

            default_map[cid] = {
                "id": cid,
                "label": self._normalize_label(str(it.get("label", ""))),
                "confidence": max(0.0, min(1.0, self._try_float(it.get("confidence", 0.0), 0.0))),
            }

        for i in range(1, len(claims) + 1):
            out.append(default_map[i])

        return out

    # -------------------------
    # Gold judge: Generated -> Gold
    # -------------------------
    def _judge_gold(self, generated: str, gold: str) -> Tuple[str, float]:
        prompt = (
            (self.judge_system or "").strip()
            + "\n\n"
            + (self.gold_template or "")
                .replace("{{GENERATED}}", generated)
                .replace("{{GOLD}}", gold)
        ).strip()

        raw = self._cached_call(prompt)
        obj = self._extract_json(raw)

        if isinstance(obj, dict):
            label = self._normalize_label(str(obj.get("label", "")))
            conf = max(0.0, min(1.0, self._try_float(obj.get("confidence", 0.0), 0.0)))
            return label, conf

        low = (raw or "").lower()
        if "contradict" in low:
            return "contradicts", 0.5
        if "entail" in low or "supported" in low:
            return "entails", 0.5
        return "not_enough_info", 0.5

    # -------------------------
    # Main evaluation
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
        tau_support = max(0.0, min(1.0, self._try_float(self.sr_threshold, 0.8)))

        max_claims = max(1, int(self.max_claims or 5))
        top_k_chunks = max(1, int(self.top_k_chunks or 4))
        max_evidence = int(self.max_evidence_chars or 4000)
        max_gold = int(self.max_gold_chars or 2500)
        max_gen = int(self.max_gen_chars or 2500)

        n = 0

        # Support summary
        sup_supported_cnt = 0
        sup_hallu_cnt = 0
        sr_sum = 0.0
        sr_count = 0
        sr_pass_cnt = 0
        sr_contra_any_cnt = 0

        # Gold summary
        gold_correct_cnt = 0
        gold_contra_cnt = 0

        out_rows: List[Dict[str, Any]] = []

        for r in rows:
            question = str(r.get(qcol, "") or "")
            gen = self._clip(str(r.get(gencol, "") or ""), max_gen)
            gold = self._clip(str(r.get(goldcol, "") or ""), max_gold)

            evidence_text, chunk_ids = self._evidence_text_and_ids(r.get(evcol, ""), top_k_chunks=top_k_chunks)
            evidence_text = self._clip(evidence_text, max_evidence)

            # -----------------
            # Support SR
            # -----------------
            sup_label = ""
            sup_confidence = ""
            sup_supported = ""
            sup_hallucinated = ""

            sr_val = None
            sr_supported = 0
            sr_total = 0
            sr_pass = ""
            sr_contradiction_any = 0
            sr_details: List[Dict[str, Any]] = []

            if bool(self.run_support):
                if not gen.strip() or not evidence_text.strip():
                    sr_val = 0.0
                    sr_supported = 0
                    sr_total = 0
                    sr_pass = 0
                    sr_contradiction_any = 0
                    sup_label = "not_enough_info"
                    sup_confidence = 0.0
                    sup_supported = 0
                    sup_hallucinated = 0 if not gen.strip() else 1
                else:
                    claims = self._split_claims(gen, max_claims=max_claims)
                    sr_total = len(claims)

                    judged = self._judge_support_batch(evidence_text, claims)

                    for idx, item in enumerate(judged):
                        lab = self._normalize_label(str(item.get("label", "")))
                        conf = max(0.0, min(1.0, self._try_float(item.get("confidence", 0.0), 0.0)))

                        is_supported = 1 if (lab == "entails" and conf >= conf_th) else 0
                        sr_supported += is_supported

                        if lab == "contradicts" and conf >= conf_th:
                            sr_contradiction_any = 1

                        sr_details.append(
                            {
                                "claim": claims[idx] if idx < len(claims) else "",
                                "label": lab,
                                "confidence": conf,
                                "supported": is_supported,
                            }
                        )

                    sr_val = (sr_supported / sr_total) if sr_total else 0.0
                    sr_pass_int = 1 if (sr_val >= tau_support and sr_contradiction_any == 0) else 0
                    sr_pass = sr_pass_int

                    if sr_contradiction_any == 1:
                        sup_label = "contradicts"
                    else:
                        sup_label = "entails" if sr_pass_int == 1 else "not_enough_info"

                    sup_confidence = float(sr_val)
                    sup_supported = sr_pass_int
                    sup_hallucinated = 1 if (gen.strip() and sr_pass_int == 0) else 0

                sup_supported_cnt += int(sup_supported) if str(sup_supported).isdigit() else 0
                sup_hallu_cnt += int(sup_hallucinated) if str(sup_hallucinated).isdigit() else 0
                sr_sum += float(sr_val or 0.0)
                sr_count += 1
                sr_pass_cnt += int(sr_pass) if str(sr_pass).isdigit() else 0
                sr_contra_any_cnt += int(sr_contradiction_any)

            # -----------------
            # Generated -> Gold
            # -----------------
            gold_label_gen_to_gold = ""
            gold_conf_gen_to_gold = ""
            gold_correct_gen_to_gold = ""
            gold_contradiction_gen_to_gold = ""

            if bool(self.run_gold):
                if gen.strip() and gold.strip():
                    glab, gconf = self._judge_gold(gen, gold)
                else:
                    glab, gconf = "not_enough_info", 0.0

                gold_ok = 1 if (glab == "entails" and float(gconf) >= conf_th) else 0
                gold_contra = 1 if (glab == "contradicts" and float(gconf) >= conf_th) else 0

                gold_label_gen_to_gold = glab
                gold_conf_gen_to_gold = gconf
                gold_correct_gen_to_gold = gold_ok
                gold_contradiction_gen_to_gold = gold_contra

                gold_correct_cnt += gold_ok
                gold_contra_cnt += gold_contra

            row_out: Dict[str, Any] = {
                "question": question,
                "generated_answer": gen,
                "gold_answer": gold,
                "chunk_ids": json.dumps(chunk_ids, ensure_ascii=False),

                # Support / hallucination
                "sup_label": sup_label,
                "sup_confidence": sup_confidence,
                "sup_supported": sup_supported,
                "sup_hallucinated": sup_hallucinated,

                "sr": sr_val,
                "sr_supported": sr_supported,
                "sr_total": sr_total,
                "sr_pass": sr_pass,
                "tau_support": tau_support,
                "sr_contradiction_any": sr_contradiction_any,
                "max_claims_used": max_claims,

                # Accuracy
                "gold_label_gen_to_gold": gold_label_gen_to_gold,
                "gold_conf_gen_to_gold": gold_conf_gen_to_gold,
                "gold_correct_gen_to_gold": gold_correct_gen_to_gold,
                "gold_contradiction_gen_to_gold": gold_contradiction_gen_to_gold,
            }

            if bool(self.include_evidence_text):
                row_out["evidence"] = evidence_text

            row_out["sr_details_json"] = json.dumps(sr_details, ensure_ascii=False)

            out_rows.append(row_out)
            n += 1

        self._summary = {
            "rows": n,
            "conf_threshold_theta": conf_th,
            "tau_support": tau_support,
            "max_claims": max_claims,
            "top_k_chunks": top_k_chunks,
            "cache_enabled": bool(self.use_cache),

            "support_rate": (sup_supported_cnt / n) if (n and bool(self.run_support)) else None,
            "hallucination_rate": (sup_hallu_cnt / n) if (n and bool(self.run_support)) else None,
            "sr_mean": (sr_sum / sr_count) if sr_count else None,
            "sr_pass_rate": (sr_pass_cnt / n) if (n and bool(self.run_support)) else None,
            "sr_contradiction_any_rate": (sr_contra_any_cnt / n) if (n and bool(self.run_support)) else None,

            "gold_gen_to_gold_accuracy": (gold_correct_cnt / n) if (n and bool(self.run_gold)) else None,
            "gold_gen_to_gold_contradiction_rate": (gold_contra_cnt / n) if (n and bool(self.run_gold)) else None,

            "cols": {
                "q": qcol,
                "gen": gencol,
                "gold": goldcol,
                "evidence": evcol,
            },
        }

        parts = [f"Rows={n}", f"θ={conf_th:.2f}"]
        if bool(self.run_support) and n:
            parts.append(f"support={sup_supported_cnt/n:.3f}")
            parts.append(f"hallu={sup_hallu_cnt/n:.3f}")
            parts.append(f"SRμ={sr_sum/sr_count:.3f}" if sr_count else "SRμ=NA")
            parts.append(f"SRpass={sr_pass_cnt/n:.3f}")
        if bool(self.run_gold) and n:
            parts.append(f"gen→gold={gold_correct_cnt/n:.3f}")

        self.status = " | ".join(parts)

        return DataFrame([Data(data=x) for x in out_rows])

    def get_summary(self) -> Data:
        return Data(data=self._summary)