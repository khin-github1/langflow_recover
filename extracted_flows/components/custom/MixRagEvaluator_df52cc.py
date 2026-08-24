# Recovered Langflow component
# type: MixRagEvaluator
# class: MixRagEvaluator
# used in 3 flow(s): Markdown Evaluation, Naive Flattening Evaluatoin , Summary Evaluation
# json path: node.data.node.template.code.value

# Langflow Custom Component: MixRAG-format RAG Evaluator (naive flattening baseline)
# -----------------------------------------------------------------------------
# Ground truth: test_qaFees.json  -> list of {question, answer, q_type, document_uid, ...}
#   * document_uid is used ONLY to look up the expected SOURCE url (not as a chunk key).
# Index: your pgvector collection (langchain_pg_embedding), chunks carry metadata {"source": url}.
#
# Produces two tables, grouped by q_type (col / cross / row / text):
#   1) Retrieval:  Type | Total | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR
#   2) Accuracy :  Type | Correct | Total | EM   (+ OVERALL)
#
# Why source-level relevance: chunks have no chunk_id and 46/59 answers are derived
# counts, so neither chunk-id nor answer-containment can define a gold chunk. A
# retrieval "hit" = the expected source's chunk appears in top-k. EM accuracy
# (retrieve -> generate -> normalized match) is where naive flattening visibly fails.
# -----------------------------------------------------------------------------

import json
import re

import pandas as pd
from langchain_community.vectorstores import PGVector

from lfx.custom.custom_component.component import Component
from lfx.io import (
    BoolInput,
    DropdownInput,
    FileInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    MultilineInput,
    Output,
    SecretStrInput,
    StrInput,
)
from lfx.schema.dataframe import DataFrame
from lfx.utils.connection_string_parser import transform_connection_string

# document_uid -> expected source URL (extracted from your two files; verify/edit as needed)
DEFAULT_DOC_SOURCE_MAP = {
    "846c299b-8e9e-4535-b163-a3d710cff248": "https://www.bu.ac.th/en/tuition-fees/bachelor-degree/2027",
    "26ee19d8-8e8c-4ba6-9f1f-941f3a729747": "https://ait.ac.th/admissions/tuition-and-fees/",
    "01002947-2dae-455a-96a0-a5f7afbf514a": "https://iis.ru.ac.th/index.php/academics/tuition-fees/tuition-fee-tables-new-rate",
}


class MixRagEvaluator(Component):
    display_name = "MixRAG Evaluator (Hit@K / EM)"
    description = (
        "Evaluates a pgvector naive-flattening index against a MixRAG-format JSON. "
        "Reports Hit@1/3/5, Recall@5, MRR and EM accuracy by q_type."
    )
    icon = "ruler"
    name = "MixRagEvaluator"

    inputs = [
        SecretStrInput(name="pg_server_url", display_name="PostgreSQL Server Connection String", required=True),
        StrInput(name="collection_name", display_name="Table", required=True, value="naive_data"),
        HandleInput(
            name="embedding",
            display_name="Embedding",
            input_types=["Embeddings"],
            required=True,
            info="Wire the same Ollama Embeddings component your index was built with.",
        ),
        IntInput(name="k", display_name="Top-K (retrieval depth)", value=5),
        FileInput(
            name="ground_truth_file",
            display_name="Ground-Truth JSON",
            file_types=["json"],
            required=True,
            info="MixRAG-format list with question / answer / q_type / document_uid.",
        ),
        DropdownInput(
            name="relevance_mode",
            display_name="Relevance Mode",
            options=["source_match", "answer_contains"],
            value="source_match",
            info=(
                "source_match: a hit = the question's expected source URL is in top-k "
                "(works for count answers; document-anchored). "
                "answer_contains: a hit = a retrieved chunk contains the gold answer "
                "(only meaningful for the ~13 string answers)."
            ),
        ),
        MultilineInput(
            name="doc_source_map",
            display_name="document_uid -> source URL (JSON)",
            value=json.dumps(DEFAULT_DOC_SOURCE_MAP, indent=2),
            info="Used only in source_match mode to map each question's document to its source URL.",
        ),
        BoolInput(
            name="run_generation",
            display_name="Run Generation (EM accuracy)",
            value=True,
            info="Retrieve -> generate -> normalized exact-match vs ground truth.",
        ),
        MessageTextInput(
            name="ollama_base_url",
            display_name="Ollama Base URL",
            value="",
            info="Same URL as your Ollama / Ollama Embeddings node (e.g. https://ollama.aitgpt.dev.brain.cs...).",
        ),
        MessageTextInput(name="gen_model", display_name="Generation Model", value="qwen3:8b"),
        # field names (advanced)
        MessageTextInput(name="q_field", display_name="Question field", value="question", advanced=True),
        MessageTextInput(name="a_field", display_name="Answer field", value="answer", advanced=True),
        MessageTextInput(name="type_field", display_name="Type field", value="q_type", advanced=True),
        MessageTextInput(name="doc_field", display_name="Document-uid field", value="document_uid", advanced=True),
    ]

    outputs = [
        Output(display_name="Retrieval Metrics", name="retrieval", method="retrieval_table"),
        Output(display_name="Accuracy (EM)", name="accuracy", method="accuracy_table"),
        Output(display_name="Per-Question CSV", name="per_question", method="per_question_csv"),
    ]

    _df = None  # cached per-question results

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _strip_think(t: str) -> str:
        return re.sub(r"<think>.*?</think>", "", t or "", flags=re.DOTALL).strip()

    @staticmethod
    def _norm(text) -> str:
        if isinstance(text, float):
            text = int(text) if text == int(text) else text
        s = str(text).lower().replace("฿", " ").replace(",", "")
        s = re.sub(r"\[.*?\]", " ", s)  # drop [Apply Now] style tags
        s = re.sub(r"[^\w\u0e00-\u0e7f.]", " ", s)  # keep word chars + Thai + dot
        s = re.sub(r"(\d)\.0\b", r"\1", s)  # 17.0 -> 17
        return re.sub(r"\s+", " ", s).strip()

    def _em(self, prediction: str, gold) -> int:
        pred = self._norm(self._strip_think(prediction))
        g = self._norm(gold)
        if not g:
            return 0
        if re.fullmatch(r"\d+", g):  # numeric gold: token must appear in prediction
            return int(g in pred.split())
        return int(g in pred)  # string gold: normalized containment

    def _ollama(self, prompt: str) -> str:
        import httpx

        base = (self.ollama_base_url or "").strip().rstrip("/")
        if not base or "localhost" in base or "127.0.0.1" in base:
            msg = "Ollama Base URL is empty/localhost. Set it to your Ollama node's URL."
            raise ValueError(msg)
        try:
            with httpx.Client(timeout=180) as client:
                resp = client.post(
                    base + "/api/generate",
                    json={"model": self.gen_model, "prompt": prompt, "stream": False, "think": False},
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
        except httpx.HTTPError as e:
            msg = f"Could not reach Ollama at '{base}/api/generate': {e}"
            raise ValueError(msg) from e

    def _generate(self, context: str, question: str) -> str:
        prompt = (
            "Use only the context to answer. Reply with just the answer value, no explanation.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return self._strip_think(self._ollama(prompt))

    # ------------------------------------------------------------------- core
    def _run(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        path = self.ground_truth_file
        if isinstance(path, list):
            path = path[0] if path else None
        if not path:
            msg = "No ground-truth JSON provided."
            raise ValueError(msg)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        conn = (self.pg_server_url or "").strip().rstrip(",").strip()
        if not conn:
            msg = "PostgreSQL connection string is empty."
            raise ValueError(msg)

        try:
            src_map = json.loads(self.doc_source_map) if self.doc_source_map else {}
        except json.JSONDecodeError as e:
            msg = f"document_uid->source map is not valid JSON: {e}"
            raise ValueError(msg) from e

        store = PGVector.from_existing_index(
            embedding=self.embedding,
            collection_name=self.collection_name,
            connection_string=transform_connection_string(conn),
        )

        k = int(self.k)
        rows = []
        for item in data:
            q = str(item[self.q_field])
            gold = item[self.a_field]
            qtype = str(item.get(self.type_field, "all")).lower()
            doc_uid = str(item.get(self.doc_field, ""))
            expected_source = src_map.get(doc_uid, "")

            docs = store.similarity_search(query=q, k=k)

            hit_rank = 0
            for idx, doc in enumerate(docs, start=1):
                meta = getattr(doc, "metadata", {}) or {}
                content = getattr(doc, "page_content", "")
                if self.relevance_mode == "source_match":
                    relevant = expected_source and meta.get("source", "") == expected_source
                else:  # answer_contains
                    gnorm = self._norm(gold)
                    relevant = bool(gnorm) and gnorm in self._norm(content)
                if relevant:
                    hit_rank = idx
                    break

            rr = (1.0 / hit_rank) if hit_rank else 0.0
            rec = {
                "q_type": qtype,
                "question": q,
                "gold_answer": gold,
                "expected_source": expected_source,
                "hit_rank": hit_rank,
                "rr": rr,
                "hit@1": int(0 < hit_rank <= 1),
                "hit@3": int(0 < hit_rank <= 3),
                "hit@5": int(0 < hit_rank <= 5),
                "recall@5": int(0 < hit_rank <= 5),  # single gold unit per question => = hit@5
            }

            if self.run_generation:
                context = "\n\n".join(getattr(d, "page_content", "") for d in docs)
                pred = self._generate(context, q)
                rec["generated_answer"] = pred[:400]
                rec["EM"] = self._em(pred, gold)

            rows.append(rec)

        self._df = pd.DataFrame(rows)
        return self._df

    # -------------------------------------------------------- formatting utils
    @staticmethod
    def _ascii(df: pd.DataFrame) -> str:
        return df.to_string(index=False)

    # ----------------------------------------------------- output: retrieval
    def retrieval_table(self) -> DataFrame:
        df = self._run()
        out = []
        for qtype, sub in sorted(df.groupby("q_type")):
            out.append(
                {
                    "Type": qtype,
                    "Total": len(sub),
                    "Hit@1": round(sub["hit@1"].mean(), 5),
                    "Hit@3": round(sub["hit@3"].mean(), 5),
                    "Hit@5": round(sub["hit@5"].mean(), 5),
                    "Recall@5": round(sub["recall@5"].mean(), 5),
                    "MRR": round(sub["rr"].mean(), 5),
                }
            )
        table = pd.DataFrame(out)
        self.status = self._ascii(table)
        return DataFrame(table)

    # ------------------------------------------------------ output: accuracy
    def accuracy_table(self) -> DataFrame:
        df = self._run()
        if "EM" not in df.columns:
            self.status = "Turn on 'Run Generation (EM accuracy)' to compute EM."
            return DataFrame(pd.DataFrame([{"note": "Run Generation is OFF"}]))
        out = []
        for qtype, sub in sorted(df.groupby("q_type")):
            correct = int(sub["EM"].sum())
            out.append(
                {"Type": qtype, "Correct": correct, "Total": len(sub), "EM": round(correct / len(sub), 5)}
            )
        total_correct = int(df["EM"].sum())
        out.append(
            {"Type": "OVERALL", "Correct": total_correct, "Total": len(df), "EM": round(total_correct / len(df), 5)}
        )
        table = pd.DataFrame(out)
        self.status = self._ascii(table)
        return DataFrame(table)

    # --------------------------------------------------- output: per-question
    def per_question_csv(self) -> DataFrame:
        df = self._run()
        cols = ["q_type", "question", "gold_answer"]
        if "generated_answer" in df.columns:
            cols += ["generated_answer", "EM"]
        cols += ["hit_rank", "hit@1", "hit@5", "rr"]
        sheet = df[cols].copy()
        self.status = sheet.to_csv(index=False)
        return DataFrame(sheet)