# Recovered Langflow component
# type: RetrievalEvaluator
# class: RetrievalEvaluator
# used in 3 flow(s): Markdown Evaluation, Naive Flattening Evaluatoin , Summary Evaluation
# json path: node.data.node.template.code.value

# Langflow Custom Component: Retrieval Evaluator (MRR / Hit@K / Accuracy)
# ----------------------------------------------------------------------
# Purpose: Loop a ground-truth QA set through the SAME retrieval stack as your
# naive baseline flow (Ollama Embeddings handle + PGVector table + k) and report
# MRR, Hit@1/3/5, and (optionally) answer accuracy, broken down by
# relationship_type (row / column / cross). This isolates the RETRIEVER so the
# structure-break failure is not masked by the LLM guessing the answer.
#
# Wiring:
#   Ollama Embeddings  --(Embeddings handle)-->  [this component].embedding
#   Provide the same PostgreSQL connection string + Table name as your PGVector.
#   Upload the ground-truth CSV (the 30-question set).
#
# Two outputs:
#   - "Per-Question Results"  : one row per question (rank, RR, hit flags, snippet)
#   - "Metrics Summary"       : overall + per-relationship_type aggregates
# ----------------------------------------------------------------------

import re

import pandas as pd

# Imports must be plain top-level statements (no try/except) so the Langflow
# custom-component validator can bind them. These lfx paths match your build.
from langchain_community.vectorstores import PGVector

from lfx.custom.custom_component.component import Component
from lfx.io import (
    BoolInput,
    DropdownInput,
    FileInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    Output,
    SecretStrInput,
    StrInput,
)
from lfx.schema.dataframe import DataFrame
from lfx.utils.connection_string_parser import transform_connection_string


class RetrievalEvaluator(Component):
    display_name = "Retrieval Evaluator (MRR/Hit@K)"
    description = (
        "Loops a ground-truth QA CSV through PGVector retrieval and reports "
        "MRR, Hit@1/3/5 and optional answer accuracy, stratified by relationship_type."
    )
    icon = "ruler"
    name = "RetrievalEvaluator"

    inputs = [
        # --- Retrieval stack (mirror your flow exactly) ---
        SecretStrInput(
            name="pg_server_url",
            display_name="PostgreSQL Server Connection String",
            required=True,
        ),
        StrInput(name="collection_name", display_name="Table", required=True, value="naive_data"),
        HandleInput(
            name="embedding",
            display_name="Embedding",
            input_types=["Embeddings"],
            required=True,
            info="Wire the SAME Ollama Embeddings component your flow uses.",
        ),
        IntInput(
            name="k",
            display_name="Top-K (retrieval depth)",
            value=5,
            info="Number of chunks retrieved per question. Hit@K is reported up to this K.",
        ),
        # --- Ground truth ---
        FileInput(
            name="ground_truth_file",
            display_name="Ground-Truth CSV",
            file_types=["csv"],
            required=True,
            info="Columns: id, question, answer, relationship_type [, gold_chunk_id, gold_keywords].",
        ),
        MessageTextInput(
            name="question_col", display_name="Question column", value="question", advanced=True
        ),
        MessageTextInput(
            name="answer_col", display_name="Answer column", value="answer", advanced=True
        ),
        MessageTextInput(
            name="stratum_col",
            display_name="Stratum column",
            value="relationship_type",
            advanced=True,
        ),
        # --- Relevance judging ---
        DropdownInput(
            name="match_mode",
            display_name="Relevance Match Mode",
            options=["answer_contains", "keyword_all", "chunk_id"],
            value="answer_contains",
            info=(
                "answer_contains: chunk text contains the gold answer (zero setup, noisy). "
                "keyword_all: chunk contains ALL pipe-separated tokens in 'gold_keywords' "
                "(best for proving row+column structure breaks). "
                "chunk_id: chunk metadata id == 'gold_chunk_id' (most rigorous, needs labels)."
            ),
        ),
        MessageTextInput(
            name="keyword_col", display_name="gold_keywords column", value="gold_keywords", advanced=True
        ),
        MessageTextInput(
            name="chunk_id_col", display_name="gold_chunk_id column", value="gold_chunk_id", advanced=True
        ),
        MessageTextInput(
            name="metadata_id_key",
            display_name="Chunk metadata id key",
            value="chunk_id",
            advanced=True,
            info="Which key in each retrieved chunk's metadata holds its id (for chunk_id mode).",
        ),
        # --- Optional end-to-end answer accuracy ---
        BoolInput(
            name="run_generation",
            display_name="Run Generation (answer accuracy)",
            value=False,
            info="If on, also calls Ollama per question and scores the answer. "
            "Note: this mixes retrieval + LLM behaviour; retrieval metrics are the clean proof.",
        ),
        MessageTextInput(
            name="ollama_base_url",
            display_name="Ollama Base URL",
            value="",
            info="Required only if Run Generation is on. Use the same URL as your "
            "Ollama / Ollama Embeddings node (e.g. https://ollama.aitgpt.dev.brain.cs...).",
        ),
        MessageTextInput(
            name="gen_model",
            display_name="Generation Model",
            value="qwen3:8b",
            info="Ollama model used for generation and (if selected) judging.",
        ),
        DropdownInput(
            name="judge_mode",
            display_name="Answer Judge",
            options=["llm_judge", "string_match"],
            value="llm_judge",
            info=(
                "How a generated answer is graded against ground truth. "
                "llm_judge: the model decides if the answer conveys the gold facts "
                "(robust for ranges / 'X vs Y' / course names). "
                "string_match: normalized numeric/substring containment (strict, brittle)."
            ),
        ),
    ]

    outputs = [
        Output(display_name="Per-Question Results", name="results", method="evaluate"),
        Output(display_name="Metrics Summary", name="summary", method="summarize"),
    ]

    # cache so both outputs reuse a single pass
    _per_question_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").lower()
        text = text.replace("฿", " ").replace(",", "")
        text = re.sub(r"[^\w\s.]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _strip_think(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _is_relevant(self, doc, row) -> bool:
        mode = self.match_mode
        content_norm = self._normalize(getattr(doc, "page_content", ""))

        if mode == "chunk_id":
            gold = str(row.get(self.chunk_id_col, "")).strip()
            meta = getattr(doc, "metadata", {}) or {}
            return bool(gold) and str(meta.get(self.metadata_id_key, "")).strip() == gold

        if mode == "keyword_all":
            raw = str(row.get(self.keyword_col, "")).strip()
            if not raw:
                return False
            tokens = [self._normalize(t) for t in raw.split("|") if t.strip()]
            return all(tok in content_norm for tok in tokens)

        # default: answer_contains
        gold = self._normalize(str(row.get(self.answer_col, "")))
        if not gold:
            return False
        # require the numeric core if present, else full substring
        nums = re.findall(r"\d+\.?\d*", gold)
        if nums:
            return all(n in content_norm for n in nums)
        return gold in content_norm

    def _answer_correct(self, prediction: str, gold: str) -> bool:
        pred = self._normalize(self._strip_think(prediction))
        g = self._normalize(gold)
        nums = re.findall(r"\d+\.?\d*", g)
        if nums:
            return all(n in pred for n in nums)
        return g in pred

    def _generate(self, context: str, question: str) -> str:
        prompt = (
            "Answer the user's question based only on the following context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Give a clear, concise answer."
        )
        return self._ollama(prompt)

    def _ollama(self, prompt: str) -> str:
        import httpx

        base = (self.ollama_base_url or "").strip().rstrip("/")
        if not base or "localhost" in base or "127.0.0.1" in base:
            msg = (
                f"Ollama Base URL is '{base or 'empty'}', which is not reachable from Langflow. "
                "Set it to the same URL your Ollama / Ollama Embeddings node uses "
                "(e.g. https://ollama.aitgpt.dev.brain.cs...)."
            )
            raise ValueError(msg)
        url = base + "/api/generate"
        payload = {"model": self.gen_model, "prompt": prompt, "stream": False, "think": False}
        try:
            with httpx.Client(timeout=180) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "")
        except httpx.HTTPError as e:
            msg = f"Could not reach Ollama generate endpoint at '{url}'. Underlying error: {e}"
            raise ValueError(msg) from e

    def _llm_judge(self, question: str, gold: str, prediction: str) -> int:
        prompt = (
            "You are grading a question-answering system. Decide whether the SYSTEM ANSWER "
            "states the same key facts/values as the REFERENCE ANSWER. Ignore wording, "
            "formatting and extra explanation; judge only the facts.\n\n"
            f"QUESTION: {question}\n"
            f"REFERENCE ANSWER: {gold}\n"
            f"SYSTEM ANSWER: {prediction}\n\n"
            "Reply with exactly one word: YES (facts match) or NO (they do not)."
        )
        verdict = self._strip_think(self._ollama(prompt)).strip().upper()
        return int(verdict.startswith("YES"))

    # --------------------------------------------------------------- main pass
    def _run_eval(self) -> pd.DataFrame:
        if self._per_question_df is not None:
            return self._per_question_df

        path = self.ground_truth_file
        if isinstance(path, list):
            path = path[0] if path else None
        if not path:
            msg = "No ground-truth CSV provided."
            raise ValueError(msg)
        gt = pd.read_csv(path)

        # Sanitize the connection string: a blank field makes langchain fall back
        # to localhost:5432 and fail with an opaque "connection refused".
        conn = (self.pg_server_url or "").strip().rstrip(",").strip()
        if not conn:
            msg = (
                "PostgreSQL connection string is empty on the Retrieval Evaluator. "
                "Paste the same string your PGVector node uses "
                "(e.g. postgresql://user:password@pgvector:5432/default)."
            )
            raise ValueError(msg)

        # Masked host:port for diagnostics (never leaks the password).
        host_match = re.search(r"@([^/]+)", conn)
        host = host_match.group(1) if host_match else "unknown-host"

        store = None
        last_err = None
        for attempt in range(2):  # one retry for a flaky SSH tunnel
            try:
                store = PGVector.from_existing_index(
                    embedding=self.embedding,
                    collection_name=self.collection_name,
                    connection_string=transform_connection_string(conn),
                )
                # force a real connection so failures surface here, clearly
                store.similarity_search(query="connectivity probe", k=1)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if store is None:
            msg = (
                f"Could not reach Postgres at '{host}' (table '{self.collection_name}'). "
                f"Your PGVector node uses the same endpoint — if that node also fails, the "
                f"database/SSH tunnel is down; if it works, re-paste the connection string here. "
                f"Underlying error: {last_err}"
            )
            raise ValueError(msg)

        k = int(self.k)
        rows = []
        for _, row in gt.iterrows():
            q = str(row[self.question_col])
            docs = store.similarity_search(query=q, k=k)

            hit_rank = 0
            for idx, doc in enumerate(docs, start=1):
                if self._is_relevant(doc, row):
                    hit_rank = idx
                    break

            rr = (1.0 / hit_rank) if hit_rank else 0.0
            rec = {
                "id": row.get("id", ""),
                self.stratum_col: row.get(self.stratum_col, "all"),
                "question": q,
                "gold_answer": row.get(self.answer_col, ""),
                "hit_rank": hit_rank,
                "reciprocal_rank": round(rr, 4),
                "hit@1": int(0 < hit_rank <= 1),
                "hit@3": int(0 < hit_rank <= 3),
                "hit@5": int(0 < hit_rank <= 5),
                "top1_snippet": (getattr(docs[0], "page_content", "")[:160] if docs else ""),
            }

            if self.run_generation:
                context = "\n\n".join(getattr(d, "page_content", "") for d in docs)
                pred = self._strip_think(self._generate(context, q))
                gold_answer = str(row[self.answer_col])
                if self.judge_mode == "llm_judge":
                    correct = self._llm_judge(q, gold_answer, pred)
                else:
                    correct = int(self._answer_correct(pred, gold_answer))
                rec["answer_pred"] = pred[:400]
                rec["answer_correct"] = correct
                rec["verdict"] = "CORRECT" if correct else "WRONG"

            rows.append(rec)

        self._per_question_df = pd.DataFrame(rows)
        return self._per_question_df

    # ------------------------------------------------------------- output: rows
    def evaluate(self) -> DataFrame:
        df = self._run_eval()
        extra = ""
        if "answer_correct" in df.columns:
            n_ok = int(df["answer_correct"].sum())
            extra = f" | answer accuracy {n_ok}/{len(df)} = {n_ok / len(df):.1%} ({len(df) - n_ok} wrong)"
        self.status = (
            f"Evaluated {len(df)} questions (k={self.k}, mode={self.match_mode}){extra}. "
            "Open the table-grid icon for per-question rows; switch to 'Metrics Summary' for the breakdown."
        )
        return DataFrame(df)

    # --------------------------------------------------------- output: summary
    def summarize(self) -> DataFrame:
        df = self._run_eval()

        def block(sub: pd.DataFrame, label: str) -> dict:
            d = {
                "stratum": label,
                "n": len(sub),
                "MRR": round(sub["reciprocal_rank"].mean(), 4),
                "Hit@1": round(sub["hit@1"].mean(), 4),
                "Hit@3": round(sub["hit@3"].mean(), 4),
                "Hit@5": round(sub["hit@5"].mean(), 4),
            }
            if "answer_correct" in sub.columns:
                n_correct = int(sub["answer_correct"].sum())
                d["Answer_Acc"] = round(sub["answer_correct"].mean(), 4)
                d["correct"] = n_correct
                d["wrong"] = len(sub) - n_correct
            return d

        out = [block(df, "overall")]
        for stratum, sub in df.groupby(self.stratum_col):
            out.append(block(sub, str(stratum)))

        summary = pd.DataFrame(out)
        self.status = summary.to_string(index=False)
        return DataFrame(summary)