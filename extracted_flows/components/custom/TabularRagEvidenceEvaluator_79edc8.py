# Recovered Langflow component
# type: TabularRagEvidenceEvaluator
# class: TabularRagEvidenceEvaluator
# used in 1 flow(s): Naive Flattening Evaluatoin 
# json path: node.data.node.template.code.value

# Langflow Custom Component: Tabular RAG Evaluator & Downstream Generator
# -----------------------------------------------------------------------------
# Evaluates pgvector collections (B-test / M1-test / M2-test / M3-test) against
# evidence_ground_truth.json using:
#   * Hit@K         - Coarse table routing check.
#   * Coverage@K    - Binding-aware fraction of evidence cells retrieved.
#   * Sufficient@K  - Binary check: were 100% of required evidence cells retrieved?
#   * Tokens@K      - Retrieved context overhead (words/tokens).
#   * Exact Match   - Downstream LLM answer verification via Ollama (EM).
#
# Outputs:
#   1. Metrics by Type     - Summary table grouped by method and q_type with EM.
#   2. Chatbot Paradox     - Contingency table: Hit@5 vs Suff@5 vs Mean EM.
#   3. Per-Question CSV    - Row-level dataset with predictions and scores.
# -----------------------------------------------------------------------------

import asyncio
import json
import re
import urllib.parse
from typing import Any
from urllib.parse import urljoin

import httpx
import pandas as pd
from langchain_community.vectorstores import PGVector
from sqlalchemy import create_engine, text as sa_text
from sqlalchemy.engine import make_url

from lfx.custom.custom_component.component import Component
from lfx.io import (
    BoolInput,
    DropdownInput,
    FileInput,
    HandleInput,
    MessageTextInput,
    MultilineInput,
    Output,
    SecretStrInput,
)
from lfx.schema.dataframe import DataFrame
from lfx.utils.connection_string_parser import transform_connection_string

HTTP_STATUS_OK = 200
Q_TYPE_ORDER = ["row", "col", "cross", "multi_cross"]


class TabularRagEvidenceEvaluator(Component):
    display_name = "End to End Evaluator"
    description = (
        "Evaluates table serialization indexes for upstream evidence sufficiency (Hit/Cov/Suff/Tok) "
        "and runs downstream Ollama generation to test Exact Match (EM) and the Chatbot Paradox."
    )
    icon = "ruler"
    name = "TabularRagEvidenceEvaluator"

    inputs = [
        SecretStrInput(
            name="pg_server_url",
            display_name="PostgreSQL Connection String",
            required=True,
            value="postgresql://user:password@localhost:5432/dbname",
            info="Format: postgresql://user:password@host:5432/dbname",
        ),
        MultilineInput(
            name="collection_names",
            display_name="Collections to Evaluate",
            value="B-test\nM1-test\nM2-test\nM3-test",
            info="One collection name per line (or comma-separated).",
        ),
        HandleInput(
            name="embedding",
            display_name="Embedding",
            input_types=["Embeddings"],
            required=True,
            info="Connect the Ollama Embeddings component your vector index was built with.",
        ),
        FileInput(
            name="ground_truth_file",
            display_name="Ground-Truth JSON",
            file_types=["json"],
            required=True,
            info="JSON dataset containing question, answer, q_type, gold_table_id, and evidence_cells.",
        ),
        MessageTextInput(
            name="k_values",
            display_name="K Values",
            value="1,3,5",
            info="Comma-separated retrieval depths (e.g., 1,3,5)."
        ),
        BoolInput(
            name="binding_aware",
            display_name="Binding-Aware Coverage",
            value=True,
            info="Enforce that cell value and entity row label must co-occur inside the SAME chunk.",
        ),
        BoolInput(
            name="run_generation",
            display_name="Run Downstream Generation (EM)",
            value=True,
            info="Pass Top-K retrieved context into Ollama to measure downstream Exact Match (EM).",
        ),
        MessageTextInput(
            name="ollama_base_url",
            display_name="Ollama Base URL",
            value="http://ollama:11434",
            info="Endpoint of the Ollama API (e.g., http://ollama:11434 or http://localhost:11434).",
            real_time_refresh=True,
        ),
        DropdownInput(
            name="gen_model",
            display_name="Generation Model",
            options=[],
            info="Available models fetched directly from your Ollama instance.",
            refresh_button=True,
            real_time_refresh=True,
            required=True,
        ),
        MessageTextInput(name="id_field", display_name="Metadata ID Field", value="gold_table_id", advanced=True),
        MessageTextInput(name="q_field", display_name="Question Field", value="question", advanced=True),
        MessageTextInput(name="a_field", display_name="Answer Field", value="answer", advanced=True),
        MessageTextInput(name="type_field", display_name="Query Type Field", value="q_type", advanced=True),
        MessageTextInput(name="gold_field", display_name="Gold Table ID Field", value="gold_table_id", advanced=True),
        MessageTextInput(name="cells_field", display_name="Evidence Cells Field", value="evidence_cells", advanced=True),
    ]

    outputs = [
        Output(display_name="Metrics by Spatial Type", name="metrics", method="metrics_table"),
        Output(display_name="Chatbot Paradox Analysis", name="paradox", method="paradox_table"),
        Output(display_name="Per-Question CSV", name="per_question", method="per_question_csv"),
    ]

    _df = None
    _errors = None
    _available = None

    # ----------------------------------------------------------------- Ollama Dynamic Dropdown Handlers
    async def is_valid_ollama_url(self, url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                url_str = str(url or "").rstrip("/").removesuffix("/v1")
                if not url_str.endswith("/"):
                    url_str = url_str + "/"
                tags_url = urljoin(url_str, "api/tags")
                response = await client.get(url=tags_url)
                return response.status_code == HTTP_STATUS_OK
        except Exception:
            return False

    async def get_models(self, base_url_value: str) -> list[str]:
        try:
            base_url = str(base_url_value or "").rstrip("/").removesuffix("/v1")
            if not base_url.endswith("/"):
                base_url = base_url + "/"
            tags_url = urljoin(base_url, "api/tags")

            async with httpx.AsyncClient(timeout=10.0) as client:
                tags_response = await client.get(url=tags_url)
                tags_response.raise_for_status()
                models = tags_response.json()

                model_ids = []
                for model in models.get("models", []):
                    model_name = model.get("name")
                    if model_name:
                        model_ids.append(model_name)
                return model_ids
        except Exception:
            return []

    async def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None):
        if field_name in {"gen_model", "ollama_base_url", None}:
            base_url_to_check = (
                field_value
                if field_name == "ollama_base_url" and field_value
                else build_config.get("ollama_base_url", {}).get("value") or self.ollama_base_url or "http://ollama:11434"
            )

            if await self.is_valid_ollama_url(base_url_to_check):
                models = await self.get_models(base_url_to_check)
                build_config["gen_model"]["options"] = models
                if models and not build_config["gen_model"].get("value"):
                    build_config["gen_model"]["value"] = models[0]
            else:
                build_config["gen_model"]["options"] = []

        return build_config

    # ----------------------------------------------------------------- Helpers & Sanitizers
    @staticmethod
    def _norm(s) -> str:
        if s is None:
            return ""
        s = str(s).lower().replace("฿", "").replace(",", "")
        s = re.sub(r"[^\w\u0e00-\u0e7f]+", " ", s)  # Keep alphanumeric + Thai characters
        return re.sub(r"\s+", " ", s).strip()

    def _usable_url(self, conn_input) -> str:
        """Unpack SecretStr and URL-encode credentials if special characters exist."""
        if hasattr(conn_input, "get_secret_value"):
            conn = conn_input.get_secret_value()
        elif hasattr(conn_input, "value"):
            conn = conn_input.value
        else:
            conn = str(conn_input)

        conn = str(conn or "").strip().rstrip(",").strip()
        if not conn:
            raise ValueError("PostgreSQL connection string is empty.")

        if conn.startswith("postgres://"):
            conn = conn.replace("postgres://", "postgresql://", 1)
        elif not conn.startswith("postgresql"):
            conn = f"postgresql://{conn}"

        try:
            make_url(conn)
            return conn
        except Exception:
            pass

        try:
            scheme, after = conn.split("://", 1)
            creds, host = after.rsplit("@", 1)
            user, _, pw = creds.partition(":")
            enc_user = urllib.parse.quote_plus(urllib.parse.unquote(user))
            enc_pw = urllib.parse.quote_plus(urllib.parse.unquote(pw))
            return f"{scheme}://{enc_user}:{enc_pw}@{host}"
        except Exception:
            return conn

    def _cell_matched(self, cell, chunks_norm, binding: bool) -> bool:
        row_raw = cell[0] if len(cell) > 0 else ""
        val_raw = cell[2] if len(cell) > 2 else ""
        row_n = self._norm(row_raw)
        val_n = self._norm(val_raw)
        empty = str(val_raw).strip() in ("", "-") or val_n == ""

        for cn in chunks_norm:
            if empty:
                if row_n and row_n in cn:
                    return True
            elif binding:
                if val_n in cn and (not row_n or row_n in cn):
                    return True
            elif val_n in cn:
                return True
        return False

    def _compute_em(self, prediction: str, gold: str) -> int:
        pred_clean = re.sub(r"<think>.*?</think>", "", str(prediction or ""), flags=re.DOTALL).strip()
        p_norm = self._norm(pred_clean)
        g_norm = self._norm(gold)
        if not g_norm:
            return 0
        if re.fullmatch(r"\d+", g_norm):
            return int(g_norm in p_norm.split())
        return int(g_norm in p_norm)

    def _query_ollama(self, context: str, question: str) -> str:
        base = (self.ollama_base_url or "").strip().rstrip("/")
        if not base or not self.gen_model:
            return ""
        prompt = (
            "You are a precise tabular assistant. Answer the question using ONLY the provided context.\n"
            "Reply with strictly the direct value or name with no conversational filler.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{base}/api/generate",
                    json={"model": self.gen_model, "prompt": prompt, "stream": False, "think": False},
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
        except Exception as e:
            return f"<Generation Error: {e}>"

    def _k_list(self):
        ks = []
        for tok in str(self.k_values).replace(" ", "").split(","):
            if tok.isdigit():
                ks.append(int(tok))
        return sorted(list(set(ks or [1, 3, 5])))

    def _collections(self):
        return [c.strip() for c in re.split(r"[\n,]", self.collection_names or "") if c.strip()]

    def _available_collections(self, conn_use):
        try:
            eng = create_engine(conn_use)
            with eng.connect() as c:
                rows = c.execute(sa_text("SELECT name FROM langchain_pg_collection ORDER BY name")).fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            return [f"<could not list collections: {e}>"]

    # ------------------------------------------------------------------- Core Execution
    def _run(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        self._errors = []

        # 1. Load Ground Truth File
        path = self.ground_truth_file
        if isinstance(path, list):
            path = path[0] if path else None
        if not path:
            raise ValueError("No ground-truth JSON provided.")
        with open(path, "r", encoding="utf-8") as f:
            gt = json.load(f)

        # 2. Database Connection
        conn_use = self._usable_url(self.pg_server_url)
        try:
            conn_parsed = transform_connection_string(conn_use)
        except Exception:
            conn_parsed = conn_use

        collections = self._collections()
        if not collections:
            raise ValueError("No collection names given.")

        if self.embedding is None or not hasattr(self.embedding, "embed_query"):
            raise ValueError("Embedding handle is not wired. Connect Ollama Embeddings.")

        ks = self._k_list()
        maxk = max(ks)
        binding = bool(self.binding_aware)

        # 3. Pre-embed questions once
        questions = [str(item.get(self.q_field, "")) for item in gt]
        try:
            q_vecs = [self.embedding.embed_query(q) for q in questions]
        except Exception as e:
            raise ValueError(f"Embedding query failed: {e}") from e

        rows = []
        for coll in collections:
            try:
                store = PGVector.from_existing_index(
                    embedding=self.embedding,
                    collection_name=coll,
                    connection_string=conn_parsed,
                )
            except Exception as e:
                self._errors.append(f"[{coll}] could not load collection: {e}")
                continue

            for i, item in enumerate(gt):
                q_uid = item.get("question_uid", i)
                q_text = questions[i]
                gold_ans = str(item.get(self.a_field, ""))
                gold_table_id = str(item.get(self.gold_field, ""))
                qtype = str(item.get(self.type_field, "all")).lower()
                cells = item.get(self.cells_field, [])
                n_cells = len(cells)

                # Similarity search
                try:
                    docs = store.similarity_search_by_vector(q_vecs[i], k=maxk)
                except AttributeError:
                    docs = store.similarity_search(q_text, k=maxk)
                except Exception as e:
                    self._errors.append(f"[{coll}] search failed on q{q_uid}: {e}")
                    docs = []

                rec = {
                    "method": coll,
                    "q_uid": q_uid,
                    "q_type": qtype,
                    "question": q_text,
                    "gold_answer": gold_ans,
                    "gold_table_id": gold_table_id,
                    "gold_cells": n_cells,
                }

                # Evaluate metrics at each K
                for k in ks:
                    top = docs[:k]
                    texts = [getattr(d, "page_content", "") for d in top]

                    ids = []
                    for d in top:
                        m = getattr(d, "metadata", {}) or {}
                        tid = m.get(self.id_field) or m.get("table_id") or m.get("doc_id") or ""
                        ids.append(str(tid))

                    chunks_norm = [self._norm(t) for t in texts]

                    hit = int(bool(gold_table_id) and any(gold_table_id in str(cid) for cid in ids))
                    matched = sum(1 for c in cells if self._cell_matched(c, chunks_norm, binding))
                    cov = (matched / n_cells) if n_cells else float(hit)
                    suff = int(matched == n_cells) if n_cells else hit
                    tokens = sum(len(t.split()) for t in texts)

                    rec[f"Hit@{k}"] = hit
                    rec[f"Cov@{k}"] = round(cov, 4)
                    rec[f"Suff@{k}"] = suff
                    rec[f"Tok@{k}"] = tokens

                # Downstream Generation on Max-K
                if self.run_generation:
                    max_top_texts = [getattr(d, "page_content", "") for d in docs[:maxk]]
                    joined_context = "\n\n".join(max_top_texts)
                    pred = self._query_ollama(joined_context, q_text)
                    rec["generated_answer"] = pred
                    rec["EM"] = self._compute_em(pred, gold_ans)

                rows.append(rec)

        self._available = self._available_collections(conn_use)
        self._df = pd.DataFrame(rows)
        return self._df

    def _diagnostic(self) -> DataFrame:
        errs = self._errors or ["No rows were produced."]
        rows = [{"issue": e} for e in errs]
        rows.append({"issue": "--- Collections found in database ---"})
        for name in (self._available or []):
            rows.append({"issue": f"available: {name}"})
        rows.append({"issue": f"requested: {', '.join(self._collections())}"})
        table = pd.DataFrame(rows)
        self.status = table.to_string(index=False)
        return DataFrame(table)

    # ---------------------------------------------------- Output 1: Metrics by Type
    def metrics_table(self) -> DataFrame:
        df = self._run()
        if df.empty or not any(c.startswith("Hit@") for c in df.columns):
            return self._diagnostic()

        ks = self._k_list()
        maxk = max(ks)
        collections = self._collections()

        def agg(sub):
            r = {"N": len(sub), "Avg_Cells": round(sub["gold_cells"].mean(), 1)}
            for k in ks:
                r[f"Hit@{k}"] = round(sub[f"Hit@{k}"].mean(), 3)
            for k in ks:
                r[f"Cov@{k}"] = round(sub[f"Cov@{k}"].mean(), 3)
            for k in ks:
                r[f"Suff@{k}"] = round(sub[f"Suff@{k}"].mean(), 3)
            if "EM" in sub.columns:
                r["EM"] = round(sub["EM"].mean(), 3)
            r[f"Tok@{maxk}"] = int(round(sub[f"Tok@{maxk}"].mean(), 0))
            return r

        out = []
        seen_types = list(dict.fromkeys(list(df["q_type"].unique())))
        ordered = [t for t in Q_TYPE_ORDER if t in seen_types] + [t for t in seen_types if t not in Q_TYPE_ORDER]

        for coll in collections:
            m = df[df["method"] == coll]
            if m.empty:
                continue
            for qt in ordered:
                sub = m[m["q_type"] == qt]
                if not sub.empty:
                    out.append({"Method": coll, "Type": qt, **agg(sub)})
            out.append({"Method": coll, "Type": "ALL", **agg(m)})

        summary_table = pd.DataFrame(out)
        self.status = summary_table.to_string(index=False)
        return DataFrame(summary_table)

    # ------------------------------------------------ Output 2: Chatbot Paradox Proof
    def paradox_table(self) -> DataFrame:
        df = self._run()
        if df.empty or "EM" not in df.columns:
            return self._diagnostic()

        maxk = max(self._k_list())

        def classify_state(row):
            h = row[f"Hit@{maxk}"]
            s = row[f"Suff@{maxk}"]
            if h == 1 and s == 1:
                return "1. Hit=1 & Suff=1 (Complete Evidence)"
            elif h == 1 and s == 0:
                return "2. Hit=1 & Suff=0 (Chatbot Paradox: Incomplete Evidence)"
            else:
                return "3. Hit=0 & Suff=0 (Retrieval Miss)"

        df_copy = df.copy()
        df_copy["Retrieval_State"] = df_copy.apply(classify_state, axis=1)

        summary = df_copy.groupby(["method", "Retrieval_State"]).agg(
            Queries=("EM", "count"),
            Mean_Cov=(f"Cov@{maxk}", "mean"),
            Chatbot_EM=("EM", "mean"),
        ).reset_index()

        summary["Mean_Cov"] = summary["Mean_Cov"].round(3)
        summary["Chatbot_EM"] = summary["Chatbot_EM"].round(3)

        self.status = summary.to_string(index=False)
        return DataFrame(summary)

    # ---------------------------------------------------- Output 3: Per-Question CSV
    def per_question_csv(self) -> DataFrame:
        df = self._run()
        if df.empty:
            return self._diagnostic()
        self.status = df.to_csv(index=False)
        return DataFrame(df)