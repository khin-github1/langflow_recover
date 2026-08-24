# Recovered Langflow component
# type: GoldTableEvaluator
# class: GoldTableEvaluator
# used in 1 flow(s): Evaluation_for_all
# json path: node.data.node.template.code.value

# Langflow Custom Component: Gold-Table RAG Evaluator
# -----------------------------------------------------------------------------
# Evaluates one or more pgvector collections (B-test / M1-test / M2-test / M3-test)
# LIVE against evidence_ground_truth.json, using:
#   * Hit@K         table routing — did a chunk whose gold_table_id == the
#                   question's gold_table_id appear in top-K?
#   * Coverage@K    fraction of the question's evidence_cells found in top-K
#                   (BINDING-AWARE: a cell's value must co-occur with its record
#                   label in the SAME chunk, so lossy summaries can't over-credit).
#   * Sufficient@K  binary — were ALL evidence_cells found? (primary metric)
#   * Tokens@K      retrieved context size (prompt overhead).
# Grouped by q_type (row / col / cross / multi_cross). GoldCells = mean evidence
# count per type.
# -----------------------------------------------------------------------------

import json
import re
from urllib.parse import quote

import pandas as pd
from langchain_community.vectorstores import PGVector
from sqlalchemy import create_engine, text as sa_text
from sqlalchemy.engine import make_url

from lfx.custom.custom_component.component import Component
from lfx.io import (
    BoolInput,
    FileInput,
    HandleInput,
    MessageTextInput,
    MultilineInput,
    Output,
    SecretStrInput,
)
from lfx.schema.dataframe import DataFrame
from lfx.utils.connection_string_parser import transform_connection_string


Q_TYPE_ORDER = ["row", "col", "cross", "multi_cross"]


class GoldTableEvaluator(Component):
    display_name = "Evaluation with GoldTableEvaluator)"
    description = (
        "Evaluates pgvector collections against evidence_ground_truth.json using "
        "gold_table_id routing and binding-aware evidence coverage, grouped by q_type."
    )
    icon = "ruler"
    name = "GoldTableEvaluator"

    inputs = [
        SecretStrInput(
            name="pg_server_url",
            display_name="PostgreSQL Server Connection String",
            required=True,
            value="postgresql://user:password@pgvector:5432/default",
        ),
        MultilineInput(
            name="collection_names",
            display_name="Collections to evaluate",
            value="B-test\nM1-test\nM2-test\nM3-test",
            info="One collection name per line (or comma-separated). Evaluated in this order.",
        ),
        HandleInput(
            name="embedding",
            display_name="Embedding",
            input_types=["Embeddings"],
            required=True,
            info="Wire the SAME Ollama Embeddings node the indexes were built with (mxbai-embed-large).",
        ),
        FileInput(
            name="ground_truth_file",
            display_name="Ground-Truth JSON",
            file_types=["json"],
            required=True,
            info="evidence_ground_truth.json (question_uid / q_type / gold_table_id / evidence_cells).",
        ),
        MessageTextInput(name="k_values", display_name="K values", value="1,3,5"),
        BoolInput(
            name="binding_aware",
            display_name="Binding-aware coverage",
            value=True,
            info="A cell counts only if its value AND its record label appear in the SAME chunk.",
        ),
        MessageTextInput(name="id_field", display_name="Metadata id field", value="gold_table_id", advanced=True),
        MessageTextInput(name="q_field", display_name="Question field", value="question", advanced=True),
        MessageTextInput(name="type_field", display_name="Type field", value="q_type", advanced=True),
        MessageTextInput(name="gold_field", display_name="Gold table-id field", value="gold_table_id", advanced=True),
        MessageTextInput(name="cells_field", display_name="Evidence-cells field", value="evidence_cells", advanced=True),
    ]

    outputs = [
        Output(display_name="Metrics by Type", name="metrics", method="metrics_table"),
        Output(display_name="Method Comparison", name="comparison", method="comparison_table"),
        Output(display_name="Per-Question CSV", name="per_question", method="per_question_csv"),
    ]

    _df = None
    _errors = None
    _available = None

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _norm(s) -> str:
        s = str(s).lower().replace("฿", "").replace(",", "")
        s = re.sub(r"[^\w\u0e00-\u0e7f]+", " ", s)  # keep alnum + Thai, punctuation -> space
        return re.sub(r"\s+", " ", s).strip()

    def _cell_matched(self, cell, chunks_norm, binding: bool) -> bool:
        row_raw = cell[0] if len(cell) > 0 else ""
        val_raw = cell[2] if len(cell) > 2 else ""
        row_n = self._norm(row_raw)
        val_n = self._norm(val_raw)
        empty = str(val_raw).strip() in ("", "-") or val_n == ""
        for cn in chunks_norm:
            if empty:
                if row_n and row_n in cn:  # empty cell -> anchor on the record label
                    return True
            elif binding:
                if val_n in cn and (not row_n or row_n in cn):  # value bound to its record
                    return True
            elif val_n in cn:
                return True
        return False

    def _k_list(self):
        ks = []
        for tok in str(self.k_values).replace(" ", "").split(","):
            if tok.isdigit():
                ks.append(int(tok))
        return ks or [1, 3, 5]

    def _collections(self):
        return [c.strip() for c in re.split(r"[\n,]", self.collection_names or "") if c.strip()]

    @staticmethod
    def _mask(conn: str) -> str:
        return re.sub(r"(://[^:/@]+:)[^@]*(@)", r"\1***\2", conn or "")

    def _usable_url(self, conn_input) -> str:
        """Unpack SecretStr and return a SQLAlchemy-parseable URL."""
        if hasattr(conn_input, "get_secret_value"):
            conn = conn_input.get_secret_value()
        elif hasattr(conn_input, "value"):
            conn = conn_input.value
        else:
            conn = str(conn_input)

        conn = str(conn or "").strip().rstrip(",").strip()
        
        try:
            make_url(conn)
            return conn
        except Exception:
            pass
        try:
            scheme, after = conn.split("://", 1)
            creds, host = after.rsplit("@", 1)
            user, _, pw = creds.partition(":")
            enc = f"{scheme}://{quote(user, safe='')}:{quote(pw, safe='')}@{host}"
            make_url(enc)
            return enc
        except Exception:
            return conn

    def _available_collections(self, conn_use):
        try:
            eng = create_engine(conn_use)
            with eng.connect() as c:
                rows = c.execute(sa_text("SELECT name FROM langchain_pg_collection ORDER BY name")).fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            return [f"<could not list collections: {e}>"]

    # ------------------------------------------------------------------- core
    def _run(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        self._errors = []

        path = self.ground_truth_file
        if isinstance(path, list):
            path = path[0] if path else None
        if not path:
            msg = "No ground-truth JSON provided. Upload evidence_ground_truth.json."
            raise ValueError(msg)
        with open(path, encoding="utf-8") as f:
            gt = json.load(f)

        conn_use = self._usable_url(self.pg_server_url)
        if not conn_use:
            msg = "PostgreSQL connection string is empty."
            raise ValueError(msg)

        try:
            make_url(conn_use)
        except Exception as e:
            self._errors = [
                f"Connection string is not a valid SQLAlchemy URL: {e}",
                f"Preview (password masked): {self._mask(conn_use)}",
                "Fix: use postgresql://USER:PASSWORD@HOST:5432/DBNAME and percent-encode "
                "any @ : / # ? or space in the password.",
            ]
            self._available = ["<skipped: URL unparseable>"]
            self._df = pd.DataFrame([])
            return self._df

        try:
            conn_parsed = transform_connection_string(conn_use)
        except Exception:
            conn_parsed = conn_use

        collections = self._collections()
        if not collections:
            msg = "No collection names given."
            raise ValueError(msg)

        if self.embedding is None or not hasattr(self.embedding, "embed_query"):
            msg = "Embedding is not wired. Connect the Ollama Embeddings node to the 'Embedding' input."
            raise ValueError(msg)

        ks = self._k_list()
        maxk = max(ks)
        binding = bool(self.binding_aware)

        # Embed every question ONCE, reuse across collections.
        questions = [str(item[self.q_field]) for item in gt]
        try:
            q_vecs = [self.embedding.embed_query(q) for q in questions]
        except Exception as e:
            msg = f"Embedding the questions failed (check the Ollama node / base URL): {e}"
            raise ValueError(msg) from e

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

            got_any = False
            for i, item in enumerate(gt):
                cells = item.get(self.cells_field) or []
                gold = str(item.get(self.gold_field, ""))
                qtype = str(item.get(self.type_field, "all")).lower()
                n_cells = len(cells)

                try:
                    docs = store.similarity_search_by_vector(q_vecs[i], k=maxk)
                except AttributeError:
                    docs = store.similarity_search(questions[i], k=maxk)
                except Exception as e:
                    self._errors.append(f"[{coll}] search failed on q{item.get('question_uid', i)}: {e}")
                    docs = []

                if docs:
                    got_any = True
                rec = {"method": coll, "q_uid": item.get("question_uid", i),
                       "q_type": qtype, "gold_table_id": gold, "gold_cells": n_cells}
                for k in ks:
                    top = docs[:k]
                    texts = [getattr(d, "page_content", "") for d in top]
                    
                    # Extract ID safely with metadata fallback
                    ids = []
                    for d in top:
                        m = getattr(d, "metadata", {}) or {}
                        tid = m.get(self.id_field) or m.get("gold_table_id") or m.get("table_id") or m.get("doc_id") or ""
                        ids.append(str(tid))

                    chunks_norm = [self._norm(t) for t in texts]

                    hit = int(bool(gold) and any(gold in str(cid) for cid in ids))
                    matched = sum(1 for c in cells if self._cell_matched(c, chunks_norm, binding))
                    cov = matched / n_cells if n_cells else float(hit)
                    suff = int(matched == n_cells) if n_cells else hit
                    tokens = sum(len(t.split()) for t in texts)

                    rec[f"Hit@{k}"] = hit
                    rec[f"Cov@{k}"] = cov
                    rec[f"Suff@{k}"] = suff
                    rec[f"Tok@{k}"] = tokens
                rows.append(rec)

            if not got_any:
                self._errors.append(f"[{coll}] loaded but returned 0 chunks for every query "
                                    "(wrong collection name, empty index, or embedding mismatch).")

        self._available = self._available_collections(conn_use)
        self._df = pd.DataFrame(rows)
        return self._df

    def _diagnostic(self) -> DataFrame:
        """Return a visible explanation instead of an empty frame."""
        errs = self._errors or ["No rows were produced."]
        rows = [{"issue": e} for e in errs]
        rows.append({"issue": "— collections found in this database —"})
        for name in (self._available or []):
            rows.append({"issue": f"available: {name}"})
        rows.append({"issue": f"requested: {', '.join(self._collections())}"})
        table = pd.DataFrame(rows)
        self.status = table.to_string(index=False)
        return DataFrame(table)

    # --------------------------------------------------------------- outputs
    def metrics_table(self) -> DataFrame:
        df = self._run()
        if df.empty or not any(c.startswith("Hit@") for c in df.columns):
            return self._diagnostic()
        ks = self._k_list()
        maxk = max(ks)
        collections = self._collections()

        def agg(sub):
            r = {"N": len(sub), "GoldCells": round(sub["gold_cells"].mean(), 2)}
            for k in ks:
                r[f"Hit@{k}"] = round(sub[f"Hit@{k}"].mean(), 3)
            for k in ks:
                r[f"Cov@{k}"] = round(sub[f"Cov@{k}"].mean(), 3)
            for k in ks:
                r[f"Suff@{k}"] = round(sub[f"Suff@{k}"].mean(), 3)
            r[f"Tok@{maxk}"] = round(sub[f"Tok@{maxk}"].mean(), 0)
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
                if sub.empty:
                    continue
                out.append({"Method": coll, "Type": qt, **agg(sub)})
            out.append({"Method": coll, "Type": "ALL", **agg(m)})

        table = pd.DataFrame(out)
        self.status = table.to_string(index=False)
        return DataFrame(table)

    def comparison_table(self) -> DataFrame:
        df = self._run()
        if df.empty or not any(c.startswith("Suff@") for c in df.columns):
            return self._diagnostic()
        maxk = max(self._k_list())
        collections = self._collections()
        seen_types = list(dict.fromkeys(list(df["q_type"].unique())))
        ordered = [t for t in Q_TYPE_ORDER if t in seen_types] + [t for t in seen_types if t not in Q_TYPE_ORDER]

        out = []
        for qt in ordered:
            row = {"Type": qt}
            for coll in collections:
                sub = df[(df["method"] == coll) & (df["q_type"] == qt)]
                if sub.empty:
                    continue
                row[f"{coll} Suff@{maxk}"] = round(sub[f"Suff@{maxk}"].mean(), 3)
                row[f"{coll} Cov@{maxk}"] = round(sub[f"Cov@{maxk}"].mean(), 3)
            out.append(row)
        # overall
        row = {"Type": "ALL"}
        for coll in collections:
            sub = df[df["method"] == coll]
            if sub.empty:
                continue
            row[f"{coll} Suff@{maxk}"] = round(sub[f"Suff@{maxk}"].mean(), 3)
            row[f"{coll} Cov@{maxk}"] = round(sub[f"Cov@{maxk}"].mean(), 3)
        out.append(row)

        table = pd.DataFrame(out)
        self.status = table.to_string(index=False)
        return DataFrame(table)

    def per_question_csv(self) -> DataFrame:
        df = self._run()
        if df.empty:
            return self._diagnostic()
        self.status = df.to_csv(index=False)
        return DataFrame(df)