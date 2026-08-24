# Recovered Langflow component
# type: MixRagEvaluator
# class: MixRagEvaluator
# used in 3 flow(s): Baseline- Naive, Evaluation_for_all, Table-aware-scraper
# json path: node.data.node.template.code.value

import json
import re
import pandas as pd
from langchain_community.vectorstores import PGVector

from lfx.custom.custom_component.component import Component
from lfx.io import (
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


class MixRagEvaluator(Component):
    display_name = "Spatial Table RAG Evaluator (SES-Recall / Hit / Tokens)"
    description = (
        "Evaluates a pgvector table index against evidence_cells ground-truth data. "
        "Reports Hit@3/5, SES-Recall@3/5, and Tokens@3/5 grouped by spatial query type (q_type)."
    )
    icon = "ruler"
    name = "MixRagEvaluator"

    inputs = [
        SecretStrInput(
            name="pg_server_url", 
            display_name="PostgreSQL Server Connection String", 
            required=True,
            info="Format: postgresql://user:password@host:5432/dbname"
        ),
        StrInput(
            name="collection_name", 
            display_name="Table / Collection Name", 
            required=True, 
            value="naive_data",
            info="The pgvector collection name (e.g., baseline, method1, method2, soung_method)."
        ),
        HandleInput(
            name="embedding",
            display_name="Embedding",
            input_types=["Embeddings"],
            required=True,
            info="Wire the same Ollama Embeddings component your index was built with.",
        ),
        IntInput(
            name="max_k", 
            display_name="Max Retrieval Depth (K)", 
            value=5,
            info="Retrieval depth to fetch chunks for evaluation (default = 5)."
        ),
        FileInput(
            name="ground_truth_file",
            display_name="Ground-Truth JSON / Dataset",
            file_types=["json"],
            required=True,
            info="JSON dataset containing question, q_type, gold_table_id, and evidence_cells tuples.",
        ),
        MessageTextInput(
            name="q_field", 
            display_name="Question Field", 
            value="question", 
            advanced=True
        ),
        MessageTextInput(
            name="type_field", 
            display_name="Query Type Field", 
            value="q_type", 
            advanced=True
        ),
        MessageTextInput(
            name="gold_table_field", 
            display_name="Gold Table ID Field", 
            value="gold_table_id", 
            advanced=True
        ),
    ]

    outputs = [
        Output(display_name="Retrieval & Sufficiency Metrics", name="retrieval", method="retrieval_table"),
        Output(display_name="Per-Question Evaluation CSV", name="per_question", method="per_question_csv"),
    ]

    _df = None  # Cache per-question results across outputs

    # ----------------------------------------------------------------- Connection String Sanitizer
    def _get_clean_connection_string(self) -> str:
        """Extracts and sanitizes the PostgreSQL URL for SQLAlchemy compatibility."""
        conn = self.pg_server_url
        
        # Unpack SecretStr if passed as an object
        if hasattr(conn, "get_secret_value"):
            conn = conn.get_secret_value()
        elif hasattr(conn, "value"):
            conn = conn.value
            
        conn_str = str(conn).strip().strip("'").strip('"')
        
        if not conn_str:
            raise ValueError("PostgreSQL connection string is empty.")
            
        # Fix deprecated 'postgres://' -> 'postgresql://'
        if conn_str.startswith("postgres://"):
            conn_str = conn_str.replace("postgres://", "postgresql://", 1)
            
        # Prepend scheme if user omitted 'postgresql://'
        if not conn_str.startswith("postgresql"):
            conn_str = f"postgresql://{conn_str}"
            
        try:
            return transform_connection_string(conn_str)
        except Exception:
            return conn_str

    # ----------------------------------------------------------------- Helpers
    @staticmethod
    def _normalize(text) -> str:
        """Strips punctuation, currency symbols, and converts to lowercase for exact matching."""
        if text is None:
            return ""
        s = str(text).lower().replace("฿", "").replace(",", "")
        s = re.sub(r"[^\w\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def _check_evidence_sufficiency(self, evidence_cells, norm_retrieved_text: str) -> float:
        """
        Calculates Structural Evidence Sufficiency Recall (SES-Recall)
        over evidence_cells triples: [Row Header, Column Header, Cell Value].
        """
        if not evidence_cells or len(evidence_cells) == 0:
            return 0.0

        matched_count = 0
        total_cells = len(evidence_cells)

        for cell in evidence_cells:
            # cell structure: [row_header, col_header, value]
            row_header = self._normalize(cell[0]) if len(cell) > 0 else ""
            val_str = self._normalize(cell[2]) if len(cell) > 2 else ""

            # Standard cell value check
            if val_str and val_str not in ["", "-"]:
                if val_str in norm_retrieved_text:
                    matched_count += 1
            # Null/dash value check anchored by row header presence
            elif val_str in ["", "-"] and row_header:
                if row_header in norm_retrieved_text:
                    matched_count += 1

        return matched_count / total_cells

    # ------------------------------------------------------------------- Core Execution
    def _run(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        # Load Ground Truth File
        path = self.ground_truth_file
        if isinstance(path, list):
            path = path[0] if path else None
        if not path:
            raise ValueError("No ground-truth JSON provided.")

        with open(path, encoding="utf-8") as f:
            gt_data = json.load(f)

        # Get Clean & Sanitized Connection String
        conn_str = self._get_clean_connection_string()

        # Database Connection via LangChain
        store = PGVector.from_existing_index(
            embedding=self.embedding,
            collection_name=self.collection_name,
            connection_string=conn_str,
        )

        max_k = max(int(self.max_k), 5)
        eval_rows = []

        for item in gt_data:
            q_uid = item.get("question_uid", "")
            q = str(item.get(self.q_field, ""))
            qtype = str(item.get(self.type_field, "all")).lower()
            gold_table_id = str(item.get(self.gold_table_field, item.get("gold_doc_id", "")))
            evidence_cells = item.get("evidence_cells", [])

            # Similarity Search Top-K Chunks
            retrieved_docs = store.similarity_search(query=q, k=max_k)

            # Extract retrieved chunk contents and metadata identifiers
            doc_contents = []
            chunk_identifiers = []

            for doc in retrieved_docs:
                meta = getattr(doc, "metadata", {}) or {}
                content = getattr(doc, "page_content", "")
                doc_contents.append(content)

                # Identify chunk table ID (supports table_id, chunk_id, or doc_id)
                tid = (
                    meta.get("table_id") 
                    or meta.get("chunk_id") 
                    or meta.get("doc_id") 
                    or meta.get("source") 
                    or ""
                )
                chunk_identifiers.append(str(tid))

            rec = {
                "question_uid": q_uid,
                "q_type": qtype,
                "question": q,
                "gold_answer": item.get("answer", ""),
                "gold_table_id": gold_table_id,
            }

            # Evaluate at cutoff K = 3 and K = 5
            for k_val in [3, 5]:
                k_contents = doc_contents[:k_val]
                k_ids = chunk_identifiers[:k_val]

                joined_text = "\n\n".join(k_contents)
                norm_joined_text = self._normalize(joined_text)

                # 1. Binary Hit@K
                hit_k = 1 if any(gold_table_id in str(cid) for cid in k_ids) else 0

                # 2. Structural Evidence Sufficiency Recall (SES-Recall@K)
                ses_recall_k = self._check_evidence_sufficiency(evidence_cells, norm_joined_text)

                # 3. Prompt Token Overhead
                token_count = len(joined_text.split())

                rec[f"hit@{k_val}"] = hit_k
                rec[f"ses_recall@{k_val}"] = round(ses_recall_k, 5)
                rec[f"tokens@{k_val}"] = token_count

            eval_rows.append(rec)

        self._df = pd.DataFrame(eval_rows)
        return self._df

    # ---------------------------------------------------- Output 1: Summary Table
    def retrieval_table(self) -> DataFrame:
        df = self._run()
        out = []

        for qtype, sub in sorted(df.groupby("q_type")):
            out.append(
                {
                    "Spatial Type": qtype,
                    "Count": len(sub),
                    "Hit@3": round(sub["hit@3"].mean(), 4),
                    "Hit@5": round(sub["hit@5"].mean(), 4),
                    "SES-Recall@3": round(sub["ses_recall@3"].mean(), 4),
                    "SES-Recall@5": round(sub["ses_recall@5"].mean(), 4),
                    "Tokens@3": round(sub["tokens@3"].mean(), 1),
                    "Tokens@5": round(sub["tokens@5"].mean(), 1),
                }
            )

        summary_table = pd.DataFrame(out)
        self.status = summary_table.to_string(index=False)
        return DataFrame(summary_table)

    # --------------------------------------------------- Output 2: Per-Question CSV
    def per_question_csv(self) -> DataFrame:
        df = self._run()
        cols = [
            "question_uid",
            "q_type",
            "question",
            "gold_answer",
            "gold_table_id",
            "hit@3",
            "hit@5",
            "ses_recall@3",
            "ses_recall@5",
            "tokens@3",
            "tokens@5",
        ]
        sheet = df[cols].copy()
        self.status = sheet.to_csv(index=False)
        return DataFrame(sheet)