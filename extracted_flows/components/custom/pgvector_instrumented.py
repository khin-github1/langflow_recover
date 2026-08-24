# Recovered Langflow component
# type: pgvector_instrumented
# class: PGVectorStoreComponent
# used in 21 flow(s): AITGPT Fees, AITGPT General, AITGPT Program, AITGPT V1.0.0, Baseline Normal AIT, Baseline Normal AIT 123, Baseline Normal Squad, Baseline Reason AIT ...
# json path: node.data.node.template.code.value

import time
from typing import Any, Dict, Optional, Tuple, List

from langchain_community.vectorstores import PGVector

from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.helpers.data import docs_to_data
from langflow.io import HandleInput, IntInput, SecretStrInput, StrInput
from langflow.schema.data import Data
from langflow.utils.connection_string_parser import transform_connection_string


class PGVectorStoreComponent(LCVectorStoreComponent):
    display_name = "PGVector (Instrumented)"
    description = "PGVector Vector Store with search + retrieval metrics embedded into returned Data."
    name = "pgvector_instrumented"
    icon = "cpu"

    inputs = [
        SecretStrInput(name="pg_server_url", display_name="PostgreSQL Server Connection String", required=True),
        StrInput(name="collection_name", display_name="Table", required=True),
        *LCVectorStoreComponent.inputs,
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"], required=True),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            value=4,
            advanced=True,
        ),
    ]

    @check_cached_vector_store
    def build_vector_store(self) -> PGVector:
        self.ingest_data = self._prepare_ingest_data()

        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                documents.append(_input)

        connection_string_parsed = transform_connection_string(self.pg_server_url)

        if documents:
            pgvector = PGVector.from_documents(
                embedding=self.embedding,
                documents=documents,
                collection_name=self.collection_name,
                connection_string=connection_string_parsed,
            )
        else:
            pgvector = PGVector.from_existing_index(
                embedding=self.embedding,
                collection_name=self.collection_name,
                connection_string=connection_string_parsed,
            )

        return pgvector

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def search_documents(self) -> list[Data]:
        vector_store = self.build_vector_store()

        q = self.search_query
        if not (q and isinstance(q, str) and q.strip()):
            return []

        q = q.strip()
        k = int(self.number_of_results or 4)

        # --- Retrieval timing includes: embedding(query) + DB search + result materialization ---
        t0 = time.perf_counter()

        docs = []
        scores: Optional[List[Optional[float]]] = None

        # Prefer scores if available
        try:
            # returns List[Tuple[Document, float]]
            docs_with_scores: List[Tuple[Any, Any]] = vector_store.similarity_search_with_score(query=q, k=k)
            docs = [d for (d, _) in docs_with_scores]
            scores = [self._safe_float(s) for (_, s) in docs_with_scores]
        except Exception:
            # Fallback: no scores
            docs = vector_store.similarity_search(query=q, k=k)
            scores = None

        wall_s = time.perf_counter() - t0

        data = docs_to_data(docs)

        # --- Retrieval metrics payload (turn-level) ---
        retrieval_metrics: Dict[str, Any] = {
            "retrieval_wall_time_s": wall_s,
            "top_k": k,
            "n_returned": len(data),
            "collection_name": self.collection_name,
            "query_len_chars": len(q),
            "has_scores": scores is not None,
        }

        # Attach metrics (and per-doc score) onto each Data item
        enriched: list[Data] = []
        for i, item in enumerate(data):
            if not isinstance(item, Data):
                enriched.append(item)
                continue

            # Copy existing payload safely
            payload = dict(item.data) if isinstance(item.data, dict) else {}

            # Per-doc score (if present)
            if scores is not None and i < len(scores):
                payload["_pgvector_score"] = scores[i]

            # Turn-level retrieval metrics (repeat on every doc so you can extract from any one)
            payload["_retrieval"] = retrieval_metrics

            # Write back
            try:
                item.data = payload
                enriched.append(item)
            except Exception:
                # If Data is immutable in your build, recreate
                enriched.append(
                    Data(
                        text_key=getattr(item, "text_key", "text"),
                        data=payload,
                        default_value=getattr(item, "default_value", ""),
                    )
                )

        self.status = enriched
        return enriched
