# Recovered Langflow component
# type: PGVectorDecomposerRetriever
# class: PGVectorDecomposerRetriever
# used in 24 flow(s): AIT Cognitive RAG, Cognitive RAG V0.5.2 backup, Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup, Cognitive RAG V1.1.0, Cognitive RAG V1.1.0 Clean, Cognitive RAG V1.1.0 Eval ...
# json path: node.data.node.template.code.value

import ast
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import PGVector

from lfx.custom import Component
from lfx.io import HandleInput, IntInput, MessageInput, Output, SecretStrInput, StrInput
from lfx.schema import Message


class PGVectorDecomposerRetriever(Component):
    display_name = "PGVector Decomposer Retriever"
    description = "Runs PGVector retrieval once or across decomposed sub-questions and returns grouped results."
    icon = "database"
    name = "PGVectorDecomposerRetriever"

    inputs = [
        SecretStrInput(
            name="pg_server_url",
            display_name="PostgreSQL Server Connection String",
            required=True,
        ),
        StrInput(
            name="collection_name",
            display_name="Table",
            required=True,
        ),
        HandleInput(
            name="embedding",
            display_name="Embedding",
            input_types=["Embeddings"],
            required=True,
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results Per Query",
            value=4,
            required=True,
        ),
        MessageInput(
            name="decomposer_message",
            display_name="Decomposer Message",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Grouped Results Message",
            name="grouped_results_message",
            method="build_output",
        ),
    ]

    def _extract_message_text(self, value: Any) -> str:
        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()

        if hasattr(value, "data") and isinstance(value.data, dict):
            if "text" in value.data:
                return str(value.data["text"]).strip()

        return str(value).strip()

    def _parse_payload(self, raw_text: str) -> Dict[str, Any]:
        if not raw_text:
            return {}

        # Try JSON first
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Try Python dict string
        try:
            parsed = ast.literal_eval(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return {}

    def _safe_float(self, x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def _resolve_queries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        decomposition_used = bool(payload.get("decomposition_used", False))
        decomposition_type = str(payload.get("decomposition_type", "none")).strip() or "none"
        notes = str(payload.get("notes", "")).strip()

        original_query = str(
            payload.get("original_query")
            or payload.get("User Query")
            or payload.get("user_query")
            or ""
        ).strip()

        sub_questions = payload.get("sub_questions", [])
        if not isinstance(sub_questions, list):
            sub_questions = []

        cleaned_sub_questions = []
        for q in sub_questions:
            qs = str(q).strip()
            if qs:
                cleaned_sub_questions.append(qs)

        if decomposition_used and cleaned_sub_questions:
            queries = cleaned_sub_questions
        else:
            queries = [original_query] if original_query else []

        return {
            "original_query": original_query,
            "decomposition_used": decomposition_used,
            "decomposition_type": decomposition_type,
            "notes": notes,
            "queries": queries,
            "final_query_count": len(queries),
        }

    def _build_vector_store(self) -> PGVector:
        return PGVector.from_existing_index(
            embedding=self.embedding,
            collection_name=self.collection_name,
            connection_string=self.pg_server_url,
        )

    def _search_one_query(
        self,
        vector_store: PGVector,
        query: str,
        query_index: int,
        total_queries: int,
    ) -> Dict[str, Any]:
        k = int(self.number_of_results or 4)

        t0 = time.perf_counter()

        docs = []
        scores = None

        try:
            docs_with_scores: List[Tuple[Any, Any]] = vector_store.similarity_search_with_score(query=query, k=k)
            docs = [d for (d, _) in docs_with_scores]
            scores = [self._safe_float(s) for (_, s) in docs_with_scores]
        except Exception:
            docs = vector_store.similarity_search(query=query, k=k)
            scores = None

        wall_s = time.perf_counter() - t0

        result_items = []
        for i, doc in enumerate(docs, start=1):
            result_items.append(
                {
                    "rank": i,
                    "score": scores[i - 1] if scores is not None and i - 1 < len(scores) else None,
                    "page_content": getattr(doc, "page_content", ""),
                    "metadata": getattr(doc, "metadata", {}) or {},
                }
            )

        return {
            "query_index": query_index,
            "query_text": query,
            "total_queries": total_queries,
            "retrieval_wall_time_s": wall_s,
            "top_k": k,
            "n_returned": len(result_items),
            "results": result_items,
        }

    def build_output(self) -> Message:
        raw_text = self._extract_message_text(self.decomposer_message)
        payload = self._parse_payload(raw_text)

        if not payload:
            error_result = {
                "_error": "Could not parse decomposer_message",
                "_raw_input": raw_text,
            }
            error_text = json.dumps(error_result, ensure_ascii=False, indent=2)
            self.status = error_text
            return Message(text=error_text, data=error_result)

        query_plan = self._resolve_queries(payload)
        queries = query_plan["queries"]

        if not queries:
            error_result = {
                "_error": "No usable queries found",
                "parsed_payload": payload,
            }
            error_text = json.dumps(error_result, ensure_ascii=False, indent=2)
            self.status = error_text
            return Message(text=error_text, data=error_result)

        try:
            vector_store = self._build_vector_store()
        except Exception as e:
            error_result = {
                "_error": "Failed to connect to PGVector",
                "details": str(e),
            }
            error_text = json.dumps(error_result, ensure_ascii=False, indent=2)
            self.status = error_text
            return Message(text=error_text, data=error_result)

        grouped_results = []
        total_returned = 0

        for idx, query in enumerate(queries, start=1):
            one_query_result = self._search_one_query(
                vector_store=vector_store,
                query=query,
                query_index=idx,
                total_queries=len(queries),
            )
            total_returned += one_query_result["n_returned"]
            grouped_results.append(one_query_result)

        final_output = {
            "original_query": query_plan["original_query"],
            "decomposition_used": query_plan["decomposition_used"],
            "decomposition_type": query_plan["decomposition_type"],
            "notes": query_plan["notes"],
            "final_query_count": query_plan["final_query_count"],
            "total_results_returned": total_returned,
            "results_by_query": grouped_results,
        }

        output_text = json.dumps(final_output, ensure_ascii=False, indent=2)
        self.status = output_text
        return Message(text=output_text, data=final_output)