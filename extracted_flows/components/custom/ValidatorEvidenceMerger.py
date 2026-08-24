# Recovered Langflow component
# type: ValidatorEvidenceMerger
# class: ValidatorEvidenceMerger
# used in 24 flow(s): AIT Cognitive RAG, Cognitive RAG V0.5.2 backup, Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup, Cognitive RAG V1.1.0, Cognitive RAG V1.1.0 Clean, Cognitive RAG V1.1.0 Eval ...
# json path: node.data.node.template.code.value

import ast
import json
from typing import Any, Dict, List

from lfx.custom import Component
from lfx.io import MessageInput, Output
from lfx.schema import Message


class ValidatorEvidenceMerger(Component):
    display_name = "Validator Evidence Merger"
    description = "Merges validator-selected evidence indices with the actual retrieved RAG chunks."
    icon = "merge"
    name = "ValidatorEvidenceMerger"

    inputs = [
        MessageInput(
            name="validator_message",
            display_name="Validator Message",
            info="Validator JSON output containing top_kept_evidences with global_retrieval_index.",
            required=True,
        ),
        MessageInput(
            name="retrieval_message",
            display_name="Retrieval Message",
            info="Original grouped retrieval output containing results_by_query with actual page_content and metadata.",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Merged Evidence Message",
            name="merged_message",
            method="build_output",
        ),
    ]

    def _extract_message_text(self, value: Any) -> str:
        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()
        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()
        if hasattr(value, "data") and isinstance(value.data, dict):
            if "text" in value.data:
                return str(value.data["text"]).strip()
        return str(value).strip()

    def _parse_payload(self, raw_text: str) -> Dict[str, Any]:
        if not raw_text:
            return {}

        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return {}

    def _build_retrieval_index_map(self, retrieval_payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Build a map:
          global_retrieval_index -> original retrieved chunk
        """
        index_map: Dict[int, Dict[str, Any]] = {}

        results_by_query = retrieval_payload.get("results_by_query", [])
        if not isinstance(results_by_query, list):
            return index_map

        # The retriever already assigned global_retrieval_index in the flattened prompt schema,
        # but the grouped retrieval output usually has results inside each query group.
        # We reconstruct a global counter in retrieval order if needed.
        global_counter = 0

        for group in results_by_query:
            query_index = group.get("query_index")
            query_text = group.get("query_text", "")
            results = group.get("results", [])
            if not isinstance(results, list):
                continue

            for item in results:
                global_counter += 1

                global_idx = item.get("global_retrieval_index", None)
                if global_idx is None:
                    global_idx = global_counter

                try:
                    global_idx = int(global_idx)
                except Exception:
                    continue

                index_map[global_idx] = {
                    "global_retrieval_index": global_idx,
                    "query_index": query_index,
                    "query_text": query_text,
                    "retrieval_rank_within_query": item.get("rank"),
                    "pgvector_score": item.get("score"),
                    "page_content": item.get("page_content", ""),
                    "metadata": item.get("metadata", {}) or {},
                }

        return index_map

    def build_output(self) -> Message:
        validator_raw = self._extract_message_text(self.validator_message)
        retrieval_raw = self._extract_message_text(self.retrieval_message)

        validator_payload = self._parse_payload(validator_raw)
        retrieval_payload = self._parse_payload(retrieval_raw)

        if not validator_payload:
            error = {
                "_error": "Could not parse validator_message",
                "_raw_validator": validator_raw,
            }
            error_text = json.dumps(error, ensure_ascii=False, indent=2)
            self.status = error_text
            return Message(text=error_text, data=error)

        if not retrieval_payload:
            error = {
                "_error": "Could not parse retrieval_message",
                "_raw_retrieval": retrieval_raw,
            }
            error_text = json.dumps(error, ensure_ascii=False, indent=2)
            self.status = error_text
            return Message(text=error_text, data=error)

        index_map = self._build_retrieval_index_map(retrieval_payload)

        kept = validator_payload.get("top_kept_evidences", [])
        if not isinstance(kept, list):
            kept = []

        merged_kept: List[Dict[str, Any]] = []

        for item in kept:
            if not isinstance(item, dict):
                continue

            global_idx = item.get("global_retrieval_index", None)
            try:
                global_idx = int(global_idx)
            except Exception:
                global_idx = None

            original_chunk = index_map.get(global_idx, {})

            merged_item = {
                "global_retrieval_index": global_idx,
                "query_index": item.get("query_index", original_chunk.get("query_index")),
                "query_text": item.get("query_text", original_chunk.get("query_text", "")),
                "validation_score": item.get("validation_score", 0.0),
                "is_answer_bearing": item.get("is_answer_bearing", False),
                "page_content": original_chunk.get("page_content", ""),
                "metadata": original_chunk.get("metadata", {}),
            }

            merged_kept.append(merged_item)

        output = {
            "original_query": validator_payload.get("original_query", ""),
            "decomposition_used": validator_payload.get("decomposition_used", False),
            "decomposition_type": validator_payload.get("decomposition_type", "none"),
            "top_kept_evidences": merged_kept,
            "evidence_sufficiency_score": validator_payload.get("evidence_sufficiency_score", 0.0),
            "conflict_penalty": validator_payload.get("conflict_penalty", 0.0),
            "rag_valid": validator_payload.get("rag_valid", False),
            "decision_hint": validator_payload.get("decision_hint", "retry"),
        }

        output_text = json.dumps(output, ensure_ascii=False, indent=2)
        self.status = output_text
        return Message(text=output_text, data=output)