# Recovered Langflow component
# type: pgvector_metrics_extractor_envelope
# class: PGVectorMetricsExtractor
# used in 1 flow(s): CRCV-v1.103 (Normal) Oak Backups
# json path: node.data.node.template.code.value

from typing import Any, Dict, List, Optional

from langflow.custom import Component
from langflow.io import HandleInput, Output
from langflow.schema.data import Data


class PGVectorMetricsExtractor(Component):
    display_name = "PGVector Metrics Extractor (Envelope)"
    description = "Returns {metrics, chunks} from a PGVector results list (supports flattened or nested metadata)."
    icon = "gauge"
    name = "pgvector_metrics_extractor_envelope"

    inputs = [
        HandleInput(
            name="results",
            display_name="Results (list[Data])",
            input_types=["Data"],
            required=True,
        )
    ]

    outputs = [
        Output(display_name="Envelope", name="envelope", method="build_envelope"),
    ]

    # Keys we do NOT want to treat as metadata when PGVector output is flattened
    _NON_METADATA_KEYS = {
        "text",
        "page_content",
        "_pgvector_score",
        "_retrieval",
        "rank",
        "score",
        "chunk_id",
        "metadata",  # nested metadata container
    }

    @staticmethod
    def _as_dict(x: Any) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    @staticmethod
    def _get_text(item: Data) -> str:
        if not isinstance(item, Data):
            return ""
        d = item.data if isinstance(item.data, dict) else {}

        # Your screenshot shows top-level "text"
        if isinstance(d.get("text"), str):
            return d["text"]

        # Some versions store chunk content in "page_content"
        if isinstance(d.get("page_content"), str):
            return d["page_content"]

        # If Data has a text_key that exists
        tk = getattr(item, "text_key", None)
        if isinstance(tk, str) and isinstance(d.get(tk), str):
            return d[tk]

        return ""

    def _get_metadata(self, item: Data) -> Dict[str, Any]:
        """
        Support BOTH:
          (A) nested:  item.data["metadata"] is a dict
          (B) flat:    item.data contains keys like 'source', 'uuid', 'custom_id', etc.
        Return a merged metadata dict where nested metadata overrides flat if conflicts.
        """
        d = item.data if isinstance(item.data, dict) else {}

        nested = d.get("metadata")
        nested_md = nested if isinstance(nested, dict) else {}

        # flattened metadata: everything except known non-metadata keys
        flat_md: Dict[str, Any] = {}
        for k, v in d.items():
            if k in self._NON_METADATA_KEYS:
                continue
            flat_md[k] = v

        # Merge: flat first, nested overrides
        merged = dict(flat_md)
        merged.update(nested_md)
        return merged

    def _get_chunk_id(self, item: Data, rank: int) -> str:
        """
        Prefer stable IDs:
          1) chunk_id (top-level or metadata)
          2) custom_id (top-level or metadata)
          3) uuid (top-level or metadata)  [unique but may not be stable across reingest]
          4) source + rank fallback
        """
        d = item.data if isinstance(item.data, dict) else {}
        md = self._get_metadata(item)

        # Try both top-level and metadata
        for key in ("chunk_id", "custom_id", "uuid", "id", "doc_id", "document_id", "source_id"):
            v = d.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, (int, float)):
                return str(v)

            v2 = md.get(key)
            if isinstance(v2, str) and v2.strip():
                return v2.strip()
            if isinstance(v2, (int, float)):
                return str(v2)

        # Source fallback
        src = d.get("source")
        if not (isinstance(src, str) and src.strip()):
            src = md.get("source")
        if isinstance(src, str) and src.strip():
            return f"{src.strip()}#r{rank}"

        return f"rank_{rank}"

    def build_envelope(self) -> Dict[str, Any]:
        r = self.results

        # Normalize to list[Data]
        items: List[Data] = []
        if isinstance(r, list):
            items = [x for x in r if isinstance(x, Data)]
        elif isinstance(r, Data):
            items = [r]

        if not items:
            return {"metrics": {}, "chunks": []}

        # Metrics are repeated on each item; take from first (your screenshot shows "_retrieval" top-level)
        metrics: Dict[str, Any] = {}
        first = items[0]
        if isinstance(first.data, dict):
            m = first.data.get("_retrieval")
            if isinstance(m, dict):
                metrics = m

        chunks: List[Dict[str, Any]] = []
        for i, item in enumerate(items):
            rank = i + 1
            d = item.data if isinstance(item.data, dict) else {}
            md = self._get_metadata(item)

            chunks.append(
                {
                    "rank": rank,
                    "chunk_id": self._get_chunk_id(item, rank),
                    "score": d.get("_pgvector_score"),
                    "text": self._get_text(item),
                    "metadata": md,  # includes flattened 'source', etc.
                }
            )

        return {"metrics": metrics, "chunks": chunks}
