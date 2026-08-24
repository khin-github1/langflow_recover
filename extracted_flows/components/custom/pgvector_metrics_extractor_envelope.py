# Recovered Langflow component
# type: pgvector_metrics_extractor_envelope
# class: PGVectorMetricsExtractor
# used in 16 flow(s): AITGPT Fees, AITGPT General, AITGPT Program, AITGPT V1.0.0, Baseline Normal AIT, Baseline Normal AIT 123, Baseline Normal Squad, Baseline Reason AIT ...
# json path: node.data.node.template.code.value

from typing import Any, Dict, List, Optional

from langflow.custom import Component
from langflow.io import HandleInput, Output
from langflow.schema.data import Data


class PGVectorMetricsExtractor(Component):
    display_name = "PGVector Metrics Extractor (Envelope)"
    description = "Returns a Data envelope: {metrics, chunks} from a PGVector results list (supports flattened or nested metadata)."
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
        Output(
            display_name="Envelope (Data)",
            name="envelope",
            method="build_envelope",
        ),
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
    def _get_text(item: Data) -> str:
        if not isinstance(item, Data):
            return ""
        d = item.data if isinstance(item.data, dict) else {}

        if isinstance(d.get("text"), str):
            return d["text"]

        if isinstance(d.get("page_content"), str):
            return d["page_content"]

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

        flat_md: Dict[str, Any] = {}
        for k, v in d.items():
            if k in self._NON_METADATA_KEYS:
                continue
            flat_md[k] = v

        merged = dict(flat_md)
        merged.update(nested_md)  # nested overrides
        return merged

    def _get_chunk_id(self, item: Data, rank: int) -> str:
        """
        Prefer stable IDs:
          1) chunk_id
          2) custom_id
          3) uuid
          4) source + rank fallback
        """
        d = item.data if isinstance(item.data, dict) else {}
        md = self._get_metadata(item)

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

        src = d.get("source")
        if not (isinstance(src, str) and src.strip()):
            src = md.get("source")

        if isinstance(src, str) and src.strip():
            return f"{src.strip()}#r{rank}"

        return f"rank_{rank}"

    @staticmethod
    def _pick_source_url(md: Dict[str, Any]) -> Optional[str]:
        # You may use either "source_url" or "source" depending on your ingestion/chunker
        for k in ("source_url", "source", "url"):
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def build_envelope(self) -> Data:
        r = self.results

        # Normalize to list[Data]
        items: List[Data] = []
        if isinstance(r, list):
            items = [x for x in r if isinstance(x, Data)]
        elif isinstance(r, Data):
            items = [r]

        # ALWAYS return Data (red) — never dict (grey)
        if not items:
            return Data(data={"metrics": {}, "chunks": []})

        # Metrics are repeated on each item; take from first
        metrics: Dict[str, Any] = {}
        first = items[0]
        if isinstance(first.data, dict):
            m = first.data.get("_retrieval")
            if isinstance(m, dict):
                metrics = dict(m)

        chunks: List[Dict[str, Any]] = []
        retrieved_chunk_ids_ranked: List[str] = []
        retrieved_source_urls_ranked: List[str] = []

        for i, item in enumerate(items):
            rank = i + 1
            d = item.data if isinstance(item.data, dict) else {}
            md = self._get_metadata(item)

            chunk_id = self._get_chunk_id(item, rank)
            retrieved_chunk_ids_ranked.append(chunk_id)

            src_url = self._pick_source_url(md)
            if src_url:
                retrieved_source_urls_ranked.append(src_url)

            chunks.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": d.get("_pgvector_score"),
                    "text": self._get_text(item),
                    "metadata": md,
                }
            )

        # Add ranked lists into metrics for convenience
        metrics["retrieved_chunk_ids_ranked"] = retrieved_chunk_ids_ranked
        metrics["retrieved_source_urls_ranked"] = retrieved_source_urls_ranked

        return Data(data={"metrics": metrics, "chunks": chunks})
