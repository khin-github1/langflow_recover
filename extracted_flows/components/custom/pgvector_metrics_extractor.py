# Recovered Langflow component
# type: pgvector_metrics_extractor
# class: PGVectorMetricsExtractor
# used in 1 flow(s): CRCV-v1.103 (Normal) Oak Backups
# json path: node.data.node.template.code.value

from typing import Any, Dict, List

from langflow.custom import Component
from langflow.io import HandleInput, Output
from langflow.schema.data import Data


class PGVectorMetricsExtractor(Component):
    display_name = "PGVector Metrics Extractor"
    description = "Extracts retrieval metrics from the first Data item in a PGVector results list."
    icon = "gauge"
    name = "pgvector_metrics_extractor"

    inputs = [
        HandleInput(
            name="results",
            display_name="Results",
            input_types=["Data"],
            required=True,
        )
    ]

    outputs = [
        Output(display_name="Metrics", name="metrics", method="build_metrics"),
    ]

    def build_metrics(self) -> Dict[str, Any]:
        r = self.results

        # LangFlow often passes lists for vector results
        if isinstance(r, list) and r:
            first = r[0]
            if isinstance(first, Data) and isinstance(first.data, dict):
                m = first.data.get("_retrieval")
                if isinstance(m, dict):
                    return m
                return {}
            # if first is dict-like
            if isinstance(first, dict):
                m = first.get("_retrieval")
                return m if isinstance(m, dict) else {}

        # Single Data fallback
        if isinstance(r, Data) and isinstance(r.data, dict):
            m = r.data.get("_retrieval")
            return m if isinstance(m, dict) else {}

        return {}
