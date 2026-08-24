# Recovered Langflow component
# type: JoinedRowToAnalyzerData
# class: JoinedRowToAnalyzerData
# used in 10 flow(s): Cognitive RAG V1.2.0 GPT, Cognitive RAG V1.2.5 GPT, Cognitive RAG V1.3.0 GPT, Cognitive RAG V1.3.0 GPT Loose AIT , Cognitive RAG V1.3.0 GPT Strict, Cognitive RAG V1.3.0 GPT Strict AIT, Retry RAG V1.3.0 GPT, Retry RAG V1.3.0 GPT Loose AIT ...
# json path: node.data.node.template.code.value

from typing import Any, Dict

from lfx.custom import Component
from lfx.io import DataInput, Output
from lfx.schema import Data


class JoinedRowToAnalyzerData(Component):
    display_name = "Joined Row → Analyzer Data"
    description = "Extracts matched analyzer payload as Data."
    icon = "database"
    name = "JoinedRowToAnalyzerData"

    inputs = [
        DataInput(
            name="joined_record",
            display_name="Joined Record",
            required=True,
        )
    ]

    outputs = [
        Output(
            display_name="Analyzer Data",
            name="analyzer_data",
            method="build_analyzer_data",
        )
    ]

    def _payload(self) -> Dict[str, Any]:
        if isinstance(self.joined_record, Data):
            return self.joined_record.data or {}
        if hasattr(self.joined_record, "data") and isinstance(self.joined_record.data, dict):
            return self.joined_record.data
        if isinstance(self.joined_record, dict):
            return self.joined_record
        return {}

    def build_analyzer_data(self) -> Data:
        joined = self._payload()
        analyzer = joined.get("analyzer_data", {}) or {}
        if not isinstance(analyzer, dict):
            analyzer = {}
        return Data(text_key="original_question", data=analyzer, default_value="")