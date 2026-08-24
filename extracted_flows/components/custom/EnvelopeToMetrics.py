# Recovered Langflow component
# type: EnvelopeToMetrics
# class: EnvelopeToMetrics
# used in 16 flow(s): AITGPT Fees, AITGPT General, AITGPT Program, AITGPT V1.0.0, Baseline Normal AIT, Baseline Normal AIT 123, Baseline Normal Squad, Baseline Reason AIT ...
# json path: node.data.node.template.code.value

from typing import Any, Dict

from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema.data import Data


class EnvelopeToMetrics(Component):
    display_name = "Envelope → Metrics"
    description = "Extracts env.data['metrics'] and emits it as Data."
    icon = "Gauge"
    name = "EnvelopeToMetrics"

    inputs = [
        DataInput(
            name="envelope",
            display_name="Envelope (Data)",
            info="Data object produced by Ollama Chat (Envelope).",
            required=True,
        )
    ]

    outputs = [
        Output(display_name="Metrics (Data)", name="metrics", method="build_metrics"),
    ]

    def build_metrics(self) -> Data:
        env: Data = self.envelope
        payload: Dict[str, Any] = env.data or {}

        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}

        # Put metrics into Data.data so downstream can read keys easily
        return Data(
            text_key="text",
            data={
                "text": "",       # keep primary text empty
                "metrics": metrics
            },
            default_value="",
        )
