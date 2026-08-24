# Recovered Langflow component
# type: JoinedRowToPromptMessage
# class: JoinedRowToPromptMessage
# used in 10 flow(s): Cognitive RAG V1.2.0 GPT, Cognitive RAG V1.2.5 GPT, Cognitive RAG V1.3.0 GPT, Cognitive RAG V1.3.0 GPT Loose AIT , Cognitive RAG V1.3.0 GPT Strict, Cognitive RAG V1.3.0 GPT Strict AIT, Retry RAG V1.3.0 GPT, Retry RAG V1.3.0 GPT Loose AIT ...
# json path: node.data.node.template.code.value

from typing import Any, Dict

from lfx.custom import Component
from lfx.io import DataInput, Output
from lfx.schema import Data, Message


class JoinedRowToPromptMessage(Component):
    display_name = "Joined Row → Prompt Message"
    description = "Extracts matched generator_prompt.raw_prompt_text as Message."
    icon = "file-text"
    name = "JoinedRowToPromptMessage"

    inputs = [
        DataInput(
            name="joined_record",
            display_name="Joined Record",
            required=True,
        )
    ]

    outputs = [
        Output(
            display_name="Prompt Message",
            name="prompt_message",
            method="build_prompt_message",
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

    def build_prompt_message(self) -> Message:
        joined = self._payload()
        text = str(joined.get("generator_prompt_text", "") or "").strip()
        return Message(text=text, data={"text": text})