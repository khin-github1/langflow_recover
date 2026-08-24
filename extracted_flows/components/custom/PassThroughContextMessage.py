# Recovered Langflow component
# type: PassThroughContextMessage
# class: PassThroughContextMessage
# used in 2 flow(s): Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup
# json path: node.data.node.template.code.value

from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema.message import Message


class PassThroughContextMessage(Component):
    display_name = "Pass Through Context Message"
    description = "Returns the incoming message text as-is."
    icon = "file-search"
    name = "PassThroughContextMessage"

    inputs = [
        MessageTextInput(
            name="input_value",
            display_name="Input",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Context",
            name="context",
            method="build_context",
        ),
    ]

    def build_context(self) -> Message:
        return Message(text=self.input_value or "")