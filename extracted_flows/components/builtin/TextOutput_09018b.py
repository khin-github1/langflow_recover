# Recovered Langflow component
# type: TextOutput
# class: TextOutputComponent
# used in 10 flow(s): CRCV, CRCV Test, CRCV V2, CRCV V3, CRCV-deprecated, CRCV-v0.1 Evaluation, CRCV-v1.0, CRCV-v1.1 ...
# json path: node.data.node.template.code.value

from langflow.base.io.text import TextComponent
from langflow.io import MultilineInput, Output
from langflow.schema.message import Message


class TextOutputComponent(TextComponent):
    display_name = "Text Output"
    description = "Sends text output via API."
    documentation: str = "https://docs.langflow.org/components-io#text-output"
    icon = "type"
    name = "TextOutput"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Inputs",
            info="Text to be passed as output.",
        ),
    ]
    outputs = [
        Output(display_name="Output Text", name="text", method="text_response"),
    ]

    def text_response(self) -> Message:
        message = Message(
            text=self.input_value,
        )
        self.status = self.input_value
        return message
