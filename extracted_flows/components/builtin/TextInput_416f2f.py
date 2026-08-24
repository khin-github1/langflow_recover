# Recovered Langflow component
# type: TextInput
# class: TextInputComponent
# used in 10 flow(s): AIT main page _full workflow, CRCV, CRCV Test, CRCV V2, CRCV V3, CRCV-v1.0, CRCV-v1.1, CRCV-v1.1.01 ...
# json path: node.data.node.template.code.value

from langflow.base.io.text import TextComponent
from langflow.io import MultilineInput, Output
from langflow.schema.message import Message


class TextInputComponent(TextComponent):
    display_name = "Text Input"
    description = "Get user text inputs."
    documentation: str = "https://docs.langflow.org/components-io#text-input"
    icon = "type"
    name = "TextInput"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Text",
            info="Text to be passed as input.",
        ),
    ]
    outputs = [
        Output(display_name="Output Text", name="text", method="text_response"),
    ]

    def text_response(self) -> Message:
        return Message(
            text=self.input_value,
        )
