# Recovered Langflow component
# type: TextInput
# class: TextInputComponent
# used in 63 flow(s): AIT Cognitive RAG, Baseline Normal AIT, Baseline Normal AIT 123, Baseline Normal Squad, Baseline Reason AIT, Baseline Reason Squad, Blog Writer, CRCV Evaluation Runner 1.0.0 (backup) ...
# json path: node.data.node.template.code.value

from lfx.base.io.text import TextComponent
from lfx.io import MultilineInput, Output
from lfx.schema.message import Message


class TextInputComponent(TextComponent):
    display_name = "Text Input"
    description = "Get user text inputs."
    documentation: str = "https://docs.langflow.org/text-input-and-output"
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
