# Recovered Langflow component
# type: MessageOrGate
# class: MessageOrGate
# used in 2 flow(s): Cognitive RAG V1.1.0 Eval, Cognitive RAG V1.1.5 Backup
# json path: node.data.node.template.code.value

from typing import Any

from langflow.custom import Component
from langflow.io import DropdownInput, HandleInput, MessageTextInput, Output
from langflow.schema.message import Message


class MessageOrGate(Component):
    display_name = "Message OR Gate"
    description = "Returns the first non-empty message, or combines both if configured."
    icon = "git-merge"
    name = "MessageOrGate"

    inputs = [
        HandleInput(
            name="primary_message",
            display_name="Primary Message",
            input_types=["Message"],
            required=False,
        ),
        HandleInput(
            name="secondary_message",
            display_name="Secondary Message",
            input_types=["Message"],
            required=False,
        ),
        DropdownInput(
            name="mode",
            display_name="Mode",
            options=["first_non_empty", "prefer_primary", "prefer_secondary", "concat_both"],
            value="first_non_empty",
            info=(
                "first_non_empty: return primary if non-empty else secondary.\n"
                "prefer_primary: always return primary if it exists, otherwise secondary.\n"
                "prefer_secondary: always return secondary if it exists, otherwise primary.\n"
                "concat_both: combine both messages."
            ),
        ),
        MessageTextInput(
            name="separator",
            display_name="Separator",
            value="\n\n",
            required=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message", method="build_message"),
    ]

    def _as_message(self, value: Any) -> Message | None:
        if value is None:
            return None

        if isinstance(value, Message):
            text = str(value.text or "").strip()
            if not text:
                return None
            return value

        if hasattr(value, "text"):
            text = str(getattr(value, "text", "") or "").strip()
            if not text:
                return None
            return Message(text=text)

        text = str(value).strip()
        if not text:
            return None
        return Message(text=text)

    def build_message(self) -> Message:
        primary = self._as_message(self.primary_message)
        secondary = self._as_message(self.secondary_message)

        mode = str(self.mode or "first_non_empty").strip().lower()
        sep = str(self.separator or "\n\n")

        if mode == "prefer_primary":
            chosen = primary or secondary
            return chosen if chosen is not None else Message(text="")

        if mode == "prefer_secondary":
            chosen = secondary or primary
            return chosen if chosen is not None else Message(text="")

        if mode == "concat_both":
            if primary and secondary:
                return Message(text=f"{primary.text}{sep}{secondary.text}")
            if primary:
                return primary
            if secondary:
                return secondary
            return Message(text="")

        # default: first_non_empty
        if primary:
            return primary
        if secondary:
            return secondary
        return Message(text="")