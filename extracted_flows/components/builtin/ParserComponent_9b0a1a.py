# Recovered Langflow component
# type: ParserComponent
# class: ParserComponent
# used in 1 flow(s): CRCV-v1.103 (Normal) Oak Backups
# json path: node.data.node.template.code.value

from langflow.custom.custom_component.component import Component
from langflow.helpers.data import safe_convert
from langflow.inputs.inputs import BoolInput, HandleInput, MessageTextInput, MultilineInput, TabInput
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame
from langflow.schema.message import Message
from langflow.template.field.base import Output


class ParserComponent(Component):
    display_name = "Parser"
    description = "Extracts text using a template."
    documentation: str = "https://docs.langflow.org/components-processing#parser"
    icon = "braces"

    inputs = [
        HandleInput(
            name="input_data",
            display_name="Data / DataFrame / List[Data]",
            input_types=["DataFrame", "Data"],
            info="Accepts DataFrame, Data, or list[Data] (e.g., from PGVector).",
            required=True,
        ),
        TabInput(
            name="mode",
            display_name="Mode",
            options=["Parser", "Stringify"],
            value="Parser",
            info="Convert into raw string instead of using a template.",
            real_time_refresh=True,
        ),
        MultilineInput(
            name="pattern",
            display_name="Template",
            info=(
                "Used only in Parser mode.\n"
                "For Data: keys in data.data, e.g. {text}\n"
                "For list[Data]: applied to each item and joined with Separator."
            ),
            value="Text: {text}",
            dynamic=True,
            show=True,
            required=True,
        ),
        MessageTextInput(
            name="sep",
            display_name="Separator",
            advanced=True,
            value="\n",
            info="String used to separate rows/items.",
        ),
    ]

    outputs = [
        Output(
            display_name="Parsed Text",
            name="parsed_text",
            info="Formatted text output.",
            method="parse_combined_text",
        ),
    ]

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name == "mode":
            is_parser = (field_value == "Parser")
            build_config["pattern"]["show"] = is_parser
            build_config["pattern"]["required"] = is_parser

            # Keep your old behavior: only add clean_data control for Stringify mode
            if field_value == "Stringify":
                clean_data = BoolInput(
                    name="clean_data",
                    display_name="Clean Data",
                    info="Enable to clean output by removing empty lines.",
                    value=True,
                    advanced=True,
                    required=False,
                )
                build_config["clean_data"] = clean_data.to_dict()
            else:
                build_config.pop("clean_data", None)

        return build_config

    def _clean_args(self):
        """
        Returns:
          df: DataFrame | None
          data: Data | None
          data_list: list[Data] | None
        """
        input_data = self.input_data

        match input_data:
            case list() if all(isinstance(item, Data) for item in input_data):
                # ACCEPT list[Data] from PGVector
                return None, None, input_data

            case DataFrame():
                return input_data, None, None

            case Data():
                return None, input_data, None

            case dict() if "data" in input_data:
                try:
                    if "columns" in input_data:  # Likely a DataFrame
                        return DataFrame.from_dict(input_data), None, None
                    return None, Data(**input_data), None
                except (TypeError, ValueError, KeyError) as e:
                    raise ValueError(f"Invalid structured input provided: {e!s}") from e

            case _:
                raise ValueError(
                    f"Unsupported input type: {type(input_data)}. Expected DataFrame, Data, or list[Data]."
                )

    @staticmethod
    def _extract_text_from_data(d: Data) -> str:
        """
        Return the chunk text only, not the whole Data object.
        Common keys: 'text' (docs_to_data), sometimes 'page_content'.
        """
        payload = d.data if isinstance(d.data, dict) else {}

        if isinstance(payload, dict):
            if isinstance(payload.get("text"), str) and payload["text"].strip():
                return payload["text"]
            if isinstance(payload.get("page_content"), str) and payload["page_content"].strip():
                return payload["page_content"]

        # fallback to safe_convert if unknown shape
        return safe_convert(d)

    @staticmethod
    def _clean_text(s: str) -> str:
        # remove empty lines and trailing spaces
        lines = [ln.rstrip() for ln in s.splitlines()]
        lines = [ln for ln in lines if ln.strip() != ""]
        return "\n".join(lines)

    def parse_combined_text(self) -> Message:
        # Stringify mode: produce clean doc context string
        if self.mode == "Stringify":
            return self.convert_to_string()

        df, data, data_list = self._clean_args()

        lines = []
        if df is not None:
            for _, row in df.iterrows():
                lines.append(self.pattern.format(**row.to_dict()))
        elif data is not None:
            payload = data.data if isinstance(data.data, dict) else {}
            lines.append(self.pattern.format(**payload))
        elif data_list is not None:
            for d in data_list:
                payload = d.data if isinstance(d.data, dict) else {}
                lines.append(self.pattern.format(**payload))

        combined_text = (self.sep or "\n").join(lines)
        self.status = combined_text
        return Message(text=combined_text)

    def convert_to_string(self) -> Message:
        """
        IMPORTANT CHANGE:
        - If list[Data] from PGVector: join only their chunk texts.
        - If single Data: extract only its text.
        - Else fallback to safe_convert.
        """
        clean = bool(getattr(self, "clean_data", False))
        sep = self.sep or "\n"

        inp = self.input_data

        if isinstance(inp, list) and all(isinstance(item, Data) for item in inp):
            parts = [self._extract_text_from_data(item) for item in inp]
            result = sep.join(parts)
        elif isinstance(inp, Data):
            result = self._extract_text_from_data(inp)
        else:
            result = safe_convert(inp)

        if clean:
            result = self._clean_text(result)

        message = Message(text=result)
        self.status = message
        return message
