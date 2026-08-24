# Recovered Langflow component
# type: SplitText
# class: SplitTextComponent
# used in 2 flow(s): multi_webpage_chunk after summary, summary with conditional
# json path: node.data.node.template.code.value

from langchain_text_splitters import CharacterTextSplitter
from lfx.custom.custom_component.component import Component
from lfx.io import DropdownInput, HandleInput, IntInput, MessageTextInput, Output
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.utils.util import unescape_string

class SplitTextComponent(Component):
    display_name: str = "Split Text"
    description: str = "Split text into chunks based on specified criteria."
    documentation: str = "https://docs.langflow.org/split-text"
    icon = "scissors-line-dashed"
    name = "SplitText"

    inputs = [
        HandleInput(
            name="data_inputs",
            display_name="Input",
            info="The data with texts to split in chunks.",
            input_types=["Data", "DataFrame", "Message"],
            required=True,
        ),
        IntInput(
            name="chunk_overlap",
            display_name="Chunk Overlap",
            info="Number of characters to overlap between chunks.",
            value=200,
        ),
        IntInput(
            name="chunk_size",
            display_name="Chunk Size",
            info=(
                "The maximum length of each chunk. Text is first split by separator, "
                "then chunks are merged up to this size. "
                "Individual splits larger than this won't be further divided."
            ),
            value=1000,
        ),
        MessageTextInput(
            name="separator",
            display_name="Separator",
            info=(
                "The character to split on. Use \\n for newline. "
                "Examples: \\n\\n for paragraphs, \\n for lines, . for sentences"
            ),
            value="\n",
        ),
        MessageTextInput(
            name="text_key",
            display_name="Text Key",
            info="The key to use for the text column.",
            value="text",
            advanced=True,
        ),
        DropdownInput(
            name="keep_separator",
            display_name="Keep Separator",
            info="Whether to keep the separator in the output chunks and where to place it.",
            options=["False", "True", "Start", "End"],
            value="False",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Chunks", name="dataframe", method="split_text"),
    ]

    def _docs_to_data(self, docs) -> list[Data]:
        return [Data(text=doc.page_content, data=doc.metadata) for doc in docs]

    def _fix_separator(self, separator: str) -> str:
        """Fix common separator issues and convert to proper format."""
        if separator == "/n":
            return "\n"
        if separator == "/t":
            return "\t"
        return separator

    def split_text_base(self):
        separator = self._fix_separator(self.separator)
        separator = unescape_string(separator)

        # 1. Handle DataFrame Input
        if isinstance(self.data_inputs, DataFrame):
            if not len(self.data_inputs):
                raise TypeError("DataFrame is empty")
            self.data_inputs.text_key = self.text_key
            try:
                documents = self.data_inputs.to_lc_documents()
            except Exception as e:
                raise TypeError(f"Error converting DataFrame to documents: {e}") from e

        # 2. Handle Message Input
        elif isinstance(self.data_inputs, Message):
            self.data_inputs = [self.data_inputs.to_data()]
            return self.split_text_base()

        # 3. Handle Data or List[Data] Input (This is where your Summarizer outputs go)
        else:
            if not self.data_inputs:
                raise TypeError("No data inputs provided")

            documents = []
            # Normalize to a list to handle single Data objects or List[Data]
            inputs_list = self.data_inputs if isinstance(self.data_inputs, list) else [self.data_inputs]
            
            for input_item in inputs_list:
                if isinstance(input_item, Data):
                    # CRITICAL FIX: Explicitly set text_key for every item in the list
                    # This ensures to_lc_document() pulls the correct content.
                    input_item.text_key = self.text_key
                    documents.append(input_item.to_lc_document())
                else:
                    # Handle cases where inputs might be raw dicts
                    try:
                        temp_data = Data(data=input_item) if isinstance(input_item, dict) else input_item
                        temp_data.text_key = self.text_key
                        documents.append(temp_data.to_lc_document())
                    except Exception as e:
                        continue

            if not documents:
                raise TypeError(f"No valid Data inputs found in {type(self.data_inputs)}")

        # 4. Perform the Splitting
        try:
            keep_sep = self.keep_separator
            if isinstance(keep_sep, str):
                if keep_sep.lower() == "false":
                    keep_sep = False
                elif keep_sep.lower() == "true":
                    keep_sep = True

            splitter = CharacterTextSplitter(
                chunk_overlap=self.chunk_overlap,
                chunk_size=self.chunk_size,
                separator=separator,
                keep_separator=keep_sep,
            )
            return splitter.split_documents(documents)
        except Exception as e:
            raise TypeError(f"Error splitting text: {e}") from e

    def split_text(self) -> DataFrame:
        return DataFrame(self._docs_to_data(self.split_text_base()))