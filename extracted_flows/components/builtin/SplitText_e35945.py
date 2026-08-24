# Recovered Langflow component
# type: SplitText
# class: SplitTextComponent
# used in 3 flow(s): multi_webpage_chunk after summary, pageindex_json_flow, summary with conditional
# json path: node.data.node.template.code.value

from langchain_text_splitters import RecursiveCharacterTextSplitter # Changed here
from lfx.custom.custom_component.component import Component
from lfx.io import DropdownInput, HandleInput, IntInput, MessageTextInput, Output
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.utils.util import unescape_string

class SplitTextComponent(Component):
    display_name: str = "Split Text (Recursive)"
    description: str = "Split text into chunks using multiple fallback separators."
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
            value=200,
        ),
        IntInput(
            name="chunk_size",
            display_name="Chunk Size",
            value=1000,
        ),
        MessageTextInput(
            name="text_key",
            display_name="Text Key",
            value="text",
        ),
    ]

    outputs = [
        Output(display_name="Chunks", name="dataframe", method="split_text"),
    ]

    def _docs_to_data(self, docs) -> list[Data]:
        return [Data(text=doc.page_content, data=doc.metadata) for doc in docs]

    def split_text_base(self):
        # 1. Prepare Documents
        documents = []
        inputs_list = self.data_inputs if isinstance(self.data_inputs, list) else [self.data_inputs]
        
        for input_item in inputs_list:
            if isinstance(input_item, Data):
                # This ensures the 'text' column in your image is found
                input_item.text_key = self.text_key
                documents.append(input_item.to_lc_document())
            elif isinstance(input_item, Message):
                documents.append(input_item.to_data().to_lc_document())

        if not documents:
            return []

        # 2. Use Recursive Splitter (Much more reliable)
        # It tries \n\n, then \n, then " ", then ""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
        
        return splitter.split_documents(documents)

    def split_text(self) -> DataFrame:
        return DataFrame(self._docs_to_data(self.split_text_base()))