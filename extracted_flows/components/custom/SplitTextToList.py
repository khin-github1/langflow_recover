# Recovered Langflow component
# type: SplitTextToList
# class: SplitTextComponent
# used in 1 flow(s): Summarization_flow_chunk
# json path: node.data.node.template.code.value

from langchain_text_splitters import CharacterTextSplitter
from lfx.custom.custom_component.component import Component
from lfx.io import DropdownInput, HandleInput, IntInput, MessageTextInput, Output
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.utils.util import unescape_string

class SplitTextComponent(Component):
    display_name: str = "AIT Splitter to List"
    description: str = "Splits text and formats output as a list of {text, source}."
    icon = "scissors-line-dashed"
    name = "SplitTextToList"

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
            name="separator",
            display_name="Separator",
            value="\n",
        ),
        MessageTextInput(
            name="text_key",
            display_name="Text Key",
            value="text",
            advanced=True,
        ),
        DropdownInput(
            name="keep_separator",
            display_name="Keep Separator",
            options=["False", "True", "Start", "End"],
            value="False",
            advanced=True,
        ),
    ]

    # CHANGED: Output type is now list[Data] instead of DataFrame
    outputs = [
        Output(display_name="Chunks List", name="chunks_list", method="split_text_to_list"),
    ]

    def _fix_separator(self, separator: str) -> str:
        if separator == "/n": return "\n"
        if separator == "/t": return "\t"
        return separator

    def split_text_base(self):
        separator = unescape_string(self._fix_separator(self.separator))
        
        # Prepare documents
        if isinstance(self.data_inputs, DataFrame):
            self.data_inputs.text_key = self.text_key
            documents = self.data_inputs.to_lc_documents()
        elif isinstance(self.data_inputs, Message):
            data_obj = self.data_inputs.to_data()
            data_obj.text_key = self.text_key
            documents = [data_obj.to_lc_document()]
        else:
            # Handle list of Data or single Data object
            inputs = self.data_inputs if isinstance(self.data_inputs, list) else [self.data_inputs]
            documents = []
            for input_ in inputs:
                if isinstance(input_, Data):
                    input_.text_key = self.text_key
                    documents.append(input_.to_lc_document())

        try:
            keep_sep = self.keep_separator
            if str(keep_sep).lower() == "false": keep_sep = False
            elif str(keep_sep).lower() == "true": keep_sep = True

            splitter = CharacterTextSplitter(
                chunk_overlap=self.chunk_overlap,
                chunk_size=self.chunk_size,
                separator=separator,
                keep_separator=keep_sep,
            )
            return splitter.split_documents(documents)
        except Exception as e:
            # Return error in the requested format if splitting fails
            url = getattr(self, "source_url", "unknown")
            return [Data(data={"text": f"Error: {str(e)}", "source": url})]

    def split_text_to_list(self) -> list[Data]:
        """Final method to return the list of {text, source} objects."""
        lc_docs = self.split_text_base()
        results = []
        
        for doc in lc_docs:
            # If an error occurred in base, it might already be a Data object
            if isinstance(doc, Data):
                results.append(doc)
                continue
                
            # Extract text and the source URL from metadata
            # Scrapers typically put the URL in 'source' or 'url' keys
            text_content = doc.page_content
            url = doc.metadata.get("source") or doc.metadata.get("url") or "Unknown"
            
            # Format exactly as you requested
            results.append(Data(data={
                "text": text_content,
                "source": url
            }))
            
        return results