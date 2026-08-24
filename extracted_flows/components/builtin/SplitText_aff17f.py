# Recovered Langflow component
# type: SplitText
# class: SplitTextComponent
# used in 1 flow(s): Table-aware-scraper
# json path: node.data.node.template.code.value

from langchain_text_splitters import CharacterTextSplitter

from lfx.custom.custom_component.component import Component
from lfx.io import DropdownInput, HandleInput, IntInput, MessageTextInput, Output
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.utils.util import unescape_string

import hashlib


class SplitTextComponent(Component):
    display_name: str = "Split Text (chunk_id + gold_table_id)"
    description: str = (
        "Split text into chunks. Carries source/table_id/gold_table_id from the "
        "incoming table Data into every chunk, and stamps a stable, table-scoped chunk_id."
    )
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
        MessageTextInput(
            name="chunk_id_key",
            display_name="Chunk ID metadata key",
            info="Metadata key under which the stable chunk id is stored.",
            value="chunk_id",
            advanced=True,
        ),
        MessageTextInput(
            name="chunk_id_prefix",
            display_name="Chunk ID prefix",
            info="Prefix for the readable chunk id, e.g. 'c' -> c0000, c0001, ...",
            value="c",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Chunks", name="dataframe", method="split_text"),
    ]

    def _docs_to_data(self, docs) -> list[Data]:
        return [Data(text=doc.page_content, data=doc.metadata) for doc in docs]

    def _fix_separator(self, separator: str) -> str:
        if separator == "/n":
            return "\n"
        if separator == "/t":
            return "\t"
        return separator

    def split_text_base(self):
        separator = self._fix_separator(self.separator)
        separator = unescape_string(separator)

        if isinstance(self.data_inputs, DataFrame):
            if not len(self.data_inputs):
                msg = "DataFrame is empty"
                raise TypeError(msg)

            self.data_inputs.text_key = self.text_key
            try:
                documents = self.data_inputs.to_lc_documents()
            except Exception as e:
                msg = f"Error converting DataFrame to documents: {e}"
                raise TypeError(msg) from e
        elif isinstance(self.data_inputs, Message):
            self.data_inputs = [self.data_inputs.to_data()]
            return self.split_text_base()
        else:
            if not self.data_inputs:
                msg = "No data inputs provided"
                raise TypeError(msg)

            documents = []
            if isinstance(self.data_inputs, Data):
                self.data_inputs.text_key = self.text_key
                documents = [self.data_inputs.to_lc_document()]
            else:
                try:
                    documents = [input_.to_lc_document() for input_ in self.data_inputs if isinstance(input_, Data)]
                    if not documents:
                        msg = f"No valid Data inputs found in {type(self.data_inputs)}"
                        raise TypeError(msg)
                except AttributeError as e:
                    msg = f"Invalid input type in collection: {e}"
                    raise TypeError(msg) from e
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
            # split_documents preserves each source document's metadata
            # (source, table_id, gold_table_id, institution) on every child chunk.
            return splitter.split_documents(documents)
        except Exception as e:
            msg = f"Error splitting text: {e}"
            raise TypeError(msg) from e

    def _stamp_chunk_ids(self, docs):
        """Table-scoped, stable chunk ids + guaranteed gold_table_id passthrough.

        chunk_id looks like 'bu-t0001-c0000' when table_id is present, so ids are
        readable, unique across tables, and idempotent on re-ingest.
        """
        per_table_counter: dict[str, int] = {}
        for global_i, doc in enumerate(docs):
            md = doc.metadata or {}

            table_id = md.get("table_id") or md.get("gold_table_id")
            if table_id and not md.get("gold_table_id"):
                md["gold_table_id"] = table_id

            if table_id:
                local_i = per_table_counter.get(table_id, 0)
                per_table_counter[table_id] = local_i + 1
                md[self.chunk_id_key] = f"{table_id}-{self.chunk_id_prefix}{local_i:04d}"
            else:
                md[self.chunk_id_key] = f"{self.chunk_id_prefix}{global_i:04d}"

            md["chunk_index"] = global_i
            md["content_hash"] = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:10]
            doc.metadata = md
        return docs

    def split_text(self) -> DataFrame:
        docs = self._stamp_chunk_ids(self.split_text_base())
        n_tables = len({d.metadata.get("gold_table_id") for d in docs if d.metadata.get("gold_table_id")})
        self.status = f"Split into {len(docs)} chunks across {n_tables} gold tables."
        return DataFrame(self._docs_to_data(docs))