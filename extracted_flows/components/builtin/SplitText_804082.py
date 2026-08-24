# Recovered Langflow component
# type: SplitText
# class: SplitTextComponent
# used in 2 flow(s): URL Test, URL Test (Backup)
# json path: node.data.node.template.code.value

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langflow.custom.custom_component.component import Component
from langflow.io import DropdownInput, HandleInput, IntInput, MessageTextInput, Output
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame
from langflow.schema.message import Message
from langflow.utils.util import unescape_string

from urllib.parse import urlsplit, urlunsplit


class SplitTextComponent(Component):
    display_name: str = "Split Text (Chunk IDs)"
    description: str = "Split text into chunks and attach stable chunk_id/doc_id/chunk_index metadata."
    documentation: str = "https://docs.langflow.org/components-processing#split-text"
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
            info="The maximum length of each chunk.",
            value=1000,
        ),
        MessageTextInput(
            name="separator",
            display_name="Separator",
            info=(
                "Primary separator. Use \\n for newline. "
                "This component will also fall back to other separators automatically."
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

    # ----------------------------
    # Helpers
    # ----------------------------
    def _docs_to_data(self, docs) -> list[Data]:
        # Keep metadata in Data.data, chunk text in Data.text
        return [Data(text=doc.page_content, data=(doc.metadata or {})) for doc in docs]

    @staticmethod
    def _fix_separator(separator: str) -> str:
        """Fix common separator issues and convert to proper format."""
        if separator == "/n":
            return "\n"
        if separator == "/t":
            return "\t"
        return separator

    @staticmethod
    def _get_source_from_metadata(md: Dict) -> Optional[str]:
        for k in ("source", "url", "page_url", "source_url"):
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    @staticmethod
    def _normalize_doc_id(source: str) -> str:
        """
        Make a stable doc_id from a URL:
          - drop query + fragment
          - strip trailing slash
        """
        try:
            parts = urlsplit(source.strip())
            cleaned = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
            cleaned = cleaned.rstrip("/")
            return cleaned if cleaned else source.strip()
        except Exception:
            return source.strip().rstrip("/")

    def _keep_sep_value(self):
        keep_sep = self.keep_separator
        if isinstance(keep_sep, str):
            low = keep_sep.lower()
            if low == "false":
                return False
            if low == "true":
                return True
            if low == "start":
                return "start"
            if low == "end":
                return "end"
        return keep_sep

    # ----------------------------
    # Core logic
    # ----------------------------
    def split_text_base(self):
        # Prepare separators
        primary_sep = unescape_string(self._fix_separator(self.separator))

        # Recursive splitter guarantees chunk_size adherence (unlike CharacterTextSplitter)
        # Put user-selected separator first, then common fallbacks.
        separators = []
        if isinstance(primary_sep, str) and primary_sep != "":
            separators.append(primary_sep)
        # common fallbacks (helps when scraper returns single-line text)
        for s in ["\n\n", "\n", ". ", " ", ""]:
            if s not in separators:
                separators.append(s)

        keep_sep = self._keep_sep_value()

        # Convert inputs -> LangChain Documents
        if isinstance(self.data_inputs, DataFrame):
            if not len(self.data_inputs):
                raise TypeError("DataFrame is empty")
            self.data_inputs.text_key = self.text_key
            try:
                documents = self.data_inputs.to_lc_documents()
            except Exception as e:
                raise TypeError(f"Error converting DataFrame to documents: {e}") from e

        elif isinstance(self.data_inputs, Message):
            # Convert Message -> Data -> recurse
            self.data_inputs = [self.data_inputs.to_data()]
            return self.split_text_base()

        else:
            if not self.data_inputs:
                raise TypeError("No data inputs provided")

            if isinstance(self.data_inputs, Data):
                self.data_inputs.text_key = self.text_key
                documents = [self.data_inputs.to_lc_document()]
            else:
                # list of Data
                try:
                    documents = [x.to_lc_document() for x in self.data_inputs if isinstance(x, Data)]
                    if not documents:
                        raise TypeError(f"No valid Data inputs found in {type(self.data_inputs)}")
                except AttributeError as e:
                    raise TypeError(f"Invalid input type in collection: {e}") from e

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(self.chunk_size),
            chunk_overlap=int(self.chunk_overlap),
            separators=separators,
            keep_separator=keep_sep,
        )

        # Split
        chunk_docs = splitter.split_documents(documents)

        # Attach stable IDs
        # chunk_index must be per-doc_id (per source page)
        per_doc_counter: Dict[str, int] = {}
        now = datetime.now(timezone.utc).isoformat()

        for d in chunk_docs:
            md = d.metadata or {}

            source = self._get_source_from_metadata(md) or "unknown_source"
            doc_id = self._normalize_doc_id(source)

            idx = per_doc_counter.get(doc_id, 0)
            per_doc_counter[doc_id] = idx + 1

            # Use a stable, readable chunk_id
            # (Pad index to keep lexical order aligned with numeric order)
            chunk_id = f"{doc_id}::c{idx:04d}"

            # Do not overwrite if already present (lets you bring your own IDs)
            md.setdefault("source", source)
            md.setdefault("doc_id", doc_id)
            md.setdefault("chunk_index", idx)
            md.setdefault("chunk_id", chunk_id)

            # Optional handy fields for debugging/QA
            md.setdefault("chunk_len_chars", len(d.page_content or ""))
            md.setdefault("ingested_at_utc", now)

            d.metadata = md

        return chunk_docs

    def split_text(self) -> DataFrame:
        return DataFrame(self._docs_to_data(self.split_text_base()))
