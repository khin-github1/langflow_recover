# Recovered Langflow component
# type: TableAwareSplitter
# class: TableAwareSplitter
# used in 1 flow(s): Method 1-Markdown Serialization
# json path: node.data.node.template.code.value

# Langflow Custom Component: Table-Aware Markdown Splitter
# -----------------------------------------------------------------------------
# Input : scraped markdown (one or many pages). Pages delimited by 'URL: <url>'
#         lines; headings by '#'/'##'; tables in GitHub-style pipe format.
# Output: a DataFrame of chunks ready for the chunk_id-aware PGVector component.
#   * Each markdown TABLE -> ONE chunk, kept whole, tagged type="table" + table_id.
#     The reassembler stitches rows that the scrape broke across physical lines.
#   * All other PROSE  -> character-split chunks, tagged type="text" + chunk_id.
# Every chunk carries metadata: chunk_id, type, source, section, content_hash
# (tables also carry table_id). chunk_id is the universal id pgvector stores.
# -----------------------------------------------------------------------------

import hashlib
import re

from langchain_text_splitters import CharacterTextSplitter

from lfx.custom.custom_component.component import Component
from lfx.io import BoolInput, HandleInput, IntInput, MessageTextInput, Output
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.utils.util import unescape_string


class TableAwareSplitter(Component):
    display_name = "Table-Aware Markdown Splitter V2"
    description = "Keeps each markdown table as one chunk (with table_id); chunks prose separately."
    icon = "scissors-line-dashed"
    name = "TableAwareSplitter"

    inputs = [
        HandleInput(
            name="data_inputs",
            display_name="Input",
            info="Markdown text to split (Data, DataFrame, or Message).",
            input_types=["Data", "DataFrame", "Message"],
            required=True,
        ),
        IntInput(name="chunk_size", display_name="Prose Chunk Size", value=1000),
        IntInput(name="chunk_overlap", display_name="Prose Chunk Overlap", value=200),
        MessageTextInput(
            name="separator",
            display_name="Prose Separator",
            info="Separator for splitting PROSE only (tables are never split). Use \\n for lines.",
            value="\n",
        ),
        BoolInput(
            name="include_section_in_table",
            display_name="Prefix table with its section heading",
            value=True,
            info="Embeds the nearest heading with the table for better retrieval context.",
        ),
        MessageTextInput(name="text_key", display_name="Text Key", value="text", advanced=True),
        MessageTextInput(name="default_source", display_name="Fallback source URL", value="", advanced=True),
        MessageTextInput(name="table_id_prefix", display_name="Table ID prefix", value="t", advanced=True),
        MessageTextInput(name="chunk_id_prefix", display_name="Text chunk ID prefix", value="c", advanced=True),
    ]

    outputs = [
        Output(display_name="Chunks", name="dataframe", method="split"),
    ]

    # ---------------------------------------------------------- markdown utils
    @staticmethod
    def _is_sep(line: str) -> bool:
        s = line.strip()
        if "-" not in s:
            return False
        return set(s.replace(" ", "")) <= set("|-:") and "|" in s

    @staticmethod
    def _is_hr(line: str) -> bool:
        """A markdown horizontal rule (e.g. '---' page separator) — not a table."""
        s = line.strip()
        return len(s) >= 3 and set(s) <= set("-")

    @staticmethod
    def _read_row(lines, k):
        """Read one logical pipe-row starting at k, reassembling broken physical lines."""
        parts = [lines[k].rstrip()]
        k += 1
        while k < len(lines):
            cur = re.sub(r"\s+", " ", " ".join(p.strip() for p in parts)).strip()
            if cur.endswith("|"):
                break
            nxt = lines[k].strip()
            if nxt == "" or nxt.startswith("#"):
                break
            parts.append(nxt)
            k += 1
        row = re.sub(r"\s+", " ", " ".join(p.strip() for p in parts)).strip()
        return row, k

    def _parse_table(self, lines, i):
        """If a table starts at line i, return (table_markdown, next_index); else None."""
        header, k = self._read_row(lines, i)
        j = k
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j >= len(lines) or not self._is_sep(lines[j]):
            return None
        rows = [header, lines[j].strip()]
        k = j + 1
        while k < len(lines):
            sline = lines[k].strip()
            if sline == "" or sline.startswith("#") or not sline.startswith("|"):
                break
            row, k = self._read_row(lines, k)
            rows.append(row)
        return "\n".join(rows), k

    # ------------------------------------------------------------- input text
    def _gather_text(self) -> str:
        di = self.data_inputs
        if isinstance(di, Message):
            return di.text or ""
        if isinstance(di, DataFrame):
            di.text_key = self.text_key
            return "\n".join(d.page_content for d in di.to_lc_documents())
        if isinstance(di, Data):
            return di.get_text() or ""
        if isinstance(di, list):
            out = []
            for x in di:
                if isinstance(x, Data):
                    out.append(x.get_text() or "")
                elif isinstance(x, Message):
                    out.append(x.text or "")
            return "\n".join(out)
        return str(di or "")

    # ------------------------------------------------------------------- main
    def split(self) -> DataFrame:
        text = self._gather_text()
        if not text.strip():
            msg = "No markdown text found on the input."
            raise ValueError(msg)

        lines = text.split("\n")
        sep = unescape_string("\n" if self.separator in ("/n", "\\n", "") else self.separator)
        prose_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator=sep
        )

        out_data: list[Data] = []
        prose_buf: list[str] = []
        source = self.default_source or ""
        section = ""
        tcount = 0
        ccount = 0

        def flush_prose():
            nonlocal ccount, prose_buf
            body = "\n".join(prose_buf).strip()
            prose_buf = []
            if not body:
                return
            for piece in prose_splitter.split_text(body):
                piece = piece.strip()
                if not piece:
                    continue
                cid = f"{self.chunk_id_prefix}{ccount:04d}"
                ccount += 1
                out_data.append(
                    Data(
                        text=piece,
                        data={
                            "chunk_id": cid,
                            "type": "text",
                            "source": source,
                            "section": section,
                            "content_hash": hashlib.md5(piece.encode("utf-8")).hexdigest()[:10],
                        },
                    )
                )

        i = 0
        while i < len(lines):
            s = lines[i].strip()
            if s.startswith("URL:"):
                flush_prose()
                source = s[4:].strip()
                i += 1
                continue
            if self._is_hr(lines[i]):  # '---' page separator from Merged Markdown
                i += 1
                continue
            if s.startswith("#"):
                flush_prose()
                section = s.lstrip("#").strip()
                prose_buf.append(s)  # keep heading text in prose chunks too
                i += 1
                continue
            if s.startswith("|") and not self._is_sep(lines[i]):
                res = self._parse_table(lines, i)
                if res:
                    table_md, k = res
                    flush_prose()
                    tid = f"{self.table_id_prefix}{tcount:04d}"
                    tcount += 1
                    body = f"{section}\n{table_md}" if (self.include_section_in_table and section) else table_md
                    out_data.append(
                        Data(
                            text=body,
                            data={
                                "chunk_id": tid,  # universal id pgvector stores
                                "table_id": tid,
                                "type": "table",
                                "source": source,
                                "section": section,
                                "n_rows": table_md.count("\n") - 1,  # minus header+sep
                                "content_hash": hashlib.md5(table_md.encode("utf-8")).hexdigest()[:10],
                            },
                        )
                    )
                    i = k
                    continue
            if s:
                prose_buf.append(lines[i])
            i += 1
        flush_prose()

        n_tables = sum(1 for d in out_data if d.data.get("type") == "table")
        n_text = len(out_data) - n_tables
        self.status = f"{len(out_data)} chunks: {n_tables} tables (whole) + {n_text} text chunks."
        return DataFrame(out_data)