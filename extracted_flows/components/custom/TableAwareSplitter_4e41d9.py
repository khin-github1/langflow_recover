# Recovered Langflow component
# type: TableAwareSplitter
# class: TableAwareSplitter
# used in 1 flow(s): Method 1-Markdown Serialization
# json path: node.data.node.template.code.value

# Langflow Custom Component: Table-Aware Markdown Splitter
# -----------------------------------------------------------------------------
# Input : scraped markdown (one or many pages). Pages delimited by 'URL: <url>'
#         lines; headings by '#'/'##'; tables in GitHub-style pipe format.
# Output: chunks ready for the chunk_id-aware PGVector component.
#   * Each markdown TABLE -> ONE chunk, kept whole, tagged type="table" + table_id.
#   * All other PROSE  -> character-split chunks, tagged type="text" + chunk_id.
# Every chunk carries: text, chunk_id, type, table_id, source, section,
# n_rows, content_hash — ALL inside the data dict, so the inspect panel renders
# a browsable grid (one row per chunk) exactly like the scraper's output.
#
# Outputs:
#   * "Chunks (rows)"      -> list[Data]  : grid view for inspection.
#   * "Chunks (DataFrame)" -> DataFrame   : feed to PGVector (port name unchanged).
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
    display_name = "V4Table-Aware Markdown Splitter"
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
        Output(display_name="Chunks (rows)", name="chunks", method="chunks"),
        Output(display_name="Chunks (DataFrame)", name="dataframe", method="split"),
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

    # ---------------------------------------------------------- chunk factory
    @staticmethod
    def _mk(text, *, chunk_id, ctype, source, section, table_id="", n_rows=0) -> Data:
        """One chunk Data with EVERYTHING in the data dict (scraper pattern),
        so the inspect panel shows a real grid column per field."""
        return Data(
            data={
                "text": text,
                "type": ctype,
                "chunk_id": chunk_id,
                "table_id": table_id,
                "section": section,
                "source": source,
                "n_rows": n_rows,
                "content_hash": hashlib.md5(text.encode("utf-8")).hexdigest()[:10],
            }
        )

    # ------------------------------------------------------------------- main
    def _build(self) -> list[Data]:
        text = self._gather_text()
        if not text.strip():
            msg = "No markdown text found on the input."
            raise ValueError(msg)
        # If the scraper marked tables explicitly, extract them verbatim (no guessing).
        if "<!-- TABLE" in text:
            chunks = self._split_with_markers(text)
        else:
            chunks = self._split_heuristic(text)
        n_tables = sum(1 for d in chunks if d.data.get("type") == "table")
        self.status = f"{len(chunks)} chunks: {n_tables} tables + {len(chunks) - n_tables} text chunks."
        return chunks

    # public outputs -------------------------------------------------------
    def chunks(self) -> list[Data]:
        """Grid view: one row per chunk (renders like the scraper's output)."""
        return self._build()

    def split(self) -> DataFrame:
        """DataFrame view for the downstream PGVector component."""
        return DataFrame(self._build())

    # ---------------------------------------------------------- split modes
    def _split_with_markers(self, text: str) -> list[Data]:
        out_data: list[Data] = []
        prose_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=unescape_string("\n" if self.separator in ("/n", "\\n", "") else self.separator),
        )
        state = {"source": self.default_source or "", "section": "", "cc": 0}

        def emit_prose(segment: str):
            for raw in segment.split("\n"):
                s = raw.strip()
                if s.startswith("URL:"):
                    state["source"] = s[4:].strip()
                elif s.startswith("#"):
                    state["section"] = s.lstrip("#").strip()
            body = re.sub(r"<!--.*?-->", "", segment, flags=re.DOTALL).strip()
            if not body:
                return
            for piece in prose_splitter.split_text(body):
                piece = piece.strip()
                if not piece or self._is_hr(piece):
                    continue
                cid = f"{self.chunk_id_prefix}{state['cc']:04d}"
                state["cc"] += 1
                out_data.append(
                    self._mk(piece, chunk_id=cid, ctype="text",
                             source=state["source"], section=state["section"])
                )

        pattern = re.compile(r"<!-- TABLE id=(\S+) source=(.*?) -->\n(.*?)\n<!-- /TABLE -->", re.DOTALL)
        last = 0
        for m in pattern.finditer(text):
            emit_prose(text[last : m.start()])
            tid, src, table_md = m.group(1), m.group(2).strip(), m.group(3).strip()
            body = f"{state['section']}\n{table_md}" if (self.include_section_in_table and state["section"]) else table_md
            out_data.append(
                self._mk(body, chunk_id=tid, ctype="table",
                         source=src or state["source"], section=state["section"],
                         table_id=tid, n_rows=max(table_md.count("\n") - 1, 0))
            )
            last = m.end()
        emit_prose(text[last:])
        return out_data

    def _split_heuristic(self, text: str) -> list[Data]:
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
                    self._mk(piece, chunk_id=cid, ctype="text", source=source, section=section)
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
                        self._mk(body, chunk_id=tid, ctype="table", source=source,
                                 section=section, table_id=tid, n_rows=table_md.count("\n") - 1)
                    )
                    i = k
                    continue
            if s:
                prose_buf.append(lines[i])
            i += 1
        flush_prose()
        return out_data