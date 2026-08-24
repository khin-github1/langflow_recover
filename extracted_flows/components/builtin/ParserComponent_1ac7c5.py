# Recovered Langflow component
# type: ParserComponent
# class: ParserComponent
# used in 1 flow(s): CRCV-v1.103 (Normal) Oak Backups
# json path: node.data.node.template.code.value

import re
from typing import Any, Dict, List, Optional, Tuple

from langflow.custom.custom_component.component import Component
from langflow.helpers.data import safe_convert
from langflow.inputs.inputs import (
    BoolInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    MultilineInput,
    TabInput,
)
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame
from langflow.schema.message import Message
from langflow.template.field.base import Output


class ParserComponent(Component):
    display_name = "Parser"
    description = "Extracts text using a template or formats RAG results into a readable context."
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
            options=["Parser", "Stringify", "RAG Context"],
            value="Parser",
            info="Parser: template formatting. Stringify: raw text. RAG Context: ranked readable blocks for LLM.",
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

        # --- Controls for Stringify / RAG Context (shown dynamically) ---
        IntInput(
            name="max_chunks",
            display_name="Max Chunks",
            value=5,
            advanced=True,
        ),
        IntInput(
            name="max_chars_per_chunk",
            display_name="Max chars per chunk",
            value=1200,
            advanced=True,
        ),
        BoolInput(
            name="include_source_url",
            display_name="Include source_url",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="include_chunk_id",
            display_name="Include chunk_id",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="include_score",
            display_name="Include score",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="dedupe_chunks",
            display_name="Dedupe near-identical chunks",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="strip_boilerplate",
            display_name="Strip boilerplate (cookies/newsletter/etc.)",
            value=True,
            advanced=True,
        ),
        MultilineInput(
            name="boilerplate_markers",
            display_name="Boilerplate markers (one per line)",
            value=(
                "we use cookies\n"
                "cookie settings\n"
                "accept all\n"
                "manage consent\n"
                "privacy overview\n"
                "subscribe to our newsletter\n"
                "follow us on social media\n"
            ),
            advanced=True,
        ),
        BoolInput(
            name="clean_data",
            display_name="Clean Data",
            info="Remove empty lines and trim whitespace.",
            value=True,
            advanced=True,
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

    # ----------------------------
    # UI dynamic config
    # ----------------------------
    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name == "mode":
            is_parser = (field_value == "Parser")
            is_stringify = (field_value == "Stringify")
            is_rag = (field_value == "RAG Context")

            # Template only in Parser mode
            build_config["pattern"]["show"] = is_parser
            build_config["pattern"]["required"] = is_parser

            # Show RAG/Stringify knobs only when relevant
            for k in [
                "max_chunks",
                "max_chars_per_chunk",
                "include_source_url",
                "include_chunk_id",
                "include_score",
                "dedupe_chunks",
                "strip_boilerplate",
                "boilerplate_markers",
                "clean_data",
            ]:
                if k in build_config:
                    build_config[k]["show"] = (is_stringify or is_rag)

        return build_config

    # ----------------------------
    # Input normalization
    # ----------------------------
    def _clean_args(self) -> Tuple[Optional[DataFrame], Optional[Data], Optional[List[Data]]]:
        """
        Returns:
          df: DataFrame | None
          data: Data | None
          data_list: list[Data] | None
        """
        input_data = self.input_data

        match input_data:
            case list() if all(isinstance(item, Data) for item in input_data):
                return None, None, input_data
            case DataFrame():
                return input_data, None, None
            case Data():
                return None, input_data, None
            case dict() if "data" in input_data:
                # Structured dict that may represent Data or DataFrame
                try:
                    if "columns" in input_data:
                        return DataFrame.from_dict(input_data), None, None
                    return None, Data(**input_data), None
                except (TypeError, ValueError, KeyError) as e:
                    raise ValueError(f"Invalid structured input provided: {e!s}") from e
            case _:
                raise ValueError(
                    f"Unsupported input type: {type(input_data)}. Expected DataFrame, Data, or list[Data]."
                )

    # ----------------------------
    # Extraction helpers
    # ----------------------------
    @staticmethod
    def _payload(d: Data) -> Dict[str, Any]:
        return d.data if isinstance(d.data, dict) else {}

    @classmethod
    def _extract_text_from_data(cls, d: Data) -> str:
        payload = cls._payload(d)

        # Common keys in LangFlow vector outputs
        for key in ("text", "page_content"):
            v = payload.get(key)
            if isinstance(v, str) and v.strip():
                return v

        # If Data has a text_key and it exists
        tk = getattr(d, "text_key", None)
        if isinstance(tk, str) and tk in payload and isinstance(payload[tk], str):
            return payload[tk]

        # fallback
        return safe_convert(d)

    @classmethod
    def _extract_metadata(cls, d: Data) -> Dict[str, Any]:
        payload = cls._payload(d)

        # docs_to_data can store metadata directly in data dict, or nested as "metadata"
        md = payload.get("metadata")
        if isinstance(md, dict):
            return md

        # If your pipeline already puts metadata fields at top-level, keep them
        # (e.g., source_url, chunk_id, doc_id, chunk_index)
        # Return the payload itself only if it looks metadata-ish
        return payload

    @staticmethod
    def _get_source_url(md: Dict[str, Any]) -> str:
        for k in ("source_url", "source", "url"):
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    @staticmethod
    def _get_chunk_id(md: Dict[str, Any]) -> str:
        for k in ("chunk_id", "custom_id", "uuid", "id", "doc_id"):
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, (int, float)):
                return str(v)
        return ""

    @staticmethod
    def _get_score(payload: Dict[str, Any]) -> Optional[float]:
        v = payload.get("_pgvector_score")
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    @staticmethod
    def _clean_text(s: str) -> str:
        # Collapse whitespace + remove empty lines
        lines = [ln.strip() for ln in s.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)

    @staticmethod
    def _strip_by_markers(text: str, markers: List[str]) -> str:
        """
        Best-effort: if boilerplate marker appears, cut content before it.
        This is intentionally aggressive to stop cookie banners dominating chunks.
        """
        low = text.lower()
        cut = None
        for m in markers:
            m = (m or "").strip().lower()
            if not m:
                continue
            idx = low.find(m)
            if idx != -1:
                cut = idx if cut is None else min(cut, idx)
        if cut is not None:
            return text[:cut].strip()
        return text

    @staticmethod
    def _dedupe_key(text: str) -> str:
        # normalize for near-duplicate detection
        t = text.lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[^a-z0-9 ]+", "", t)
        return t[:500]  # enough for duplicates

    # ----------------------------
    # Main output
    # ----------------------------
    def parse_combined_text(self) -> Message:
        mode = self.mode or "Parser"

        if mode == "Stringify":
            return self.convert_to_string()

        if mode == "RAG Context":
            return self.build_rag_context()

        # Parser mode
        df, data, data_list = self._clean_args()
        lines: List[str] = []

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
        clean = bool(getattr(self, "clean_data", True))
        sep = self.sep or "\n"

        df, data, data_list = self._clean_args()

        if data_list is not None:
            parts = [self._extract_text_from_data(item) for item in data_list]
            result = sep.join(parts)
        elif data is not None:
            result = self._extract_text_from_data(data)
        elif df is not None:
            # stringify DF
            try:
                result = df.to_string(index=False)
            except Exception:
                result = safe_convert(df)
        else:
            result = safe_convert(self.input_data)

        if clean:
            result = self._clean_text(result)

        msg = Message(text=result)
        self.status = msg
        return msg

    def build_rag_context(self) -> Message:
        clean = bool(getattr(self, "clean_data", True))
        max_chunks = int(getattr(self, "max_chunks", 5) or 5)
        max_chars = int(getattr(self, "max_chars_per_chunk", 1200) or 1200)

        include_source_url = bool(getattr(self, "include_source_url", True))
        include_chunk_id = bool(getattr(self, "include_chunk_id", True))
        include_score = bool(getattr(self, "include_score", True))

        dedupe = bool(getattr(self, "dedupe_chunks", True))
        strip_boilerplate = bool(getattr(self, "strip_boilerplate", True))

        markers_raw = getattr(self, "boilerplate_markers", "") or ""
        markers = [m.strip() for m in markers_raw.splitlines() if m.strip()]

        sep_block = "\n\n---\n\n"

        df, data, data_list = self._clean_args()

        # Normalize to list[Data]
        items: List[Data] = []
        if data_list is not None:
            items = data_list
        elif data is not None:
            items = [data]
        elif df is not None:
            # Convert DF rows to "pseudo chunks"
            # (rare for RAG, but keep it safe)
            items = []
            for _, row in df.iterrows():
                items.append(Data(data=row.to_dict()))
        else:
            msg = Message(text="")
            self.status = msg
            return msg

        seen = set()
        blocks: List[str] = []

        for i, item in enumerate(items[:max_chunks]):
            payload = self._payload(item)
            md = self._extract_metadata(item)

            text = self._extract_text_from_data(item)
            if strip_boilerplate and text:
                text = self._strip_by_markers(text, markers)
            if clean and text:
                text = self._clean_text(text)

            if not text.strip():
                continue

            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "…"

            if dedupe:
                key = self._dedupe_key(text)
                if key in seen:
                    continue
                seen.add(key)

            src = self._get_source_url(md)
            cid = self._get_chunk_id(md)
            score = self._get_score(payload)

            header_parts = [f"[{len(blocks)+1}]"]
            if include_source_url and src:
                header_parts.append(f"source_url={src}")
            if include_chunk_id and cid:
                header_parts.append(f"chunk_id={cid}")
            if include_score and score is not None:
                header_parts.append(f"score={score:.6f}")

            header = " ".join(header_parts)

            block = f"{header}\n{text}"
            blocks.append(block)

        combined = sep_block.join(blocks)
        msg = Message(text=combined)
        self.status = msg
        return msg
