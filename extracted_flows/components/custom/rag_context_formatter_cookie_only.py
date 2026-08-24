# Recovered Langflow component
# type: rag_context_formatter_cookie_only
# class: RAGContextFormatter
# used in 19 flow(s): AITGPT Fees, AITGPT General, AITGPT Program, AITGPT V1.0.0, Baseline Normal AIT, Baseline Normal AIT 123, Baseline Normal Squad, Baseline Reason AIT ...
# json path: node.data.node.template.code.value

import re
from typing import Any, Dict, List, Optional, Tuple

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, HandleInput, MessageTextInput, Output
from langflow.schema.data import Data
from langflow.schema.message import Message


class RAGContextFormatter(Component):
    display_name = "RAG Context Formatter (Cookie Strip Only)"
    description = "Formats PGVector list[Data] into readable context. Removes ONLY cookie banner text."
    icon = "braces"
    name = "rag_context_formatter_cookie_only"

    inputs = [
        HandleInput(
            name="results",
            display_name="Results (list[Data])",
            input_types=["Data"],  # PGVector returns list[Data] under Data handle
            required=True,
        ),
        BoolInput(
            name="strip_cookie_banner",
            display_name="Strip Cookie Banner",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="clean_empty_lines",
            display_name="Clean Empty Lines",
            value=True,
            advanced=True,
        ),
        MessageTextInput(
            name="chunk_separator",
            display_name="Chunk Separator",
            value="\n\n---\n\n",
            advanced=True,
        ),
        BoolInput(
            name="include_headers",
            display_name="Include [rank] source_url chunk_id score header",
            value=True,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Context", name="context", method="build_context"),
    ]

    # -----------------------------
    # Extraction helpers
    # -----------------------------
    @staticmethod
    def _payload(item: Data) -> Dict[str, Any]:
        return item.data if isinstance(item.data, dict) else {}

    @staticmethod
    def _metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
        md = payload.get("metadata")
        return md if isinstance(md, dict) else {}

    @staticmethod
    def _get_text(item: Data) -> str:
        payload = item.data if isinstance(item.data, dict) else {}
        # Most common shapes:
        # - payload["text"] (docs_to_data)
        # - payload["page_content"]
        if isinstance(payload.get("text"), str) and payload["text"].strip():
            return payload["text"]
        if isinstance(payload.get("page_content"), str) and payload["page_content"].strip():
            return payload["page_content"]

        # Fallback: if Data has its own text field (depends on LF version)
        t = getattr(item, "text", None)
        if isinstance(t, str) and t.strip():
            return t

        return ""

    @staticmethod
    def _get_score(payload: Dict[str, Any]) -> Optional[float]:
        # You stored it as _pgvector_score in your instrumented node
        s = payload.get("_pgvector_score")
        try:
            return float(s) if s is not None else None
        except Exception:
            return None

    @staticmethod
    def _get_source_url(payload: Dict[str, Any]) -> str:
        md = RAGContextFormatter._metadata(payload)

        for key in ("source_url", "source", "url"):
            v = md.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        for key in ("source_url", "source", "url"):
            v = payload.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        return ""

    @staticmethod
    def _get_chunk_id(payload: Dict[str, Any], rank: int) -> str:
        md = RAGContextFormatter._metadata(payload)

        for key in ("chunk_id", "custom_id", "uuid", "id", "doc_id"):
            v = md.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        for key in ("chunk_id", "custom_id", "uuid", "id", "doc_id"):
            v = payload.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # fallback: stable enough for logging
        src = RAGContextFormatter._get_source_url(payload)
        return f"{src}::r{rank}" if src else f"rank_{rank}"

    # -----------------------------
    # Cookie stripping ONLY
    # -----------------------------
    @staticmethod
    def _strip_cookie_block(text: str) -> str:
        """
        Removes cookie-banner tail text, but does NOT dedupe and does NOT enforce max length.
        Strategy:
          - Find first occurrence of strong cookie markers.
          - Cut from that marker to end (cookie banner is almost always at the end).
        """
        if not text:
            return text

        lower = text.lower()

        # Strong markers that reliably indicate cookie banner start
        markers = [
            "we use cookies",
            "cookie settings",
            "accept all",
            "manage consent",
            "privacy overview",
            "consent to the use",
            "strictly necessary",
            "stored on your browser",
            "third-party cookies",
        ]

        positions = [lower.find(m) for m in markers if lower.find(m) != -1]
        if not positions:
            return text

        cut_at = min(positions)

        # Extra safety: only cut if the tail actually looks cookie-related
        tail = lower[cut_at : min(len(lower), cut_at + 600)]
        cookieish = any(k in tail for k in ("cookie", "consent", "privacy"))
        if not cookieish:
            return text

        return text[:cut_at].rstrip()

    @staticmethod
    def _clean_lines(text: str) -> str:
        # Just removes empty lines and trims right spaces; doesn't shrink content otherwise
        lines = [ln.rstrip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln.strip() != ""]
        return "\n".join(lines)

    def build_context(self) -> Message:
        r = self.results

        # Normalize to list[Data]
        items: List[Data] = []
        if isinstance(r, list):
            items = [x for x in r if isinstance(x, Data)]
        elif isinstance(r, Data):
            items = [r]

        if not items:
            return Message(text="")

        out_parts: List[str] = []
        for i, item in enumerate(items):
            rank = i + 1
            payload = self._payload(item)
            text = self._get_text(item)

            if self.strip_cookie_banner:
                text = self._strip_cookie_block(text)

            if self.clean_empty_lines:
                text = self._clean_lines(text)

            if self.include_headers:
                src = self._get_source_url(payload)
                cid = self._get_chunk_id(payload, rank)
                score = self._get_score(payload)
                score_s = f"{score:.6f}" if isinstance(score, float) else "None"

                header = f"[{rank}] source_url={src} chunk_id={cid} score={score_s}"
                out_parts.append(header)
                out_parts.append(text)
            else:
                out_parts.append(text)

        sep = self.chunk_separator or "\n\n---\n\n"
        final = sep.join(out_parts).strip()
        self.status = final
        return Message(text=final)
