# Recovered Langflow component
# type: GeneratorInputExtractor
# class: GeneratorInputExtractorComponent
# used in 23 flow(s): AIT Cognitive RAG, Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup, Cognitive RAG V1.1.0, Cognitive RAG V1.1.0 Clean, Cognitive RAG V1.1.0 Eval, Cognitive RAG V1.1.5 ...
# json path: node.data.node.template.code.value

import ast
import json
from typing import Any, Dict, List

from lfx.custom import Component
from lfx.io import MessageInput, Output
from lfx.schema import Message


class GeneratorInputExtractorComponent(Component):
    display_name = "Generator Input Extractor"
    description = "Extracts user_query and rag_results and formats them as plain text for the generation LLM."
    icon = "filter"
    name = "GeneratorInputExtractor"

    inputs = [
        MessageInput(
            name="controller_message",
            display_name="Controller Message",
            info="Controller output containing user_query and rag_results.",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Generator Prompt Message",
            name="generator_prompt_message",
            method="build_output",
        ),
    ]

    def _extract_message_text(self, value: Any) -> str:
        if hasattr(value, "text") and value.text is not None:
            return str(value.text).strip()
        if hasattr(value, "content") and value.content is not None:
            return str(value.content).strip()
        if hasattr(value, "data") and isinstance(value.data, dict) and "text" in value.data:
            return str(value.data["text"]).strip()
        return str(value).strip()

    def _parse_payload(self, raw_text: str) -> Dict[str, Any]:
        if not raw_text:
            return {}

        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return {}

    def _format_context(self, rag_results: List[Dict[str, Any]]) -> str:
        if not isinstance(rag_results, list) or not rag_results:
            return "Context:\n[No retrieved evidence available]"

        lines = ["Context:"]
        for i, item in enumerate(rag_results, start=1):
            if not isinstance(item, dict):
                continue

            page_content = str(item.get("page_content", "")).strip()
            metadata = item.get("metadata", {}) or {}
            source = str(metadata.get("source", "")).strip()

            lines.append(f"[{i}] {page_content}")
            if source:
                lines.append(f"Source: {source}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def build_output(self) -> Message:
        controller_raw = self._extract_message_text(self.controller_message)
        payload = self._parse_payload(controller_raw)

        if not payload:
            error_text = "User Query: \nContext:\n[Could not parse controller_message]"
            self.status = error_text
            return Message(text=error_text, data={"text": error_text})

        user_query = str(payload.get("user_query", "")).strip()
        rag_results = payload.get("rag_results", [])

        formatted_text = f"User Query: {user_query}\n\n{self._format_context(rag_results)}"

        self.status = formatted_text
        return Message(
            text=formatted_text,
            data={
                "user_query": user_query,
                "rag_results": rag_results,
            },
        )