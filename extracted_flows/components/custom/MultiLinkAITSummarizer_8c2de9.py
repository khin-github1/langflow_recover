# Recovered Langflow component
# type: MultiLinkAITSummarizer
# class: MultiLinkAITSummarizer
# used in 1 flow(s): Method2- Summary
# json path: node.data.node.template.code.value

import hashlib
import json
import re
from typing import List

import httpx
from langchain_text_splitters import CharacterTextSplitter

from langflow.custom import Component
from langflow.io import (
    DropdownInput,
    FloatInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    Output,
)
from langflow.schema import Data


# Auditor prompt. {source} and {content} are filled per table; the doubled
# braces {{ }} are the literal JSON braces the model must emit.
AUDITOR_PROMPT = """Role: You are a Senior Academic Auditor reviewing official university tuition, fee, and admissions pages. Your goal is to transform raw scraped text into a high-fidelity intelligence report.
Instructions:
Deep Scan: Analyze the provided text from {source}. Look specifically for numerical data, deadlines, eligibility criteria, and contact protocols.
EXTRACTIVE Requirements: Do not summarize this section. List every single factual requirement and deadline found in the text as full, standalone sentences. If a table of fees or room types exists, extract the key categories.
Paragraph 1: High-level purpose and mission of the page.
Paragraph 2: Core procedural details (How it works).
Paragraph 3: Important caveats, warnings, or "next steps" for the student.
Constraint: If the source text is long, do not skip the end of the document.
Output Protocol:
Respond ONLY in valid JSON format.
Keys: "extractive" (string),
Target Length: Minimum 300 words total for a complete report.
JSON STRUCTURE: {{ "extractive": "Full factual list here..."}}

TEXT:
{content}
"""


class MultiLinkAITSummarizer(Component):
    display_name = "V1Extractive Table Summarizer (M2)"
    description = "Summarizes each detected table with the Auditor prompt (tagged with gold_table_id); carries page prose through as raw text chunks."
    icon = "Ollama"

    inputs = [
        MessageTextInput(
            name="base_url",
            display_name="Ollama API URL",
            value="https://ollama.aitgpt.dev.brain.cs.ait.ac.th",
        ),
        DropdownInput(name="model_name", display_name="Model Name", options=[], refresh_button=True),
        HandleInput(
            name="input_data",
            display_name="Scraped Data (List)",
            info="Connect the Markdown Web Scraper output here.",
            input_types=["Data"],
            is_list=True,
        ),
        MessageTextInput(
            name="system_message",
            display_name="System Message",
            value="Follow the user's instructions exactly. Base every statement only on the provided "
            "text and never invent numbers, dates, or facts. Respond ONLY with valid JSON of the "
            "form {\"extractive\": \"...\"}.",
        ),
        FloatInput(name="temperature", display_name="Temperature", value=0.1),
        IntInput(name="chunk_size", display_name="Prose Chunk Size", value=1000),
        IntInput(name="chunk_overlap", display_name="Prose Chunk Overlap", value=200),
    ]

    outputs = [
        Output(display_name="Summary List", name="summary_list", method="process_summaries"),
    ]

    async def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None):
        if field_name == "model_name" or not build_config["model_name"]["options"]:
            try:
                base_url = build_config["base_url"]["value"].rstrip("/")
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{base_url}/api/tags")
                    if response.status_code == 200:
                        build_config["model_name"]["options"] = [m["name"] for m in response.json().get("models", [])]
            except Exception:
                build_config["model_name"]["options"] = ["qwen2.5:7b", "llama3"]
        return build_config

    # ---------------------------------------------------------- table parsing
    @staticmethod
    def _is_sep(line: str) -> bool:
        s = line.strip()
        return "-" in s and set(s.replace(" ", "")) <= set("|-:") and "|" in s

    @staticmethod
    def _is_hr(line: str) -> bool:
        s = line.strip()
        return len(s) >= 3 and set(s) <= set("-")

    @classmethod
    def _read_row(cls, lines, k):
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
        return re.sub(r"\s+", " ", " ".join(p.strip() for p in parts)).strip(), k

    @classmethod
    def _parse_table(cls, lines, i):
        header, k = cls._read_row(lines, i)
        j = k
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j >= len(lines) or not cls._is_sep(lines[j]):
            return None
        rows = [header, lines[j].strip()]
        k = j + 1
        while k < len(lines):
            sl = lines[k].strip()
            if sl == "" or sl.startswith("#") or not sl.startswith("|"):
                break
            row, k = cls._read_row(lines, k)
            rows.append(row)
        return "\n".join(rows), k

    def _segments(self, text: str):
        """Yield (kind, content, table_id) in document order.
        table_id is the gold id from the scraper marker (None for prose / heuristic)."""
        if "<!-- TABLE" in text:
            pattern = re.compile(r"<!-- TABLE id=(\S+) source=(.*?) -->\n(.*?)\n<!-- /TABLE -->", re.DOTALL)
            last = 0
            for m in pattern.finditer(text):
                yield ("prose", text[last:m.start()], None)
                yield ("table", m.group(3).strip(), m.group(1))  # m.group(1) == gold_table_id
                last = m.end()
            yield ("prose", text[last:], None)
            return
        # heuristic fallback (no markers -> no gold id available)
        lines = text.split("\n")
        prose: list[str] = []
        i = 0
        while i < len(lines):
            s = lines[i].strip()
            if s.startswith("|") and not self._is_sep(lines[i]):
                res = self._parse_table(lines, i)
                if res:
                    yield ("prose", "\n".join(prose), None)
                    prose = []
                    table_md, k = res
                    yield ("table", table_md, None)
                    i = k
                    continue
            prose.append(lines[i])
            i += 1
        yield ("prose", "\n".join(prose), None)

    # ------------------------------------------------------------ summarize
    @staticmethod
    def _unwrap(text: str) -> str:
        """Pull the prose out of {"extractive": "..."} or code fences."""
        t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
        if t.startswith("{"):
            try:
                obj = json.loads(t)
                if isinstance(obj, dict):
                    if isinstance(obj.get("extractive"), str):
                        return obj["extractive"].strip()
                    strings = [v for v in obj.values() if isinstance(v, str)]
                    if strings:
                        return max(strings, key=len).strip()
            except json.JSONDecodeError:
                pass
        return t

    async def _summarize_text(self, client, content: str, source: str) -> str:
        prompt = AUDITOR_PROMPT.format(source=source or "the source page", content=content)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": self.system_message,
            "stream": False,
            "format": "json",  # ask Ollama to constrain output to JSON
            "options": {"temperature": self.temperature, "num_ctx": 8192},
        }
        resp = await client.post(f"{self.base_url.rstrip('/')}/api/generate", json=payload)
        resp.raise_for_status()
        return self._unwrap(resp.json().get("response", "") or "")

    # --------------------------------------------------------------- main
    async def process_summaries(self) -> List[Data]:
        if not self.input_data:
            return []
        pages = self.input_data if isinstance(self.input_data, list) else [self.input_data]
        prose_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator="\n"
        )

        results: list[Data] = []
        tcount = 0
        ccount = 0
        n_summarized = 0
        n_fallback = 0

        async with httpx.AsyncClient(timeout=180.0) as client:
            for page in pages:
                pdata = page.data if hasattr(page, "data") else {}
                text = pdata.get("text", "") or ""
                source = pdata.get("source", "Unknown")
                if not text or "Error:" in text[:20]:
                    continue

                section = ""
                for kind, payload, tid in self._segments(text):
                    if kind == "prose":
                        # prose stays raw, char-split into text chunks
                        for raw in payload.split("\n"):
                            st = raw.strip()
                            if st.startswith("#"):
                                section = st.lstrip("#").strip()
                        body = re.sub(r"<!--.*?-->", "", payload).strip()
                        if not body:
                            continue
                        for piece in prose_splitter.split_text(body):
                            piece = piece.strip()
                            if not piece or self._is_hr(piece):
                                continue
                            results.append(
                                Data(
                                    data={
                                        "text": piece,
                                        "chunk_id": f"c{ccount:04d}",
                                        "gold_table_id": "",
                                        "table_id": "",
                                        "type": "text",
                                        "source": source,
                                        "section": section,
                                        "n_rows": 0,
                                        "content_hash": hashlib.md5(piece.encode("utf-8")).hexdigest()[:10],
                                    }
                                )
                            )
                            ccount += 1
                    else:  # table -> SUMMARIZE with the auditor prompt
                        table_md = payload
                        gold_id = tid if tid else f"t{tcount:04d}"  # prefer scraper's gold id
                        tcount += 1
                        content = f"Section: {section}\n\n{table_md}" if section else table_md
                        try:
                            summary = await self._summarize_text(client, content, source)
                        except Exception as e:
                            summary = ""
                            self.status = f"Table summarize error: {e}"
                        # fall back to the raw table only if the model returned nothing
                        if summary:
                            n_summarized += 1
                        else:
                            n_fallback += 1
                        chunk_text = summary if summary else table_md
                        results.append(
                            Data(
                                data={
                                    "text": f"{section}\n{chunk_text}" if section else chunk_text,
                                    "chunk_id": gold_id,          # universal id pgvector stores
                                    "gold_table_id": gold_id,     # the annotation
                                    "table_id": gold_id,
                                    "type": "table_summary" if summary else "table",
                                    "source": source,
                                    "section": section,
                                    "n_rows": max(table_md.count("\n") - 1, 0),
                                    "content_hash": hashlib.md5(table_md.encode("utf-8")).hexdigest()[:10],
                                }
                            )
                        )

        n_txt = sum(1 for d in results if d.data.get("type") == "text")
        self.status = (
            f"{len(results)} items: {n_summarized} table summaries"
            + (f" ({n_fallback} raw fallbacks — check Ollama)" if n_fallback else "")
            + f" + {n_txt} text chunks."
        )
        return results