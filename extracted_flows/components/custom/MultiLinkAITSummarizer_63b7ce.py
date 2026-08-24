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


class MultiLinkAITSummarizer(Component):
    display_name = "V2 Extractive Table Summarizer (M2)"
    description = "Summarizes each table extractively and carries the page prose through as text chunks."
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
            value="You summarize tables faithfully in plain prose. Never invent numbers. "
            "Respond with the summary text only — no JSON, no markdown, no key labels.",
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
        """Yield ('prose', str) and ('table', table_md) segments in document order."""
        if "<!-- TABLE" in text:
            pattern = re.compile(r"<!-- TABLE id=(\S+) source=(.*?) -->\n(.*?)\n<!-- /TABLE -->", re.DOTALL)
            last = 0
            for m in pattern.finditer(text):
                yield ("prose", text[last : m.start()])
                yield ("table", m.group(3).strip())
                last = m.end()
            yield ("prose", text[last:])
            return
        # heuristic fallback
        lines = text.split("\n")
        prose: list[str] = []
        i = 0
        while i < len(lines):
            s = lines[i].strip()
            if s.startswith("|") and not self._is_sep(lines[i]):
                res = self._parse_table(lines, i)
                if res:
                    yield ("prose", "\n".join(prose))
                    prose = []
                    table_md, k = res
                    yield ("table", table_md)
                    i = k
                    continue
            prose.append(lines[i])
            i += 1
        yield ("prose", "\n".join(prose))

    # ------------------------------------------------------------ summarize
    @staticmethod
    def _unwrap(text: str) -> str:
        """Models often wrap the answer in JSON like {"extractive": "..."} or code fences.
        Return just the prose."""
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

    async def _summarize_table(self, client, table_md: str, section: str) -> str:
        prompt = (
            f"Section: {section}\n"
            "Write a concise EXTRACTIVE summary of the table below as PLAIN PROSE. "
            "State the important row labels, column headers and key values as sentences. "
            "Do not invent data. Do not output JSON, markdown, code fences, or key labels — "
            "return only the summary sentences.\n\n"
            f"TABLE:\n{table_md}\n\nSummary:"
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": self.system_message,
            "stream": False,
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

        async with httpx.AsyncClient(timeout=180.0) as client:
            for page in pages:
                pdata = page.data if hasattr(page, "data") else {}
                text = pdata.get("text", "") or ""
                source = pdata.get("source", "Unknown")
                if not text or "Error:" in text[:20]:
                    continue

                section = ""
                for kind, payload in self._segments(text):
                    if kind == "prose":
                        # track latest heading for section context
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
                    else:  # table
                        table_md = payload
                        try:
                            summary = await self._summarize_table(client, table_md, section)
                        except Exception as e:
                            summary = ""
                            self.status = f"Table summarize error: {e}"
                        # fallback to the raw table if the model returned nothing, so data is never lost
                        chunk_text = summary if summary else table_md
                        tid = f"t{tcount:04d}"
                        results.append(
                            Data(
                                data={
                                    "text": f"{section}\n{chunk_text}" if section else chunk_text,
                                    "chunk_id": tid,
                                    "table_id": tid,
                                    "type": "table_summary" if summary else "table",
                                    "source": source,
                                    "section": section,
                                    "n_rows": max(table_md.count("\n") - 1, 0),
                                    "content_hash": hashlib.md5(table_md.encode("utf-8")).hexdigest()[:10],
                                }
                            )
                        )
                        tcount += 1

        n_tbl = sum(1 for d in results if d.data.get("type", "").startswith("table"))
        preview = [f"{len(results)} items: {n_tbl} table summaries + {len(results) - n_tbl} text chunks", ""]
        for d in results:
            m = d.data
            preview.append(f"[{m['chunk_id']}] {m['type']} | section: {m.get('section', '')} | {m.get('source', '')}")
            preview.append(m.get("text", ""))
            preview.append("-" * 60)
        self.status = "\n".join(preview)
        return results