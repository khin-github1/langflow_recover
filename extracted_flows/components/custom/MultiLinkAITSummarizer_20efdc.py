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
    BoolInput,
    DropdownInput,
    FloatInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    Output,
)
from langflow.schema import Data


# Value-preserving EXTRACTIVE prompt. One sentence per row, copy every value.
# {{ }} are the literal JSON braces the model must emit.
EXTRACTIVE_PROMPT = """You convert ONE tuition/fee table into COMPLETE factual sentences. You are NOT summarizing — you must keep every value.

Source: {source}
Section: {section}
{group_line}Column headers: {headers}

Table rows (markdown):
{rows_md}

Write EXACTLY ONE sentence per data row, in the same order. Each sentence MUST:
- begin with the row label (the first column){group_clause}
- then state every other column as "<column header>: <value>"
- copy every number, comma, currency symbol, dash, and unit EXACTLY as written — never round, translate, merge, or skip a value
- never invent a row or a number that is not in the table above

Respond ONLY with JSON: {{"extractive": "<sentence 1>\\n<sentence 2>\\n..."}}
No commentary, no headings, no markdown.{retry_note}
"""


class MultiLinkAITSummarizer(Component):
    display_name = "V3 Extractive Table Summarizer (M2)"
    description = "Summarizes each table into COMPLETE value-preserving sentences (group-aware, coverage-checked); carries prose through as text chunks."
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
            value="You are a data extractor. Copy every number and value from the table exactly. "
            "Never omit a row and never invent a value. Respond ONLY with valid JSON of the form "
            "{\"extractive\": \"...\"}.",
        ),
        FloatInput(name="temperature", display_name="Temperature", value=0.1),
        IntInput(name="chunk_size", display_name="Prose Chunk Size", value=1000),
        IntInput(name="chunk_overlap", display_name="Prose Chunk Overlap", value=200),
        FloatInput(name="coverage_threshold", display_name="Min value coverage", value=0.9, advanced=True,
                   info="If the LLM summary preserves fewer than this fraction of the table's numbers, retry once then fall back."),
        IntInput(name="max_rows_per_call", display_name="Max rows per LLM call", value=20, advanced=True,
                 info="Tall groups are summarized in row-batches of this size so nothing is truncated."),
        BoolInput(name="deterministic_fallback", display_name="Deterministic fallback on low coverage", value=True, advanced=True,
                  info="If the LLM still drops values after a retry, substitute a complete deterministic linearization (flagged in metadata)."),
        BoolInput(name="chunk_per_group", display_name="One chunk per program group", value=False, advanced=True,
                  info="Emit a separate chunk per detected group (e.g. per IIS program) instead of one chunk per table."),
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
                build_config["model_name"]["options"] = ["qwen3:8b", "qwen2.5:7b"]
        return build_config

    # ------------------------------------------------- markdown segmentation
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
        """Yield (kind, content, table_id) in document order (table_id from marker)."""
        if "<!-- TABLE" in text:
            pattern = re.compile(r"<!-- TABLE id=(\S+) source=(.*?) -->\n(.*?)\n<!-- /TABLE -->", re.DOTALL)
            last = 0
            for m in pattern.finditer(text):
                yield ("prose", text[last:m.start()], None)
                yield ("table", m.group(3).strip(), m.group(1))
                last = m.end()
            yield ("prose", text[last:], None)
            return
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

    # ------------------------------------------------------- table structure
    @staticmethod
    def _parse_md_table(md: str):
        lines = [l for l in md.strip().split("\n") if l.strip()]

        def cells(line):
            parts = re.split(r"(?<!\\)\|", line)
            parts = [p.strip().replace("\\|", "|") for p in parts]
            if parts and parts[0] == "":
                parts = parts[1:]
            if parts and parts[-1] == "":
                parts = parts[:-1]
            return parts

        if not lines:
            return [], []
        header = cells(lines[0])
        data = [cells(l) for l in lines[2:]] if len(lines) > 2 else []
        w = max([len(header)] + [len(r) for r in data]) if data else len(header)
        header = header + [""] * (w - len(header))
        data = [r + [""] * (w - len(r)) for r in data]
        return header, data

    @staticmethod
    def _drop_headerless_cols(header, rows):
        boiler = {"", "more information", "more info", "details", "view", "apply", "read more"}
        drop = set()
        for j, h in enumerate(header):
            if h.strip():
                continue
            col = [r[j] for r in rows]
            ne = [c for c in col if c.strip()]
            if not ne or all(c.strip().lower() in boiler for c in ne):
                drop.add(j)
        if not drop:
            return header, rows
        keep = [j for j in range(len(header)) if j not in drop]
        if len(keep) < 2:
            return header, rows
        return [header[j] for j in keep], [[r[j] for j in keep] for r in rows]

    @staticmethod
    def _split_groups(rows):
        """A row with only col0 filled is a group sub-header (e.g. IIS program)."""
        groups = []
        cur_label = None
        cur = []
        for r in rows:
            if r[0].strip() and not any(c.strip() for c in r[1:]):
                if cur:
                    groups.append((cur_label, cur))
                cur_label = r[0].strip()
                cur = []
            else:
                cur.append(r)
        if cur:
            groups.append((cur_label, cur))
        return groups

    @staticmethod
    def _rows_to_md(header, rows):
        lines = ["| " + " | ".join(header) + " |",
                 "| " + " | ".join(["---"] * len(header)) + " |"]
        for r in rows:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    @staticmethod
    def _linearize_group(header, glabel, rows):
        """Complete deterministic linearization (guaranteed value-preserving)."""
        out = []
        for r in rows:
            label = r[0].strip() or "(row)"
            prefix = f"{glabel}, {label}" if glabel else label
            parts = [f"{header[j]}: {r[j].strip()}" for j in range(1, len(header)) if r[j].strip()]
            out.append(f"{prefix} — " + "; ".join(parts) + ".")
        return "\n".join(out)

    _NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")

    @classmethod
    def _numbers(cls, s: str) -> set:
        return {n.rstrip(".") for n in cls._NUM.findall(s or "")}

    @classmethod
    def _coverage(cls, source_md: str, summary: str):
        tn = cls._numbers(source_md)
        if not tn:
            return 1.0, set()
        sn = cls._numbers(summary)
        return len(tn & sn) / len(tn), (tn - sn)

    # ------------------------------------------------------------ summarize
    @staticmethod
    def _unwrap(text: str) -> str:
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

    async def _call(self, client, header, glabel, rows, source, section, retry_note=""):
        rows_md = self._rows_to_md(header, rows)
        group_line = f"Program group: {glabel}\n" if glabel else ""
        group_clause = f', prefixed with the program group "{glabel}"' if glabel else ""
        prompt = EXTRACTIVE_PROMPT.format(
            source=source or "the source page", section=section or "-",
            group_line=group_line, headers=" | ".join(header),
            rows_md=rows_md, group_clause=group_clause, retry_note=retry_note,
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": self.system_message,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_ctx": 8192},
        }
        resp = await client.post(f"{self.base_url.rstrip('/')}/api/generate", json=payload)
        resp.raise_for_status()
        return self._unwrap(resp.json().get("response", "") or "")

    async def _summarize_group(self, client, header, glabel, rows, source, section):
        """LLM extraction with coverage guard + retry + deterministic fallback.
        Returns (text, coverage, method)."""
        max_rows = max(1, int(self.max_rows_per_call))
        batches = [rows[i:i + max_rows] for i in range(0, len(rows), max_rows)] or [rows]
        texts, methods = [], []
        for batch in batches:
            batch_md = self._rows_to_md(header, batch)
            try:
                out = await self._call(client, header, glabel, batch, source, section)
            except Exception:
                out = ""
            cov, missing = self._coverage(batch_md, out)
            method = "llm"
            if out and cov < self.coverage_threshold:
                note = ("\nYou previously omitted these values — include every one of them, "
                        f"each bound to its row: {', '.join(sorted(missing))}.")
                try:
                    out2 = await self._call(client, header, glabel, batch, source, section, retry_note=note)
                    if self._coverage(batch_md, out2)[0] > cov:
                        out, cov = out2, self._coverage(batch_md, out2)[0]
                except Exception:
                    pass
            if (not out or cov < self.coverage_threshold) and self.deterministic_fallback:
                out = self._linearize_group(header, glabel, batch)
                cov, method = 1.0, "deterministic_fallback"
            texts.append(out)
            methods.append(method)
        full = "\n".join(t for t in texts if t)
        overall_cov = self._coverage(self._rows_to_md(header, rows), full)[0]
        method = "deterministic_fallback" if "deterministic_fallback" in methods else "llm"
        return full, overall_cov, method

    # --------------------------------------------------------------- main
    async def process_summaries(self) -> List[Data]:
        if not self.input_data:
            return []
        pages = self.input_data if isinstance(self.input_data, list) else [self.input_data]
        prose_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator="\n"
        )
        results: list[Data] = []
        ccount = 0
        tcount = 0
        last_error = ""

        def mk(text, *, chunk_id, gold_id, ctype, source, section, n_rows=0, coverage=None, method=""):
            data = {
                "text": text,
                "chunk_id": chunk_id,
                "table_id": gold_id,
                "gold_table_id": gold_id,
                "type": ctype,
                "source": source,
                "section": section,
                "n_rows": n_rows,
                "content_hash": hashlib.md5(text.encode("utf-8")).hexdigest()[:10],
            }
            if coverage is not None:
                data["value_coverage"] = round(coverage, 3)
                data["summarizer"] = method
            return Data(data=data)

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
                            results.append(mk(piece, chunk_id=f"c{ccount:04d}", gold_id="",
                                              ctype="text", source=source, section=section))
                            ccount += 1
                        continue

                    # ---- table: extract into complete value-preserving sentences ----
                    gold_id = tid if tid else f"t{tcount:04d}"
                    tcount += 1
                    header, rows = self._parse_md_table(payload)
                    header, rows = self._drop_headerless_cols(header, rows)
                    if not header or not rows:
                        continue
                    groups = self._split_groups(rows)
                    n_data = sum(len(g_rows) for _, g_rows in groups)

                    group_texts = []
                    covs = []
                    methods = []
                    try:
                        for glabel, g_rows in groups:
                            gtext, gcov, gmethod = await self._summarize_group(
                                client, header, glabel, g_rows, source, section
                            )
                            group_texts.append((glabel, gtext))
                            covs.append(gcov)
                            methods.append(gmethod)
                    except Exception as e:
                        last_error = str(e)

                    table_cov = round(sum(covs) / len(covs), 3) if covs else 0.0
                    table_method = "deterministic_fallback" if "deterministic_fallback" in methods else "llm"

                    if self.chunk_per_group and len(groups) > 1:
                        for k, (glabel, gtext) in enumerate(group_texts):
                            head = f"{section} — {glabel}" if section and glabel else (glabel or section)
                            body = f"{head}\n{gtext}" if head else gtext
                            results.append(mk(body, chunk_id=f"{gold_id}-g{k:02d}", gold_id=gold_id,
                                              ctype="table_summary", source=source,
                                              section=glabel or section, n_rows=len(group_texts[k][1].splitlines()),
                                              coverage=covs[k], method=methods[k]))
                    else:
                        summary = "\n".join(t for _, t in group_texts if t)
                        body = f"{section}\n{summary}" if section else summary
                        results.append(mk(body, chunk_id=gold_id, gold_id=gold_id,
                                          ctype="table_summary", source=source, section=section,
                                          n_rows=n_data, coverage=table_cov, method=table_method))

        # Show the ROWS in the panel (grid), not a count sentence.
        if results:
            self.status = results
        elif last_error:
            self.status = f"Table summarize error: {last_error}"
        return results