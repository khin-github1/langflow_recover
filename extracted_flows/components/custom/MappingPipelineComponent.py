# Recovered Langflow component
# type: MappingPipelineComponent
# class: MappingPipelineComponent
# used in 1 flow(s): New Flow
# json path: node.data.node.template.code.value

import io
import re
from pathlib import Path
import pandas as pd
from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema.message import Message


class MappingPipelineComponent(Component):
    display_name = "Mapping Pipeline (P1+P2)"
    description = "Map responses to A/B/C/D and generate summary counts"
    name = "MappingPipelineComponent"

    inputs = [
        MessageTextInput(
            name="responses_file",
            display_name="Responses File (CSV text or path)",
            required=True,
        ),
        MessageTextInput(
            name="mapping_file",
            display_name="Mapping File (CSV text or path)",
            required=True,
        ),
    ]

    outputs = [
        Output(
            name="result",
            display_name="Result",
            method="build_output",
            type="message",
        ),
    ]

    VECTOR_NAME_MAP = {
        "vector without summary":         "Vector without summary",
        "vector less of pageindex chat":   "Vector less of pageindex",
        "vector less of pageindex":        "Vector less of pageindex",
        "vector less of pageindex local":  "Vector less of pageindex local",
        "vector with summary table":       "Vector with Summary",
        "vector with summary":             "Vector with Summary",
    }

    def _to_text(self, value) -> str:
        if isinstance(value, str):
            return value
        for attr in ("text", "data", "content"):
            v = getattr(value, attr, None)
            if v is not None:
                return str(v)
        s = str(value)
        return "" if s in ("None", "nan", "NaN") else s

    def _recover_collapsed_csv(self, s: str) -> pd.DataFrame:
       
        parts = s.split(" ")
        ncols = parts[0].count(",") + 1
        rows, current, acc = [], [], 0
        for token in parts:
            current.append(token)
            acc += token.count(",") + 1
            if acc == ncols:
                rows.append(" ".join(current))
                current, acc = [], 0
        if len(rows) < 2:
            raise ValueError(
                f"Space-collapsed CSV: could not reconstruct rows "
                f"(got {len(rows)}, need ≥2). Input: {s[:200]}"
            )
      
        try:
            df = pd.read_csv(
                io.StringIO("\n".join(rows)), header=0,
                sep=None, engine="python"
            )
            if not df.empty:
                return df
        except Exception:
            pass
        
        fixed = []
        for row in rows:
            need = ncols - 1 - row.count(",")
            for _ in range(need):
                row = row.replace(" ", ",", 1)
            fixed.append(row)
        return pd.read_csv(io.StringIO("\n".join(fixed)), header=0)

    def _csv_text_to_df(self, raw: str) -> pd.DataFrame:
        s = raw.strip()
        if not s or s in ("None", "nan"):
            raise ValueError("Input is empty.")
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        if r"\n" in s and "\n" not in s:
            s = s.replace(r"\n", "\n")
        lines = [ln for ln in s.split("\n") if ln.strip()]
        if len(lines) >= 2:
            return pd.read_csv(io.StringIO("\n".join(lines)), header=0)
        return self._recover_collapsed_csv(s)

    def read_input_to_df(self, value) -> pd.DataFrame:
        s = self._to_text(value).strip()
        if not s or s in ("None", "nan"):
            raise ValueError("Input is empty — connect a Text Input node with CSV content.")
        if "\n" not in s and len(s) < 300:
            p = Path(s)
            if p.exists():
                ext = p.suffix.lower()
                if ext == ".csv":
                    return pd.read_csv(p, header=0, encoding="utf-8-sig")
                if ext in (".xlsx", ".xls"):
                    return pd.read_excel(p, header=0)
                raise ValueError(f"Unsupported file type: {ext}")
        return self._csv_text_to_df(s)

    def make_unique_columns(self, columns):
        seen: dict = {}
        result = []
        for col in columns:
            name = str(col).strip()
            seen.setdefault(name, 0)
            if seen[name]:
                result.append(f"{name}_{seen[name]}")
            else:
                result.append(name)
            seen[name] += 1
        return result

    def normalize_response_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Responses DataFrame is empty.")
        first_row = [str(v).strip().lower() for v in df.iloc[0].tolist()]
        headers   = [str(c).strip().lower() for c in df.columns.tolist()]
        overlap   = sum(1 for a, b in zip(first_row, headers) if a == b)
        if overlap >= max(1, len(headers) // 2):
            df = df.copy()
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
        df.columns = self.make_unique_columns(df.columns)
        return df

    def find_question_columns(self, df: pd.DataFrame) -> list:
        skip = [r"^timestamp$", r"^score$", r"^email", r"^name$", r"^student.?id$"]
        return [
            c for c in df.columns
            if not any(re.search(p, str(c).strip().lower()) for p in skip)
        ]

    def parse_cell(self, value) -> list:
        if pd.isna(value):
            return []
        raw = re.sub(r"\s+", "", str(value).strip().upper())
        return [p for p in re.split(r",+", raw) if p] if raw else []

    def map_answer_parts(self, parts: list, q_index: int, mapping_dict: dict) -> str:
        mapper = mapping_dict.get(q_index, {})
        mapped = set()
        for p in parts:
            if p in ("A", "B", "C", "D"):
                mapped.add(p)
            elif p.isdigit():
                letter = mapper.get(int(p))
                if letter:
                    mapped.add(letter)
        return ", ".join(sorted(mapped)) if mapped else ""

    def _col_is_all_letters(self, series: pd.Series) -> bool:
        """True if every non-null value in the series is one of A/B/C/D."""
        vals = series.dropna().astype(str).str.strip().str.upper()
        return len(vals) > 0 and vals.isin(["A", "B", "C", "D"]).all()

    def extract_mapping_dict(self, mapping_df: pd.DataFrame):
        
        all_cols  = list(mapping_df.columns)
        data_cols = all_cols[1:]   # skip the Question label column

        letter_cols = [c for c in data_cols if self._col_is_all_letters(mapping_df[c])]
        text_cols   = [c for c in data_cols if not self._col_is_all_letters(mapping_df[c])]

        if len(text_cols) == 0 and len(letter_cols) >= 2:
            vector_col_names = letter_cols
            canonical = [
                self.VECTOR_NAME_MAP.get(str(c).strip().lower(), str(c).strip())
                for c in vector_col_names
            ]
            mapping_dict: dict = {}
            qvl: dict = {}
            for idx, row in mapping_df.iterrows():
                q_idx = idx + 1
                letters = [str(row[c]).strip().upper() for c in vector_col_names]
                valid = [l for l in letters if l in ("A", "B", "C", "D")]
                if len(valid) == len(vector_col_names):
                    mapping_dict[q_idx] = {i + 1: valid[i] for i in range(len(valid))}
                    qvl[q_idx] = {canonical[i]: valid[i] for i in range(len(valid))}
                else:
                    print(f"[WARN] Q{q_idx}: found {len(valid)}/{len(vector_col_names)} letters, skipping.")
            if not mapping_dict:
                raise ValueError("No valid mapping rows found (Layout B).")
            return mapping_dict, canonical, qvl

        if len(text_cols) > 0 and len(letter_cols) > 0 and len(text_cols) == len(letter_cols):
            # Pair text cols (way names) with letter cols (A/B/C/D) in column order
            text_q   = [c for c in data_cols if c in text_cols]
            letter_q = [c for c in data_cols if c in letter_cols]
            canonical = [
                self.VECTOR_NAME_MAP.get(str(c).strip().lower(), str(c).strip())
                for c in text_q
            ]
            mapping_dict = {}
            qvl = {}
            for idx, row in mapping_df.iterrows():
                q_idx = idx + 1
                letters = [str(row[lc]).strip().upper() for lc in letter_q]
                valid = [l for l in letters if l in ("A", "B", "C", "D")]
                if len(valid) == len(letter_q):
                    mapping_dict[q_idx] = {i + 1: valid[i] for i in range(len(valid))}
                    qvl[q_idx] = {canonical[i]: valid[i] for i in range(len(valid))}
                else:
                    print(f"[WARN] Q{q_idx}: found {len(valid)}/{len(letter_q)} letters, skipping.")
            if not mapping_dict:
                raise ValueError("No valid mapping rows found (Layout A).")
            return mapping_dict, canonical, qvl

        raise ValueError(
            f"Cannot determine mapping layout.\n"
            f"  Columns: {all_cols}\n"
            f"  Letter-only cols ({len(letter_cols)}): {letter_cols}\n"
            f"  Text cols ({len(text_cols)}): {text_cols}\n"
            f"  Expected either ALL letter-only (Layout B) or equal counts (Layout A)."
        )
    def build_output(self) -> Message:
        # Load
        try:
            responses = self.read_input_to_df(self.responses_file)
        except Exception as e:
            return Message(text=f"[ERROR] Responses input: {e}")
        try:
            mapping = self.read_input_to_df(self.mapping_file)
        except Exception as e:
            return Message(text=f"[ERROR] Mapping input: {e}")

        responses = responses.dropna(how="all").reset_index(drop=True)

        try:
            responses = self.normalize_response_headers(responses)
            question_cols = self.find_question_columns(responses)
            mapping_dict, vector_names, question_vector_letter = self.extract_mapping_dict(mapping)
        except Exception as e:
            return Message(text=f"[ERROR] Parsing structure: {e}")

        if not question_cols:
            return Message(text="[ERROR] No question columns found in responses.")

        mapped_series: dict = {}
        changed_cells = 0
        for i, col in enumerate(question_cols, start=1):
            original = responses[col].fillna("").astype(str).str.strip()
            mapped = responses[col].apply(
                lambda v, qi=i: self.map_answer_parts(self.parse_cell(v), qi, mapping_dict)
            )
            changed_cells += int((original != mapped.astype(str).str.strip()).sum())
            mapped_series[i] = mapped.reset_index(drop=True)

        base_cols = [c for c in ("Timestamp", "Score") if c in responses.columns]
        mapped_output_df = responses[base_cols].copy().reset_index(drop=True)
        for i in range(1, len(question_cols) + 1):
            mapped_output_df[f"Q{i}"] = mapped_series[i]

        summary: dict = {}
        for i in range(1, len(question_cols) + 1):
            counts = {n: 0 for n in vector_names}
            vlm = question_vector_letter.get(i, {})
            for val in mapped_output_df[f"Q{i}"]:
                chosen = {p for p in self.parse_cell(val) if p in ("A", "B", "C", "D")}
                for vname, letter in vlm.items():
                    if letter in chosen:
                        counts[vname] += 1
            summary[f"Q{i}"] = counts

        summary_df = (
            pd.DataFrame.from_dict(summary, orient="index")
            [vector_names]  
            .fillna(0)
            .astype(int)
        )
        summary_df.index.name = "Question"

        text = (
            f"Mapped cells changed: {changed_cells}\n"
            f"Total rows: {len(mapped_output_df)}\n"
            f"Total questions: {len(question_cols)}\n\n"
            f"=== mapped_output.csv ===\n{mapped_output_df.to_csv(index=False)}\n"
            f"=== summary_counts.csv ===\n{summary_df.to_csv()}"
        )
        return Message(text=text)