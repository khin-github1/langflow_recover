# Recovered Langflow component
# type: entity_exact_match_evaluator
# class: EntityExactMatchEvaluator
# used in 2 flow(s): CRCV Evaluation Runner 1.0.0, CRCV Evaluation Runner 1.0.0 (backup)
# json path: node.data.node.template.code.value

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Set

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, FloatInput, HandleInput, MessageTextInput, Output
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame

MONTHS = (
    "jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|"
    "sep|sept|september|oct|october|nov|november|dec|december"
)

# Basic date patterns
RE_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
RE_SLASH_DATE = re.compile(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b")
RE_TEXT_MONTH_1 = re.compile(rf"\b({MONTHS})\s+(\d{{1,2}})(?:,)?\s+(\d{{4}})\b", re.IGNORECASE)
RE_TEXT_MONTH_2 = re.compile(rf"\b(\d{{1,2}})\s+({MONTHS})\s+(\d{{4}})\b", re.IGNORECASE)

# NEW: Year-only "dates" (e.g., 1959). Range chosen to avoid tons of false positives.
RE_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2})\b")

RE_NUMBER = re.compile(r"\b-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b-?\d+(?:\.\d+)?\b")
RE_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
RE_LONG_ID = re.compile(r"\b\d{6,}\b")  # long numeric ids
RE_CODE = re.compile(r"\b[A-Z]{2,}[A-Z0-9\-]{2,}\b")  # e.g., ABC-123, CS2026, etc.

# Heuristic: Capitalized phrase (2+ words) or TitleCase single token
RE_CAP_PHRASE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
RE_TITLE_SINGLE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _coerce_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, Data):
        v = v.data
    if isinstance(v, dict):
        for k in ("text", "content", "message", "value"):
            if k in v and isinstance(v[k], str):
                return v[k]
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _lower_keymap(keys: Sequence[str]) -> Dict[str, str]:
    return {k.lower(): k for k in keys}


def _resolve_col(requested: str, keys: Sequence[str], fallbacks: Sequence[str]) -> Optional[str]:
    requested = (requested or "").strip()
    km = _lower_keymap(keys)
    if requested and requested in keys:
        return requested
    if requested and requested.lower() in km:
        return km[requested.lower()]
    for f in fallbacks:
        f = (f or "").strip()
        if not f:
            continue
        if f in keys:
            return f
        if f.lower() in km:
            return km[f.lower()]
    return None


def _try_parse_structured(text: str) -> Optional[dict]:
    s = (text or "").strip()
    if not s:
        return None
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _norm_date_yyyy_mm_dd(y: int, m: int, d: int) -> Optional[str]:
    if y < 100:
        y = 2000 + y
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def extract_dates(text: str, *, dayfirst: bool = False) -> Set[str]:
    """
    Return a set of normalized dates:
      - full dates normalized as YYYY-MM-DD
      - year-only dates stored as 'YYYY'
    """
    out: Set[str] = set()
    s = text or ""

    for m in RE_ISO_DATE.finditer(s):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        nd = _norm_date_yyyy_mm_dd(y, mo, d)
        if nd:
            out.add(nd)

    for m in RE_SLASH_DATE.finditer(s):
        a, b, c = m.group(1), m.group(2), m.group(3)
        x, y, z = int(a), int(b), int(c)
        if len(c) == 2:
            z = 2000 + z
        if dayfirst:
            dd, mm, yy = x, y, z
        else:
            mm, dd, yy = x, y, z
        nd = _norm_date_yyyy_mm_dd(yy, mm, dd)
        if nd:
            out.add(nd)

    for m in RE_TEXT_MONTH_1.finditer(s):
        mon = MONTH_MAP.get(m.group(1).lower(), None)
        d = int(m.group(2))
        y = int(m.group(3))
        if mon:
            nd = _norm_date_yyyy_mm_dd(y, mon, d)
            if nd:
                out.add(nd)

    for m in RE_TEXT_MONTH_2.finditer(s):
        d = int(m.group(1))
        mon = MONTH_MAP.get(m.group(2).lower(), None)
        y = int(m.group(3))
        if mon:
            nd = _norm_date_yyyy_mm_dd(y, mon, d)
            if nd:
                out.add(nd)

    # NEW: year-only extraction, but skip if that year is clearly part of a longer date token (e.g., 2026-02-03)
    for m in RE_YEAR.finditer(s):
        start, end = m.start(1), m.end(1)
        prev_ch = s[start - 1] if start - 1 >= 0 else ""
        next_ch = s[end] if end < len(s) else ""
        if prev_ch in "-/." or next_ch in "-/.":
            continue
        out.add(m.group(1))

    return out


def extract_numbers(text: str) -> List[float]:
    out: List[float] = []
    for m in RE_NUMBER.finditer(text or ""):
        raw = m.group(0).replace(",", "")
        try:
            out.append(float(raw))
        except Exception:
            continue
    return out


def extract_ids(text: str) -> Set[str]:
    s = text or ""
    out: Set[str] = set()
    for m in RE_UUID.finditer(s):
        out.add(m.group(0).lower())
    for m in RE_LONG_ID.finditer(s):
        out.add(m.group(0))
    for m in RE_CODE.finditer(s):
        out.add(m.group(0))
    return out


def extract_locations_heuristic(text: str, *, allowlist: Optional[Set[str]] = None) -> Set[str]:
    s = text or ""
    found: Set[str] = set()

    for m in RE_CAP_PHRASE.finditer(s):
        found.add(m.group(0).strip())

    for m in RE_TITLE_SINGLE.finditer(s):
        found.add(m.group(0).strip())

    if allowlist:
        norm_allow = {a.strip().lower() for a in allowlist if a.strip()}
        found = {x for x in found if x.lower() in norm_allow}

    return found


def set_em(gold_set: Set[str], pred_set: Set[str], *, penalize_extra: bool) -> int:
    if not gold_set and not pred_set:
        return 1
    if not gold_set and pred_set:
        return 0 if penalize_extra else 1
    if gold_set and not pred_set:
        return 0
    if penalize_extra:
        return 1 if pred_set == gold_set else 0
    return 1 if gold_set.issubset(pred_set) else 0


def numbers_em(gold_nums: List[float], pred_nums: List[float], *, tol: float, penalize_extra: bool) -> int:
    if not gold_nums and not pred_nums:
        return 1
    if not gold_nums and pred_nums:
        return 0 if penalize_extra else 1
    if gold_nums and not pred_nums:
        return 0

    used = [False] * len(pred_nums)

    def _match_one(g: float) -> bool:
        for i, p in enumerate(pred_nums):
            if used[i]:
                continue
            if abs(p - g) <= tol:
                used[i] = True
                return True
        return False

    ok = all(_match_one(g) for g in gold_nums)
    if not ok:
        return 0

    if penalize_extra:
        extra = any(not u for u in used)
        return 0 if extra else 1

    return 1


def _year_set_from_dates(dates: Set[str]) -> Set[str]:
    """Extract just 'YYYY' entries from extracted date set."""
    return {x for x in dates if re.fullmatch(r"\d{4}", x or "")}


def _filter_year_numbers(nums: List[float], years: Set[str]) -> List[float]:
    """
    Prevent double-counting: if a year was extracted as a date (e.g., '1959'),
    remove that integer from numeric entity lists.
    """
    if not nums or not years:
        return nums
    out: List[float] = []
    for n in nums:
        if float(int(n)) == n:
            y = f"{int(n):04d}"
            if y in years:
                continue
        out.append(n)
    return out


class EntityExactMatchEvaluator(Component):
    display_name = "Evaluator (Entity EM: dates/numbers/ids/locations)"
    description = (
        "Entity-level exactness instead of full-string EM. "
        "Checks dates, numbers, ids, and heuristic locations between generated_answer and gold_answer."
    )
    icon = "check-circle"
    name = "entity_exact_match_evaluator"

    inputs = [
        HandleInput(
            name="results_df",
            display_name="Runner Results (DataFrame)",
            input_types=["DataFrame", "Data"],
            required=True,
        ),
        MessageTextInput(name="question_col", display_name="Question Column", value="question", advanced=True),
        MessageTextInput(name="generated_col", display_name="Generated Answer Column", value="generated_answer", advanced=True),
        MessageTextInput(name="gold_col", display_name="Gold Answer Column", value="gold_answer", advanced=True),

        BoolInput(name="check_dates", display_name="Check Dates", value=True, advanced=True),
        BoolInput(name="check_numbers", display_name="Check Numbers", value=True, advanced=True),
        BoolInput(name="check_ids", display_name="Check IDs", value=False, advanced=True),
        BoolInput(name="check_locations", display_name="Check Locations (heuristic)", value=False, advanced=True),

        BoolInput(
            name="penalize_extra_entities",
            display_name="Penalize extra entities in prediction",
            value=False,
            advanced=True,
            info="If ON: requires exact set equality. If OFF: only requires gold entities appear in prediction.",
        ),
        BoolInput(
            name="dayfirst_slash_dates",
            display_name="Interpret slash dates as DD/MM/YYYY",
            value=False,
            advanced=True,
        ),
        FloatInput(
            name="number_tolerance",
            display_name="Number tolerance",
            value=0.0,
            advanced=True,
            info="0.0 means exact numeric match; set e.g. 0.01 for rounding tolerance.",
        ),
        MessageTextInput(
            name="location_allowlist",
            display_name="Location allowlist (comma-separated, optional)",
            value="",
            advanced=True,
            info="If provided, only these location strings are considered (improves reliability).",
        ),

        BoolInput(name="write_csv", display_name="Write CSV file", value=False, advanced=True),
        MessageTextInput(name="csv_filename", display_name="CSV filename", value="eval_entity_em.csv", advanced=True),
    ]

    outputs = [
        Output(display_name="Eval DataFrame", name="eval_df", method="evaluate"),
        Output(display_name="CSV File (Data)", name="csv_file", method="get_csv_file"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._csv_path: Optional[str] = None

    def _iter_rows(self) -> List[Dict[str, Any]]:
        t = self.results_df
        if isinstance(t, DataFrame):
            return [row.to_dict() for _, row in t.iterrows()]
        if isinstance(t, Data) and isinstance(t.data, dict):
            return [dict(t.data)]
        return []

    def _write_csv(self, rows: List[Dict[str, Any]]) -> Optional[str]:
        if not rows:
            return None
        filename = (self.csv_filename or "eval_entity_em.csv").strip() or "eval_entity_em.csv"
        out_dir = "/mnt/data" if os.path.isdir("/mnt/data") else None
        if out_dir:
            path = os.path.join(out_dir, filename)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            path = tmp.name
            tmp.close()

        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        return path

    def evaluate(self) -> DataFrame:
        self._csv_path = None
        rows = self._iter_rows()
        if not rows:
            self.status = "No rows received."
            return DataFrame([])

        keys = list(rows[0].keys())
        qcol = _resolve_col(self.question_col, keys, fallbacks=["question", "query", "user_question"]) or "question"
        pcol = _resolve_col(self.generated_col, keys, fallbacks=["generated_answer", "assistant_text", "answer"]) or "generated_answer"
        gcol = _resolve_col(self.gold_col, keys, fallbacks=["gold_answer", "ground_truth", "reference"]) or "gold_answer"

        penalize_extra = bool(self.penalize_extra_entities)
        dayfirst = bool(self.dayfirst_slash_dates)
        tol = float(self.number_tolerance or 0.0)

        allowlist: Optional[Set[str]] = None
        if (self.location_allowlist or "").strip():
            allowlist = {x.strip() for x in self.location_allowlist.split(",") if x.strip()}

        out_rows: List[Dict[str, Any]] = []

        enabled_checks = [
            ("dates", bool(self.check_dates)),
            ("numbers", bool(self.check_numbers)),
            ("ids", bool(self.check_ids)),
            ("locations", bool(self.check_locations)),
        ]
        enabled_count = sum(1 for _, on in enabled_checks if on)

        for r in rows:
            q = _coerce_text(r.get(qcol, ""))
            pred_raw = _coerce_text(r.get(pcol, ""))
            gold_raw = _coerce_text(r.get(gcol, ""))

            # kept for future use
            _ = _try_parse_structured(pred_raw)
            _ = _try_parse_structured(gold_raw)

            pred_text = pred_raw
            gold_text = gold_raw

            # Extract entities (only if the check is enabled)
            gold_dates = extract_dates(gold_text, dayfirst=dayfirst) if bool(self.check_dates) else set()
            pred_dates = extract_dates(pred_text, dayfirst=dayfirst) if bool(self.check_dates) else set()

            gold_nums = extract_numbers(gold_text) if bool(self.check_numbers) else []
            pred_nums = extract_numbers(pred_text) if bool(self.check_numbers) else []

            gold_ids = extract_ids(gold_text) if bool(self.check_ids) else set()
            pred_ids = extract_ids(pred_text) if bool(self.check_ids) else set()

            gold_locs = extract_locations_heuristic(gold_text, allowlist=allowlist) if bool(self.check_locations) else set()
            pred_locs = extract_locations_heuristic(pred_text, allowlist=allowlist) if bool(self.check_locations) else set()

            # NEW: avoid double-counting years as both dates and numbers
            if bool(self.check_dates) and bool(self.check_numbers):
                gold_years = _year_set_from_dates(gold_dates)
                pred_years = _year_set_from_dates(pred_dates)
                gold_nums = _filter_year_numbers(gold_nums, gold_years)
                pred_nums = _filter_year_numbers(pred_nums, pred_years)

            # Applicability (per-row): enabled AND gold contains at least one entity of that type
            dates_applicable = bool(self.check_dates) and bool(gold_dates) 
            numbers_applicable = bool(self.check_numbers) and bool(gold_nums)
            ids_applicable = bool(self.check_ids) and bool(gold_ids)
            locs_applicable = bool(self.check_locations) and bool(gold_locs)

            # Compute EM only if applicable; else blank string
            date_em = set_em(gold_dates, pred_dates, penalize_extra=penalize_extra) if dates_applicable else ""
            num_em = numbers_em(gold_nums, pred_nums, tol=tol, penalize_extra=penalize_extra) if numbers_applicable else ""
            id_em = set_em(gold_ids, pred_ids, penalize_extra=penalize_extra) if ids_applicable else ""
            loc_em = set_em(gold_locs, pred_locs, penalize_extra=penalize_extra) if locs_applicable else ""

            applicable_checks = sum([dates_applicable, numbers_applicable, ids_applicable, locs_applicable])

            em_sum = 0
            if dates_applicable:
                em_sum += int(date_em)
            if numbers_applicable:
                em_sum += int(num_em)
            if ids_applicable:
                em_sum += int(id_em)
            if locs_applicable:
                em_sum += int(loc_em)

            em_rate = (em_sum / applicable_checks) if applicable_checks else None

            out_rows.append(
                {
                    "question": q,
                    "generated_answer": pred_raw,
                    "gold_answer": gold_raw,

                    "date_em": date_em,
                    "number_em": num_em,
                    "id_em": id_em,
                    "location_em": loc_em,

                    "em_applicable_checks": applicable_checks,
                    "em_rate": em_rate,

                    # Backward compatibility (same as em_rate now)
                    "entity_em": em_rate,

                    "gold_dates": json.dumps(sorted(gold_dates), ensure_ascii=False),
                    "pred_dates": json.dumps(sorted(pred_dates), ensure_ascii=False),
                }
            )

        if bool(self.write_csv):
            self._csv_path = self._write_csv(out_rows)

        self.status = (
            f"rows={len(out_rows)} | checks_enabled={enabled_count} | "
            f"csv={self._csv_path or 'none'}"
        )
        return DataFrame([Data(data=x) for x in out_rows])

    def get_csv_file(self) -> Data:
        return Data(data={"file_path": self._csv_path, "file_name": (self.csv_filename or "eval_entity_em.csv")})
