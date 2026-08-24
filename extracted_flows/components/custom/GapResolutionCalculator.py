# Recovered Langflow component
# type: GapResolutionCalculator
# class: GapResolutionCalculator
# used in 1 flow(s): Cognitive RAG V1.1.5 Backup
# json path: node.data.node.template.code.value

from __future__ import annotations

import json
from typing import Any, Dict, List

from langflow.custom import Component
from langflow.io import DataInput, MessageTextInput, Output
from langflow.schema.data import Data


class GapResolutionCalculator(Component):
    display_name = "Gap Resolution Calculator"
    description = (
        "Deterministically resolves USER-missing slots with HISTORY fills, merges clarification tracking, "
        "and outputs the final analyzer-ready gap payload."
    )
    icon = "calculator"
    name = "GapResolutionCalculator"

    inputs = [
        DataInput(
            name="slot_gap_data",
            display_name="Slot Gap Data",
            required=True,
        ),
        DataInput(
            name="clarification_data",
            display_name="Clarification Tracking Data",
            required=True,
        ),
        MessageTextInput(
            name="expected_to_required_map_json",
            display_name="Expected→Required Slot Map JSON",
            value=json.dumps(
                {
                    "contact_information": {"general": [], "critical": []},
                    "university_overview": {"general": [], "critical": []},
                    "program_info": {"general": ["program"], "critical": ["program"]},
                    "admission_requirements": {"general": ["program", "degree_level"], "critical": ["program", "degree_level"]},
                    "eligibility": {"general": ["program", "degree_level"], "critical": ["program", "degree_level"]},
                    "required_documents": {"general": ["program", "degree_level"], "critical": ["program", "degree_level"]},
                    "application_process": {"general": ["program", "degree_level"], "critical": ["program", "degree_level"]},
                    "deadline": {"general": ["program", "degree_level", "intake_year_or_term"], "critical": ["intake_year_or_term"]},
                    "fees": {"general": ["program", "degree_level", "fee_type"], "critical": ["program", "fee_type"]},
                    "tuition_fees": {"general": ["program", "degree_level", "fee_type"], "critical": ["program", "fee_type"]},
                    "scholarship": {"general": ["scholarship_type", "degree_level"], "critical": ["scholarship_type"]},
                    "scholarship_opportunities": {"general": ["scholarship_type", "degree_level"], "critical": ["scholarship_type"]},
                    "dorm_overview": {"general": [], "critical": []},
                    "dorm_fees": {"general": ["accommodation_type", "fee_type"], "critical": ["accommodation_type"]},
                    "location": {"general": [], "critical": []},
                    "hours": {"general": [], "critical": []},
                    "other": {"general": [], "critical": []},
                },
                ensure_ascii=False,
                indent=2,
            ),
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Final Gap Data",
            name="final_gap_data",
            method="build_output",
        ),
    ]

    @staticmethod
    def _safe_list(x: Any) -> List[str]:
        if not isinstance(x, list):
            return []
        out: List[str] = []
        for it in x:
            s = str(it).strip()
            if s:
                out.append(s)
        return out

    @staticmethod
    def _safe_slot_dict(x: Any, source_expected: str | None = None) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        if not isinstance(x, dict):
            return out
        for k, v in x.items():
            kk = str(k).strip()
            if not isinstance(v, dict):
                continue
            val = str(v.get("value", "")).strip()
            src = str(v.get("source", "")).strip().lower()
            if not val or src not in {"user", "history"}:
                continue
            if source_expected is not None and src != source_expected:
                continue
            out[kk] = {"value": val, "source": src}
        return out

    @staticmethod
    def _safe_clarification_tracking(x: Any) -> Dict[str, bool]:
        if not isinstance(x, dict):
            return {
                "clarification_asked_in_history": False,
                "user_is_answering_clarification": False,
            }
        return {
            "clarification_asked_in_history": bool(x.get("clarification_asked_in_history", False)),
            "user_is_answering_clarification": bool(x.get("user_is_answering_clarification", False)),
        }

    @staticmethod
    def _union_required_slots(
        expected_general: List[str],
        expected_critical: List[str],
        expected_to_required_map: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        required_general: List[str] = []
        required_critical: List[str] = []

        for slot in expected_general + expected_critical:
            cfg = expected_to_required_map.get(slot, {})
            if isinstance(cfg, dict):
                required_general.extend(
                    [str(x).strip() for x in cfg.get("general", []) if str(x).strip()]
                )
                required_critical.extend(
                    [str(x).strip() for x in cfg.get("critical", []) if str(x).strip()]
                )

        return {
            "required_general": list(dict.fromkeys(required_general)),
            "required_critical": list(dict.fromkeys(required_critical)),
        }

    @staticmethod
    def _subtract_filled(raw_missing: List[str], history_fills: Dict[str, Dict[str, str]]) -> List[str]:
        filled = set(history_fills.keys())
        return [s for s in raw_missing if s not in filled]

    def build_output(self) -> Data:
        slot_gap_payload = self.slot_gap_data.data if hasattr(self.slot_gap_data, "data") else {}
        clarification_payload = self.clarification_data.data if hasattr(self.clarification_data, "data") else {}

        if not isinstance(slot_gap_payload, dict):
            slot_gap_payload = {}
        if not isinstance(clarification_payload, dict):
            clarification_payload = {}

        try:
            expected_to_required_map = json.loads(self.expected_to_required_map_json or "{}")
            if not isinstance(expected_to_required_map, dict):
                expected_to_required_map = {}
        except Exception:
            expected_to_required_map = {}

        expected_general = self._safe_list(slot_gap_payload.get("expected_general_slots", []))
        expected_critical = self._safe_list(slot_gap_payload.get("expected_critical_slots", []))

        raw_missing_general = self._safe_list(slot_gap_payload.get("missing_general", []))
        raw_missing_critical = self._safe_list(slot_gap_payload.get("missing_critical", []))

        user_resolved_slots = self._safe_slot_dict(slot_gap_payload.get("user_resolved_slots", {}), "user")
        history_resolved_slots = self._safe_slot_dict(slot_gap_payload.get("history_resolved_slots", {}), "history")

        refinement = slot_gap_payload.get("refinement", {})
        if not isinstance(refinement, dict):
            refinement = {"enabled": False, "used": False, "refined_query": "", "used_slots": []}

        clarification_tracking = self._safe_clarification_tracking(clarification_payload)

        required = self._union_required_slots(expected_general, expected_critical, expected_to_required_map)
        allowed = set(required["required_general"] + required["required_critical"])

        user_resolved_slots = {k: v for k, v in user_resolved_slots.items() if k in allowed}
        history_resolved_slots = {k: v for k, v in history_resolved_slots.items() if k in allowed}

        # only let history fill slots that were actually missing from USER
        history_resolved_slots = {
            k: v for k, v in history_resolved_slots.items()
            if k in set(raw_missing_general + raw_missing_critical)
        }

        final_missing_general = self._subtract_filled(raw_missing_general, history_resolved_slots)
        final_missing_critical = self._subtract_filled(raw_missing_critical, history_resolved_slots)

        resolved_slots = {}
        resolved_slots.update(user_resolved_slots)
        resolved_slots.update(history_resolved_slots)

        result = {
            "expected_general_slots": expected_general,
            "expected_critical_slots": expected_critical,
            "missing_general": final_missing_general,
            "missing_critical": final_missing_critical,
            "raw_missing_general": raw_missing_general,
            "raw_missing_critical": raw_missing_critical,
            "resolved_slots": resolved_slots,
            "user_resolved_slots": user_resolved_slots,
            "history_resolved_slots": history_resolved_slots,
            "refinement": refinement,
            "clarification_tracking": clarification_tracking,
        }

        self.status = (
            f"expG={len(expected_general)} expC={len(expected_critical)} "
            f"rawMissG={len(raw_missing_general)} rawMissC={len(raw_missing_critical)} "
            f"histFill={len(history_resolved_slots)} "
            f"finalMissG={len(final_missing_general)} finalMissC={len(final_missing_critical)}"
        )

        return Data(text_key="final_gap", data=result, default_value="")