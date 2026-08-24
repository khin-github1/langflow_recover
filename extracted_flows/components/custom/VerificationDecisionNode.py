# Recovered Langflow component
# type: VerificationDecisionNode
# class: VerificationDecisionNode
# used in 33 flow(s): AIT Cognitive RAG, Cognitive RAG V1.0.0 Eval Flow, Cognitive RAG V1.0.0 GPT Version, Cognitive RAG V1.0.0 backup, Cognitive RAG V1.1.0, Cognitive RAG V1.1.0 Clean, Cognitive RAG V1.1.0 Eval, Cognitive RAG V1.1.5 ...
# json path: node.data.node.template.code.value

import json
import re
from typing import Any, Dict, List

from langflow.custom import Component
from langflow.io import FloatInput, MessageTextInput, Output
from langflow.schema.message import Message


class VerificationDecisionNode(Component):
    display_name = "Verification Decision Node"
    description = (
        "Consumes generation_mode and verifier message output, computes weighted final support score, "
        "and outputs accept / retry / abstain as a Message."
    )
    icon = "check-circle"
    name = "VerificationDecisionNode"

    inputs = [
        MessageTextInput(
            name="generation_mode",
            display_name="Generation Mode",
            info='Message text like "normal" / "reason" or JSON like {"generation_mode":"normal"}',
            required=True,
        ),
        MessageTextInput(
            name="verification_message",
            display_name="Verification Message",
            info="Verifier output as a message containing JSON.",
            required=True,
        ),
        FloatInput(
            name="general_weight",
            display_name="General Weight",
            value=1.0,
            info="Weight for general claims.",
        ),
        FloatInput(
            name="critical_weight",
            display_name="Critical Weight",
            value=2.0,
            info="Weight for critical claims.",
        ),
        FloatInput(
            name="final_support_threshold",
            display_name="Final Support Threshold",
            value=0.75,
            info="If final_support_score >= this threshold, decision is accept.",
        ),
    ]

    outputs = [
        Output(
            display_name="Decision",
            name="decision",
            method="build_decision",
        ),
    ]

    def _parse_generation_mode(self, raw: Any) -> str:
        text = str(raw or "").strip()
        if not text:
            return "normal"

        if text.lower() in {"normal", "reason"}:
            return text.lower()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                mode = str(obj.get("generation_mode", "normal")).strip().lower()
                if mode in {"normal", "reason"}:
                    return mode
        except Exception:
            pass

        return "normal"

    def _strip_fences(self, text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    def _parse_json_obj(self, text: str) -> Dict[str, Any]:
        text = self._strip_fences(text)

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        return {}

    def _get_verifier_payload(self) -> Dict[str, Any]:
        raw_text = str(self.verification_message or "").strip()
        obj = self._parse_json_obj(raw_text)

        if not obj:
            return {}

        # Case 1: direct verifier JSON
        if "claims" in obj or "slot_coverage" in obj:
            return obj

        # Case 2: outer envelope with nested verifier_json
        nested = obj.get("verifier_json")
        if isinstance(nested, dict):
            return nested

        return obj

    def build_decision(self) -> Message:
        mode = self._parse_generation_mode(self.generation_mode)
        payload = self._get_verifier_payload()

        claims = payload.get("claims", [])
        if not isinstance(claims, list):
            claims = []

        slot_coverage = payload.get("slot_coverage", {})
        if not isinstance(slot_coverage, dict):
            slot_coverage = {}

        general_slots = slot_coverage.get("general", [])
        critical_slots = slot_coverage.get("critical", [])
        if not isinstance(general_slots, list):
            general_slots = []
        if not isinstance(critical_slots, list):
            critical_slots = []

        general_weight = float(self.general_weight or 1.0)
        critical_weight = float(self.critical_weight or 2.0)
        final_support_threshold = float(self.final_support_threshold or 0.75)

        total_claims = 0
        total_general_claims = 0
        total_critical_claims = 0

        weighted_score_sum = 0.0
        weighted_total = 0.0
        raw_score_sum = 0.0

        claims_per_slot: Dict[str, int] = {}
        claim_rows: List[Dict[str, Any]] = []

        for idx, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                continue

            total_claims += 1

            claim_id = str(claim.get("claim_id", f"c{idx}")).strip()
            claim_text = str(claim.get("claim_text", "")).strip()
            slot_type = str(claim.get("slot_type", "other")).strip().lower()
            slot_importance = str(claim.get("slot_importance", "other")).strip().lower()

            try:
                support_score = float(claim.get("support_score", 0.0))
            except Exception:
                support_score = 0.0

            support_score = max(0.0, min(1.0, support_score))

            if slot_importance == "critical":
                weight = critical_weight
                total_critical_claims += 1
            else:
                weight = general_weight
                total_general_claims += 1

            weighted_score_sum += weight * support_score
            weighted_total += weight
            raw_score_sum += support_score

            claims_per_slot[slot_type] = claims_per_slot.get(slot_type, 0) + 1

            claim_rows.append(
                {
                    "claim_id": claim_id,
                    "claim_text": claim_text,
                    "slot_type": slot_type,
                    "slot_importance": slot_importance,
                    "support_score": support_score,
                    "weight": weight,
                    "weighted_contribution": weight * support_score,
                }
            )

        raw_mean_support_score = (raw_score_sum / total_claims) if total_claims > 0 else 0.0
        final_support_score = (weighted_score_sum / weighted_total) if weighted_total > 0 else 0.0

        # Slot rows are kept for inspection only, not for decision
        general_slots_total = 0
        general_slots_answered = 0
        general_slots_missing = 0
        general_slot_rows: List[Dict[str, Any]] = []

        for row in general_slots:
            if not isinstance(row, dict):
                continue

            slot_name = str(row.get("slot_name", "")).strip().lower()
            status = str(row.get("status", "missing")).strip().lower()

            try:
                support_score = float(row.get("support_score", 0.0))
            except Exception:
                support_score = 0.0

            support_score = max(0.0, min(1.0, support_score))
            general_slots_total += 1

            answered = status == "answered"
            if answered:
                general_slots_answered += 1
            elif status != "not_applicable":
                general_slots_missing += 1

            general_slot_rows.append(
                {
                    "slot_name": slot_name,
                    "status": status,
                    "support_score": support_score,
                    "answered": answered,
                }
            )

        critical_slots_total = 0
        critical_slots_answered = 0
        critical_slots_missing = 0
        critical_slot_rows: List[Dict[str, Any]] = []

        for row in critical_slots:
            if not isinstance(row, dict):
                continue

            slot_name = str(row.get("slot_name", "")).strip().lower()
            status = str(row.get("status", "missing")).strip().lower()

            try:
                support_score = float(row.get("support_score", 0.0))
            except Exception:
                support_score = 0.0

            support_score = max(0.0, min(1.0, support_score))

            if status != "not_applicable":
                critical_slots_total += 1
                answered = status == "answered"
                if answered:
                    critical_slots_answered += 1
                else:
                    critical_slots_missing += 1
            else:
                answered = False

            critical_slot_rows.append(
                {
                    "slot_name": slot_name,
                    "status": status,
                    "support_score": support_score,
                    "answered": answered,
                }
            )

        # Decision logic:
        # 1) accept if final_support_score >= threshold
        # 2) else retry only if current mode is normal
        # 3) else abstain
        if final_support_score >= final_support_threshold:
            decision = "accept"
        elif mode == "normal":
            decision = "retry"
        else:
            decision = "abstain"

        next_generation_mode = "reason" if decision == "retry" else mode

        result = {
            "decision": decision,
            "generation_mode_in": mode,
            "next_generation_mode": next_generation_mode,

            "general_weight": general_weight,
            "critical_weight": critical_weight,
            "final_support_threshold": final_support_threshold,

            "total_claims": total_claims,
            "total_general_claims": total_general_claims,
            "total_critical_claims": total_critical_claims,

            "raw_mean_support_score": raw_mean_support_score,
            "weighted_score_sum": weighted_score_sum,
            "weighted_total": weighted_total,
            "final_support_score": final_support_score,

            "general_slots_total": general_slots_total,
            "general_slots_answered": general_slots_answered,
            "general_slots_missing": general_slots_missing,

            "critical_slots_total": critical_slots_total,
            "critical_slots_answered": critical_slots_answered,
            "critical_slots_missing": critical_slots_missing,

            "claims_per_slot": claims_per_slot,
            "claim_rows": claim_rows,
            "general_slot_rows": general_slot_rows,
            "critical_slot_rows": critical_slot_rows,
        }

        self.status = (
            f"decision={decision} | mode={mode} | "
            f"score={final_support_score:.3f} | "
            f"claims={total_claims}"
        )

        return Message(
            text=json.dumps(result, ensure_ascii=False, indent=2),
            sender="System",
            sender_name="VerificationDecisionNode",
        )