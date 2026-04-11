from typing import Dict, Any

class ExplainabilityModule:
    def explain(self, state: Dict[str, Any], action: str) -> Dict[str, str]:
        lactate = state["labs"]["lactate"]
        ph = state["labs"]["pH"]
        rr = state["vitals"]["RR"]
        gcs = state["vitals"]["GCS"]
        qsofa = state.get("qSOFA", 0)
        spo2 = state["vitals"]["SpO2"]
        bp = state["vitals"]["BP"]
        fio2 = state["resources"]["fio2"]

        reason = f"Action '{action}' because "
        risk = "Low"
        outcome = "Stable"

        if lactate > 2.0:
            reason += f"lactate elevated ({lactate:.1f} mmol/L, normal<2). "
            risk = "Moderate"
            outcome = "May improve with fluids/vasopressors"
        if ph < 7.35:
            reason += f"pH low ({ph:.2f}, acidosis). "
            risk = "High" if ph < 7.2 else "Moderate"
            outcome = "Needs ventilation or bicarbonate"
        if rr >= 22:
            reason += f"respiratory rate {rr} (tachypnea). "
            risk = "High"
            outcome = "Consider non-invasive ventilation"
        if spo2 < 92:
            reason += f"hypoxemia (SpO2 {spo2:.0f}%, FiO2 {fio2:.0%}). "
            risk = "High"
            outcome = "Increase FiO2 or ventilation"
        if bp < 90:
            reason += f"hypotension (BP {bp:.0f}). "
            risk = "High"
            outcome = "Start vasopressors or fluids"
        if gcs < 15:
            reason += f"low GCS ({gcs:.0f}, altered mentation). "
            risk = "High" if gcs < 13 else "Moderate"
            outcome = "Protect airway, consider ICU transfer"

        if reason.endswith("because "):
            reason = "Routine monitoring, no acute abnormalities."
        else:
            reason = reason.rstrip(" ")

        citation = "Based on Sepsis-3, NEWS, and GCS: lactate, pH, RR, BP, and GCS are key predictors of mortality and deterioration."
        return {
            "decision_reason": reason,
            "risk_assessment": f"{risk} risk (qSOFA={qsofa})",
            "expected_outcome": outcome,
            "clinical_evidence": citation
        }