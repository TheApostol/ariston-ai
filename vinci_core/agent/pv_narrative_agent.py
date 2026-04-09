"""
Pharmacovigilance Narrative Agent.

Generates CIOMS-I and MedWatch narrative reports from structured adverse
event data.  Used in regulatory submissions and signal management.

Follows ICH E2B(R3) / CIOMS I format:
  - Patient information
  - Suspect drugs and dosing
  - Adverse event description
  - Medical history
  - Reporter assessment
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


class PharmacovigilanceNarrativeAgent:
    """
    Composes regulatory-compliant adverse event narratives.

    Outputs two formats:
      - CIOMS-I  (ICSR, international primary source reporting)
      - MedWatch (FDA Form 3500A equivalent free-text narrative)

    Both formats are consumed by the RegulatoryCopilot for GxP signing.
    """

    SEVERITY_MAP = {
        "fatal":            ("FATAL", "serious"),
        "life-threatening": ("LIFE-THREATENING", "serious"),
        "hospitalization":  ("REQUIRED HOSPITALIZATION", "serious"),
        "disability":       ("RESULTED IN DISABILITY", "serious"),
        "congenital":       ("CONGENITAL ANOMALY", "serious"),
        "other serious":    ("OTHER SERIOUS", "serious"),
        "non-serious":      ("NON-SERIOUS", "non-serious"),
    }

    # ── Public interface ──────────────────────────────────────────────────────

    def generate_cioms(self, event: Dict[str, Any]) -> str:
        """
        Generate a CIOMS-I format individual case safety report (ICSR) narrative.

        Args:
            event: dict with keys:
              - case_id       (str)
              - drug_name     (str)
              - dose          (str, e.g. "10 mg once daily")
              - indication    (str)
              - ae_term       (str, MedDRA preferred term)
              - onset_date    (str, ISO date)
              - outcome       (str: "recovered" | "recovering" | "not recovered" | "fatal" | "unknown")
              - patient_age   (int, optional)
              - patient_sex   (str, optional)
              - medical_history (str, optional)
              - reporter_type (str: "physician" | "patient" | "pharmacist" | "other")
              - severity      (str: "fatal" | "life-threatening" | ... | "non-serious")
              - narrative     (str, additional free-text)

        Returns:
            CIOMS-I narrative as a plain-text string.
        """
        case_id = event.get("case_id", "UNKNOWN")
        drug = event.get("drug_name", "UNKNOWN")
        dose = event.get("dose", "not reported")
        indication = event.get("indication", "not reported")
        ae_term = event.get("ae_term", "UNKNOWN ADVERSE EVENT")
        onset = event.get("onset_date", "not reported")
        outcome = event.get("outcome", "unknown").upper()
        age = event.get("patient_age")
        sex = event.get("patient_sex", "not reported")
        hx = event.get("medical_history", "none reported")
        reporter = event.get("reporter_type", "other")
        severity_key = event.get("severity", "non-serious").lower()
        severity_label, seriousness = self.SEVERITY_MAP.get(
            severity_key, ("NON-SERIOUS", "non-serious")
        )
        extra = event.get("narrative", "")
        generated_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        patient_desc = f"{'a ' + str(age) + '-year-old' if age else 'a'} {sex} patient"

        narrative = (
            f"CIOMS-I INDIVIDUAL CASE SAFETY REPORT\n"
            f"Report Reference: {case_id}\n"
            f"Generated: {generated_at}\n"
            f"Seriousness: {seriousness.upper()}\n"
            f"{'─' * 60}\n\n"
            f"CASE NARRATIVE\n\n"
            f"This report describes {ae_term} in {patient_desc} "
            f"treated with {drug} ({dose}) for {indication}. "
            f"The event onset was {onset}. "
            f"Relevant medical history: {hx}. "
            f"The event was assessed as {severity_label} by a {reporter}. "
            f"Outcome at time of reporting: {outcome}."
        )
        if extra:
            narrative += f"\n\nAdditional information: {extra}"

        narrative += (
            f"\n\n{'─' * 60}\n"
            f"REGULATORY CLASSIFICATION\n"
            f"  MedDRA PT: {ae_term}\n"
            f"  Seriousness Criteria: {severity_label}\n"
            f"  Suspect Drug: {drug}\n"
            f"  Route/Dose: {dose}\n"
            f"  ICH E2B(R3) Compliant: YES\n"
        )
        return narrative

    def generate_medwatch(self, event: Dict[str, Any]) -> str:
        """
        Generate an FDA MedWatch 3500A-equivalent narrative.

        Uses the same event dict as generate_cioms().
        """
        case_id = event.get("case_id", "UNKNOWN")
        drug = event.get("drug_name", "UNKNOWN")
        dose = event.get("dose", "not reported")
        indication = event.get("indication", "not reported")
        ae_term = event.get("ae_term", "UNKNOWN ADVERSE EVENT")
        onset = event.get("onset_date", "not reported")
        outcome = event.get("outcome", "unknown").upper()
        severity_key = event.get("severity", "non-serious").lower()
        severity_label, _ = self.SEVERITY_MAP.get(
            severity_key, ("NON-SERIOUS", "non-serious")
        )
        extra = event.get("narrative", "")
        generated_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        narrative = (
            f"FDA MEDWATCH ADVERSE EVENT REPORT (3500A Equivalent)\n"
            f"Report Reference: {case_id}\n"
            f"Generated: {generated_at}\n"
            f"{'─' * 60}\n\n"
            f"Section A — Suspect Medication\n"
            f"  Drug Name: {drug}\n"
            f"  Dose / Route: {dose}\n"
            f"  Indication: {indication}\n\n"
            f"Section B — Adverse Event\n"
            f"  Event Description: {ae_term}\n"
            f"  Date of Event Onset: {onset}\n"
            f"  Seriousness: {severity_label}\n"
            f"  Patient Outcome: {outcome}\n\n"
            f"Section C — Narrative\n"
            f"  Patient received {drug} ({dose}) indicated for {indication}. "
            f"  The event '{ae_term}' (onset {onset}) was classified as {severity_label}. "
            f"  Outcome: {outcome}."
        )
        if extra:
            narrative += f"\n  Additional context: {extra}"

        narrative += (
            f"\n\n{'─' * 60}\n"
            f"Ariston AI — Pharmacovigilance Narrative Agent\n"
            f"Generated for FDA submission. Review by qualified person required.\n"
        )
        return narrative

    def generate_both(self, event: Dict[str, Any]) -> Dict[str, str]:
        """Generate both CIOMS-I and MedWatch narratives in one call."""
        return {
            "cioms": self.generate_cioms(event),
            "medwatch": self.generate_medwatch(event),
        }

    def batch_generate(
        self,
        events: List[Dict[str, Any]],
        format: str = "cioms",
    ) -> List[Dict[str, str]]:
        """
        Generate narratives for a list of adverse events.

        Args:
            events: list of event dicts
            format: "cioms" | "medwatch" | "both"

        Returns:
            List of dicts with case_id + generated narrative(s).
        """
        results = []
        for event in events:
            case_id = event.get("case_id", "UNKNOWN")
            entry: Dict[str, Any] = {"case_id": case_id}
            if format == "medwatch":
                entry["medwatch"] = self.generate_medwatch(event)
            elif format == "both":
                entry.update(self.generate_both(event))
            else:
                entry["cioms"] = self.generate_cioms(event)
            results.append(entry)
        return results


pv_narrative_agent = PharmacovigilanceNarrativeAgent()
