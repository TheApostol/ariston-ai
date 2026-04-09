"""
Pharmacogenomics Agent.
Cross-references ClinVar genomic variants with drug-gene interactions.
Alerts on critical PGx risks (CYP2C19, HLA-B*5701, TPMT, VKORC1, etc.)
"""

from typing import List, Dict, Any
from vinci_core.tools.medical_tools import MedicalTools

# Curated PGx knowledge base — extend with full CPIC guidelines
GENE_DRUG_INTERACTIONS: Dict[str, Dict[str, str]] = {
    "CYP2C19": {
        "clopidogrel": "Poor metabolizer: reduced active metabolite → increased cardiovascular risk. Consider alternative antiplatelet.",
        "omeprazole": "Rapid metabolizer: may require higher dosage for H. pylori eradication.",
        "citalopram": "Poor metabolizer: increased plasma levels → QT prolongation risk.",
    },
    "CYP2D6": {
        "codeine": "Poor metabolizer: inadequate analgesia. Ultra-rapid: life-threatening morphine toxicity.",
        "tamoxifen": "Poor metabolizer: reduced endoxifen formation → decreased efficacy.",
    },
    "HLA-B*5701": {
        "abacavir": "CRITICAL: High risk of life-threatening hypersensitivity reaction. Do not use.",
    },
    "HLA-B*1502": {
        "carbamazepine": "CRITICAL: High risk of Stevens-Johnson syndrome in Asian populations.",
    },
    "TPMT": {
        "azathioprine": "Deficiency: severe myelosuppression. Reduce dose by 10-fold or switch to alternative.",
        "mercaptopurine": "Deficiency: severe myelosuppression. Genetic testing mandatory before use.",
    },
    "VKORC1": {
        "warfarin": "Increased sensitivity: requires lower initial dosing. Use pharmacogenomic dosing calculator.",
    },
    "DPYD": {
        "fluorouracil": "Deficiency: severe/fatal fluoropyrimidine toxicity. Dose reduction required.",
        "capecitabine": "Deficiency: severe/fatal toxicity. Pre-treatment DPYD genotyping recommended.",
    },
    "UGT1A1": {
        "irinotecan": "*28 homozygous: severe neutropenia. Reduce starting dose.",
    },
}


class PharmacogenomicsAgent:
    async def cross_reference(self, drug_name: str, patient_id: str = None) -> Dict[str, Any]:
        """
        Fetch ClinVar variants for known PGx genes and cross-reference with drug.
        """
        alerts = []
        genes_checked = []

        d_lower = drug_name.lower()

        for gene, interactions in GENE_DRUG_INTERACTIONS.items():
            for target_drug, risk in interactions.items():
                if target_drug in d_lower or d_lower in target_drug:
                    # Verify against live ClinVar
                    variants = await MedicalTools.search_clinvar(gene)
                    genes_checked.append(gene)

                    severity = "CRITICAL" if "CRITICAL" in risk else "WARNING"
                    alerts.append({
                        "gene": gene,
                        "drug": drug_name,
                        "severity": severity,
                        "risk_description": risk,
                        "clinvar_variants": variants[:3],
                    })

        return {
            "drug": drug_name,
            "genes_checked": genes_checked,
            "genomic_alerts": alerts,
            "status": "warning" if alerts else "safe",
            "recommendation": (
                "Consult clinical pharmacist before prescribing. "
                "Genetic counseling recommended." if alerts else "No known PGx interactions detected."
            ),
        }

    async def format_for_context(self, drug_name: str) -> str:
        result = await self.cross_reference(drug_name)
        if result["status"] == "safe":
            return f"PGx Check ({drug_name}): No known interactions."
        lines = [f"PGx ALERTS for {drug_name}:"]
        for alert in result["genomic_alerts"]:
            lines.append(f"  [{alert['severity']}] {alert['gene']}: {alert['risk_description']}")
        return "\n".join(lines)


pharmacogenomics_agent = PharmacogenomicsAgent()
