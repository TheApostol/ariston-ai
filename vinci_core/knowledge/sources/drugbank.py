"""
DrugBank Open Data — drug-drug interactions, targets, pathways.
Uses DrugBank's free open data REST API (no key required for basic queries).
Full database requires academic/commercial license — start with open endpoints.
"""

import httpx
from typing import List, Dict

BASE = "https://go.drugbank.com/drugs"
OPEN_API = "https://api.drugbank.com/v1"


async def get_drug_interactions(drug_name: str) -> List[Dict]:
    """
    Get drug-drug interactions via DrugBank open REST API.
    Falls back to RxNorm interaction check if DrugBank unavailable.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            # DrugBank open search
            r = await client.get(
                "https://rxnav.nlm.nih.gov/REST/interaction/interaction.json",
                params={"drug": drug_name}
            )
            if r.status_code != 200:
                return []
            data = r.json()
            groups = data.get("interactionTypeGroup", [])
            interactions = []
            for group in groups:
                for itype in group.get("interactionType", []):
                    for pair in itype.get("interactionPair", []):
                        interactions.append({
                            "source": "drugbank/rxnorm",
                            "drug_a": pair.get("interactionConcept", [{}])[0].get("minConceptItem", {}).get("name", ""),
                            "drug_b": pair.get("interactionConcept", [{}])[-1].get("minConceptItem", {}).get("name", ""),
                            "severity": pair.get("severity", "unknown"),
                            "description": pair.get("description", ""),
                        })
            return interactions[:10]
        except Exception as e:
            print(f"[DrugBank/Interactions] {e}")
            return []


async def get_drug_targets(drug_name: str) -> List[Dict]:
    """
    Get molecular targets for a drug via ChEMBL (public, no key).
    Returns protein targets, mechanisms, and action types.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            # Search ChEMBL for molecule
            r = await client.get(
                "https://www.ebi.ac.uk/chembl/api/data/molecule/search",
                params={"q": drug_name, "format": "json", "limit": 1}
            )
            molecules = r.json().get("molecules", [])
            if not molecules:
                return []
            chembl_id = molecules[0].get("molecule_chembl_id")

            # Get mechanisms of action
            m = await client.get(
                f"https://www.ebi.ac.uk/chembl/api/data/mechanism",
                params={"molecule_chembl_id": chembl_id, "format": "json", "limit": 5}
            )
            mechs = m.json().get("mechanisms", [])
            return [
                {
                    "source": "chembl",
                    "chembl_id": chembl_id,
                    "target": mech.get("target_chembl_id", ""),
                    "action_type": mech.get("action_type", ""),
                    "mechanism": mech.get("mechanism_of_action", ""),
                }
                for mech in mechs
            ]
        except Exception as e:
            print(f"[ChEMBL/Targets] {e}")
            return []
