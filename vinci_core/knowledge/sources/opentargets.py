"""
Open Targets Platform — drug → target → disease associations.
Free GraphQL API, no key required. Industry standard for target validation.
"""

import httpx
from typing import List, Dict

GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"


async def get_drug_disease_associations(drug_name: str, limit: int = 5) -> List[Dict]:
    """
    Get disease associations for a drug via Open Targets.
    Returns diseases, evidence scores, and clinical phase data.
    """
    query = """
    query DrugAssociations($name: String!, $limit: Int!) {
      search(queryString: $name, entityNames: ["drug"], page: {index: 0, size: 1}) {
        hits {
          id
          name
          object {
            ... on Drug {
              id
              name
              mechanismsOfAction { rows { actionType mechanismOfAction } }
              indications {
                rows(page: {index: 0, size: $limit}) {
                  disease { id name }
                  maxPhaseForIndication
                }
              }
            }
          }
        }
      }
    }
    """
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.post(GRAPHQL_URL, json={
                "query": query,
                "variables": {"name": drug_name, "limit": limit}
            })
            hits = r.json().get("data", {}).get("search", {}).get("hits", [])
            if not hits:
                return []

            drug_obj = hits[0].get("object", {})
            results = []

            for row in drug_obj.get("indications", {}).get("rows", []):
                results.append({
                    "source": "opentargets",
                    "drug": drug_name,
                    "disease_id": row.get("disease", {}).get("id", ""),
                    "disease_name": row.get("disease", {}).get("name", ""),
                    "clinical_phase": row.get("maxPhaseForIndication", 0),
                    "content": f"{drug_name} → {row.get('disease', {}).get('name', '')} (Phase {row.get('maxPhaseForIndication', '?')})",
                })
            return results
        except Exception as e:
            print(f"[OpenTargets] {e}")
            return []


async def get_target_disease_associations(target_symbol: str, limit: int = 5) -> List[Dict]:
    """
    Get disease associations for a gene/protein target.
    Useful for pharma layer to validate drug targets.
    """
    query = """
    query TargetAssociations($symbol: String!, $limit: Int!) {
      target(ensemblId: $symbol) {
        id
        approvedSymbol
        associatedDiseases(page: {index: 0, size: $limit}) {
          rows {
            disease { id name }
            score
          }
        }
      }
    }
    """
    # First resolve symbol to Ensembl ID
    search_query = """
    query SearchTarget($symbol: String!) {
      search(queryString: $symbol, entityNames: ["target"], page: {index: 0, size: 1}) {
        hits { id name }
      }
    }
    """
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            s = await client.post(GRAPHQL_URL, json={"query": search_query, "variables": {"symbol": target_symbol}})
            hits = s.json().get("data", {}).get("search", {}).get("hits", [])
            if not hits:
                return []
            ensembl_id = hits[0]["id"]

            r = await client.post(GRAPHQL_URL, json={"query": query, "variables": {"symbol": ensembl_id, "limit": limit}})
            rows = r.json().get("data", {}).get("target", {}).get("associatedDiseases", {}).get("rows", [])
            return [
                {
                    "source": "opentargets",
                    "target": target_symbol,
                    "disease_id": row.get("disease", {}).get("id", ""),
                    "disease_name": row.get("disease", {}).get("name", ""),
                    "association_score": round(row.get("score", 0), 3),
                    "content": f"{target_symbol} → {row.get('disease', {}).get('name', '')} (score: {round(row.get('score', 0), 3)})",
                }
                for row in rows
            ]
        except Exception as e:
            print(f"[OpenTargets/Target] {e}")
            return []
