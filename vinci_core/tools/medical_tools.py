"""
Medical Tools — live public API integrations for knowledge grounding.
PubMed · RxNorm · OpenFDA · ClinicalTrials.gov · ClinVar
"""

import httpx
import urllib.parse
from typing import Dict, List


class MedicalTools:

    @staticmethod
    async def search_pubmed(query: str, max_results: int = 3) -> List[Dict[str, str]]:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get(f"{base}/esearch.fcgi", params={
                    "db": "pubmed", "term": query, "retmode": "json", "retmax": max_results
                })
                ids = r.json().get("esearchresult", {}).get("idlist", [])
                if not ids:
                    return []
                s = await client.get(f"{base}/esummary.fcgi", params={
                    "db": "pubmed", "id": ",".join(ids), "retmode": "json"
                })
                data = s.json().get("result", {})
                return [
                    {"title": data[uid].get("title", ""), "source": data[uid].get("source", ""),
                     "pubdate": data[uid].get("pubdate", ""), "link": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"}
                    for uid in ids if uid in data
                ]
            except Exception as e:
                print(f"[PubMed] {e}")
                return []

    @staticmethod
    async def get_drug_classes(drug_name: str) -> List[str]:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get("https://rxnav.nlm.nih.gov/REST/rxcui.json", params={"name": drug_name})
                rxcui_list = r.json().get("idGroup", {}).get("rxnormId", [])
                if not rxcui_list:
                    return []
                c = await client.get("https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json",
                                     params={"rxcui": rxcui_list[0]})
                classes = [i.get("rxclassMinConceptItem", {}).get("className", "")
                           for i in c.json().get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])]
                return list(set(classes))[:5]
            except Exception as e:
                print(f"[RxNorm] {e}")
                return []

    @staticmethod
    async def get_fda_drug_info(drug_name: str) -> Dict[str, str]:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get("https://api.fda.gov/drug/label.json",
                                     params={"search": f"openfda.generic_name:{urllib.parse.quote(drug_name)}", "limit": 1})
                if r.status_code != 200:
                    return {"warning": f"No FDA data for {drug_name}"}
                result = r.json().get("results", [{}])[0]
                return {
                    "indications": str(result.get("indications_and_usage", ["N/A"])[0])[:500],
                    "side_effects": str(result.get("adverse_reactions", ["N/A"])[0])[:500],
                    "warnings": str(result.get("warnings", ["N/A"])[0])[:500],
                    "brand_name": str(result.get("openfda", {}).get("brand_name", ["Unknown"])[0]),
                }
            except Exception as e:
                print(f"[FDA] {e}")
                return {"error": str(e)}

    @staticmethod
    async def get_clinical_trials(condition: str, max_results: int = 3) -> List[Dict[str, str]]:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get("https://clinicaltrials.gov/api/v2/studies",
                                     params={"query.cond": condition, "pageSize": max_results})
                studies = []
                for s in r.json().get("studies", []):
                    p = s.get("protocolSection", {})
                    studies.append({
                        "nct_id": p.get("identificationModule", {}).get("nctId", ""),
                        "title": p.get("identificationModule", {}).get("briefTitle", ""),
                        "status": p.get("statusModule", {}).get("overallStatus", ""),
                    })
                return studies
            except Exception as e:
                print(f"[ClinicalTrials] {e}")
                return []

    @staticmethod
    async def search_clinvar(gene: str) -> List[Dict[str, str]]:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get(f"{base}/esearch.fcgi",
                                     params={"db": "clinvar", "term": gene, "retmode": "json", "retmax": 3})
                ids = r.json().get("esearchresult", {}).get("idlist", [])
                if not ids:
                    return []
                s = await client.get(f"{base}/esummary.fcgi",
                                     params={"db": "clinvar", "id": ",".join(ids), "retmode": "json"})
                data = s.json().get("result", {})
                return [
                    {"title": data[uid].get("title", ""),
                     "significance": data[uid].get("germline_classification", {}).get("description", "Unknown")}
                    for uid in ids if uid in data
                ]
            except Exception as e:
                print(f"[ClinVar] {e}")
                return []

    @staticmethod
    async def get_adverse_events(drug_name: str) -> List[str]:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get("https://api.fda.gov/drug/event.json", params={
                    "search": f"patient.drug.medicinalproduct:{urllib.parse.quote(drug_name)}",
                    "count": "patient.reaction.reactionmeddrapt.exact", "limit": 5,
                })
                return [term["term"] for term in r.json().get("results", [])]
            except Exception as e:
                print(f"[FDA AE] {e}")
                return []

    @staticmethod
    async def get_symptom_research(symptom: str) -> List[str]:
        papers = await MedicalTools.search_pubmed(f"{symptom} clinical diagnosis", max_results=3)
        return [f"{p['title']} [{p['source']}]" for p in papers]
