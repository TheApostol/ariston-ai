import httpx
import urllib.parse
from typing import Dict, Any, List

class MedicalTools:
    """
    Public integrations with PubMed and RxNorm to provide factual grounding.
    """
    @staticmethod
    async def search_pubmed(query: str, max_results: int = 3) -> List[Dict[str, str]]:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        async with httpx.AsyncClient() as client:
            try:
                # 1. Search for IDs
                search_url = f"{base_url}/esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmode=json&retmax={max_results}"
                response = await client.get(search_url, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                id_list = data.get("esearchresult", {}).get("idlist", [])
                
                if not id_list:
                    return []

                # 2. Fetch Summaries
                ids_str = ",".join(id_list)
                summary_url = f"{base_url}/esummary.fcgi?db=pubmed&id={ids_str}&retmode=json"
                summary_response = await client.get(summary_url, timeout=10.0)
                summary_response.raise_for_status()
                summary_data = summary_response.json()
                
                results = []
                for uid in id_list:
                    meta = summary_data.get("result", {}).get(uid, {})
                    if meta:
                        results.append({
                            "title": meta.get("title", ""),
                            "source": meta.get("source", ""),
                            "pubdate": meta.get("pubdate", ""),
                            "link": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
                        })
                return results

            except Exception as e:
                print(f"PubMed Tool Error: {e}")
                return []

    @staticmethod
    async def get_drug_classes(drug_name: str) -> List[str]:
        # Uses RxNorm REST API to find pharmacological classes
        async with httpx.AsyncClient() as client:
            try:
                # 1. Get RxCUI
                search_url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={urllib.parse.quote(drug_name)}"
                res = await client.get(search_url, timeout=10.0)
                res.raise_for_status()
                data = res.json()
                
                id_group = data.get("idGroup", {})
                rxnorm_id = id_group.get("rxnormId", [])
                if not rxnorm_id:
                    return ["Unknown drug or no RxCUI found."]

                rxcui = rxnorm_id[0]

                # 2. Get RxClass
                class_url = f"https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui={rxcui}"
                c_res = await client.get(class_url, timeout=10.0)
                c_res.raise_for_status()
                c_data = c_res.json()
                
                classes = []
                for concept in c_data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", []):
                    classes.append(concept.get("rxclassMinConceptItem", {}).get("className", ""))
                
                return list(set(classes))[:5]
            except Exception as e:
                print(f"RxNorm Tool Error: {e}")
                return []
