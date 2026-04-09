import httpx
import urllib.parse
from typing import Dict, List

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
                            "title": str(meta.get("title", "")),
                            "source": str(meta.get("source", "")),
                            "pubdate": str(meta.get("pubdate", "")),
                            "link": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
                        })
                return results

            except Exception as e:
                print(f"PubMed Tool Error: {e}")
                return []
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
                    classes.append(str(concept.get("rxclassMinConceptItem", {}).get("className", "")))
                
                unique_classes = list(set(classes))
                if len(unique_classes) > 5:
                    return unique_classes[:5]  # pyre-ignore
                return unique_classes
            except Exception as e:
                print(f"RxNorm Tool Error: {e}")
                return []
        return []
    @staticmethod
    async def get_fda_drug_info(drug_name: str) -> Dict[str, str]:
        # Uses OpenFDA to find drug indications and adverse events
        async with httpx.AsyncClient() as client:
            try:
                url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{urllib.parse.quote(drug_name)}&limit=1"
                res = await client.get(url, timeout=10.0)
                if res.status_code != 200:
                    return {"warning": f"No FDA data found for {drug_name}."}
                
                data = res.json()
                result = data.get("results", [{}])[0]
                
                return {
                    "indications": str(result.get("indications_and_usage", ["N/A"])[0]),
                    "side_effects": str(result.get("adverse_reactions", ["N/A"])[0]),
                    "warnings": str(result.get("warnings", ["N/A"])[0]),
                    "brand_name": str(result.get("openfda", {}).get("brand_name", ["Unknown"])[0])
                }
            except Exception as e:
                print(f"OpenFDA Tool Error: {e}")
                return {"error": f"Failed to reach OpenFDA: {str(e)}"}

    @staticmethod
    async def get_clinical_trials(condition: str) -> List[Dict[str, str]]:
        # Uses ClinicalTrials.gov API to find relevant active trials
        async with httpx.AsyncClient() as client:
            try:
                url = f"https://clinicaltrials.gov/api/v2/studies?query.cond={urllib.parse.quote(condition)}&pageSize=3"
                res = await client.get(url, timeout=10.0)
                res.raise_for_status()
                data = res.json()
                
                studies = []
                for study in data.get("studies", []):
                    protocol = study.get("protocolSection", {})
                    id_sec = protocol.get("identificationModule", {})
                    status_sec = protocol.get("statusModule", {})
                    
                    studies.append({
                        "nct_id": str(id_sec.get("nctId", "N/A")),
                        "title": str(id_sec.get("briefTitle", "N/A")),
                        "status": str(status_sec.get("overallStatus", "N/A"))
                    })
                return studies
            except Exception as e:
                print(f"ClinicalTrials Tool Error: {e}")
                return []
    @staticmethod
    async def search_clinvar(gene: str) -> List[Dict[str, str]]:
        # Uses NCBI E-Utils to search ClinVar for gene significance
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        async with httpx.AsyncClient() as client:
            try:
                search_url = f"{base_url}/esearch.fcgi?db=clinvar&term={urllib.parse.quote(gene)}&retmode=json&retmax=3"
                res = await client.get(search_url, timeout=10.0)
                ids = res.json().get("esearchresult", {}).get("idlist", [])
                
                if not ids: return []
                
                summary_url = f"{base_url}/esummary.fcgi?db=clinvar&id={','.join(ids)}&retmode=json"
                s_res = await client.get(summary_url, timeout=10.0)
                data = s_res.json().get("result", {})
                
                results = []
                for uid in ids:
                    meta = data.get(uid, {})
                    if meta:
                        results.append({
                            "title": str(meta.get("title", "")),
                            "significance": str(meta.get("germline_classification", {}).get("description", "Unknown"))
                        })
                return results
            except Exception as e:
                print(f"ClinVar Tool Error: {e}")
                return []

    @staticmethod
    async def get_adverse_events(drug_name: str) -> List[str]:
        # Uses OpenFDA to find top 3 recorded adverse events
        async with httpx.AsyncClient() as client:
            try:
                url = f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{urllib.parse.quote(drug_name)}&count=patient.reaction.reactionmeddrapt.exact&limit=3"
                res = await client.get(url, timeout=10.0)
                data = res.json()
                events = [str(term['term']) for term in data.get("results", [])]
                return events
            except Exception as e:
                print(f"FDA AE Tool Error: {e}")
                return []

    @staticmethod
    async def get_vademecum_data(drug_name: str) -> Dict[str, str]:
        """
        Ariston Vademecum: Deep pharmacological compendium.
        """
        # Simulation of a high-depth vadi-mecum (drug manual)
        return {
            "mechanism": "High-affinity binding to mu-opioid receptors in the myenteric plexus.",
            "pharmacokinetics": "Low system bioavailability due to P-glycoprotein efflux.",
            "contraindications": "Acute ulcerative colitis, bacterial enterocolitis.",
            "interactions": "Inhibitors of P-glycoprotein may increase plasma concentrations."
        }

    @staticmethod
    async def get_symptom_research(symptom: str) -> List[str]:
        """
        Deep symptom-to-paper correlation research.
        """
        # Querying PubMed specifically for recent clinical papers on a symptom
        try:
            papers = await MedicalTools.search_pubmed(f"{symptom} clinical diagnosis 2024", max_results=3)
            return [f"{p['title']} [{p['source']}]" for p in papers]
        except:
            return ["No recent papers found for this symptom profile."]

    @staticmethod
    async def search_public_records(query: str) -> str:
        """
        Simulated harvesting of public medical records and case studies.
        """
        # In a real system, this would crawl public clinical registries or de-identified datasets.
        return (
            "Source: Public Clinical Registry Alpha - Case ID: RX-441\n"
            "Summary: Similar symptom presentation noted in 45yo male; response to medication X was optimal.\n"
            "Source: Global Health Open Data - Study ID: P-99\n"
            "Conclusion: High correlation between symptom Y and genetic variant Z."
        )
