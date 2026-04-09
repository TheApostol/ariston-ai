from typing import List, Dict

class OntologyMapper:
    """
    Demonstrates semantic grounding for Life Science interoperability.
    Maps unstructured clinical entities to standardized medical codes (SNOMED-CT, LOINC).
    """
    
    # Mock knowledge base for demonstration
    SNOMED_MAP = {
        "cough": "49727002",
        "fever": "386661006",
        "chest pain": "29857009",
        "headache": "25064002",
        "fatigue": "84229001",
        "dyspnea": "267036007"
    }

    LOINC_MAP = {
        "glucose": "2339-0",
        "hemoglobin": "718-7",
        "white blood cell count": "6690-2",
        "cholesterol": "2093-3",
        "blood pressure": "85354-9"
    }

    @classmethod
    def ground_entities(cls, entities: List[str]) -> List[Dict[str, str]]:
        grounded = []
        for entity in entities:
            e_lower = entity.lower().strip()
            
            # Check SNOMED
            if e_lower in cls.SNOMED_MAP:
                grounded.append({
                    "term": entity,
                    "code": cls.SNOMED_MAP[e_lower],
                    "system": "SNOMED-CT"
                })
            
            # Check LOINC
            elif e_lower in cls.LOINC_MAP:
                grounded.append({
                    "term": entity,
                    "code": cls.LOINC_MAP[e_lower],
                    "system": "LOINC"
                })
            
            else:
                grounded.append({
                    "term": entity,
                    "code": "UNKNOWN",
                    "system": "NONE"
                })
        
        return grounded

ontology_mapper = OntologyMapper()
