import requests
from rdkit import Chem
from typing import List, Tuple
from exmol.stoned import largest_mol

def name_molecule(
    origin_smiles: str,
) -> Tuple[List[str], List[float]]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{requests.utils.quote(origin_smiles)}/property/IsomericSMILES,IUPACName/JSON"
    reply = requests.get(
        url,
        params={"Threshold": 25, "MaxRecords": 10},
        headers={"accept": "text/json"},
        timeout=10,
    )

    try:
        data = reply.json()
    except:
        return [], []
    smiles = [d["IsomericSMILES"] for d in data["PropertyTable"]["Properties"]]
    names = [d["IUPACName"] for d in data["PropertyTable"]["Properties"]]
    return smiles, names
