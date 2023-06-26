import requests
from rdkit import Chem
from typing import List, Tuple
from exmol.stoned import largest_mol
from tenacity import retry, stop_after_attempt, wait_fixed
import requests_cache

session = requests_cache.CachedSession('pubchem-cache')

@retry(wait=wait_fixed(2)   , stop=stop_after_attempt(3))
def name_molecule(
    origin_smiles: str,
) -> Tuple[List[str], List[float]]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{requests.utils.quote(origin_smiles)}/property/IsomericSMILES,IUPACName/JSON"
    reply = session.get(
        url,
        params={"Threshold": 25, "MaxRecords": 10},
        headers={"accept": "text/json"},
        timeout=10,
    )

    try:
        data = reply.json()
        smiles = [d["IsomericSMILES"] for d in data["PropertyTable"]["Properties"]]
        names = [d["IUPACName"] for d in data["PropertyTable"]["Properties"]]
        return smiles, names
    except:
        return [], []
