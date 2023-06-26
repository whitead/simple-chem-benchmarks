from exmol import run_stoned
from exmol.stoned import sanitize_smiles
from rdkit import Chem
from typing import List


def make_distractors(mol_smiles: str, n: int = 4) -> List[str]:
    """
    Returns a list of distractors for a given SMILES string
    """
    smiles, scores = run_stoned(
        mol_smiles, return_selfies=False, min_mutations=2, max_mutations=4
    )
    # just take highest score
    top = sorted(zip(smiles, scores), key=lambda x: x[1], reverse=True)[:n]
    return [m for m, _ in top]
