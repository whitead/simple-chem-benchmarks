import click
from datasets import load_dataset
import os
from functools import partial
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors

block = BlockLogs()

SCRIPT_ROOT = os.path.dirname(os.path.realpath(__file__))


def product_eval(yhat, y):
    m1 = Chem.MolFromSmiles(yhat)
    m2 = Chem.MolFromSmiles(y)
    # return tanimoto similarity
    return DataStructs.TanimotoSimilarity(
        rdMolDescriptors.GetMorganFingerprintAsBitVect(m1, 2),
        rdMolDescriptors.GetMorganFingerprintAsBitVect(m2, 2),
    )


def valid_mol_eval(yhat, prompt):
    try:
        m1 = Chem.MolFromSmiles(prompt + yhat)
    except:
        return 0
    if m1 is None:
        return 0
    return 1


def product_task(full=False):
    """
    Generator that produces a tuple of prompt and evaluation function. The evaluation function takes a string and returns a similarity score (higher is better)
    """
    data = load_dataset(
        f"{SCRIPT_ROOT}/ord.py", "full" if full else "small", split="train"
    )
    output = []
    for i, example in enumerate(data):
        s = example["text"].split(">")
        yield ">".join(s[:-1]) + ">", partial(product_eval, y=s[-1])


def valid_mol_task(full=False):
    """
    Generator that produces a tuple of prompt and evaluation function. The evaluation function takes a string and returns a success score (1 or 0)
    """
    data = load_dataset(
        f"{SCRIPT_ROOT}/coconut.py", "full" if full else "small", split="train"
    )
    output = []
    for i, example in enumerate(data):
        mol = example["text"]
        # split in half
        s = mol[: len(mol) // 2]
        yield s, partial(valid_mol_eval, prompt=s)


if __name__ == "__main__":
    for prompt, eval in product_task():
        print(prompt)
        print(eval("CCO"))
        break

    for prompt, eval in valid_mol_task():
        print(prompt)
        print(eval("Pi"))
        break
