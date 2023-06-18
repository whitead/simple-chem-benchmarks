import click
from datasets import load_dataset
import os
from functools import partial
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from exmol.stoned import sanitize_smiles
from distractor import make_distractors
from pubchem import name_molecule
import random
from typing import List, Tuple

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
        if '.' in s:
            continue # only doing single product reactions for now
        product = sanitize_smiles(s[-1])[1]
        if not product:
            continue
        yield ">".join(s[:-1]) + ">", partial(product_eval, y=product), product, make_distractors(product, 4)


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
        mol = sanitize_smiles(mol)[1]
        if not mol:
            continue
        # split in half
        s = mol[: len(mol) // 2]

        # now we find distractors that if appended to s, will be invalid molecules
        ds_mol = make_distractors(mol, n=25)
        ds = [d[len(d) // 2:] for d in ds_mol]
        invalid_ds = []
        for i in range(len(ds)):
            if not Chem.MolFromSmiles(s + ds[i]):
                invalid_ds.append(ds[i])
                if len(invalid_ds) == 4:
                    break
        if len(invalid_ds) < 4:
            continue
        yield s, partial(valid_mol_eval, prompt=s), mol[len(mol) // 2:], invalid_ds

def name_mol_task(full=False):
    data = load_dataset(
        f"{SCRIPT_ROOT}/coconut.py", "full" if full else "small", split="train"
    )
    observed = set()
    for i, example in enumerate(data):
        mol = example["text"]
        mol = sanitize_smiles(mol)[1]
        if not mol:
            continue
        smiles, names = name_molecule(mol)
        if len(smiles) < 5 or smiles[0] in observed:
            continue
        observed.add(smiles[0])
        yield smiles[0], lambda x: x == names[0], names[0], names[1:5]

def make_options(ref: str, distractors: List[str]) -> Tuple[str, str]:
    """Return string of options (as letters) and correct answer"""
    options = [ref] + distractors
    random.shuffle(options)
    return (
        "\n".join([f"{chr(65 + i)}) {o}" for i, o in enumerate(options)]),
        chr(65 + options.index(ref)),
    )


template_1 = """
What is the product from this reaction:
{prompt}?

Options:
{options}

Answer:"""

template_2 = """
What is a valid completion of this molecule:
{prompt}?

Options:
{options}

Answer:"""

template_3 = """
What is the IUPAC name of this molecule:
{prompt}?

Options:
{options}

Answer:"""
if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    TASK_COUNT = 1
    count = 0
    for prompt, eval, ref, distractors in product_task():
        options, answer = make_options(ref, distractors)
        print(template_1.format(prompt=prompt, options=options))
        print(answer)
        count += 1
        if count == TASK_COUNT:
            break

    count = 0
    for prompt, eval, ref, distractors in valid_mol_task():
        options, answer = make_options(ref, distractors)
        print(template_2.format(prompt=prompt, options=options))
        print(answer)
        count += 1
        if count == TASK_COUNT:
            break

    count = 0
    for prompt, eval, ref, distractors in name_mol_task():
        options, answer = make_options(ref, distractors)
        print(template_3.format(prompt=prompt, options=options))
        print(answer)
        count += 1
        if count == TASK_COUNT:
            break