import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
import argparse
def smiles_to_pdbqt(smiles, out_pdbqt):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return False

    mol = Chem.AddHs(mol)

    res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    if res != 0:
        print(f"Embedding failed: {smiles}")
        return False

    # ===== UFF optimization =====
    try:

        if AllChem.UFFHasAllMoleculeParams(mol):

            AllChem.UFFOptimizeMolecule(mol)

        else:
            print(f"UFF parameters missing: {smiles}")
            return False

    except Exception as e:

        print(f"UFF optimization failed: {smiles}")
        print(e)

        return False

    # ===== PDBQT =====
    try:

        preparator = MoleculePreparation()
        preparator.prepare(mol)
        preparator.write_pdbqt_file(out_pdbqt)

    except Exception as e:

        print(f"PDBQT conversion failed: {smiles}")
        print(e)

        return False

    return True

    return True
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--pathin', default='model_base.h5', type=str)
arg_parser.add_argument('--pathout', default='model_base.h5', type=str)
args = arg_parser.parse_args()
# ===== main =====
input_csv = os.path.abspath(args.pathin)

output_dir = os.path.abspath(args.pathout)

os.makedirs(output_dir, exist_ok=True)

# فایل SMILES تولیدشده
df = pd.read_csv(input_csv, header=None, names=['smiles'])

# ===============================
# فایل دیتاست مرجع برای Novelty
# ===============================
from utils import read_delimited_file
mol_list0_train = list(read_delimited_file('train.smi'))
mol_list0_valid = list(read_delimited_file('valid.smi'))
mol_list0_test = list(read_delimited_file('test.smi'))
    
mol_list1 = list(read_delimited_file('HTR1A_train.txt'))
mol_list1.extend(list(read_delimited_file('HTR1A_test.txt')))
mol_list1.extend(list(read_delimited_file('HTR1A_valid.txt')))
mol_list1.extend(list(read_delimited_file('DRD2_train.txt')))
mol_list1.extend(list(read_delimited_file('DRD2_test.txt')))
mol_list1.extend(list(read_delimited_file('DRD2_valid.txt')))

mol_list = mol_list0_train
mol_list.extend(mol_list0_valid)
mol_list.extend(mol_list0_test) 
mol_list.extend(mol_list1)
reference_smiles = set(mol_list)

# ===============================
# Canonicalize
# ===============================
def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


# ===============================
# Unique و Novel
# ===============================
unique_smiles = set()
novel_smiles = []

success_count = 0

for i, row in df.iterrows():

    smiles = str(row["smiles"]).strip()
    
    mol_id = i

    out_path = os.path.join(output_dir, f"{mol_id}_lig.pdbqt")

    # اگر قبلا ساخته شده، ردش کن
    if os.path.exists(out_path):
        print(f"Skipping existing: {mol_id}")
        continue

    canon = canonicalize(smiles)

    if canon is None:
        print(f"Invalid SMILES skipped: {smiles}")
        continue

    # ذخیره یونیک‌ها
    unique_smiles.add(canon)

    # بررسی Novel
    if canon not in reference_smiles:
        novel_smiles.append(canon)

    mol_id = i

    out_path = os.path.join(output_dir, f"{mol_id}_lig.pdbqt")

    ok = smiles_to_pdbqt(canon, out_path)

    if ok:
        success_count += 1
        print(f"Done: {mol_id}")

# ===============================
# ذخیره Unique
# ===============================
unique_path = os.path.join(output_dir, "unique_smiles.txt")

with open(unique_path, "w") as f:
    for smi in sorted(unique_smiles):
        f.write(smi + "\n")

# ===============================
# ذخیره Novel
# ===============================
novel_path = os.path.join(output_dir, "novel_smiles.txt")

with open(novel_path, "w") as f:
    for smi in sorted(set(novel_smiles)):
        f.write(smi + "\n")

# ===============================
# آمار
# ===============================
print("\n========== Statistics ==========")
print(f"Total input: {len(df)}")
print(f"Successful PDBQT: {success_count}")
print(f"Unique molecules: {len(unique_smiles)}")
print(f"Novel molecules: {len(set(novel_smiles))}")

print(f"\nSaved unique SMILES to:\n{unique_path}")
print(f"\nSaved novel SMILES to:\n{novel_path}")
