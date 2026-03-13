import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from pathlib import Path

INPUT = "01_data/frozen/dataset_v1_frozen.csv"
OUTPUT = "01_data/processed/scaffold_groups.csv"

df = pd.read_csv(INPUT)

scaffolds = []

for smiles in df["Canonical_SMILES"]:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        scaffolds.append("INVALID")
        continue

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)

    scaffolds.append(scaffold_smiles)

df["scaffold_smiles"] = scaffolds

# benzersiz scaffold ID üret
scaffold_map = {
    s: f"SCF_{i:03d}"
    for i, s in enumerate(sorted(df["scaffold_smiles"].unique()), start=1)
}

df["Murcko_scaffold_ID"] = df["scaffold_smiles"].map(scaffold_map)

out = df[["Compound_ID", "Murcko_scaffold_ID"]]

Path("01_data/processed").mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False)

print("OK ✅ Murcko scaffolds generated")
print("Unique scaffolds:", len(scaffold_map))
print("Output:", OUTPUT)