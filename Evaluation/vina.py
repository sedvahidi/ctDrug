import os
import glob
import subprocess
import csv
import argparse

def extract_affinity(stdout):
    for line in stdout.splitlines():
        if line.strip().startswith("1 "):
            parts = line.split()
            return float(parts[1])
    return None

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--ligpath', default="C:/vahidi/Results/20data/layered_FFN_1024/FiLM/HTR1A/ligands", type=str)
arg_parser.add_argument('--target', default='HTR1A', type=str)
args = arg_parser.parse_args()

ligand_dir = os.path.abspath(args.ligpath)
ligand_files = sorted(glob.glob(os.path.join(ligand_dir, "*_lig.pdbqt")))

if args.target == "HTR1A":
    center_x = "102.972"
    center_y = "114.777"
    center_z = "108.403"
    size_x = "25.0"
    size_y = "25.0"
    size_z = "25.0"
    receptor = os.path.abspath("C:/vahidi/Results/7E2Y_H.pdbqt")
elif args.target == "DRD2":
    center_x = "9.002"
    center_y = "6.721"
    center_z = "-9.659"
    size_x = "25.0"
    size_y = "25.0"
    size_z = "25.0"
    receptor = os.path.abspath("C:/vahidi/Results/6LUQ_H.pdbqt")

out_dir = os.path.join(args.ligpath, "out_"+args.target)
os.makedirs(out_dir, exist_ok=True)

csv_path = os.path.join(out_dir, "vina_summary.csv")

# -------- resume: load already done ligands ----------
done = set()
if os.path.exists(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if row:
                done.add(row[0])

# -------- open CSV in append mode ----------
file_exists = os.path.exists(csv_path)
f = open(csv_path, "a", newline="")
writer = csv.writer(f)

if not file_exists:
    writer.writerow(["ligand", "affinity"])
    f.flush()

results = []

for lig_file in ligand_files:

    lig_file = os.path.abspath(lig_file).replace("\\", "/")
    base = os.path.basename(lig_file).replace("_lig.pdbqt", "")

    # -------- skip if already processed ----------
    if base in done:
        print("SKIP:", base)
        continue

    out_file = os.path.join(out_dir, f"{base}_out.pdbqt").replace("\\", "/")

    cmd = [
        "vina",
        "--receptor", receptor,
        "--ligand", lig_file,
        "--center_x", center_x,
        "--center_y", center_y,
        "--center_z", center_z,
        "--size_x", size_x,
        "--size_y", size_y,
        "--size_z", size_z,
        "--out", out_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("ERROR:", lig_file)
        print(result.stderr)
        continue

    affinity = extract_affinity(result.stdout)

    if affinity is None:
        print("NO AFFINITY:", lig_file)
        continue

    # -------- store immediately ----------
    writer.writerow([base, affinity])
    f.flush()          # مهم: همون لحظه روی دیسک
    os.fsync(f.fileno())

    print(base, affinity)

f.close()
