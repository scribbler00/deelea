import os
import sys, json
from fastrenewables.tabular.core import get_samples_per_day, read_files
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import tqdm

sys.path.append("../tlt/")
sys.path.append("./tlt/")
from deelea.utils import str_to_path, get_blacklist


SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SERVER = False
else:
    SERVER = True

if SERVER:
    folders = [
        "/mnt/work/transfer/data/DAF_ICON_Synthetic_Wind_Power_processed/",
        "/mnt/work/transfer/data/enercast/wind/",
        "/mnt/work/transfer/data/enercast/pv/",
    ]
    base_folder = "/mnt/work/transfer/similarity/"
else:
    folders = [
        # "/home/scribbler/data/DAF_ICON_Synthetic_Wind_Power_processed/",
        # "/home/scribbler/data/DAF_ICON_Synthetic_PV_Power_processed/",
        # "/home/scribbler/data/enercast/wind/",
        # "/home/scribbler/data/enercast/pv/",
        "/home/scribbler/data/prophesy-data/PVSandbox2015/",
        "/home/scribbler/data/prophesy-data/WindSandbox2015/",
    ]

    base_folder = "./results/"


names = [
    # "WINDSYN",
    # "PVSYN"
    # "WINDREAL",
    # "PVREAL",
    "OPENPV",
    "OPENWIND",
]

for n in names:
    Path(base_folder + n).mkdir(exist_ok=True, parents=True)


def create_splits(folder, res_folder):
    data_folder = str_to_path(folder)
    files = data_folder.ls()
    # files = [f for f in files if f.suffix == ".h5" or f.suffix == ".csv"]
    files_new = []
    blacklist = get_blacklist()
    for file in tqdm.tqdm(files):
        file_string = str(file)
        if file.stem in blacklist:
            print("Skipped:", file_string)
            continue

        if (str(file).endswith(".csv") or str(file).endswith(".h5")) and (
            (
                (
                    "solar" not in file_string.lower()
                    and "wind" in str(res_folder).lower()
                )
                or ("solar" in file_string.lower() and "pv" in str(res_folder).lower())
            )
            or "gefcom" not in str(folder).lower()
        ):
            files_new.append(file)
    files = files_new
    print(f"we have {len(files)} files for folder {res_folder}")
    n_splits = 5
    if len(files) < n_splits:
        n_splits = len(files)
    kf = KFold(n_splits=n_splits, shuffle=True)
    splits = dict()
    for run_id, (source_file_ids, target_file_ids) in enumerate(kf.split(files)):
        source_files = [str(f) for f in np.array(files)[source_file_ids]]
        target_files = [str(f) for f in np.array(files)[target_file_ids]]
        splits[run_id] = {"sources": source_files, "targets": target_files}

    p = str_to_path(f"{base_folder}/{res_folder}")

    p.mkdir(exist_ok=True, parents=True)
    with open(f"{base_folder}/{res_folder}/splits.json", "w") as f:
        json.dump(splits, f)


for folder_input, folder_ouput in zip(folders, names):
    create_splits(folder_input, folder_ouput)
