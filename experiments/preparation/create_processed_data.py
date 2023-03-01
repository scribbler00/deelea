import os
from pathlib import Path

from fastai.data.block import CategoryBlock, RegressionBlock
from deelea.utils_models import ModelArchitecture

os.environ["CUDA_VISIBLE_DEVICES"] = ""
SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = "4"
    SERVER = False
else:
    SERVER = True

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)
torch.set_num_threads(SLURM_CPUS_PER_TASK)

from deelea.utils_data import Datatype, FeaturesConfig, source_data
import tqdm
import argparse, json
import pandas as pd
import json
import pickle as pkl

from deelea.utils import (
    add_args,
    dump_dict_as_json,
    get_blacklist,
    str_to_path,
)

from fastrenewables.tabular.model import EmbeddingType


def create_source_files_mtl(result_folder, features_config, smoke_test):
    with open(f"{result_folder}/splits.json") as f:
        splits = json.load(f)

    result_folder = result_folder / "mtl_files"
    result_folder.mkdir(exist_ok=True, parents=True)

    for fold_id in tqdm.tqdm(list(splits.keys())):

        cur_splits = splits[fold_id]
        source_files = cur_splits["sources"]

        # TODO: move to separate function?
        final_source_files = []
        blacklist = get_blacklist()
        for file in source_files:
            if Path(file).stem in blacklist:
                print("Skipped file", file.stem)
                continue
            else:
                final_source_files.append(file)
        source_files = final_source_files

        if smoke_test:
            final_source_files = final_source_files[0:3]

        to_train, to_test, task_id_to_name = source_data(
            final_source_files,
            features_config,
            y_block=RegressionBlock,
            is_mtl_dataset=True,
            limit_data=features_config.datatype
            not in [Datatype.PVOPEN, Datatype.WINDOPEN],
        )

        pkl.dump(
            to_train, open(result_folder / f"fold_{int(fold_id):02d}_train.pkl", "wb")
        )
        pkl.dump(
            to_test, open(result_folder / f"fold_{int(fold_id):02d}_test.pkl", "wb")
        )

        dump_dict_as_json(
            task_id_to_name, result_folder / f"fold_{int(fold_id):02d}_taskIdToName.pkl"
        )


def create_source_files_stl(result_folder, data_folder, features_config, smoke_test):
    result_folder = result_folder / "stl_files"
    result_folder.mkdir(exist_ok=True, parents=True)
    single_files = [f for f in data_folder.ls() if f.suffix in [".h5", ".csv"]]

    if smoke_test:
        single_files = single_files[1:3]
    blacklist = get_blacklist()
    for file in tqdm.tqdm(single_files):

        if file.stem in blacklist:
            print("Skipped file", file.stem)
            continue

        to_train, to_test, _ = source_data(
            file, features_config, y_block=RegressionBlock, is_mtl_dataset=False
        )
        pkl.dump(to_train, open(result_folder / f"{file.stem}_train.pkl", "wb"))
        pkl.dump(to_test, open(result_folder / f"{file.stem}_test.pkl", "wb"))


#
#%%
def main(parser):
    args = parser.parse_args()
    data_folder = str_to_path(args.data_folder)
    smoke_test = True if str(args.smoke_test).lower() == "true" else False

    result_folder = str_to_path(args.result_folder)
    data_type = Datatype.datafolder_to_datatype(result_folder)

    features_config_stl = FeaturesConfig(data_type, ModelArchitecture.STL)
    create_source_files_stl(result_folder, data_folder, features_config_stl, smoke_test)

    features_config_mtl = FeaturesConfig(data_type, ModelArchitecture.MTL)
    create_source_files_mtl(result_folder, features_config_mtl, smoke_test)


if __name__ == "__main__":
    DEBUG = True
    parser = argparse.ArgumentParser()
    add_args(parser)
    main(parser)

# %%
