from deelea.config_environment import SETUP_ENVIRONMENT

# SHOULD ALWAYS BE THE FIRST TO INCLUDE
SLURM_CPUS_PER_TASK, RESSOURCES_PER_TRIAL, SERVER = SETUP_ENVIRONMENT(
    RESSOURCES_PER_TRIAL=4  # SLURM_CPUS_PER_TASK - 1
)
import argparse
from pathlib import Path
import time
from fastai.torch_core import to_np
from fastrenewables.utils import filter_preds

import gridfs
from deelea.config_environment import SETUP_ENVIRONMENT
from deelea.forecasts import (
    create_single_df,
    get_indexes_and_targets,
    get_prediction_setup,
    get_predictions,
)
from deelea.mongo import (
    get_model_from_artifacts,
    get_results_of_db,
)
from deelea.utils_sacred import SacredConfig

# %%
import numpy as np
import pandas as pd
import pickle as pkl
from deelea.utils import add_args
from deelea.utils_data import FeaturesConfig, DataConfig, TLType, read_renewables_data
from deelea.utils_models import ModelArchitecture, ModelConfig
from tqdm import tqdm


# %%
HOSTNAME = "alatar.ies.uni-kassel.de"
sacred_config_tcn_models = SacredConfig(
    experiment_name=f"sourcestltcn",
    db_name=f"sourcetcn",
    mongo_observer=True,
    hostname=HOSTNAME,
    port="27031",
)

sacred_config_mlp_models = SacredConfig(
    experiment_name=f"sourcestlmlp",
    db_name=f"sourcemlp",
    mongo_observer=True,
    hostname=HOSTNAME,
    port="27031",
)


def create_stl_forecasts(
    data_config: DataConfig, model_config: ModelConfig, resources_per_trial: dict
):
    dest_folfer = "/media/scribbler/1B5D7DBC354FB150/forecasts_phd/stl/"
    if model_config.model_architecture == ModelArchitecture.MTL:
        raise NotImplementedError

    train_file = data_config.train_file
    test_file = data_config.test_file
    # assure that we get all the data
    to_train = read_renewables_data(train_file, model_config, is_test_data=True)
    to_test = read_renewables_data(test_file, model_config, is_test_data=True)

    sacred_config_model, fast_prediction = get_prediction_setup(
        model_config,
        sacred_config_mlp_models,
        sacred_config_tcn_models,
    )
    (
        indexes_train,
        indexes_test,
        targets_train,
        targets_test,
    ) = get_indexes_and_targets(model_config.model_type, to_train, to_test)

    df_power = create_single_df(
        targets_train,
        targets_test,
        indexes_train,
        indexes_test,
        "PowerGeneration",
        include_test_flag=True,
    )

    df_res, db = get_results_of_db(model_config, data_config, sacred_config_model)
    df_res = df_res[df_res.Status == "COMPLETED"]
    grid_fs = gridfs.GridFS(db)

    # take only those with matching datatype
    df_res = df_res[df_res.result_folder == data_config.result_folder.name]

    df_results = []

    file_name = f"{dest_folfer}/{data_config.data_type.name}/{data_config.data_file_name}_{model_config.model_architecture.name.lower()}_{model_config.model_type.name.lower()}.csv"
    file_name = Path(file_name)
    file_name.parent.mkdir(exist_ok=True, parents=True)

    if file_name.exists():
        print(file_name, "exists already, skipping it.")
        return

    for id_cur_row in tqdm(df_res.index):

        cur_df_row = df_res.loc[id_cur_row]
        model = get_model_from_artifacts(grid_fs, cur_df_row)
        preds_train, preds_test = get_predictions(
            to_train, to_test, fast_prediction, model, return_targets=False
        )
        df_preds = create_single_df(
            preds_train,
            preds_test,
            indexes_train,
            indexes_test,
            cur_df_row.data_file_name,
        )
        df_results.append(df_preds)

    df_results = pd.concat(df_results + [df_power], axis=1).sort_index(axis=0)

    df_results.to_csv(file_name, sep=";")


if __name__ == "__main__":
    DEBUG = False
    parser = argparse.ArgumentParser()

    add_args(parser)

    parser.add_argument(
        "--model_architecture",
        help="Single-task learning [stl]",
        default="stl",
    )
    base_folder = "/media/scribbler/1B5D7DBC354FB150/phd-data/"
    args = parser.parse_args()
    for data_folder in [
        "WINDREAL",
        "WINDOPEN",
        "WINDSYN",
        "PVREAL",
        "PVOPEN",
        "PVSYN",
    ]:
        for model_type in ["tcn", "mlp"]:
            model_config = ModelConfig(
                "normal",
                "false",
                model_type,
                model_architecture=args.model_architecture,
                smoke_test=args.smoke_test,
            )

            data_config = DataConfig(
                fold_id=0,
                result_folder=base_folder + data_folder,
                model_config=model_config,
                SERVER=SERVER,
                tl_type=TLType.SOURCE,
            )

            features_config = FeaturesConfig(
                data_config.data_type, model_config.model_architecture
            )

            for file_name in data_config.get_all_files():
                print(model_type, file_name.stem)
                data_config.set_file_name(
                    file_name.stem.replace("_train", "").replace("_test", "")
                )

                create_stl_forecasts(data_config, model_config, None)
