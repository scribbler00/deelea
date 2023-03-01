from pathlib import Path
from fastrenewables.utils import filter_preds
from deelea.config_environment import SETUP_ENVIRONMENT
from deelea.target import load_target_data

# SHOULD ALWAYS BE THE FIRST TO INCLUDE
SLURM_CPUS_PER_TASK, RESSOURCES_PER_TRIAL, SERVER = SETUP_ENVIRONMENT(
    RESSOURCES_PER_TRIAL=1
)

import os
from time import sleep
from deelea.utils_sacred import (
    SacredConfig,
    add_sacred_artifact,
    delete_table,
    log_dict_as_scalar,
)

from fastcore.foundation import L
import pandas as pd
from deelea.utils_eval import get_error_dict
from fastai.torch_core import no_random
from fastcore.transform import Pipeline
from fastrenewables.tabular.core import FilterDays, FilterMonths


import argparse

from deelea.utils_training import (
    TargetTraining,
    run_stl_experiment,
    sacred_experiment,
)

from deelea.utils_data import DataConfig, TLType
from deelea.utils_models import (
    ModelArchitecture,
    ModelConfig,
)
from deelea.utils import (
    add_args,
    get_tmp_dir,
)
from deelea.target import load_target_data
import pickle as pkl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def gbrt_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    resources_per_trial: dict,
    optuna_study,
    sacred_experiment,
    **kwargs,
):
    target_training = TargetTraining()
    seasons, num_days_trainings, errors = [], [], []
    for cur_season, months_to_train_on, num_days_training in target_training.params():
        if data_config.create_forecasts and num_days_training != 365:
            continue
        to_train = load_target_data(data_config, months_to_train_on, num_days_training)

        no_random(seed=42)
        if to_train is None:
            continue

        model = GradientBoostingRegressor()

        grid = GridSearchCV(
            estimator=model,
            param_grid=model_config.get_optimization_config(),
            cv=3,
            n_jobs=-1,
            refit=True,
        )
        grid = grid.fit(to_train.conts.values, to_train.ys.values.ravel())

        to_test = pkl.load(open(data_config.test_file, "rb"))

        test_indices = list(to_test.items.index)
        for cur_index in to_train.items.index:
            if cur_index in test_indices:
                raise ValueError("This should not happen.")
        preds = grid.predict(to_test.conts.values)
        targets = to_test.ys.values.ravel()
        targets, preds = filter_preds(targets, preds)

        error_dict = get_error_dict(targets, preds, prefix="test_")

        seasons += [cur_season]
        num_days_trainings += [num_days_training]
        errors += [error_dict]

        log_dict_as_scalar(
            sacred_experiment,
            {
                **{"season": cur_season, "num_days_training": num_days_training},
                **error_dict,
            },
        )

        model_folder = Path(data_config.tmp_dir)
        model_folder.mkdir(exist_ok=True, parents=True)

        file_name = f"{model_config.model_type.name}_model_season_{cur_season}_ndays_{num_days_training}.jbl"
        add_sacred_artifact(grid, str(model_folder / file_name), sacred_experiment)

        if data_config.create_forecasts:
            indexes = to_test.items.index.ravel()

            create_forecasts(
                data_config,
                targets,
                preds,
                indexes,
                model_config,
            )

        print("********************************")
        print(cur_season, num_days_training, error_dict)
        print("********************************")

    result = pd.DataFrame(errors)
    result["Season"] = seasons
    result["NumDaysTraining"] = num_days_trainings
    result["FoldID"] = data_config.fold_id
    add_sacred_artifact(result, "gbrt_results.csv", sacred_experiment)

    return result.to_json()


def create_forecasts(data_config, targets, yhat, indexes, model_config):
    dest_forecast_folder = f"../../doc/forecasts/gbrt/"
    Path(dest_forecast_folder).mkdir(exist_ok=True, parents=True)
    yhat[yhat < 0] = 0
    yhat[yhat > 1.1] = 1.1
    df_forecast = pd.DataFrame(
        {
            "PowerGeneration": targets.ravel(),
            "Preds": yhat.ravel(),
        },
        index=indexes.ravel(),
    )
    df_forecast.to_csv(
        f"{dest_forecast_folder}/{data_config.data_file_name}.csv", sep=";"
    )


if __name__ == "__main__":
    DEBUG = False
    parser = argparse.ArgumentParser()

    add_args(parser)
    tmp_dir = get_tmp_dir(SERVER)

    parser.add_argument(
        "--model_architecture",
        help="Single-task learning [stl] or multi-task learning [mtl]",
        default="stl",
    )
    parser.add_argument(
        "--create_forecasts",
        help="Create forecasts or not.",
        default=False,
    )
    model_type = "gbrt"

    args = parser.parse_args()

    model_config = ModelConfig(
        args.embedding_type,
        args.full_bayes,
        model_type,
        args.model_architecture,
        debug=DEBUG,
        smoke_test=args.smoke_test,
    )
    port = "27031"
    if model_config.smoke_test:
        port = "27029"

    sacred_config = SacredConfig(
        experiment_name="gbrttarget",
        db_name="targetbaselinesmodel",
        # db_name="test",
        mongo_observer=True,
        hostname="alatar",
        port=port,
    )

    Path("log").mkdir(exist_ok=True)
    sleep(1)
    if args.fold_id == "all":
        folds_ids = ["0", "1", "2", "3", "4"]
    else:
        folds_ids = [args.fold_id]

    for fold_id in folds_ids:

        data_config = DataConfig(
            fold_id,
            args.result_folder,
            model_config,
            SERVER,
            tl_type=TLType.TARGET,
            create_forecasts=args.create_forecasts,
        )

        run_stl_experiment(
            sacred_config,
            data_config,
            model_config,
            {"cpu": 1},
            train_function=gbrt_training,
            use_optuna=False,
            check_db_entries=False,
        )
