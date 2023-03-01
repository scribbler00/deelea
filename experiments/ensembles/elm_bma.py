import os

SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = 3
SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)

from deelea.config_environment import SETUP_ENVIRONMENT

# SHOULD ALWAYS BE THE FIRST TO INCLUDE
SLURM_CPUS_PER_TASK, RESSOURCES_PER_TRIAL, SERVER = SETUP_ENVIRONMENT(
    RESSOURCES_PER_TRIAL=3  # SLURM_CPUS_PER_TASK - 1
)


import numpy as np
import pandas as pd
from functools import partial
from pathlib import Path
import time
import argparse
import pickle as pkl
import copy
from fastai.torch_core import no_random
from fastrenewables.utils import filter_preds

from deelea.mongo import (
    get_object_id_in_artefact,
    read_artifact,
)
from deelea.utils_eval import get_crps_error, get_error_dict
from deelea.utils_data import DataConfig, Datatype, FeaturesConfig, TLType
from deelea.target import load_target_data, read_and_filter_source_results
from deelea.utils_sacred import (
    SacredConfig,
    add_sacred_artifact,
    delete_table,
    log_dict_as_scalar,
)
from deelea.utils_training import (
    TargetTraining,
    run_stl_experiment,
)
from deelea.utils_models import ModelConfigTarget, SimilarityMeasure
from deelea.utils import (
    add_args,
    get_tmp_dir,
)

from fastrenewables.models.ensembles import BayesModelAveraing

HOSTNAME = "alatar.ies.uni-kassel.de"


def bma_elm_target(
    data_config: DataConfig,
    model_config: ModelConfigTarget,
    resources_per_trial: dict,
    optuna_study,
    sacred_experiment,
    pre_loaded_elms,
    **kwargs,
):

    target_training = TargetTraining()

    seasons, num_days_trainings, errors = [], [], []
    for cur_season, months_to_train_on, num_days_training in target_training.params():
        if data_config.create_forecasts and num_days_training != 365:
            continue

        to_train_target = load_target_data(
            data_config, months_to_train_on, num_days_training
        )

        no_random(seed=42)
        if to_train_target is None:
            continue

        to_train_target.split = None
        dls_train = to_train_target.dataloaders()
        bma_ensemble = create_bma_model(
            pre_loaded_elms,
            dls_train,
            model_config.similarity_measure_type,
        )
        # valid deterministic
        preds = bma_ensemble.predict(dls_train).ravel()
        targets = dls_train.train_ds.ys.values.ravel()
        targets, preds = filter_preds(targets, preds)
        error_dict_train = get_error_dict(targets, preds, prefix="valid_")
        # valid proba
        yhat, yhat_std = bma_ensemble.predict_proba(dls_train)
        crps_error_valid = get_crps_error(targets, yhat, yhat_std, prefix="valid_")

        # test deterministic
        to_test = pkl.load(open(data_config.test_file, "rb"))
        to_test.split = None
        dls_target = to_test.dataloaders()
        preds = bma_ensemble.predict(dls_target).ravel()
        targets = dls_target.train_ds.ys.values.ravel()
        targets, preds = filter_preds(targets, preds)
        # test proba
        yhat, yhat_std = bma_ensemble.predict_proba(dls_target)
        crps_error_test = get_crps_error(targets, yhat, yhat_std, prefix="test_")
        if data_config.create_forecasts:
            create_forecasts(data_config, targets, yhat, yhat_std, to_test)

        error_dict = get_error_dict(targets, preds, prefix="test_")
        error_dict = {
            **error_dict,
            **error_dict_train,
            **crps_error_valid,
            **crps_error_test,
        }

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

        print("********************************")
        print(cur_season, num_days_training, error_dict)
        print("********************************")

    result = pd.DataFrame(errors)
    result["Season"] = seasons
    result["NumDaysTraining"] = num_days_trainings
    result["FoldID"] = data_config.fold_id
    add_sacred_artifact(result, "results.csv", sacred_experiment)

    return result.to_json()


def create_forecasts(data_config, targets, yhat, yhat_std, to_test):
    dest_forecast_folder = "../../doc/forecasts/elmbma"
    Path(dest_forecast_folder).mkdir(exist_ok=True, parents=True)
    yhat[yhat < 0] = 0
    yhat[yhat > 1.1] = 1.1
    df_forecast = pd.DataFrame(
        {
            "PowerGeneration": targets.ravel(),
            "Preds": yhat.ravel(),
            "PredsStd": yhat_std.ravel(),
        },
        index=to_test.items.index.ravel(),
    )
    df_forecast.to_csv(
        f"{dest_forecast_folder}/{data_config.data_file_name}.csv", sep=";"
    )


def pre_load_source_elms(grid_fs, df_converted_filtered):
    elms, file_names = [], list(df_converted_filtered.data_file_name)
    for idx in range(len(df_converted_filtered)):
        object_id = get_object_id_in_artefact(
            df_converted_filtered.iloc[idx].artifacts, filter="model"
        )
        elm = read_artifact(grid_fs, object_id, "pkl")
        elms.append(elm)
    return elms, file_names


def create_bma_model(
    models,
    dls,
    similarity_measure_type: SimilarityMeasure,
):
    X_train_complete = dls.train_ds.conts.values
    y_train_complete = dls.train_ds.ys.values.ravel()

    updated_models = []
    for elm in models:

        alpha = elm.alpha
        elm = setup_elm_target_model(elm, alpha=alpha)
        elm = elm.update(X_train_complete, y_train_complete.ravel())
        updated_models.append(elm)

    bma_ensemble = BayesModelAveraing(
        source_models=updated_models,
        rank_measure="none",
        weighting_strategy=similarity_measure_type.name.lower().replace("log", ""),
        n_best_models=-1,
    )
    cats, conts, targets = bma_ensemble.conversion_to_tensor(dls.valid_ds)
    bma_ensemble.fit_tensors(cats, conts, targets)
    # bma_ensemble.fit(dls)

    return bma_ensemble


def setup_elm_target_model(elm, alpha):
    elm = copy.deepcopy(elm)
    elm.alpha = alpha

    return elm


if __name__ == "__main__":
    DEBUG = True
    parser = argparse.ArgumentParser()

    add_args(parser)
    tmp_dir = get_tmp_dir(SERVER)

    parser.add_argument(
        "--model_architecture",
        help="Single-task learning [stl] or multi-task learning [mtl]",
        default="stl",
    )

    parser.add_argument(
        "--model_type",
        help="Model type that is used as ensemble member. Only ELM is supported.",
        default="elm",
    )

    parser.add_argument(
        "--similarity_measure",
        help="The kind on which the we create the ensemble. Either logevidence, posterior,  or all",
        default="all",
    )
    parser.add_argument(
        "--create_forecasts",
        help="Create forecasts or not.",
        default=False,
    )

    args = parser.parse_args()

    if args.create_forecasts and not args.smoke_test:
        raise ValueError
    elif not args.smoke_test:
        args.create_forecasts = False

    if args.similarity_measure == "all":
        similarity_measures = ["uncertainty"]
    else:
        similarity_measures = [args.similarity_measure]

    for similarity_measure in similarity_measures:

        model_config = ModelConfigTarget(
            args.embedding_type,
            args.full_bayes,
            args.model_type,
            args.model_architecture,
            debug=DEBUG,
            smoke_test=args.smoke_test,
            similarity_measure=similarity_measure,
            ensemble_type="bma",
        )
        Path("./log").mkdir(exist_ok=True)
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
                tl_type=TLType.SOURCE,
                create_forecasts=args.create_forecasts,
            )

            sacred_config_models = SacredConfig(
                experiment_name="ELMSource",
                db_name="sourceelm",
                mongo_observer=True,
                hostname=HOSTNAME,
                port="27031",
            )

            grid_fs, df_converted_filtered = read_and_filter_source_results(
                data_config, sacred_config_models, is_mtl=False
            )
            if model_config.smoke_test:
                df_converted_filtered = df_converted_filtered.iloc[0:5]

            pre_loaded_models, _ = pre_load_source_elms(grid_fs, df_converted_filtered)
            ts = data_config.target_splits
            ts = [Path(t).stem for t in ts]
            for d in df_converted_filtered.data_file_name:
                if d in ts:
                    raise ValueError(f"{d} should not be in source models")

            port = "27031"
            if model_config.smoke_test:
                port = "27029"

            db_name = "ensemble_bma_elm"
            sacred_config = SacredConfig(
                experiment_name="ensemble_bma_elm",
                db_name=db_name,
                mongo_observer=True,
                hostname=HOSTNAME,
                port=port,
            )
            # delete_table(sacred_config, db_name)
            run_stl_experiment(
                sacred_config,
                data_config,
                model_config,
                {"cpu": 1},
                train_function=partial(
                    bma_elm_target, pre_loaded_elms=pre_loaded_models
                ),
                use_optuna=False,
                check_db_entries=False,
            )

            time.sleep(2)
