from deelea.config_environment import SETUP_ENVIRONMENT
import os

res_per_trial = 4
SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is not None:
    res_per_trial = int(SLURM_CPUS_PER_TASK) - 1
# SHOULD ALWAYS BE THE FIRST TO INCLUDE
SLURM_CPUS_PER_TASK, RESSOURCES_PER_TRIAL, SERVER = SETUP_ENVIRONMENT(
    RESSOURCES_PER_TRIAL=res_per_trial
)

from fastrenewables.timeseries.core import Timeseries
from fastai.callback.core import TrainEvalCallback
from fastcore.foundation import L
from fastai.learner import Learner
from fastai.metrics import rmse
from fastcore.xtras import delete
from fastrenewables.timeseries.learner import RenewableTimeseriesLearner
from fastrenewables.utils import filter_preds
from deelea.target import load_target_data, read_and_filter_source_results


SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = 3
SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)


from fastai.torch_core import no_random, to_np
from fastcore.transform import Pipeline
from fastrenewables.tabular.core import FilterDays, FilterMonths

import gridfs
import numpy as np
import pandas as pd

from deelea.mongo import (
    convert_mongo_df,
    get_mongo_db_connection,
    get_object_id_in_artefact,
    read_artifact,
    sacred_db_to_df,
)
from deelea.utils_eval import get_crps_error, get_error_dict
from deelea.utils_data import DataConfig, Datatype, FeaturesConfig, TLType

from functools import partial
from pathlib import Path
import time

from deelea.utils_sacred import (
    SacredConfig,
    add_sacred_artifact,
    delete_table,
    log_dict_as_scalar,
)

import argparse

from deelea.utils_training import (
    TargetTraining,
    run_stl_experiment,
)

from deelea.utils_models import ModelConfigTarget, SimilarityMeasure
from deelea.utils import (
    add_args,
    get_tmp_dir,
)

import pickle as pkl
import copy
from fastrenewables.models.ensembles import BayesModelAveraing
from fastrenewables.baselines import BayesLinReg
from fastrenewables.models.transfermodels import (
    LinearTransferModel,
    reduce_layers_tcn_model,
)

HOSTNAME = "alatar.ies.uni-kassel.de"
from sklearn.model_selection import train_test_split

import numpy as np


def setup_ann_target_model(source_model, is_ts=False):

    name_layers_or_function_to_remove = "layers"
    if is_ts:
        name_layers_or_function_to_remove = reduce_layers_tcn_model
    # ann = copy.deepcopy(ann)
    target_model = LinearTransferModel(
        source_model,
        num_layers_to_remove=1,
        name_layers_or_function_to_remove=name_layers_or_function_to_remove,
        use_original_weights=False,
        prediction_model=BayesLinReg(
            alpha=1,
            beta=1,
            empirical_bayes=False,
            use_fixed_point=True,
            n_iter=15,
        ),
        as_multivariate_target=False,
    )

    return target_model


def create_bma_model(
    models,
    cats,
    conts,
    targets,
    similarity_measure_type: SimilarityMeasure,
    is_ts=False,
):
    target_models = []

    for ann in models:
        target_model = setup_ann_target_model(ann, is_ts=is_ts)
        x_transformed = target_model(cats, conts)
        target_model = target_model.update(x_transformed, targets)

        target_models.append(target_model)

    bma_ensemble = BayesModelAveraing(
        source_models=target_models,
        rank_measure="none",
        weighting_strategy=similarity_measure_type.name.lower().replace("log", ""),
        n_best_models=-1,
        is_timeseries=is_ts,
    )

    bma_ensemble.fit_tensors(cats, conts, targets)

    return bma_ensemble


def bma_ann_target(
    data_config: DataConfig,
    model_config: ModelConfigTarget,
    resources_per_trial: dict,
    optuna_study,
    sacred_experiment,
    pre_loaded_models,
    **kwargs,
):

    target_training = TargetTraining()

    seasons, num_days_trainings, errors = [], [], []
    to_test = pkl.load(open(data_config.test_file, "rb"))

    if model_config.is_timeseries_model:
        to_test = Timeseries(to_test, splits=None)
        to_test.split = range(len(to_test))
    else:
        to_test.split = None

    dls_target = to_test.dataloaders(bs=len(to_test), shuffle=False)
    cats_test, conts_test, targets_test = dls_target.one_batch()

    for cur_season, months_to_train_on, num_days_training in target_training.params():
        if data_config.create_forecasts and num_days_training != 365:
            continue

        to_train_target = load_target_data(
            data_config, months_to_train_on, num_days_training
        )

        no_random(seed=42)
        if to_train_target is None:
            continue

        if model_config.is_timeseries_model:
            to_train_target.split = range(len(to_train_target))
        else:
            to_train_target.split = None

        dls_train = to_train_target.dataloaders(bs=len(to_train_target))
        cats_train, conts_train, targets_train = dls_train.one_batch()
        bma_ensemble = create_bma_model(
            pre_loaded_models,
            cats_train,
            conts_train,
            targets_train,
            model_config.similarity_measure_type,
            is_ts=model_config.is_timeseries_model,
        )

        # deterministic valid
        preds = bma_ensemble.forward(cats_train, conts_train).ravel()
        targets_train, preds = filter_preds(to_np(targets_train).ravel(), preds.ravel())
        error_dict_train = get_error_dict(targets_train, preds, prefix="valid_")
        # proba valid
        yhat, yhat_std = bma_ensemble.forward(cats_train, conts_train, pred_proba=True)
        crps_error_train = get_crps_error(
            targets_train, yhat, yhat_std, prefix="valid_"
        )

        # deterministic test
        preds = bma_ensemble.forward(cats_test, conts_test).ravel()
        targets, preds = filter_preds(to_np(targets_test).ravel(), preds.ravel())
        error_dict_test = get_error_dict(targets, preds, prefix="test_")
        # proba test
        yhat, yhat_std = bma_ensemble.forward(cats_test, conts_test, pred_proba=True)
        crps_error_test = get_crps_error(targets_test, yhat, yhat_std, prefix="test_")
        if data_config.create_forecasts:

            if model_config.is_timeseries_model:
                indexes = dls_target.train_ds.indexes.ravel()
            else:
                indexes = dls_target.train_ds.items.index.ravel()

            create_forecasts(
                data_config,
                targets_test,
                yhat,
                yhat_std,
                indexes,
                model_config.model_type.name,
            )

        error_dict = {
            **error_dict_test,
            **error_dict_train,
            **crps_error_train,
            **crps_error_test,
        }
        import pickle

        pkl.dump(
            bma_ensemble,
            open("test", "wb"),
        )

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


def create_forecasts(data_config, targets, yhat, yhat_std, indexes, model_type):
    dest_forecast_folder = f"../../doc/forecasts/{model_type.lower()}bma/"
    Path(dest_forecast_folder).mkdir(exist_ok=True, parents=True)
    yhat[yhat < 0] = 0
    yhat[yhat > 1.1] = 1.1
    df_forecast = pd.DataFrame(
        {
            "PowerGeneration": targets.ravel(),
            "Preds": yhat.ravel(),
            "PredsStd": yhat_std.ravel(),
        },
        index=indexes.ravel(),
    )
    df_forecast.to_csv(
        f"{dest_forecast_folder}/{data_config.data_file_name}.csv", sep=";"
    )


def pre_load_source_models(grid_fs, df_converted_filtered, data_type="pkl"):
    models, file_names = [], list(df_converted_filtered.data_file_name)
    for idx in range(len(df_converted_filtered)):
        object_id = get_object_id_in_artefact(
            df_converted_filtered.iloc[idx].artifacts, filter="model"
        )
        model = read_artifact(grid_fs, object_id, data_type)
        if data_type == "learner":
            model = model.model
        models.append(model)
    return models, file_names


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
        help="Model type that is used as ensemble member. Either mlp or tcn",
        default="all",
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

    if args.model_type == "all":
        model_types = [
            "tcn",
            "mlp",
        ]
    else:
        model_types = [args.model_type]
    for model_type in model_types:
        for similarity_measure in similarity_measures:

            model_config = ModelConfigTarget(
                args.embedding_type,
                args.full_bayes,
                model_type,
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

                experiment_name = f"source{args.model_architecture}{model_type}"
                db_name = f"source{model_type}"
                sacred_config_models = SacredConfig(
                    experiment_name=experiment_name,
                    db_name=db_name,
                    mongo_observer=True,
                    hostname=HOSTNAME,
                    port="27031",
                )

                grid_fs, df_converted_filtered = read_and_filter_source_results(
                    data_config, sacred_config_models, is_mtl=False
                )
                if model_config.smoke_test and not args.create_forecasts:
                    df_converted_filtered = df_converted_filtered.iloc[0:5]

                pre_loaded_models, _ = pre_load_source_models(
                    grid_fs, df_converted_filtered, data_type="learner"
                )
                ts = data_config.target_splits
                ts = [Path(t).stem for t in ts]
                for d in df_converted_filtered.data_file_name:
                    if d in ts:
                        raise ValueError(f"{d} should not be in source models")

                port = "27031"
                if model_config.smoke_test:
                    port = "27029"

                experiment_name = f"ensemble_bma_ann_{model_type}"
                db_name = f"ensemble_bma_ann"
                sacred_config = SacredConfig(
                    experiment_name=experiment_name,
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
                        bma_ann_target, pre_loaded_models=pre_loaded_models
                    ),
                    use_optuna=False,
                    check_db_entries=False,
                )

                time.sleep(2)
