import os
from deelea.config_environment import SETUP_ENVIRONMENT

SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = 3
SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)

# SHOULD ALWAYS BE THE FIRST TO INCLUDE
SLURM_CPUS_PER_TASK, RESSOURCES_PER_TRIAL, SERVER = SETUP_ENVIRONMENT(
    RESSOURCES_PER_TRIAL=4  # SLURM_CPUS_PER_TASK - 1
)
import os
import torch
import gridfs
import numpy as np
import pandas as pd
import pickle as pkl
import copy
import argparse

from functools import partial
from pathlib import Path
import time
from fastcore.foundation import L
from fastai.callback.core import TrainEvalCallback
from fastai.learner import Learner
from fastai.metrics import rmse
from fastai.optimizer import SGD, Adam
from fastai.torch_core import no_random, to_np

from fastrenewables.utils_pytorch import freeze
from fastrenewables.timeseries.learner import RenewableTimeseriesLearner
from fastrenewables.utils import filter_preds
from fastrenewables.timeseries.core import Timeseries
from fastrenewables.models.ensembles import (
    CSGE,
    LocalErrorPredictor,
    BayesModelAveraing,
    rank_by_evidence,
    simple_local_error_estimator,
)
from fastrenewables.baselines import BayesLinReg
from fastrenewables.models.transfermodels import (
    LinearTransferModel,
    reduce_layers_tcn_model,
)
from fastrenewables.utils_pytorch import unfreeze
from fastrenewables.models.ensembles import TorchSklearnWrapper
from deelea.mongo import (
    convert_mongo_df,
    convert_py_reduce_to_type,
    get_mongo_db_connection,
    get_object_id_in_artefact,
    read_artifact,
    sacred_db_to_df,
)
from deelea.utils_eval import get_error_dict
from deelea.utils_data import DataConfig, Datatype, FeaturesConfig, TLType
from deelea.target import (
    load_target_data,
    read_and_filter_source_results,
    setup_and_fit_linear_target_model,
)
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

import warnings

warnings.filterwarnings("ignore")
HOSTNAME = "alatar.ies.uni-kassel.de"
from sklearn.model_selection import train_test_split


def create_target_learner(learner_type, dls_train, target_model):
    learner = learner_type(
        dls_train, copy.deepcopy(target_model), metrics=rmse, opt_func=Adam
    )
    learner.cbs = L(c for c in learner.cbs if type(c) == TrainEvalCallback)
    return learner


def unfreeze_final_layer(model_config, csge_ensemble: CSGE):

    argsortid = csge_ensemble.global_weights.ravel().argsort()[-1]
    blended_model = csge_ensemble.source_models[argsortid]

    if model_config.is_timeseries_model:
        unfreeze(blended_model.layers.temporal_blocks[-1])
    else:
        unfreeze(blended_model.layers[-1])


def csge_ann_target(
    data_config: DataConfig,
    model_config: ModelConfigTarget,
    resources_per_trial: dict,
    optuna_study,
    sacred_experiment,
    pre_loaded_models,
    df_results_gbrt=None,
    gridfs_gbrt=None,
    **kwargs,
):
    if df_results_gbrt is not None:
        df_results_gbrt = copy.copy(df_results_gbrt)
        df_results_gbrt = df_results_gbrt[
            df_results_gbrt.data_file_name == data_config.data_file_name
        ]

    target_training = TargetTraining()

    seasons, num_days_trainings, errors = [], [], []
    to_test = pkl.load(open(data_config.test_file, "rb"))

    if model_config.is_timeseries_model:

        to_test = Timeseries(to_test, splits=None)
        to_test.split = range(len(to_test))
    else:
        to_test.split = None

    # dls_target = to_test.dataloaders(bs=len(to_test))
    dls_target = to_test.dataloaders()

    from fastrenewables.tabular.learner import RenewableLearner, convert_to_tensor
    from fastrenewables.timeseries.learner import (
        RenewableTimeseriesLearner,
        convert_to_tensor_ts,
    )

    for idx, (cur_season, months_to_train_on, num_days_training) in enumerate(
        target_training.params()
    ):
        if data_config.create_forecasts and num_days_training != 365:
            continue

        to_train_target = load_target_data(
            data_config, months_to_train_on, num_days_training
        )
        no_random(seed=42)
        if to_train_target is None:
            continue

        gbrt_model = None
        if df_results_gbrt is not None:
            expected_model_name = f"/mnt/work/transfer/tmp_result/GBRT_model_season_{cur_season}_ndays_{num_days_training}"
            mask = df_results_gbrt.iloc[0].artifacts.name == expected_model_name
            gbrt_name = df_results_gbrt.iloc[0].artifacts.name[mask]
            object_id = df_results_gbrt.iloc[0].artifacts.file_id[mask]
            gbrt_model = read_artifact(gridfs_gbrt, object_id.iloc[0], "jbl")
            if (
                f"season_{cur_season}_ndays_{num_days_training}"
                not in gbrt_name.iloc[0]
            ):
                print(
                    f"Mismatch of expected GBRT model name {gbrt_name} for season {cur_season} and ndays {num_days_training}."
                )
                continue

        if model_config.is_timeseries_model:
            bs = max(len(to_train_target.train) / 10, 1)
            bs = int(bs)
            learner_type = RenewableTimeseriesLearner
            convert_dl_to_tensor = convert_to_tensor_ts
        else:
            learner_type = RenewableLearner
            convert_dl_to_tensor = convert_to_tensor

            bs = max(len(to_train_target.train) / 10, 1)
            bs = int(bs)

        if model_config.is_timeseries_model:
            to_train_target.split = range(len(to_train_target))
        else:
            to_train_target.split = None

        dls_train = to_train_target.dataloaders(bs=len(to_train_target), num_workers=0)

        ts_length = 96
        if data_config.data_type == Datatype.PVOPEN:
            ts_length = 24

        etas = {"eta_t": 2, "eta_g": 2, "eta_l": 2}
        csge_ensemble = create_csge_model(
            pre_loaded_models,
            dls_train,
            model_config.similarity_measure_type,
            is_ts=model_config.is_timeseries_model,
            etas=etas,
            ts_length=ts_length,
            csge_type=model_config.ensemble_type,
            convert_dl_to_tensor=convert_dl_to_tensor,
            gbrt_model=gbrt_model,
        )

        lrs = [0.5, 0.1, 1e-3, 1e-5]
        if "nofit" in model_config.ensemble_type:
            lrs = [1]
        best_val_error = {"valid_rmse": 1e16}
        for eta_0 in [1, 2]:
            # for eta_0 in [2, 4]:
            for lr in lrs:
                tmp_csge_ensemble = copy.deepcopy(csge_ensemble)
                if "blended" in model_config.ensemble_type:
                    unfreeze_final_layer(model_config, tmp_csge_ensemble)

                with torch.no_grad():
                    tmp_csge_ensemble.eta_global = torch.nn.Parameter(
                        torch.Tensor([eta_0])
                    )
                    tmp_csge_ensemble.eta_local = torch.nn.Parameter(
                        torch.Tensor([eta_0])
                    )
                    tmp_csge_ensemble.eta_time = torch.nn.Parameter(
                        torch.Tensor([eta_0])
                    )
                learner = create_target_learner(
                    learner_type, dls_train, tmp_csge_ensemble
                )
                if "nofit" not in model_config.ensemble_type:
                    learner.fit(1, lr=lr)
                preds, targets = learner.predict(ds_idx=1)
                error_dict_train = get_error_dict(targets, preds, prefix="valid_")

                if error_dict_train["valid_rmse"] < best_val_error["valid_rmse"]:
                    best_val_error = error_dict_train
                    best_learner = learner

        best_etas = {
            "eta_t": to_np(best_learner.model.eta_local)[0],
            "eta_g": to_np(best_learner.model.eta_global)[0],
            "eta_l": to_np(best_learner.model.eta_time)[0],
        }
        # print("after", best_etas)

        preds, targets = best_learner.predict(test_dl=dls_target)
        error_dict = get_error_dict(targets, preds, prefix="test_")

        if data_config.create_forecasts:

            if model_config.is_timeseries_model:
                indexes = dls_target.train_ds.indexes.ravel()
            else:
                indexes = dls_target.train_ds.items.index.ravel()

            create_forecasts(
                data_config,
                targets,
                preds,
                indexes,
                model_config,
            )

        error_dict = {**error_dict, **best_val_error}

        seasons += [cur_season]
        num_days_trainings += [num_days_training]
        errors += [error_dict]

        log_dict_as_scalar(
            sacred_experiment,
            {
                **{"season": cur_season, "num_days_training": num_days_training},
                **error_dict,
                **best_etas,
            },
        )

        print("********************************")
        print(cur_season, num_days_training, error_dict, best_etas)
        print("********************************")

    result = pd.DataFrame(errors)
    result["Season"] = seasons
    result["NumDaysTraining"] = num_days_trainings
    result["FoldID"] = data_config.fold_id
    add_sacred_artifact(result, "results.csv", sacred_experiment)

    return result.to_json()


def create_forecasts(data_config, targets, yhat, indexes, model_config):
    dest_forecast_folder = f"../../doc/forecasts/{model_config.model_type.name.lower()}_{model_config.ensemble_type}/"
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


def fit_linear_models(models, cats, conts, targets, is_ts, num_layers_to_remove=1):
    linear_target_models = []

    for ann in models:
        target_model = models
        lt_model = setup_and_fit_linear_target_model(
            ann,
            cats,
            conts,
            targets,
            is_ts=is_ts,
            reuse_weights=False,
            num_layers_to_remove=num_layers_to_remove,
        )
        linear_target_models += [lt_model]

    return linear_target_models


def create_csge_model(
    models,
    dls,
    similarity_measure_type: SimilarityMeasure,
    is_ts=False,
    etas=None,
    ts_length=1,
    csge_type="csgelin",
    convert_dl_to_tensor=None,
    gbrt_model=None,
):
    target_models = []

    for ann in models:
        target_model = setup_ann_target_model(ann, is_ts=is_ts)
        target_models.append(target_model)

    if "linearoutput" in csge_type:
        cats, conts, targets = convert_dl_to_tensor(dls.train_ds)
        target_models = fit_linear_models(models, cats, conts, targets, is_ts)

    if "gbrt" in csge_type:
        gbrt_torch = TorchSklearnWrapper(gbrt_model)
        target_models += [gbrt_torch]

    if "csgelin" in csge_type:
        local_error_estimator = LocalErrorPredictor(len(target_models), use_elm=False)
    elif "csgeelm" in csge_type:
        local_error_estimator = LocalErrorPredictor(
            len(target_models), use_elm=True, n_hidden=50
        )
    elif "csgepca" in csge_type:
        local_error_estimator = simple_local_error_estimator(
            param_dict={"n_components": 2, "n_neighbors": 3}
        )
    else:
        raise ValueError("Unknown CSGE type.")

    csge_ensemble = CSGE(
        source_models=target_models,
        local_error_estimator=local_error_estimator,
        eta_global=etas["eta_g"],  # TODO as parameter?
        eta_time=etas["eta_t"],
        eta_local=etas["eta_l"],
        is_timeseries_data=is_ts,
        is_timeseries_model=is_ts,
        ts_length=ts_length,
    )

    csge_ensemble.fit(dls)

    return csge_ensemble


def setup_ann_target_model(source_model, is_ts=False):
    target_model = copy.deepcopy(source_model)
    freeze(target_model)

    return target_model


def prep_gbrt_results(HOSTNAME):
    sacred_config_models_gbrt = SacredConfig(
        experiment_name="gbrttarget",
        db_name="targetbaselinesmodel",
        mongo_observer=True,
        hostname=HOSTNAME,
        port="27031",
    )

    db, mongo_client = get_mongo_db_connection(sacred_config_models_gbrt)
    df_results_gbrt = sacred_db_to_df(db.runs)
    grid_fs = gridfs.GridFS(db)
    mask = df_results_gbrt.status == "COMPLETED"
    df_results_gbrt = df_results_gbrt[mask].reset_index()
    df_results_gbrt["data_file_name"] = df_results_gbrt.data_config.apply(
        lambda x: x["data_file_name"]
    )
    df_results_gbrt["DataType"] = df_results_gbrt.data_config.apply(
        lambda x: convert_py_reduce_to_type(x["result_folder"]).stem
    )

    return df_results_gbrt, grid_fs


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
        help="The kind on which the we create the ensemble. Options: RMSE",
        default="rmse",
    )
    parser.add_argument(
        "--create_forecasts",
        help="Create forecasts or not.",
        default=False,
    )

    args = parser.parse_args()

    if args.model_type == "all":
        model_types = [
            "tcn",
            "mlp",
        ]
    else:
        model_types = [args.model_type]

    similarity_measure = "rmse"

    if args.create_forecasts and not args.smoke_test:
        raise ValueError
    elif not args.smoke_test:
        args.create_forecasts = False

    for model_type in model_types:
        for csge_type in [
            # "csgelinnofit",
            # "csgepcanofit",
            # "csgeelmnofit",
            # "csgelinnofitlinearoutput",
            "csgepcanofitlinearoutput",
            # "csgeelmnofitlinearoutput",
            # "csgelin",
            # "csgepca",
            # "csgeelm",
            # "csgepcablended",
            # "csgelinblended",
            # "csgeelmblended",
            # "csgepcagbrt",
            # "csgelinnofitlinearoutputgbrt",
            # "csgepcanofitlinearoutputgbrt",
        ]:
            df_results_gbrt, gridfs_gbrt = None, None
            if "gbrt" in csge_type:
                df_results_gbrt, gridfs_gbrt = prep_gbrt_results(HOSTNAME)

            model_config = ModelConfigTarget(
                args.embedding_type,
                args.full_bayes,
                model_type,
                args.model_architecture,
                debug=DEBUG,
                smoke_test=args.smoke_test,
                similarity_measure=similarity_measure,
                ensemble_type=csge_type,
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
                if "gbrt" in csge_type:
                    df_results_gbrt = df_results_gbrt[
                        data_config.data_type.name == df_results_gbrt.DataType
                    ]

                experiment_name = f"source{args.model_architecture}_{model_type}"
                db_name = f"source{model_type}"
                sacred_config_models = SacredConfig(
                    experiment_name=experiment_name,
                    db_name=db_name,
                    mongo_observer=True,
                    hostname=HOSTNAME,
                    port="27031",
                )

                (
                    grid_fs,
                    df_converted_filtered,
                    mongo_client,
                ) = read_and_filter_source_results(
                    data_config,
                    sacred_config_models,
                    is_mtl=False,
                    return_mongo_client=True,
                )

                if model_config.smoke_test and not args.create_forecasts:
                    df_converted_filtered = df_converted_filtered.iloc[0:5]

                pre_loaded_models, _ = pre_load_source_models(
                    grid_fs, df_converted_filtered, data_type="learner"
                )
                mongo_client.close()
                ts = data_config.target_splits
                ts = [Path(t).stem for t in ts]
                for d in df_converted_filtered.data_file_name:
                    if d in ts:
                        raise ValueError(f"{d} should not be in source models")

                port = "27031"
                if model_config.smoke_test:
                    port = "27029"

                experiment_name = f"ensemble_csge_ann_{csge_type}_{model_type}"
                db_name = f"ensemble_csge_ann"

                sacred_config = SacredConfig(
                    experiment_name=experiment_name,
                    db_name=db_name,
                    mongo_observer=True,
                    hostname=HOSTNAME,
                    port=port,
                )

                run_stl_experiment(
                    sacred_config,
                    data_config,
                    model_config,
                    {"cpu": 1},
                    train_function=partial(
                        csge_ann_target,
                        pre_loaded_models=pre_loaded_models,
                        df_results_gbrt=df_results_gbrt,
                        gridfs_gbrt=gridfs_gbrt,
                    ),
                    use_optuna=False,
                    check_db_entries=True and model_config.smoke_test,
                )

                time.sleep(2)
