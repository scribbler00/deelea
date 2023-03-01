from xml.parsers.expat import model
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

import argparse
import pickle as pkl
import copy
import warnings
import torch
from functools import partial
from pathlib import Path
import time
import gridfs
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from fastai.torch_core import no_random, params, to_np

from fastrenewables.timeseries.core import Timeseries
from fastrenewables.utils_pytorch import (
    freeze,
    unfreeze,
)
from fastrenewables.models.transfermodels import BTuningModel
from fastrenewables.utils import filter_preds
from fastrenewables.losses import L2SPLoss

from deelea.utils_eval import get_error_dict, get_crps_error
from deelea.utils_data import DataConfig, TLType
from deelea.configs import (
    get_stl_target_btunning_optimization,
    get_stl_target_optimization,
    get_stl_target_optimization_wd_source,
)
from deelea.target import (
    create_target_learner,
    load_target_data,
    pre_load_source_models,
    read_and_filter_source_results,
    setup_and_fit_linear_target_model,
    setup_helper_functions_stl,
    setup_training_data_stl,
    stl_target_similarity_by_evidence,
    stl_target_similarity_by_rmse,
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
from deelea.utils_models import AdaptionStrategy, ModelConfigTarget, SimilarityMeasure
from deelea.utils import (
    add_args,
    get_tmp_dir,
)

warnings.filterwarnings("ignore")
HOSTNAME = "alatar.ies.uni-kassel.de"
# HOSTNAME = "localhost"


def ann_stl(
    data_config: DataConfig,
    model_config: ModelConfigTarget,
    resources_per_trial: dict,
    optuna_study,
    sacred_experiment,
    pre_loaded_models,
    model_names,
    **kwargs,
):

    target_training = TargetTraining()

    seasons, num_days_trainings, errors, selected_park_names, similarity_values = (
        [],
        [],
        [],
        [],
        [],
    )
    to_test = pkl.load(open(data_config.test_file, "rb"))

    if model_config.is_timeseries_model:
        to_test = Timeseries(to_test, splits=None)
        to_test.split = range(len(to_test))
    else:
        to_test.split = None

    dls_target_test = to_test.dataloaders(
        bs=len(to_test), num_workers=0, drop_last=False, shuffle=False
    )
    is_bayes_tune = model_config.adaption_strategy == AdaptionStrategy.BAYESTUNE

    for cur_season, months_to_train_on, num_days_training in target_training.params():
        if data_config.create_forecasts and num_days_training != 365:
            continue

        to_train_target = setup_training_data_stl(
            data_config, months_to_train_on, num_days_training
        )

        no_random(seed=42)
        if to_train_target is None:
            continue

        bs, learner_type, convert_dl_to_tensor = setup_helper_functions_stl(
            model_config, to_train_target
        )

        #     lrs = (1e-6, 1e-4)
        dls_target_train = to_train_target.dataloaders(
            bs=bs, num_workers=0, drop_last=True, shuffle=True
        )

        cats_train, conts_train, targets_train = convert_dl_to_tensor(
            dls_target_train.train_ds
        )

        cats_valid, conts_valid, targets_valid = convert_dl_to_tensor(
            dls_target_train.valid_ds
        )

        if (
            model_config.adaption_strategy
            in [
                AdaptionStrategy.DIRECT,
                AdaptionStrategy.DIRECTLINEAR,
                AdaptionStrategy.DIRECTLINEARAUTO,
            ]
            or model_config.similarity_measure_type == SimilarityMeasure.LOGEVIDENCE
        ):
            if len(cats_train) > 0:
                cats_model_selection = torch.cat([cats_train, cats_valid], axis=0)
            else:
                cats_model_selection = cats_train
            conts_model_selection = torch.cat([conts_train, conts_valid], axis=0)
            targets_model_selection = torch.cat([targets_train, targets_valid], axis=0)
        elif model_config.similarity_measure_type == SimilarityMeasure.RMSE:
            cats_model_selection, conts_model_selection, targets_model_selection = (
                cats_valid,
                conts_valid,
                targets_valid,
            )
        else:
            raise ValueError(
                "Unknown combination of similarity measure and adaption stratgey."
            )

        target_model, selected_park_name, similarity_value = create_model(
            pre_loaded_models,
            cats_model_selection,
            conts_model_selection,
            targets_model_selection,
            model_config.similarity_measure_type,
            model_config.adaption_strategy,
            is_ts=model_config.is_timeseries_model,
            park_names=model_names,
        )

        freeze_model(model_config, target_model)

        best_grid_search_error = {"valid_rmse": 1e12}

        best_learner = create_target_learner(
            learner_type,
            dls_target_train,
            target_model,
            {"lambd": 0, "wd": 0.1},
            is_bayes_tune,
        )
        best_learner.model.eval()
        preds, targets = get_test_preds(dls_target_test, is_bayes_tune, best_learner)
        error_dict_before = get_error_dict(targets, preds, prefix="before_ft_test_")
        best_config = {}

        if model_config.adaption_strategy in [
            AdaptionStrategy.WEIGHTDECAY,
            AdaptionStrategy.WEIGHTDECAYALL,
        ]:
            grid_search_configs = get_stl_target_optimization()
            freeze_model(model_config, target_model)

            best_grid_search_error, best_learner, best_config = do_grid_search_wd(
                learner_type,
                dls_target_train,
                target_model,
                grid_search_configs,
                best_grid_search_error,
                model_config,
                adapt_complete_network=model_config.adaption_strategy
                == AdaptionStrategy.WEIGHTDECAYALL,
            )
        elif model_config.adaption_strategy in [
            AdaptionStrategy.WEIGHTDECAYSOURCE,
            AdaptionStrategy.WEIGHTDECAYSOURCEALL,
        ]:
            grid_search_configs = get_stl_target_optimization_wd_source()
            freeze_model(model_config, target_model)

            (
                best_grid_search_error,
                best_learner,
                best_config,
            ) = do_grid_search_wd_source(
                learner_type,
                dls_target_train,
                target_model,
                grid_search_configs,
                best_grid_search_error,
                model_config,
                adapt_complete_network=model_config.adaption_strategy
                == AdaptionStrategy.WEIGHTDECAYSOURCEALL,
            )
        elif model_config.adaption_strategy == AdaptionStrategy.BAYESTUNE:
            freeze_model(model_config, target_model)
            grid_search_configs = get_stl_target_btunning_optimization()
            (
                best_grid_search_error,
                best_learner,
                best_config,
            ) = do_grid_search_bayestune(
                learner_type,
                dls_target_train,
                target_model,
                grid_search_configs,
                best_grid_search_error,
                model_config,
            )

        preds, targets = get_test_preds(dls_target_test, is_bayes_tune, best_learner)
        error_dict = get_error_dict(targets, preds, prefix="test_")
        crps_errors = create_crps_error(
            model_config,
            best_learner,
            convert_dl_to_tensor,
            dls_target_train,
            dls_target_test,
            data_config=data_config,
        )

        error_dict = {
            **error_dict,
            **best_grid_search_error,
            **error_dict_before,
            **crps_errors,
        }

        seasons += [cur_season]
        num_days_trainings += [num_days_training]
        errors += [error_dict]
        selected_park_names += [selected_park_name]
        similarity_values += [similarity_value]

        log_dict_as_scalar(
            sacred_experiment,
            {
                **{"season": cur_season, "num_days_training": num_days_training},
                **error_dict,
                **{"source_model_park_name": selected_park_name},
                **{"similarity_value": similarity_value},
            },
        )

        print("********************************")
        print("best config", best_config)
        print(
            cur_season,
            num_days_training,
            error_dict,
            {"source_model_park_name": selected_park_name},
            {"similarity_value": similarity_value},
        )
        print("********************************")

    result = pd.DataFrame(errors)
    result["Season"] = seasons
    result["NumDaysTraining"] = num_days_trainings
    result["FoldID"] = data_config.fold_id
    result["SelectedParkNames"] = selected_park_names
    result["SimilarityValues"] = similarity_values

    add_sacred_artifact(result, "results.csv", sacred_experiment)

    return result.to_json()


def create_crps_error(
    model_config: ModelConfigTarget,
    best_learner,
    convert_dl_to_tensor,
    dls_target_train,
    dls_target_test,
    data_config: DataConfig,
):
    crps_errors = {}

    if (
        model_config.adaption_strategy == AdaptionStrategy.DIRECTLINEAR
        and model_config.similarity_measure_type == SimilarityMeasure.LOGEVIDENCE
    ):
        proba_model = best_learner.model
        cats, conts, targets = convert_dl_to_tensor(dls_target_train.train_ds)
        yhat, yhat_std = proba_model.predict_proba(cats, conts)
        crps_error_valid = get_crps_error(targets, yhat, yhat_std, prefix="valid_")

        cats, conts, targets = convert_dl_to_tensor(dls_target_test.train_ds)
        yhat, yhat_std = proba_model.predict_proba(cats, conts)
        crps_error_test = get_crps_error(targets, yhat, yhat_std, prefix="test_")

        crps_errors = {**crps_error_valid, **crps_error_test}

        if data_config.create_forecasts:
            if model_config.is_timeseries_model:
                indexes = dls_target_test.train_ds.indexes.ravel()
            else:
                indexes = dls_target_test.train_ds.items.index.ravel()

            create_forecasts(
                data_config,
                targets,
                yhat,
                yhat_std,
                indexes,
                model_config.model_type.name.lower(),
            )

    return crps_errors


def create_forecasts(data_config, targets, yhat, yhat_std, indexes, model_type):
    dest_forecast_folder = f"../../doc/forecasts/{model_type}/"
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


def get_test_preds(dls_target_test, is_bayes_tune, best_learner):
    preds, targets = best_learner.predict(
        test_dl=dls_target_test, filter=False, flatten=False
    )
    if is_bayes_tune:
        preds = preds[:, 0]

    targets, preds = filter_preds(targets.ravel(), preds.ravel())
    return preds, targets


def freeze_model(model_config, target_model):

    freeze(target_model)

    unfreeze_final_layer = model_config.adaption_strategy not in [
        AdaptionStrategy.DIRECTLINEAR,
        AdaptionStrategy.DIRECTLINEARAUTO,
    ]

    pointer_to_model = target_model
    if model_config.adaption_strategy == AdaptionStrategy.BAYESTUNE:
        pointer_to_model = pointer_to_model.source_model

    if model_config.is_timeseries_model and unfreeze_final_layer:
        unfreeze(pointer_to_model.layers.temporal_blocks[-1])
    elif unfreeze_final_layer:
        unfreeze(pointer_to_model.layers[-1])


def do_grid_search_wd_source(
    learner_type,
    dls_train,
    target_model,
    grid_search_configs,
    best_grid_search_error,
    model_config,
    adapt_complete_network=False,
):
    best_config = None
    for config in grid_search_configs:
        target_model_tmp = copy.deepcopy(target_model)
        reference_parameters, source_parameters = target_model_tmp, target_model_tmp

        if model_config.is_timeseries_model and not adapt_complete_network:
            reference_parameters = target_model_tmp.layers.temporal_blocks[-1]
            source_parameters = target_model_tmp.layers.temporal_blocks[-1]
        elif not adapt_complete_network:
            reference_parameters = target_model_tmp.layers[-1]
            source_parameters = target_model_tmp.layers[-1]

        reference_parameters = copy.deepcopy(reference_parameters)

        freeze(reference_parameters)

        if adapt_complete_network:
            unfreeze(source_parameters)

        loss_func = L2SPLoss(
            reference_parameters, source_parameters, lambd=config["wd"]
        )
        learner = create_target_learner(
            learner_type,
            dls_train,
            target_model_tmp,
            config,
            is_bayes_tune=False,
            loss_func=loss_func,
        )
        train_n_epochs = config["epochs"]
        if model_config.train_n_epochs is not None:
            train_n_epochs = model_config.train_n_epochs
        config["epochs"] = train_n_epochs
        learner.fit(train_n_epochs, wd=0, lr=config["lr"])
        preds, targets = learner.predict(ds_idx=1)
        error_dict_train = get_error_dict(targets, preds, prefix="valid_")
        if error_dict_train["valid_rmse"] < best_grid_search_error["valid_rmse"]:
            best_grid_search_error = error_dict_train
            best_learner = copy.deepcopy(learner)
            best_config = config

    return best_grid_search_error, best_learner, best_config


def do_grid_search_wd(
    learner_type,
    dls_train,
    target_model,
    grid_search_configs,
    best_grid_search_error,
    model_config,
    adapt_complete_network=False,
):
    best_config = None
    for config in grid_search_configs:
        learner = create_target_learner(
            learner_type, dls_train, target_model, config, is_bayes_tune=False
        )
        if adapt_complete_network:
            unfreeze(learner.model)
        train_n_epochs = config["epochs"]
        if model_config.train_n_epochs is not None:
            train_n_epochs = model_config.train_n_epochs
        config["epochs"] = train_n_epochs
        learner.fit(train_n_epochs, wd=config["wd"], lr=config["lr"])
        preds, targets = learner.predict(ds_idx=1)
        error_dict_train = get_error_dict(targets, preds, prefix="valid_")
        if error_dict_train["valid_rmse"] < best_grid_search_error["valid_rmse"]:
            best_grid_search_error = error_dict_train
            best_learner = copy.deepcopy(learner)
            best_config = config

    return best_grid_search_error, best_learner, best_config


def do_grid_search_bayestune(
    learner_type,
    dls_train,
    target_model,
    grid_search_configs,
    best_grid_search_error,
    model_config,
):
    best_config = None
    for config in grid_search_configs:
        learner = create_target_learner(
            learner_type, dls_train, target_model, config, is_bayes_tune=True
        )
        train_n_epochs = config["epochs"]
        if model_config.train_n_epochs is not None:
            train_n_epochs = model_config.train_n_epochs
        config["epochs"] = train_n_epochs
        learner.fit(train_n_epochs, wd=config["wd"], lr=config["lr"])
        preds, targets = learner.predict(ds_idx=1, filter=False, flatten=False)
        preds = preds[:, 0]
        targets, preds = filter_preds(targets.ravel(), preds.ravel())

        error_dict_train = get_error_dict(targets, preds, prefix="valid_")
        if error_dict_train["valid_rmse"] < best_grid_search_error["valid_rmse"]:
            best_grid_search_error = error_dict_train
            best_learner = copy.deepcopy(learner)
            best_config = config

    return best_grid_search_error, best_learner, best_config


def create_model(
    models,
    cats,
    conts,
    targets,
    similarity_measure_type: SimilarityMeasure,
    adaption_strategy: AdaptionStrategy,
    is_ts=False,
    park_names=[],
):
    if similarity_measure_type == SimilarityMeasure.RMSE:
        target_models, similarity_values, sort_ids = stl_target_similarity_by_rmse(
            models, cats, conts, targets, is_ts
        )
    elif (
        similarity_measure_type == SimilarityMeasure.LOGEVIDENCE
        and adaption_strategy
        in [
            AdaptionStrategy.DIRECT,
            AdaptionStrategy.DIRECTLINEAR,
            AdaptionStrategy.WEIGHTDECAY,
            AdaptionStrategy.WEIGHTDECAYALL,
            AdaptionStrategy.WEIGHTDECAYSOURCE,
            AdaptionStrategy.WEIGHTDECAYSOURCEALL,
            AdaptionStrategy.BAYESTUNE,
        ]
    ):
        (
            target_models,
            linear_target_models,
            similarity_values,
            sort_ids,
        ) = stl_target_similarity_by_evidence(models, cats, conts, targets, is_ts)

    elif (
        similarity_measure_type == SimilarityMeasure.LOGEVIDENCE
        and adaption_strategy == AdaptionStrategy.DIRECTLINEARAUTO
    ):
        (
            target_models,
            linear_target_models,
            similarity_values,
            sort_ids,
        ) = stl_target_similarity_by_evidence(
            models, cats, conts, targets, is_ts, num_layers_to_remove=1
        )
        # remove two layers and replace by linear model
        (
            target_models2,
            linear_target_models2,
            similarity_values2,
            sort_ids2,
        ) = stl_target_similarity_by_evidence(
            models, cats, conts, targets, is_ts, num_layers_to_remove=2
        )
        if max(similarity_values2) > max(similarity_values):
            target_models, linear_target_models, similarity_values, sort_ids = (
                target_models2,
                linear_target_models2,
                similarity_values2,
                sort_ids2,
            )

    target_model = np.array(target_models)[sort_ids][0]
    park_name = np.array(park_names)[sort_ids][0]
    similarity_value = float(np.array(similarity_values)[sort_ids][0])

    if adaption_strategy in [
        AdaptionStrategy.DIRECTLINEAR,
        AdaptionStrategy.DIRECTLINEARAUTO,
    ]:
        target_model = np.array(linear_target_models)[sort_ids][0]
    elif adaption_strategy == AdaptionStrategy.BAYESTUNE:
        sort_ids = sort_ids[0:10]
        source_model = np.array(target_models)[sort_ids][0]
        target_models = np.array(linear_target_models)[sort_ids]
        target_model = BTuningModel(source_model, target_models)

    return target_model, park_name, similarity_value


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
        help="Model type that is used as source. Either mlp or tcn",
        default="all",
    )

    parser.add_argument(
        "--similarity_measure",
        help="The kind of similarity measurement to select a source model.",
        default="all",
    )

    parser.add_argument(
        "--train_n_epochs",
        help="The number of epochs to finetune..",
        default=5,
    )

    parser.add_argument(
        "--create_forecasts",
        help="Create forecasts or not.",
        default=False,
    )

    args = parser.parse_args()

    if args.similarity_measure == "all":
        similarity_measures = [
            "logevidence",
            "rmse",
        ]
    else:
        similarity_measures = [args.similarity_measure]

    if args.model_type == "all":
        model_types = [
            "tcn",
            "mlp",
        ]
    else:
        model_types = [args.model_type]

    adaption_strategies = [
        "directlinear",
        "weightdecaysourceall",
        "weightdecaysource",
        # "directlinearauto",
        "direct",
        "weightdecay",
        "bayestune",
        # "weightdecayall",
    ]
    if args.create_forecasts and not args.smoke_test:
        raise ValueError

    for model_type in model_types:
        for adaption_strategy in adaption_strategies:
            for similarity_measure in similarity_measures:
                if similarity_measure == "rmse" and (
                    ("directlinear" in adaption_strategy)
                    or "bayestune" in adaption_strategy
                ):
                    continue

                print("*************")
                print(adaption_strategy, similarity_measure)
                print("*************")
                n_epochs = 1
                if adaption_strategies == "directlinear":
                    n_epochs = 5

                model_config = ModelConfigTarget(
                    args.embedding_type,
                    args.full_bayes,
                    model_type,
                    args.model_architecture,
                    debug=DEBUG,
                    smoke_test=args.smoke_test,
                    similarity_measure=similarity_measure,
                    adaption_strategy=adaption_strategy,
                    ensemble_type="",
                    train_n_epochs=n_epochs,
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
                        df_converted_filtered = df_converted_filtered.iloc[0:3]

                    pre_loaded_models, model_names = pre_load_source_models(
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

                    experiment_name = f"target_stl_{model_type}"
                    db_name = f"target_stl"
                    sacred_config = SacredConfig(
                        experiment_name=experiment_name,
                        db_name=db_name,
                        mongo_observer=True,
                        hostname=HOSTNAME,
                        port=port,
                    )
                    # delete_table(sacred_config, "target_stl")
                    run_stl_experiment(
                        sacred_config,
                        data_config,
                        model_config,
                        {"cpu": 1},
                        train_function=partial(
                            ann_stl,
                            pre_loaded_models=pre_loaded_models,
                            model_names=model_names,
                        ),
                        use_optuna=False,
                        check_db_entries=True,
                    )

                    time.sleep(2)
