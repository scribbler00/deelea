import os

from fastcore.xtras import delete
from fastrenewables.utils import filter_preds

from deelea.target import load_target_data, read_and_filter_source_results


SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = 3
SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)

from deelea.config_environment import SETUP_ENVIRONMENT

# SHOULD ALWAYS BE THE FIRST TO INCLUDE
SLURM_CPUS_PER_TASK, RESSOURCES_PER_TRIAL, SERVER = SETUP_ENVIRONMENT(
    RESSOURCES_PER_TRIAL=3  # SLURM_CPUS_PER_TASK - 1
)

from fastai.torch_core import no_random
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

HOSTNAME = "alatar.ies.uni-kassel.de"
from sklearn.model_selection import train_test_split


def elm_target(
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

        # ("********************************
        # note on lg evidence:
        # the log evidence is sometimes also called the marginal likelihood as a likelihodd function over the space pf model in which the parameters have been marginalized out
        # the ratio of model evidences for two models is knownas a bayes factor, see page 162 bishop
        # "********************************

        best_idx, act_source_model = select_target_model(
            pre_loaded_elms,
            to_train_target,
            model_config.similarity_measure_type,
        )
        # valid deterministic
        preds = act_source_model.predict(to_train_target.conts.values)
        targets = to_train_target.ys.values.ravel()
        targets, preds = filter_preds(targets, preds)
        error_dict_train = get_error_dict(targets, preds, prefix="valid_")
        # valid proba
        yhat, yhat_std = act_source_model.predict_proba(to_train_target.conts.values)
        crps_error_valid = get_crps_error(targets, yhat, yhat_std, prefix="valid_")

        # test deterministic
        to_test = pkl.load(open(data_config.test_file, "rb"))
        preds = act_source_model.predict(to_test.conts.values)
        targets = to_test.ys.values.ravel()
        targets, preds = filter_preds(targets, preds)
        error_dict = get_error_dict(targets, preds, prefix="test_")
        # test proba
        yhat, yhat_std = act_source_model.predict_proba(to_test.conts.values)
        crps_error_test = get_crps_error(targets, yhat, yhat_std, prefix="test_")
        if data_config.create_forecasts:
            create_forecasts(data_config, targets, yhat, yhat_std, to_test)

        error_dict = {
            **error_dict,
            **error_dict_train,
            **crps_error_valid,
            **crps_error_test,
        }

        error_dict["test_logevidence"] = act_source_model.log_evidence(
            to_test.conts.values, to_test.ys.values.ravel()
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
        print(cur_season, num_days_training, error_dict, best_idx)
        print("********************************")

    result = pd.DataFrame(errors)
    result["Season"] = seasons
    result["NumDaysTraining"] = num_days_trainings
    result["FoldID"] = data_config.fold_id
    add_sacred_artifact(result, "results.csv", sacred_experiment)

    return result.to_json()


def create_forecasts(data_config, targets, yhat, yhat_std, to_test):
    dest_forecast_folder = "../../doc/forecasts/elm"
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


def select_target_model(
    models,
    to_train_target,
    similarity_measure_type: SimilarityMeasure,
):

    cur_best_similarity_value = -1e12
    similarity = cur_best_similarity_value
    best_idx = -1
    best_model = None

    X_train_complete = to_train_target.conts.values
    y_train_complete = to_train_target.ys.values.ravel()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_complete, y_train_complete, random_state=42, train_size=0.8
    )

    for cur_id, elm in enumerate(models):
        alpha = elm.alpha
        elm = setup_elm_target_model(elm, alpha=alpha)

        if similarity_measure_type == SimilarityMeasure.LOGEVIDENCE:
            elm = elm.update(X_train_complete, y_train_complete.ravel())
            similarity = elm.log_evidence(X_train_complete, y_train_complete)  #
        elif similarity_measure_type == SimilarityMeasure.RMSE:
            elm = elm.update(X_train, y_train.ravel())
            similarity = ((y_valid - elm.predict(X_valid)) ** 2).mean() ** 0.5
            similarity = similarity * -1
        elif similarity_measure_type == SimilarityMeasure.LIKELIHOOD:
            elm = elm.update(X_train_complete, y_train_complete.ravel())
            similarity = elm.log_likelihood(X_train_complete, y_train_complete)
        elif similarity_measure_type == SimilarityMeasure.POSTERIOR:
            elm = elm.update(X_train_complete, y_train_complete.ravel())
            similarity = elm.log_posterior(X_train_complete, y_train_complete)

        if similarity > cur_best_similarity_value:
            best_idx = cur_id
            cur_best_similarity_value = similarity
            best_model = elm

    return best_idx, copy.deepcopy(best_model)


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
        help="elm or linear",
        default="elm",
    )

    parser.add_argument(
        "--similarity_measure",
        help="logevidence, rmse, posterior,  or all",
        default="logevidence",
    )
    parser.add_argument(
        "--create_forecasts",
        help="Create forecasts or not.",
        default=False,
    )

    args = parser.parse_args()

    if args.create_forecasts and not args.smoke_test:
        raise ValueError

    if args.similarity_measure == "all":
        similarity_measures = ["logevidence", "rmse", "posterior", "likelihood"]
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
            pre_loaded_models, _ = pre_load_source_elms(grid_fs, df_converted_filtered)
            ts = data_config.target_splits
            ts = [Path(t).stem for t in ts]
            for d in df_converted_filtered.data_file_name:
                if d in ts:
                    raise ValueError(f"{d} should not be in source models")

            port = "27031"
            if model_config.smoke_test:
                port = "27029"

            sacred_config = SacredConfig(
                experiment_name="ELMTarget",
                db_name="targetelm",
                mongo_observer=True,
                hostname=HOSTNAME,
                port=port,
            )
            # delete_table(sacred_config, "targetelm")
            run_stl_experiment(
                sacred_config,
                data_config,
                model_config,
                {"cpu": 1},
                train_function=partial(elm_target, pre_loaded_elms=pre_loaded_models),
                use_optuna=False,
                check_db_entries=False,
            )

            time.sleep(5)
