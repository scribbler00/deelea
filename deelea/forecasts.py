from fastai.torch_core import to_np
from fastrenewables.tabular.learner import fast_prediction as fast_prediction_mlp
from fastrenewables.timeseries.learner import fast_prediction_ts as fast_prediction_tcn
from fastrenewables.utils import filter_preds
import gridfs
import pandas as pd
from deelea.mongo import (
    filter_results_by_config,
    get_model_from_artifacts,
    get_results_of_db,
)

from deelea.utils_models import ModelType


def get_indexes_and_targets(model_type, to_train, to_test):
    if model_type == ModelType.MLP:
        indexes_train, indexes_test = to_train.items.index, to_test.items.index
        targets_train, targets_test = (
            to_train.ys.values.ravel(),
            to_test.ys.values.ravel(),
        )
    elif model_type == ModelType.TCN:
        indexes_train, indexes_test = (
            to_train.indexes.reshape(-1),
            to_test.indexes.reshape(-1),
        )

        targets_train, targets_test = (
            to_np(to_train.ys).ravel(),
            to_np(to_test.ys).ravel(),
        )
    else:
        raise ValueError

    return (
        indexes_train,
        indexes_test,
        targets_train,
        targets_test,
    )


def get_prediction_setup(
    model_config, sacred_config_mlp_models, sacred_config_tcn_models
):
    if model_config.model_type == ModelType.MLP:
        sacred_config = sacred_config_mlp_models
        fast_prediction = fast_prediction_mlp
    elif model_config.model_type == ModelType.TCN:
        sacred_config = sacred_config_tcn_models
        fast_prediction = fast_prediction_tcn
    else:
        raise ValueError

    return (
        sacred_config,
        fast_prediction,
    )


def get_predictions(to_train, to_test, fast_prediction, model, return_targets=True):
    preds_train, targets_train = fast_prediction(
        model.model, to_train, flatten=True, filter=False
    )
    preds_test, targets_test = fast_prediction(
        model.model, to_test, flatten=True, filter=False
    )

    preds_train, targets_train = filter_preds(
        preds_train, targets_train, filter_nas=False
    )

    preds_test, targets_test = filter_preds(preds_test, targets_test, filter_nas=False)
    if return_targets:
        return preds_train, preds_test, targets_train, targets_test
    else:
        return preds_train, preds_test


def get_model_by_config(data_config, model_config, sacred_config_model):
    df_res, db = get_results_of_db(model_config, data_config, sacred_config_model)
    df_res = df_res[df_res.Status == "COMPLETED"]
    grid_fs = gridfs.GridFS(db)

    df_res_filtered = filter_results_by_config(model_config, data_config, df_res)
    source_model = get_model_from_artifacts(grid_fs, df_res_filtered)
    return source_model


def create_single_df(
    preds_train, preds_test, indexes_train, indexes_test, name, include_test_flag=False
):
    df_train = pd.DataFrame({name: preds_train}, index=indexes_train)
    df_test = pd.DataFrame({name: preds_test}, index=indexes_test)
    df_train.index.names = ["TimeUTC"]
    df_test.index.names = ["TimeUTC"]
    if include_test_flag:
        df_train["IsTest"] = False
        df_test["IsTest"] = True

    return pd.concat([df_train, df_test], axis=0)
