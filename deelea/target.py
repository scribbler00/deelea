import copy
import pickle as pkl
import gridfs
import numpy as np
import pandas as pd
import torch

from fastcore.foundation import L
from fastcore.transform import Pipeline

from fastai.torch_core import tensor, to_np
from fastai.optimizer import SGD
from fastai.callback.core import TrainEvalCallback
from fastai.metrics import rmse

from fastrenewables.timeseries.core import Timeseries
from fastrenewables.tabular.core import FilterDays, FilterMonths
from fastrenewables.tabular.learner import fast_prediction
from fastrenewables.tabular.core import get_samples_per_day
from fastrenewables.baselines import BayesLinReg
from fastrenewables.models.transfermodels import (
    LinearTransferModel,
    reduce_layers_tcn_model,
)
from fastai.data.transforms import RandomSplitter
from fastrenewables.tabular.core import TrainTestSplitByDays

from fastrenewables.timeseries.learner import (
    RenewableTimeseriesLearner,
    convert_to_tensor_ts,
)
from fastrenewables.tabular.learner import RenewableLearner, convert_to_tensor
from fastrenewables.models.ensembles import rank_by_evidence
from fastrenewables.utils import filter_preds
from fastrenewables.losses import *

from deelea.utils_models import ModelArchitecture
from deelea.mongo import (
    convert_mongo_df,
    get_mongo_db_connection,
    get_object_id_in_artefact,
    read_artifact,
    sacred_db_to_df,
)
from sklearn.metrics import mean_squared_error as mse


def load_target_data(data_config, months_to_train_on, num_days_training):
    to_train = pkl.load(open(data_config.train_file, "rb"))
    n_samples_per_day = get_samples_per_day(to_train.items)

    filter_months = FilterMonths(months=months_to_train_on)
    filter_days = FilterDays(num_days_training)
    filter_days.setups(to_train)

    target_pipeline = Pipeline([filter_months, filter_days])
    to_train = target_pipeline(to_train)

    # we should have at least XX% of the expected data
    if len(to_train) < (num_days_training * n_samples_per_day * 0.75):
        print(
            f"Insufficient training data ({data_config.data_file_name}) with length: {len(to_train.items)}. --> Skipping"
        )
        to_train = None

    elif data_config._model_config.is_timeseries_model:
        to_train = Timeseries(to_train, splits=None)

    return to_train


def read_and_filter_source_results(
    data_config,
    sacred_config,
    is_mtl,
    file_name=None,
    model_type=None,
    return_mongo_client=False,
):
    mongo_client, grid_fs, df_converted = read_source_results(sacred_config)
    mask_res_folder = df_converted.result_folder == data_config.result_folder.stem

    if file_name is not None:
        mask_fold_id_or_name = df_converted.data_file_name == file_name
        mask_arch = df_converted.model_architecture == ModelArchitecture.STL
    elif is_mtl:
        mask_fold_id_or_name = df_converted.fold_id == data_config.fold_id
        mask_arch = df_converted.model_architecture == ModelArchitecture.MTL
    else:
        mask_fold_id_or_name = df_converted.fold_id != data_config.fold_id
        mask_arch = df_converted.model_architecture == ModelArchitecture.STL

    df_converted_filtered = df_converted[
        mask_res_folder & mask_fold_id_or_name & mask_arch
    ]

    if model_type is not None:
        df_converted_filtered = df_converted_filtered[
            df_converted_filtered.model_type == model_type
        ]
    if return_mongo_client:
        return grid_fs, df_converted_filtered, mongo_client
    else:
        return grid_fs, df_converted_filtered


def read_source_results(sacred_config):
    db, mongo_client = get_mongo_db_connection(sacred_config)
    df = sacred_db_to_df(db.runs)
    grid_fs = gridfs.GridFS(db)
    mask = df.status == "COMPLETED"
    df = df[mask].reset_index()

    df_converted = convert_mongo_df(df, include_results=False)

    return mongo_client, grid_fs, df_converted


def predict_all_sources(db_entries, grid_fs, to_train, to_test):

    df_predictions = []
    for idx in db_entries.index.values:
        object_id = get_object_id_in_artefact(db_entries.artifacts[idx], filter="model")
        source_model = read_artifact(grid_fs, object_id, "learner")

        df_cur_results = predict_single_source(
            source_model.model,
            to_train,
            to_test,
            is_timeseries=False,
            model_name=db_entries.data_file_name[idx],
        )

        df_predictions.append(df_cur_results)

    df_predictions = pd.concat(df_predictions, axis=1)

    return df_predictions


def predict_single_source(
    model, data_train, data_test, is_timeseries, model_name, device="cpu", flatten=True
):

    if is_timeseries:
        raise NotImplementedError
    else:
        data_train.splits = None
        preds, _ = fast_prediction(
            model, data_train, filter=True, flatten=True, device=device
        )

        df_result_train = pd.DataFrame(
            {f"Preds{model_name}": preds}, index=data_train.items.index
        )
        preds, _ = fast_prediction(
            model, data_test, flatten=True, filter=True, device=device
        )

        df_result_test = pd.DataFrame(
            {f"Preds{model_name}": preds}, index=data_test.items.index
        )

        df_result = pd.concat([df_result_train, df_result_test])
        df_result.sort_index(inplace=True)

    return df_result


def setup_and_fit_linear_target_model(
    source_model,
    cats,
    conts,
    targets,
    is_ts=False,
    reuse_weights=False,
    num_layers_to_remove=1,
):
    alpha, beta = 1, 1
    if reuse_weights:
        alpha, beta = 1, 1

    name_layers_or_function_to_remove = "layers"
    if is_ts:
        name_layers_or_function_to_remove = reduce_layers_tcn_model
    # ann = copy.deepcopy(ann)
    target_model = LinearTransferModel(
        source_model,
        num_layers_to_remove=num_layers_to_remove,
        name_layers_or_function_to_remove=name_layers_or_function_to_remove,
        use_original_weights=reuse_weights,
        prediction_model=BayesLinReg(
            alpha=alpha,
            beta=beta,
            empirical_bayes=False,
            use_fixed_point=not reuse_weights,
            n_iter=15,
        ),
        as_multivariate_target=False,
    )
    x_tr = target_model(cats, conts)
    target_model = target_model.update(x_tr, targets)
    # target_model = target_model.eval()

    return target_model


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


def setup_training_data_stl(data_config, months_to_train_on, num_days_training):
    to_train_target = load_target_data(
        data_config, months_to_train_on, num_days_training
    )

    if to_train_target is not None:
        if data_config._model_config.is_timeseries_model:
            to_new = to_train_target.to.new(
                to_train_target.to.items,
            )
            to_new = Timeseries(to_new, splits=RandomSplitter(0.3))
        else:
            to_new = to_train_target.new(
                to_train_target.items, splits=TrainTestSplitByDays(0.3)
            )
            # to_new.process()
    else:
        to_new = None

    return to_new


def setup_helper_functions_stl(model_config, to_train_target):
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
    return bs, learner_type, convert_dl_to_tensor


def create_target_learner(
    learner_type, dls_train, target_model, config, is_bayes_tune=False, loss_func=None
):
    if is_bayes_tune:
        learner = learner_type(
            dls_train,
            copy.deepcopy(target_model),
            loss_func=BTuningLoss(lambd=config["lambd"]),
            metrics=btuning_rmse,
            opt_func=SGD,
        )
    else:
        if loss_func is None:
            target_model = copy.deepcopy(target_model)
        learner = learner_type(
            dls_train,
            target_model,
            metrics=rmse,
            opt_func=SGD,
            loss_func=loss_func,
        )
    learner.cbs = L(c for c in learner.cbs if type(c) == TrainEvalCallback)
    return learner


def stl_target_similarity_by_rmse(models, cats, conts, targets, is_ts):
    similarity_values, target_models = [], []

    for ann in models:
        target_model = copy_target_model(ann, is_ts=is_ts)
        target_models.append(target_model)
        target_model.eval()
        with torch.no_grad():
            preds = target_model(cats, conts)
        targets_cur, preds_cur = filter_preds(
            to_np(targets.reshape(-1)), to_np(preds.reshape(-1))
        )
        similarity_values.append(mse(targets_cur, preds_cur) ** 0.5)
    similarity_values = np.array(similarity_values)
    sort_ids = similarity_values.argsort()
    return target_models, similarity_values, sort_ids


def stl_target_similarity_by_evidence(
    models, cats, conts, targets, is_ts, num_layers_to_remove=1
):
    similarity_values, target_models, linear_target_models = [], [], []

    for ann in models:
        target_model = copy_target_model(ann, is_ts=is_ts)
        lt_model = setup_and_fit_linear_target_model(
            ann,
            cats,
            conts,
            targets,
            is_ts=is_ts,
            reuse_weights=False,
            num_layers_to_remove=num_layers_to_remove,
        )
        target_models += [target_model]
        linear_target_models += [lt_model]

    similarity_values, sort_ids = rank_by_evidence(
        cats, conts, targets, linear_target_models
    )

    return target_models, linear_target_models, similarity_values, sort_ids


def copy_target_model(source_model, is_ts=False):
    target_model = copy.deepcopy(source_model)

    return target_model
