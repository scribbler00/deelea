# from operator import is_
from fastai.tabular.core import TabDataLoader, TabularPandas
from fastrenewables.models.autoencoders import AutoencoderForecast
from fastrenewables.tabular.core import TabularRenewables
import numpy as np
from numpy.lib import math

import pickle as pkl
import torch

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

from fastcore.foundation import L
from fastai.learner import Learner
from fastai.torch_core import tensor, to_np
from fastai.data.load import DataLoader

from fastrenewables.tabular.learner import RenewableLearner
from fastrenewables.utils import filter_preds
from fastrenewables.utils_pytorch import monte_carlo_dropout
from fastrenewables.metrics import (
    normalized_sum_of_squared_residuals_torch,
    distance_ideal_curve,
)
from deelea.utils_data import DataConfig, read_data_as_dl
from deelea.utils_models import ModelConfig
from fastrenewables.metrics import crps_for_quantiles
from scipy import stats


def get_distance_curve(learner, ds_idx=0, dl=None, n_samples=100):

    loss_func = learner.loss_func
    if type(learner.model) == AutoencoderForecast:
        forecast_model = learner.model.forecast_model
    else:
        forecast_model = learner.model
        # raise ValueError("Model type not yet supported.")
    if not isinstance(dl, DataLoader) and dl is not None and len(list(dl)) > 1:
        dl = dl[0]

    monte_carlo_dropout(forecast_model, True)

    cats, conts, targets = dl_to_tensors(learner.dls, ds_idx=ds_idx, test_dl=dl)

    with torch.no_grad():
        distance_curve = 0

        samples = []
        for _ in range(n_samples):
            single_sample = learner.model(cats, conts)
            norm_dist_sample = torch.distributions.Normal(
                single_sample[:, 0], (single_sample[:, 1].exp()) * 2
            )
            samples_normal_dist = norm_dist_sample.rsample(
                sample_shape=[min(int(n_samples / 10), 10)]
            ).T

            samples.append(samples_normal_dist)

    learner.model.eval()
    monte_carlo_dropout(forecast_model, False)

    p_prediction, p_expected = normalized_sum_of_squared_residuals_torch(
        torch.stack(samples, dim=1).reshape(targets.shape[0], 1, -1).detach(),
        targets.reshape(-1, 1),
    )

    distance_curve = distance_ideal_curve(p_prediction, p_expected)

    if math.isnan(distance_curve):
        distance_curve = 1e12

    return distance_curve


def get_errors(
    learner: RenewableLearner,
    ds_idx: int = 0,
    dl: DataLoader = None,
    is_bayes: bool = False,
    prefix: str = "",
    filter_preds: bool = True,
):
    learner.model.eval()

    if dl is not None and len(list(dl)) > 1:
        dl = dl[0]

    preds, targets = get_preds(
        learner,
        ds_idx=ds_idx,
        dl=dl,
        flatten=True,
        is_bayes=is_bayes,
        filter_predictions=filter_preds,
    )

    mask_nans = ~np.isnan(preds)
    preds, targets = preds[mask_nans], targets[mask_nans]

    distance_curve = 0
    if is_bayes:
        distance_curve = get_distance_curve(learner, ds_idx=ds_idx, dl=dl)

    res_dict = get_error_dict(targets, preds, distance_curve, prefix)

    return res_dict


def get_preds(
    learner, ds_idx=1, dl=None, flatten=True, is_bayes=False, filter_predictions=True
):
    preds, targets = learner.get_preds(ds_idx=ds_idx, dl=dl)

    preds, targets = to_np(preds), to_np(targets)

    if is_bayes:
        preds = preds[:, 0]

    if flatten:
        preds, targets = preds.reshape(-1), targets.reshape(-1)

    if filter_predictions:
        targets, preds = filter_preds(targets, preds)

    return preds, targets


def dl_to_tensors(dls, ds_idx=0, test_dl=None):

    if test_dl is not None:
        to = test_dl.dataset
    elif ds_idx == 0:
        to = dls.train_ds
    elif ds_idx == 1:
        to = dls.valid_ds

    # to increase speed we direclty predict on all tensors
    if isinstance(to, (TabularPandas, TabularRenewables, TabDataLoader)):
        if getattr(to, "regression_setup", False):
            ys_type = np.float32
        else:
            ys_type = np.long

        cats = tensor(to.cats.values.astype(np.long))
        conts = tensor(to.conts.values.astype(np.float32))
        targets = tensor(to.ys.values.astype(ys_type))

    return cats, conts, targets


def get_errors_bayes_sklearn_model(learner, dataset, prefix=""):
    X, y = dataset.conts.values, dataset.ys.values.ravel()
    preds = learner.predict(X)

    y, preds = filter_preds(y, preds)

    error_dict = get_error_dict(y, preds, prefix=prefix)
    error_dict[f"{prefix}log_evidence"] = learner.log_evidence(X, y)

    return error_dict


def get_crps_error(targets, yhat, yhat_std, prefix):
    if len(yhat) > 0:
        quantiles = np.linspace(0.01, 0.99, 99)
        quantile_forecasts = stats.norm(
            yhat.ravel().reshape(-1, 1), yhat_std.ravel().reshape(-1, 1)
        ).ppf(quantiles)
        quantile_forecasts[quantile_forecasts < 0] = 0
        quantile_forecasts[quantile_forecasts > 1.1] = 1.1
        # avoid quantile crossing
        # quantile_forecasts = quantile_forecasts[quantile_forecasts.argsort(1)]
        crps_error = crps_for_quantiles(quantile_forecasts, targets.ravel(), quantiles)[
            0
        ]
    else:
        crps_error = 1e12

    return {f"{prefix}crps": crps_error}


def get_error_dict(targets, preds, distance_curve=0, prefix=""):
    if len(preds) > 0:
        targets, preds = targets.reshape(-1), preds.reshape(-1)
        rmse = mse(targets, preds, squared=False)
        r2 = r2_score(targets, preds)
        mae = mean_absolute_error(targets, preds)
        bias = np.mean(targets - preds)
    else:
        rmse, r2, mae, bias = 1e12, 0, 1e12, 1e12

    res_dict = {
        f"{prefix}rmse": rmse,
        f"{prefix}r2": r2,
        f"{prefix}mae": mae,
        f"{prefix}bias": bias,
        f"{prefix}distance_ideal_curve": distance_curve,
    }

    return res_dict


def get_all_errors_sklearn_bayes_model(
    learner: Learner, data_config: DataConfig, model_config: ModelConfig
):
    dl_train = read_data_as_dl(
        data_config.train_file,
        model_config,
        one_batch=False,
        drop_last=False,
    )

    dl_test = read_data_as_dl(
        data_config.test_file,
        model_config,
        one_batch=False,
        drop_last=False,
    )[0]

    error_dict_train = get_errors_bayes_sklearn_model(
        learner, dl_train.train_ds, prefix="train_"
    )
    error_dict_valid = get_errors_bayes_sklearn_model(
        learner, dl_train.valid_ds, prefix="valid_"
    )
    error_dict_test = get_errors_bayes_sklearn_model(
        learner, dl_test.dataset, prefix="test_"
    )

    result_dict = {**error_dict_train, **error_dict_valid, **error_dict_test}

    return result_dict


def get_all_errors_dl_model(
    model_config: ModelConfig,
    test_file: str,
    learner: Learner,
    filter_preds: bool = True,
):
    train_errors = get_errors(
        learner,
        ds_idx=0,
        is_bayes=model_config.full_bayes,
        prefix="train_",
        filter_preds=filter_preds,
    )

    val_errors = get_errors(
        learner,
        ds_idx=1,
        is_bayes=model_config.full_bayes,
        prefix="valid_",
        filter_preds=filter_preds,
    )

    dl_test = read_data_as_dl(
        test_file,
        model_config,
        one_batch=False,
        drop_last=False,
    )

    test_errors = get_errors(
        learner,
        dl=dl_test,
        is_bayes=model_config.is_bayes,
        prefix="test_",
    )

    return {**train_errors, **val_errors, **test_errors}
