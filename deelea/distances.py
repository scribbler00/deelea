from pathlib import Path
from typing import Callable

from scipy.spatial.distance import euclidean
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm, expon, weibull_min, poisson, beta, chi2
from sklearn.metrics import mean_squared_error as mse
from sklearn.neighbors import KernelDensity
from dtaidistance import dtw

from deelea.utils_data import Datatype, FeaturesConfig


def distance_between_features(
    df_one: pd.DataFrame, df_two: pd.DataFrame, distance_function: Callable
) -> list:
    results = []
    for c in df_one.columns:
        try:
            res = distance_function(
                df_one[[c]].values.ravel(), df_two[[c]].values.ravel()
            )
        except:
            res = np.nan
        results.append(res)
    return results


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return ((x.ravel() - y.ravel()) ** 2).mean() ** 0.5


def dtw_eucl(x: np.ndarray, y: np.ndarray) -> float:
    if len(x.shape) > 1 or len(y.shape) > 1:
        raise ValueError("Only accepting one dimensional arrays")

    return dtw(x.ravel().astype(np.double), y.ravel().astype(np.double))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    # expresses linear correleation. in case x and y increase linearly at the same but with different amounts the the value is less than 1
    return pearsonr(x.ravel(), y.ravel())[0]  # only return value


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    # expresses linear correlation: in case that x and y increase at the same time but with different amounts the spearman correlation might be still 1
    return spearmanr(x.ravel(), y.ravel())[0]  # only return value


def kld(x: np.ndarray, y: np.ndarray) -> float:
    return calculate_measure(
        x.ravel(), y.ravel(), measure="kld", fit_func=KernelDensity
    )


def hellinger(x: np.ndarray, y: np.ndarray) -> float:
    # return hellinger_closed_form(x.ravel(), y.ravel(), fit_func=norm)
    return calculate_measure(
        x.ravel(), y.ravel(), measure="hellinger", fit_func=KernelDensity
    )


def calculate_measure(
    x,
    y,
    measure="kld",
    fit_func=KernelDensity,
    eps=1.0e-5,
    min_prob=1e-256,
    quadrature=False,
    bandwidth=1,
):
    """
    Calculate difference between x and y, according to given measure.
    Parameters
    ----------
    x : numpy array
        dataset
    y : numpy array
        dataset
    measure : string
        'kld' or 'hellinger', default is 'kld'
    fit_func : function
      KernelDensity or chi2, default is KernelDensity
    eps : float
        used for numerical stability
    min_prob : float
        used to avoid division by zero
    quadrature : boolean
        define method of integration
    bandwidth : float
        parameter for size of bandwidth used in KDE
    """
    min_v, max_v = _get_min_max(x, y)
    rvs_x, rvs_y = _fit_stats_func(x, y, fit_func=fit_func, bandwidth=bandwidth)

    if fit_func == KernelDensity:

        def pdf_x(k):
            return np.exp(rvs_x.score_samples(_reshape(k)))

        def pdf_y(k):
            return np.exp(rvs_y.score_samples(_reshape(k)))

    else:

        def pdf_x(k):
            return rvs_x.pdf(k)

        def pdf_y(k):
            return rvs_y.pdf(k)

    if measure == "kld":

        def int_func(k):
            return (pdf_x(k) + min_prob) * np.log(
                (pdf_x(k) + min_prob) / (pdf_y(k) + min_prob)
            )

        if quadrature:
            return integrate.quadrature(int_func, min_v, max_v, maxiter=1000, tol=eps)[
                0
            ]
        else:
            return integrate.quad(int_func, min_v, max_v)[0]

    elif measure == "hellinger":

        def int_func(k):
            return np.sqrt(pdf_x(k) * pdf_y(k))

        if quadrature:
            return (
                1
                - integrate.quadrature(int_func, min_v, max_v, maxiter=1000, tol=eps)[0]
            )
        else:
            return 1 - integrate.quad(int_func, min_v, max_v)[0]


def hellinger_closed_form(x, y, fit_func=norm):
    """
    Closed form of hellinger-distance computation
    """
    if fit_func == norm:
        mu_x, sigma_x = fit_func.fit(x)
        mu_y, sigma_y = fit_func.fit(y)
        a = np.sqrt((2 * sigma_x * sigma_y) / (sigma_x ** 2 + sigma_y ** 2))
        b = np.exp((-1 / 4) * ((mu_x - mu_y) ** 2) / (sigma_x ** 2 + sigma_y ** 2))
        return 1 - a * b

    elif fit_func == expon:
        _, alpha = fit_func.fit(x)
        _, bet = fit_func.fit(y)
        a = 2 * np.sqrt(alpha * bet)
        b = alpha + bet
        return 1 - a / b

    elif fit_func == weibull_min:
        c_x, _, alpha = fit_func.fit(x)
        _, _, bet = fit_func.fit(y)
        a = 2 * (alpha * bet) ** (c_x / 2)
        b = alpha + bet
        return 1 - a / b

    elif fit_func == beta:
        a1, b1, _, _ = fit_func.fit(x)
        a2, b2, _, _ = fit_func.fit(y)
        a = fit_func(a=(a1 + a2) / 2, b=(b1 + b2) / 2).expect()
        b = np.sqrt(fit_func(a=a1, b=b1).expect() * fit_func(a=a2, b=b2).expect())
        return 1 - a / b


def _get_min_max(x, y):
    min_v = np.min([np.min(x), np.min(y)])
    max_v = np.max([np.max(x), np.max(y)])
    return min_v, max_v


def _fit_stats_func(x, y, fit_func=KernelDensity, bandwidth=1):

    if fit_func == KernelDensity:
        kde = fit_func(bandwidth=bandwidth)
        rvs_x = kde.fit(x.reshape(-1, 1))

        kde = fit_func(bandwidth=bandwidth)
        rvs_y = kde.fit(y.reshape(-1, 1))

    else:
        pars = fit_func.fit(x)
        rvs_x = fit_func(*pars)

        pars = fit_func.fit(y)
        rvs_y = fit_func(*pars)

    return rvs_x, rvs_y


def _reshape(x):
    x = np.array(x)
    x = x.reshape(-1, 1)
    return x


import pickle as pkl


def get_bayes_pwr_similarity(
    model_config, data_config, file_name, to_train_target, n_samples=1000
):
    # file_names = data_config.source_splits
    # file_name = ""  # file_names[0]
    file_name = Path(file_name).stem  # TODO loop over
    file_name = data_config._get_data_file(
        data_config._model_config.model_architecture, "train", file_name
    )
    to_train_source = pkl.load(open(file_name, "rb"))
    features_config = FeaturesConfig(
        Datatype.datafolder_to_datatype(data_config.result_folder),
        model_config.model_architecture,
    )
    most_relevant_features = features_config.most_relevant_features
    from deelea.modelselection import PWR

    df_source = to_train_source.items
    df_source = df_source.loc[df_source.index.minute == 0, :]
    X_source, y_source = (
        df_source.loc[:, most_relevant_features].values,
        df_source.loc[:, "PowerGeneration"].values,
    )

    model = PWR(
        metric_dict={"gamma": 500},
        prior_exp_mean=0.1,
        prior_obs_mean=1e-5 * len(X_source),
        prior_exp_prec=1e-1,
        prior_obs_prec=1e-5 * len(X_source),
    )
    model.fit(X_source, y_source)

    df_target = to_train_target.items
    df_target = df_target.loc[df_target.index.minute == 0, :]
    X_target, y_target = (
        df_target.loc[:, most_relevant_features].values,
        df_target.loc[:, "PowerGeneration"].values,
    )
    mu, sigma = model.predict_freq(X_target)
    y_sampled = np.array(
        [np.random.normal(loc=mu, scale=sigma) for _ in range(n_samples)]
    ).T
    similarity = ((y_target[:, None] - y_sampled) ** 2).mean() ** 0.5
    similarity = similarity * -1

    return similarity
