import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from scipy.stats import t


class PWR:
    METRICS = list(KERNEL_PARAMS.keys()) + ["precomputed"]

    def __init__(
        self,
        prior_exp_mean=0,
        prior_exp_prec=1,
        prior_obs_mean=1,
        prior_obs_prec=1,
        n_neighbors=None,
        metric="rbf",
        metric_dict=None,
    ):
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.metric_dict = {} if metric_dict is None else metric_dict
        self.prior_exp_mean = prior_exp_mean
        self.prior_exp_prec = prior_exp_prec
        self.prior_obs_mean = prior_obs_mean
        self.prior_obs_prec = prior_obs_prec

    def fit(self, X, y):
        self.X_ = X.copy()
        self.y_ = y.copy()
        return self

    def predict_freq(self, X):
        K = pairwise_kernels(X, self.X_, metric=self.metric, **self.metric_dict)

        # maximum likelihood
        N = np.sum(K, axis=1)
        mu_ml = K @ self.y_ / N
        sigma_ml = np.sqrt((K @ self.y_ ** 2 / N) - mu_ml ** 2)

        # prior parameters
        mu_0 = self.prior_exp_mean
        lmbda_0 = self.prior_obs_mean
        alpha_0 = self.prior_obs_prec * 0.5
        beta_0 = alpha_0 / self.prior_exp_prec

        # posterior update
        mu_N = (lmbda_0 * mu_0 + N * mu_ml) / (lmbda_0 + N)
        lmbda_N = lmbda_0 + N
        alpha_N = alpha_0 + N / 2
        beta_N = (
            beta_0
            + 0.5 * N * sigma_ml ** 2
            + 0.5 * (lmbda_0 * N * (mu_ml - mu_0) ** 2) / (lmbda_0 + N)
        )
        df = alpha_N
        loc = mu_N
        scale = (beta_N * (lmbda_N + 1)) / (alpha_N * lmbda_N)
        mean, var, skew, kurt = t.stats(
            df=2 * alpha_N, loc=loc, scale=scale, moments="mvsk"
        )

        return mu_N, np.sqrt(beta_N / alpha_N)
