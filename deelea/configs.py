import copy
import warnings
import numpy as np
from optuna.trial import Trial
import deelea.utils_data as utils_data


def get_elm_config(model_config):
    config = {
        "empirical_bayes": True,
        "alpha": 1,
        "beta": 10,
        "activations": ["relu"],
        "n_hidden": 10,
        "n_iter": 100,
        "include_original_features": True,
        "batch_size": 4,  # just for compatibility with dl training
        "minimum_hidden_size": 11,
    }

    return config


def get_elm_config_optimization(trial: Trial, model_config):

    config = get_elm_config(model_config)

    config["include_original_features"] = trial.suggest_categorical(
        "include_original_features", [True, False]
    )

    if model_config.smoke_test:
        config["n_hidden"] = trial.suggest_uniform("n_hidden", 10, 15)
    else:
        config["n_hidden"] = trial.suggest_uniform("n_hidden", 10, 1000)

    config["activations"] = trial.suggest_categorical(
        "activations", ["relu", "sigmoid", "relu,sigmoid"]
    )

    return config


def get_mlp_config(model_config):

    config = {
        "lr": 1e-3,
        "wd": 0.0,
        "epochs": 5,
        "batch_size": 4096,
        "dropout": 0.0,
        "embed_dropout": 0.0,
        # "hidden_layers": [200, 100, 10],
        "minimum_hidden_size": 11,
        "filter_preds": True,
        "percental_reduce": 50,
        "input_size_multiplier": 5,
        "second_last_layer_size": 3,
        "use_embedding": True,
    }

    if model_config.is_bayes:
        config = _get_simple_bayes_config(config)

    return config


def get_mlp_config_optimization(trial: Trial, model_config):
    config = get_mlp_config(model_config)

    config["input_size_multiplier"] = trial.suggest_int("input_size_multiplier", 1, 20)

    config = _suggest_base_config(trial, config, model_config)

    config = _suggest_wd_config(trial, config, model_config)

    config = _suggest_epochs_config(trial, config, model_config)

    config = _suggest_dropout_config(trial, config, model_config)

    return config


def get_aemlp_config(model_config):
    config = {
        "lr": 1e-3,
        "wd": 0.0,
        "epochs": 5,
        "batch_size": 4096,
        "dropout": 0.0,
        "embed_dropout": 0.0,
        "minimum_hidden_size": 9,
        "filter_preds": True,
        "percental_reduce": 80,
        "input_size_multiplier": 5,
        "use_embedding": False,
    }

    if model_config.is_bayes:
        raise NotImplementedError

    return config


def get_aemlp_config_optimization(trial: Trial, model_config):
    config = get_aemlp_config(model_config)
    # raise NotImplementedError
    config["epochs"] = [10]
    config["lr"] = [1e-1]

    return config


def get_tcn_config(model_config):
    config = get_mlp_config(model_config)

    return config


def get_aetcntcn_config(model_config):

    config = get_aemlp_config(model_config)

    config["emb_at_ith_layer"] = "first"

    return config


def get_aetcntcn_config_optimization(trial: Trial, model_config):
    config = get_aemlp_config_optimization(trial, model_config)
    # TODO: should we do this or rather have one model for each type?
    config["emb_at_ith_layer"] = trial.suggest_categorical(
        "emb_at_ith_layer", ["first", "all"]
    )

    return config


def get_tcn_config_optimization(trial: Trial, model_config):
    config = get_tcn_config(model_config)

    if model_config.is_mtl:
        config["input_size_multiplier"] = trial.suggest_categorical(
            "input_size_multiplier", [1, 5, 10, 15, 20]
        )
    else:
        config["input_size_multiplier"] = trial.suggest_int(
            "input_size_multiplier", 1, 20
        )

    config = _suggest_base_config(trial, config, model_config)

    config = _suggest_wd_config(trial, config, model_config)

    config = _suggest_epochs_config(trial, config, model_config)

    config = _suggest_dropout_config(
        trial,
        config,
        model_config,
    )

    config["emb_at_ith_layer"] = model_config.emb_at_ith_layer

    return config


def get_ae_config(model_config):
    config = {
        "lr": 1e-3,
        "wd": 0.0,
        "epochs": 5,
        "batch_size": 2048,
        "dropout": 0.0,
        "embed_dropout": 0.0,
        "minimum_hidden_size": 9,
        "filter_preds": True,
        "percental_reduce": 70,
        "input_size_multiplier": 0.7,
        "use_embedding": False,
    }

    if model_config.is_bayes:
        raise NotImplementedError

    # config["input_size_multiplier"] = 0.5
    config["percental_reduce"] = 70
    config["minimum_hidden_size"] = 2
    # size latent space
    config["second_last_layer_size"] = 5
    config["filter_preds"] = False
    config["use_embedding"] = False

    return config


def get_ae_config_optimization(trial: Trial, model_config):
    config = get_ae_config(model_config)

    if model_config.is_mtl:
        config["epochs"] = [50, 100, 200, 300, 400]
    else:
        config["epochs"] = [50, 100, 200]

    config["lr"] = [1e-2, 1e-3, 1e-4]

    if model_config.data_type == utils_data.Datatype.PVOPEN:
        config["batch_size"] = 512

    reduce_factor_timeseries = 1
    if (
        model_config.is_timeseries_model
        and model_config.data_type == utils_data.Datatype.PVOPEN
    ):
        # PVOPEN has a one hour resolution
        reduce_factor_timeseries = 24
    elif model_config.is_timeseries_model:
        # all other datasets have a one 15 minute resolution
        reduce_factor_timeseries = 96

    if model_config.is_mtl:
        if model_config.data_type in [
            utils_data.Datatype.WINDSYN,
            utils_data.Datatype.WINDREAL,
            utils_data.Datatype.PVSYN,
        ]:
            config["batch_size"] = int(4096 * 30 // reduce_factor_timeseries)
        else:
            config["batch_size"] = int(4096 * 20 // reduce_factor_timeseries)
    else:
        config["batch_size"] = int(config["batch_size"] // reduce_factor_timeseries)

    return config


def _get_simple_bayes_config(config: dict):
    config["epochs"] = 200
    config["dropout"] = 0.2
    config["wd"] = 0.1
    config["embed_dropout"] = config["dropout"]

    return config


def _suggest_dropout_config(
    trial: Trial, config: dict, model_config, min_value=0, max_value=0.9
):
    if model_config.is_bayes:
        config["dropout"] = trial.suggest_uniform("dropout", 0.05, 0.6)
        config["embed_dropout"] = config["dropout"]
    elif model_config.is_mtl and model_config.is_timeseries_model:
        config["dropout"] = trial.suggest_categorical(
            "dropout", [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        )
    else:
        config["dropout"] = trial.suggest_uniform("dropout", min_value, max_value)
    return config


def _suggest_wd_config(trial: Trial, config: dict, model_config):
    if model_config.is_bayes:
        config["wd"] = trial.suggest_uniform("wd", 0.05, 0.5)
    else:
        config["wd"] = trial.suggest_categorical("wd", [0.0, 0.1, 0.2, 0.4])
    return config


def _suggest_base_config(trial: Trial, config: dict, model_config):
    from deelea.utils_models import ModelType

    # for MLP models we do not reduce the batch size
    reduce_factor_timeseries = 1

    if (
        model_config.is_timeseries_model
        and model_config.data_type == utils_data.Datatype.PVOPEN
    ):
        # PVOPEN has a one hour resolution
        reduce_factor_timeseries = 24
    elif model_config.is_timeseries_model:
        # all other datasets have a one 15 minute resolution
        reduce_factor_timeseries = 96

    if model_config.is_mtl:

        if model_config.data_type in [
            utils_data.Datatype.PVSYN,
            utils_data.Datatype.PVREAL,
            utils_data.Datatype.WINDSYN,
            utils_data.Datatype.WINDREAL,
        ] and model_config.model_type in [ModelType.AE, ModelType.AETCN]:
            config["batch_size"] = trial.suggest_categorical(
                "batch_size",
                [4096 * 15, 4096 * 25, 4096 * 35],
            )
        elif model_config.data_type in [
            None,
            utils_data.Datatype.PVSYN,
            utils_data.Datatype.PVREAL,
            utils_data.Datatype.PVOPEN,
        ]:
            config["batch_size"] = trial.suggest_categorical(
                "batch_size",
                [
                    4096 * 15 // reduce_factor_timeseries,
                    4096 * 20 // reduce_factor_timeseries,
                    4096 * 25 // reduce_factor_timeseries,
                ],
            )
        elif model_config.data_type in [
            utils_data.Datatype.WINDSYN,
            utils_data.Datatype.WINDREAL,
            utils_data.Datatype.WINDOPEN,
        ]:
            config["batch_size"] = trial.suggest_categorical(
                "batch_size",
                [
                    4096 * 35 // reduce_factor_timeseries,
                    4096 * 40 // reduce_factor_timeseries,
                    4096 * 45 // reduce_factor_timeseries,
                ],
            )
        else:
            raise ValueError("Unknown datatype.")
    else:
        config["batch_size"] = trial.suggest_categorical(
            "batch_size",
            [
                1024 // reduce_factor_timeseries,
                2048 // reduce_factor_timeseries,
                4096 // reduce_factor_timeseries,
            ],
        )

    if model_config.smoke_test:
        config["lr"] = 1e-3
    else:
        config["lr"] = trial.suggest_loguniform("lr", 1e-6, 1e-1)

    return config


def _suggest_epochs_config(
    trial: Trial, config: dict, model_config, min_epochs=10, max_epochs=100
):
    if model_config.smoke_test:
        config["epochs"] = trial.suggest_int("epochs", 1, 5)
    elif model_config.is_bayes:
        config["epochs"] = trial.suggest_uniform("epochs", 10.0, 2000)
    else:
        config["epochs"] = trial.suggest_int("epochs", min_epochs, max_epochs)

    return config

def get_gbrt_config(model_config):

    parameters_gbrt = {
        "loss": "ls",
        "learning_rate": 1e-3,  # selected
        "n_estimators": 50,  # selected
        "subsample": 1,
        "criterion": "friedman_mse",
        "min_samples_split": 2,
        "min_samples_leaf": 10,  # selected
        "min_weight_fraction_leaf": 0,
        "max_depth": 2,  # selected
        "min_impurity_decrease": 0,
        "init": None,
        "max_features": "auto",  # if “auto”, then max_features=n_features.
        "max_leaf_nodes": None,
        "validation_fraction": 0.1,
        "n_iter_no_change": None,  # Only used if n_iter_no_change is set to an integer.
        "tol": 1e-4,  # Only used if n_iter_no_change is set to an integer.
        "ccp_alpha": 0.0,
    }

    return parameters_gbrt


def get_gbrt_config_optimization(_, model_config):
    """
    Returns the to be optimized hyperparameters of the gbrt baseline model. First argument is for compability and can be ignored.
    """
    parameters_gbrt = get_gbrt_config(model_config)
    for k, v in parameters_gbrt.items():
        parameters_gbrt[k] = [v]

    if model_config.smoke_test:
        parameters_gbrt["max_depth"] = [2, 4]
    else:
        parameters_gbrt["learning_rate"] = np.logspace(-6, 0, num=13)
        parameters_gbrt["max_depth"] = [2, 4, 6, 8]
        parameters_gbrt["n_estimators"] = [300]

    return parameters_gbrt


def get_stl_target_optimization(config={}):
    configs = []
    for epochs in [1]:
        for wd in list(np.logspace(start=-1, stop=-3, num=7, base=10)):
            for lr in list(np.logspace(start=-1, stop=-4, num=7, base=10)):
                new_conf = copy.copy(config)
                new_conf["epochs"] = epochs
                new_conf["wd"] = float(wd)
                new_conf["lr"] = float(lr)
                configs.append(new_conf)
    return configs


def get_stl_target_optimization_wd_source(config={}):
    configs = []
    for epochs in [1]:
        for wd in [1, 0.1]:
            for lr in list(np.logspace(start=-1, stop=-4, num=7, base=10)):
                new_conf = copy.copy(config)
                new_conf["epochs"] = epochs
                new_conf["wd"] = float(wd)
                new_conf["lr"] = float(lr)
                configs.append(new_conf)
    return configs


def get_stl_target_btunning_optimization(config={"wd": 0}):
    configs = []
    for epochs in [1]:
        for lambd in [0.1, 0.25, 0.5, 1, 2, 4, 8]:
            for lr in list(np.logspace(start=-1, stop=-4, num=7, base=10)):
                new_conf = copy.copy(config)
                new_conf["epochs"] = epochs
                new_conf["lambd"] = lambd
                new_conf["lr"] = float(lr)
                configs.append(new_conf)
    return configs
