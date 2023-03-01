from pathlib import Path
import pickle as pkl
import pymongo

from sklearn.base import BaseEstimator

from sacred.experiment import Experiment
from sacred.observers.file_storage import FileStorageObserver
from sacred.observers import MongoObserver, FileStorageObserver

from fastai.learner import Learner
import fastrenewables

from deelea.utils import dump_dict_as_json
from deelea.utils_eval import (
    get_all_errors_dl_model,
    get_all_errors_sklearn_bayes_model,
)
from deelea.utils_data import DataConfig
import deelea.utils_models as utils_models


class SacredConfig:
    def __init__(
        self,
        experiment_name,
        db_name,
        mongo_observer=True,
        hostname="alatar",
        username="transfer",
        password="transfer",
        port="27031",
    ):
        self.experiment_name = experiment_name
        self.db_name = db_name
        self.mongo_observer = mongo_observer
        self.username = username
        self.password = password
        self.hostname = hostname
        self.port = port


def sacred_config_as_connection_string(sacred_config: SacredConfig):
    if sacred_config.hostname in ["localhost", "127.0.0.1"]:
        hostname = sacred_config.hostname
    elif "ies" not in sacred_config.hostname:
        hostname = f"{sacred_config.hostname}.ies.uni-kassel.de"
    else:
        hostname = sacred_config.hostname

    con_str = f"mongodb://{sacred_config.username}:{sacred_config.password}@{hostname}:{sacred_config.port}"

    return con_str


def delete_table(sacred_config: SacredConfig, table_name: str):
    con_str = sacred_config_as_connection_string(sacred_config)
    mongo_client = pymongo.MongoClient(con_str)
    mongo_client.drop_database(table_name)


def create_sacred_experiment(sacred_config: SacredConfig):
    experiment = Experiment(sacred_config.experiment_name)

    if sacred_config.mongo_observer:

        # for omniboard run e.g. `omniboard -m localhost:27029:test`
        experiment.observers.append(
            MongoObserver(
                url=sacred_config_as_connection_string(sacred_config),
                db_name=sacred_config.db_name,
            )
        )
    else:
        experiment.observers.append(
            FileStorageObserver(
                f"{sacred_config.db_name}_{sacred_config.experiment_name}"
            )
        )

    experiment.add_package_dependency("fastrenewables", fastrenewables.__version__)

    return experiment


def store_hyperparameter_results(
    experiment, hyperparameter_results, best_config, tmp_dir
):

    add_sacred_artifact(
        hyperparameter_results, tmp_dir + "/hyperparameter_results.csv", experiment
    )

    add_sacred_artifact(
        best_config, tmp_dir + "/best_hyperparameter_results.json", experiment
    )


import joblib


def add_sacred_artifact(
    result_to_store, file_name: str, sacred_experiment: Experiment, content_type=None
):
    if file_name.endswith(".csv"):
        result_to_store.to_csv(file_name, sep=";")
    elif file_name.endswith(".json"):
        dump_dict_as_json(result_to_store, file_name)
    elif file_name.endswith(".pkl"):
        pkl.dump(result_to_store, open(file_name, "wb"))
    elif file_name.endswith(".jbl"):
        joblib.dump(result_to_store, open(file_name, "wb"))
    else:
        raise NotImplementedError

    sacred_experiment.add_artifact(
        file_name,
        name=file_name.replace(".csv", "")
        .replace(".json", "")
        .replace(".pkl", "")
        .replace(".jbl", ""),
        content_type=content_type,
    )


def log_dict_as_scalar(sacred_experiment, dict_to_log):
    for key in dict_to_log.keys():
        sacred_experiment.log_scalar(key, dict_to_log[key])


def save_and_log_sklearn_bayes_results(
    model_config: utils_models.ModelConfig,
    data_config: DataConfig,
    learner: BaseEstimator,
    sacred_experiment,
):

    result_dict = get_all_errors_sklearn_bayes_model(learner, data_config, model_config)

    log_dict_as_scalar(sacred_experiment, result_dict)

    add_sacred_artifact(
        learner,
        f"{data_config.tmp_dir}/{model_config.model_type.name}_model.pkl",
        sacred_experiment,
    )

    return result_dict


def save_and_log_dl_results(
    model_config: utils_models.ModelConfig,
    data_config: DataConfig,
    learner: Learner,
    sacred_experiment: Experiment,
    filter_preds: bool = True,
):
    result_dict = get_all_errors_dl_model(model_config, data_config.test_file, learner)

    log_dict_as_scalar(sacred_experiment, result_dict)

    model_folder = Path(data_config.tmp_dir)
    model_folder.mkdir(exist_ok=True, parents=True)

    learner.path = model_folder
    file_name = f"{model_config.model_type.name}_model.pkl"

    learner.export(fname=file_name)

    sacred_experiment.add_artifact(
        model_folder / file_name, name=file_name.replace(".pkl", "")
    )

    return result_dict
