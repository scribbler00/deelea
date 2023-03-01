import copy
import importlib
import pathlib
import warnings
from fastai.learner import load_learner
import joblib
import numpy as np
from pandas.core.series import Series
import pymongo
import gridfs
import pandas as pd
from io import BytesIO
import pickle as pkl
from collections import OrderedDict
import re
import json
from deelea.utils_models import ModelArchitecture
from deelea.utils_sacred import SacredConfig, sacred_config_as_connection_string


def convert_py_reduce_to_type(py_reduce):

    py_reduce_keys = list(py_reduce.keys())
    if len(py_reduce_keys) == 1 and "py/reduce" in py_reduce_keys:
        py_reduce = py_reduce["py/reduce"]

    if len(py_reduce) > 3:
        raise ValueError(
            "Do not know how to handle py reduce with more than three keys."
        )

    py_type = None
    py_tuple = None
    if type(py_reduce) == dict and "py/object" in list(py_reduce.keys()):
        py_type = py_reduce["py/object"]
        py_tuple = [py_reduce["value"]]
    else:
        for value in py_reduce:
            assert type(value) == dict
            key = list(value.keys())[0]
            if key == "py/type" or key == "py/object":
                py_type = value[key]
            elif key == "py/tuple":
                py_tuple = value[key]
            elif key != "dtype":
                ValueError(f"Unexpected key {key} in py_reduce.")

    module_names = py_type.split(".")
    module_name, module_type = module_names[0:-1], module_names[-1]
    module_name, module_type

    if len(module_name) > 1:
        module_name = ".".join(module_name)
    else:
        module_name = module_name[0]

    module = importlib.import_module(module_name)
    module_type = getattr(module, module_type, False)

    if module_type == pathlib.PosixPath:
        module_data = py_tuple[0] + py_tuple[0].join(py_tuple[1:])
    elif len(py_tuple) == 1:
        module_data = py_tuple[0]

    final_data = module_type(module_data)
    return final_data


def get_model_from_artifacts(grid_fs, cur_df_row):
    if type(cur_df_row.artifacts) == pd.DataFrame:
        object_id = get_object_id_in_artefact(cur_df_row.artifacts, filter="model")
    elif type(cur_df_row.artifacts) == pd.Series:
        object_id = get_object_id_in_artefact(
            cur_df_row.artifacts[cur_df_row.index[0]], filter="model"
        )
    else:
        raise ValueError("Unknown type.")
    model = read_artifact(grid_fs, object_id, "learner")

    return model


def get_data_from_keys(data, keys: list):
    new_data = {}
    for k in keys:
        if k not in data.keys():
            # print(f"key {k} not in {data.keys()} keys.")
            continue

        if type(data[k]) in [str, int, bool]:
            new_data[k] = data[k]
        elif type(data[k]) == dict and (
            "py/reduce" in list(data[k].keys()) or "py/object" in list(data[k].keys())
        ):
            new_data[k] = convert_py_reduce_to_type(data[k])

    return new_data


def read_artifact(grid_fs, object_id, data_type):
    bytes_object = BytesIO(grid_fs.get(object_id).read())
    if data_type == "csv":
        data = pd.read_csv(bytes_object, sep=";")
    elif data_type == "hdf":
        data = pd.read_hdf(bytes_object)
    elif data_type in ["pkl", "pickle"]:
        data = pkl.load(bytes_object)
    elif data_type == "json":
        data = json.loads(json.load(bytes_object))
    elif data_type == "learner":
        data = load_learner(bytes_object)
    elif data_type == "jbl":
        data = joblib.load(bytes_object)
    else:
        raise ValueError("Unknown type")

    return data


# based on
# https://github.com/yuvalatzmon/SACRED_HYPEROPT_Example/blob/master/mongo_queries.ipynb


def slice_dict(result_dict, keys):
    """Returns a dictionary ordered and sliced by given keys
    keys can be a list, or a CSV string
    """
    if isinstance(keys, str):
        keys = keys[:-1] if keys[-1] == "," else keys
        keys = re.split(", |[, ]", keys)

    return dict((k, result_dict[k]) for k in keys)


def _summarize_single_experiment(cur_experiment, relevant_keys):
    """
    Take only the relevant columns.
    """
    o = OrderedDict()
    o["_id"] = cur_experiment["_id"]
    o["name"] = cur_experiment["experiment"]["name"]
    o.update(cur_experiment["config"])
    for key, val in cur_experiment["info"].items():
        if key != "metrics":
            o[key] = val

    o.update(
        slice_dict(
            cur_experiment.to_dict(),
            relevant_keys,
        )
    )
    # o.update(slice_dict(s.to_dict(), "result, status, start_time, artifacts"))
    return pd.Series(o)


def sacred_db_to_df(
    db_runs,
    mongo_query={},
    additional_keys: str = "",  # e.g. start_time and heartbeat
    default_keys: str = "result, status, artifacts",
):
    """
    db_runs is usually db.runs
    returns a dataframe that summarizes the experiments, where
    config and info fields are flattened to their keys.
    Summary DF contains the following columns:
    _id, experiment.name, **config, result, **info, status, start_time
    """
    # get all experiment according to mongo query and represent as a pandas DataFrame

    df = pd.DataFrame(
        list(
            db_runs.find(
                mongo_query,
            )
        )
    )

    all_entries = []
    relevant_keys = (
        default_keys + "," + additional_keys
        if not default_keys.endswith(",") and len(additional_keys) > 0
        else default_keys
    )
    for _, cur_db_entry in df.iterrows():
        all_entries.append(_summarize_single_experiment(cur_db_entry, relevant_keys))
    df_summary = pd.DataFrame(all_entries).set_index("_id")

    if "heartbeat" in df_summary.columns:
        df_summary.heartbeat = pd.to_datetime(
            df_summary.heartbeat, utc=True, infer_datetime_format=True
        )
    if "start_time" in df_summary.columns:
        df_summary.start_time = pd.to_datetime(
            df_summary.start_time, utc=True, infer_datetime_format=True
        )

    df_summary.artifacts = df_summary.artifacts.apply(lambda x: pd.DataFrame(x))

    df_summary["model_config"] = df_summary["data_config"].apply(
        lambda x: x["_model_config"]
    )

    return df_summary


def get_mongo_db_connection(sacred_config: SacredConfig):
    con_str = sacred_config_as_connection_string(sacred_config)
    mongo_client = pymongo.MongoClient(con_str)
    db = mongo_client[sacred_config.db_name]
    db
    # grid_fs = gridfs.GridFS(db)
    return db, mongo_client


def prep_single_res_base(
    df: pd.DataFrame, cur_result_id: int, additional_keys_model_config: list = []
):
    df_artifacts = df.artifacts[cur_result_id]
    keys = ["data_file_name", "result_folder", "fold_id"]

    data_config = get_data_from_keys(df.data_config[cur_result_id], keys)

    keys = [
        "embedding_type",
        "full_bayes",
        "model_type",
        "model_architecture",
        "mtl_type",
    ] + additional_keys_model_config

    model_config = get_data_from_keys(df.model_config[cur_result_id], keys)

    df_res = pd.DataFrame.from_dict({**model_config, **data_config}, orient="index").T

    if len(df_artifacts) > 0:
        df_res.insert(0, "artifacts", [df_artifacts])

    if "result" in df.columns:
        if df.result.loc[cur_result_id] is not None and "py/object" in str(
            df.result.loc[cur_result_id]
        ):
            df_errors = get_data_from_keys(
                df.result.loc[cur_result_id],
                list(df.result.loc[cur_result_id].keys()),
            )
            df_errors = pd.DataFrame([df_errors])
        elif (
            df.result.loc[cur_result_id] is not None
            and not df.result.isna().loc[cur_result_id]
        ):
            df_errors = pd.DataFrame.from_records(
                json.loads(df.result.loc[cur_result_id])
            )
        else:
            df_errors = None
        df_res.insert(0, "results", [df_errors])

    return df_res


def convert_mongo_df(
    df, additional_keys_model_config=[], include_results=True, set_db_index=True
):
    df_results = []

    db_indexes = []
    for cur_result_id in df.index:
        try:
            df_res = prep_single_res_base(
                df,
                cur_result_id,
                additional_keys_model_config=additional_keys_model_config,
            )
            df_results.append(df_res)
            db_indexes.append(cur_result_id)
        except:
            warnings.warn(
                f"Could not read database entry {cur_result_id}. Skipping entry."
            )

    df_results = pd.concat(df_results, axis=0)
    if set_db_index:
        df_results.index = db_indexes
    df_results.result_folder = df_results.result_folder.apply(
        lambda x: str(x).split("/")[-1]
    )
    return df_results


def get_object_id_in_artefact(df_artifact, filter):
    object_id = None
    for i in df_artifact.index:
        if filter in df_artifact.name.iloc[i]:
            object_id = df_artifact.file_id.iloc[i]
            # we always take the last one
            # break
    return object_id


# %%
def get_all_best_hyperparams(df, grid_fs, filter="best_hyper", artifact_type="json"):
    df_best_hyperparams = []
    for idx in df.index:
        if len(df.iloc[idx].artifacts) > 0:
            object_id = get_object_id_in_artefact(df.iloc[idx].artifacts, filter)
            best_hyperparams = read_artifact(grid_fs, object_id, artifact_type)
            df_best_hyperparam = pd.DataFrame.from_dict(
                best_hyperparams, orient="index"
            ).T
            df_best_hyperparams.append(df_best_hyperparam)

    df_best_hyperparams = pd.concat(df_best_hyperparams, axis=0)

    return df_best_hyperparams


def is_experiment_done(df_res):
    if len(df_res) == 1:
        return True
    elif len(df_res) > 1:
        warnings.warn(
            f"There are more than two entries in the database with the same configuration. {df_res}"
        )
        return True
    else:
        return False


def is_configuration_running_or_completed(model_config, data_config, sacred_config):
    df_res, _ = get_results_of_db(model_config, data_config, sacred_config)

    df_res = filter_results_by_config(model_config, data_config, df_res)

    return is_experiment_done(df_res)


def get_results_of_db(
    model_config, data_config, sacred_config, set_db_index=True, additional_keys=[]
):
    db, mongo_client = get_mongo_db_connection(sacred_config)
    df = sacred_db_to_df(db.runs, additional_keys="heartbeat")
    df = df[(df.status == "RUNNING") | (df.status == "COMPLETED")]
    # %%
    df_res = convert_mongo_df(
        df,
        additional_keys_model_config=["emb_at_ith_layer"] + additional_keys,
        set_db_index=set_db_index,
    )
    df_res["Status"] = df.status
    if "emb_at_ith_layer" in df_res.columns:
        df_res.emb_at_ith_layer = df_res.emb_at_ith_layer.replace(np.nan, "first")
    else:
        df_res["emb_at_ith_layer"] = "first"

    df_res = df_res[df_res.model_architecture == model_config.model_architecture]

    return df_res, db


def filter_results_by_config(model_config, data_config, df_res):
    if len(df_res) == 0:
        return df_res

    df_res = df_res[df_res.fold_id == data_config.fold_id]
    df_res = df_res[data_config.result_folder.name == df_res.result_folder]
    df_res = df_res[df_res.fold_id == data_config.fold_id]

    if model_config.model_architecture == ModelArchitecture.STL:
        df_res = df_res[df_res.data_file_name == data_config.data_file_name]
    else:
        df_res = df_res[df_res.mtl_type == model_config.mtl_type]

    df_res = df_res[df_res.model_type == model_config.model_type]

    df_res = df_res[df_res.emb_at_ith_layer == model_config.emb_at_ith_layer]

    return df_res


def filter_results_by_config_target(model_config, data_config, df_res):
    if len(df_res) == 0:
        return df_res

    df_res = df_res[
        df_res.similarity_measure_type == model_config.similarity_measure_type
    ]
    df_res = df_res[df_res.adaption_strategy == model_config.adaption_strategy]
    df_res = df_res[df_res.ensemble_type == model_config.ensemble_type]

    if model_config.train_n_epochs not in [None]:
        df_res = df_res[df_res.ensemble_type == model_config.train_n_epochs]

    return df_res


def get_results_from_db(
    sacred_config,
    include_running_experiments=False,
    include_status=False,
    set_db_index=True,
):
    db, mongo_client = get_mongo_db_connection(sacred_config)
    df_db = sacred_db_to_df(db.runs)
    grid_fs = gridfs.GridFS(db)
    if include_running_experiments:
        mask = (df_db.status == "COMPLETED") | (df_db.status == "RUNNING")
    else:
        mask = df_db.status == "COMPLETED"

    df_db = df_db[mask]  # .reset_index()
    df_converted = convert_mongo_df(
        df_db,
        additional_keys_model_config=["emb_at_ith_layer"],
        set_db_index=set_db_index,
    )
    if include_status:
        df_converted["Status"] = df_db.status

    return df_converted, grid_fs


# %%
# %%
def extract_results(
    experiment_name="sourcetcn",
    include_error_results=True,
    include_running_experiments=False,
    include_status=False,
    hostname="localhost",
    set_db_index=True,
    arch_type=ModelArchitecture.MTL,
):
    sacred_config = SacredConfig("", experiment_name, hostname=hostname, port=27031)
    df_converted, grid_fs = get_results_from_db(
        sacred_config,
        include_running_experiments=include_running_experiments,
        include_status=include_status,
        set_db_index=set_db_index,
    )

    df_converted = copy.copy(df_converted)[df_converted.model_architecture == arch_type]

    df_converted.emb_at_ith_layer = df_converted.emb_at_ith_layer.fillna("first").values

    if include_error_results:
        df_experiements = pd.concat(df_converted.results.values)
        df_experiements.index = df_converted.index
        df_experiements = pd.concat([df_converted, df_experiements], axis=1)
    else:
        df_experiements = df_converted

    df_experiements.mtl_type = df_experiements.mtl_type.apply(lambda x: x.name)
    df_experiements.model_type = df_experiements.model_type.apply(lambda x: x.name)
    df_experiements["ModelName"] = (
        df_experiements.model_type
        + df_experiements.mtl_type
        + df_experiements.emb_at_ith_layer
    )
    return df_experiements, grid_fs
