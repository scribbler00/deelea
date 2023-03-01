import json
from os import error
from pathlib import Path
from fastcore.basics import *
from fastrenewables.tabular.core import str_to_path
import pickle as pkl

import pandas as pd


def read_csv(file):
    df = pd.read_csv(file, sep=";")
    if "TimeUTC" in df.columns:
        df.TimeUTC = pd.to_datetime(df.TimeUTC, utc=True, infer_datetime_format=True)
        df.set_index("TimeUTC", inplace=True)

    df.drop("Unnamed: 0", inplace=True, axis=1, errors="ignore")

    return df


_season_id_to_name = {
    0: "winter",
    1: "spring",
    2: "summer",
    3: "autumn",
    4: "complete year",
}


def get_blacklist():
    # 02100 doesn't have any test data
    return ["02100"]


def adapt_dict_path_to_backend(dict_to_transform, server):
    match_pattern = "/mnt/work/prophesy/"
    replace_pattern = "~/data/"

    if server:
        match_pattern = "/home/scribbler/data"
        replace_pattern = "/mnt/work/prophesy/"

    for k in dict_to_transform:
        cur_element = dict_to_transform[k]
        cur_element = listify(cur_element)

        cur_element = [
            str_to_path(f.replace(match_pattern, replace_pattern), create_folder=False)
            for f in cur_element
        ]

        dict_to_transform[k] = cur_element


def str_to_enum_type(name, enum):
    if isinstance(name, enum):
        return name

    for enum_type in enum:
        if name.lower() == str(enum_type.name).lower():
            return enum_type
    raise ValueError(f"Could not find enum value {name} in enum {enum}.")


def str_to_boolean(bool_string):
    if isinstance(bool_string, bool):
        return bool_string
    else:
        return True if bool_string.lower() in ["yes", "true"] else False


def read_json(data_file_name):
    data_file_name = Path(data_file_name)
    data = None
    if data_file_name.exists():
        with open(data_file_name) as f:
            data = json.load(f)
    return data


def dump_dict_as_json(result_to_store, file_name):
    result_to_store = json.dumps(result_to_store, indent=4)
    with open(file_name, "w") as outfile:
        json.dump(result_to_store, outfile)


def add_args(parser):
    parser.add_argument(
        "--data_folder",
        help="Path to the data prepared data folder.",
        default="/home/scribbler/data/enercast/wind/",
        # default="/home/scribbler/data/prophesy-data/WindSandbox2015/",
    )

    parser.add_argument(
        "--result_folder",
        help="Path to the store results.",
        default="/media/scribbler/1B5D7DBC354FB150/phd-data/WINDSYN/"
        # default="/media/scribbler/88ee70f4-9b04-4c17-b979-b93df0a29118/phd-data/PVOPEN/",
    )

    parser.add_argument(
        "--embedding_type",
        help="Bayes or normal embedding. [bayes/normal]",
        default="normal",
    )

    parser.add_argument(
        "--finetune",
        help="whether to fine tune the model or not [yes/no]",
        default="yes",
    )

    parser.add_argument(
        "--run_id",
        help="Whether to run all or 0/1/... Only used for target.",
        default="0",
    )

    parser.add_argument(
        "--smoke_test",
        help="Smoke test with limited data.",
        default=True,
    )

    parser.add_argument(
        "--full_bayes",
        help="whether to use a full bayesian network [yes/no]",
        default="no",
    )

    parser.add_argument(
        "--mtl_type",
        help="task (uses task id), meta (uses meta information), metatask (uses both information), unified (no task specific information is used)",
        default="task",
    )

    parser.add_argument(
        "--emb_at_ith_layer",
        help="first or all.",
        default="all",
    )

    parser.add_argument("--fold_id", help="0,1,2...", default="0")


def find_file_in_files(file: Path, files: list):
    file = Path(file)
    for cur_file in files:
        if cur_file is None:
            continue
        if file.stem in cur_file.stem:
            return cur_file


def file_as_processed_file(result_folder, file, suffix):
    file_name = Path(file).stem.split("_")[0]
    return result_folder / f"stl_files/{file_name}_{suffix}.pkl"


def get_model_files(result_folder, source_files):
    all_source_models = (result_folder / f"trained_models/").ls()
    relevant_source_models = []
    for sf in source_files:
        relevant_source_models.append(find_file_in_files(sf, all_source_models))
    return all_source_models, relevant_source_models


def get_source_and_target_files(result_folder, run_id: str = "0"):
    splits = json.load(open(result_folder / "splits.json", "r"))

    cur_split = splits[str(run_id)]
    source_files = filter_none(cur_split["sources"])
    target_files = filter_none(cur_split["targets"])

    return source_files, target_files


def filter_none(to_filter):
    return [c for c in to_filter if c is not None]


def pickle_dump(object, file_name):
    pkl.dump(object, open(file_name, "wb"))


def pickle_load(file_name):
    result_object = None
    file_name = Path(file_name)
    if file_name.exists():
        with open(file_name, "rb") as file:
            result_object = pkl.load(file)

    return result_object


def match_time_stamps(df_1, df_2):
    mask_df1 = df_1.index.isin(df_2.index)
    df_1 = df_1[mask_df1]

    mask_df2 = df_2.index.isin(df_1.index)
    df_2 = df_2[mask_df2]

    return df_1, df_2


def season_id_to_name(df, colname="Season"):
    df.loc[:, colname] = df.loc[:, colname].apply(lambda x: _season_id_to_name[int(x)])
    return df


def get_tmp_dir(SERVER):
    tmp_dir = Path("/tmp/tmp_result/")
    # tmp_dir = Path("/mnt/1B5D7DBC354FB150/forecasts_phd/")

    if SERVER:
        tmp_dir = Path("/mnt/work/transfer/tmp_result/")

    tmp_dir.mkdir(exist_ok=True, parents=True)

    return str(tmp_dir)


def flatten_results(df, include_columns=[]):
    results = []
    for idx in df.index:
        cur_res = df.loc[idx].results
        cur_res["ParkName"] = df.loc[idx].data_file_name
        for c in include_columns:
            cur_res[c] = df.loc[idx][c]
        results.append(cur_res)
    results = pd.concat(results, axis=0)

    return results


def rename_columns_for_eval(df: pd.DataFrame):
    rename_dict = {
        "result_folder": "DataType",
        "similarity_measure_type": "SimilarityMeasureType",
        "model_type": "ModelType",
    }
    df.rename(rename_dict, inplace=True, axis=1)
    return df


def sim_measure_as_str(df, colname="similarity_measure_type"):
    df.loc[:, colname] = df.loc[:, colname].apply(lambda x: x.name.lower())

    return df
