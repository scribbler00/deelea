import json
import inspect
import os
from fastai.data.load import DataLoader
import numpy as np
import pandas as pd
from enum import Enum, auto
from pathlib import Path
import pickle as pkl
from sklearn.model_selection import train_test_split

from fastcore.foundation import L
from fastcore.basics import listify

from fastai.data.block import RegressionBlock
from fastai.data.transforms import RandomSplitter
from fastai.tabular.core import Categorify
from fastai.tabular.core import df_shrink

from fastrenewables.tabular.core import (
    AddSeasonalFeatures,
    BinFeatures,
    CreateTimeStampIndex,
    DropCols,
    FilterByCol,
    FilterDays,
    FilterEinsmanWind,
    FilterInconsistentSamplesPerDay,
    FilterYear,
    Interpolate,
    NormalizePerTask,
    TabularRenewables,
    VerifyAndNormalizeTarget,
    TrainTestSplitByDays,
    FilterInfinity,
    FilterOutliers,
    read_files,
    str_to_path,
)

from fastrenewables.timeseries.core import *

# from deelea.config_features import *

import phd
import deelea.utils as utils

# import deelea.utils_models as utils_models
from deelea.config_features import _temporal_features
import deelea.config_features as config_features
import deelea.utils_models as phd_utils_models


class Datatype(Enum):
    WINDSYN = auto()
    PVSYN = auto()
    WINDREAL = auto()
    PVREAL = auto()
    PVOPEN = auto()
    WINDOPEN = auto()

    @staticmethod
    def datafolder_to_datatype(datafolder):
        datafolder = str(datafolder).lower()
        if "wind" in datafolder and "syn" in datafolder:
            return Datatype.WINDSYN
        elif "pv" in datafolder and "syn" in datafolder:
            return Datatype.PVSYN
        elif "wind" in datafolder and "real" in datafolder:
            return Datatype.WINDREAL
        elif "pv" in datafolder and "real" in datafolder:
            return Datatype.PVREAL
        elif "wind" in datafolder and "open" in datafolder:
            return Datatype.WINDOPEN
        elif "pv" in datafolder and "open" in datafolder:
            return Datatype.PVOPEN
        else:
            raise NotImplementedError


class FeaturesConfig:
    def __init__(self, datatype, model_architecture):
        """[Class that handles the preprocessing steps for a data type and contains also information about the most relevant features within a dataset based on domain knowledge.]

        Args:
            datatype ([Datatype]): [Enum of the data type.]
            model_architecture ([Modelarchitecture]): [Enum of the model architecture (STL/MTL).]
        """
        self.datatype = datatype
        self.model_architecture = model_architecture

    @property
    def most_relevant_features(self) -> list:
        """[Returns the most relevant features of data type base on domain knowledge and previous experiments. For solar data we return the direct and the diffuse radiation. For wind datasets we return wind speeds at different height and the pressure.]

        Raises:
            NotImplementedError: [In case of an unknown data type]


        Returns:
            [list]: [The most relevant features of the data type.]
        """

        if self.datatype == Datatype.WINDSYN:
            return ["WindSpeed58m", "WindSpeed60m", "PS_SFC_0_M"]
        elif self.datatype == Datatype.WINDREAL:
            return ["WindSpeed58m", "WindSpeed10m", "ICON_EU_PRESSURE_SURFACE_AT0"]
        elif self.datatype == Datatype.WINDOPEN:
            return ["WindSpeed100m", "WindSpeed10m", "AirPressure"]
        elif self.datatype == Datatype.PVREAL:
            return [
                "ICON_EU_SURFACE_DOWN_DIRECT_RADIATION_AVG_SINCE_MODELSTART_SURFACE_AT0_INSTANT",
                "ICON_EU_SURFACE_DOWN_DIFFUSE_RADIATION_AVG_SINCE_MODELSTART_SURFACE_AT0_INSTANT",
            ]
        elif self.datatype == Datatype.PVSYN:
            return [
                "ASWDIRS_SFC_0_M_INSTANT",
                "ASWDIFDS_SFC_0_M_INSTANT",
            ]
        elif self.datatype == Datatype.PVOPEN:
            return ["SolarRadiationDirect", "SolarRadiationDiffuse"]
        else:
            raise NotImplementedError

    @property
    def cont_names(self):
        if self.datatype == Datatype.WINDSYN:
            return config_features._cont_names_syn_wind
        elif self.datatype == Datatype.WINDREAL:
            return config_features._cont_names_real_wind
        elif self.datatype == Datatype.WINDOPEN:
            return config_features._cont_names_open_wind
        elif self.datatype == Datatype.PVREAL:
            return config_features._cont_names_real_pv
        elif self.datatype == Datatype.PVSYN:
            return config_features._cont_names_syn_pv
        elif self.datatype == Datatype.PVOPEN:
            return config_features._cont_names_open_pv
        else:
            raise NotImplementedError

    @property
    def cat_names(self):
        if self.model_architecture == phd_utils_models.ModelArchitecture.STL:
            return []
        elif self.datatype == Datatype.WINDSYN:
            return config_features._cat_names_syn_wind
        elif self.datatype == Datatype.WINDREAL:
            return config_features._cat_names_real_wind
        elif self.datatype == Datatype.WINDOPEN:
            return config_features._cat_names_open_wind
        elif self.datatype == Datatype.PVREAL:
            return config_features._cat_names_real_pv
        elif self.datatype == Datatype.PVSYN:
            return config_features._cat_names_syn_pv
        elif self.datatype == Datatype.PVOPEN:
            return config_features._cat_names_open_pv
        else:
            raise NotImplementedError

    @property
    def y_names(self):
        if self.model_architecture == phd_utils_models.ModelArchitecture.STL:
            return ["PowerGeneration"]
        else:
            return ["PowerGeneration"]

    @property
    def timestamps_name(self):
        # currently the same for all datasets
        return config_features._timestamp_name

    @property
    def procs(self):
        if self.datatype in [Datatype.WINDREAL, Datatype.PVREAL]:
            return [
                FilterInfinity(
                    ["ICON_EU_SNOW_DEPTH_SURFACE_AT0", "ICON_EU_SNOW_DEPTH_SURFACE_AT0"]
                )
            ]
        else:
            return []

    @property
    def pre_process_train(self):
        if self.datatype in [Datatype.WINDSYN, Datatype.PVSYN]:
            pre_procs = _pre_process_syn_train(self.timestamps_name)
        elif self.datatype in [Datatype.WINDREAL, Datatype.PVREAL]:
            pre_procs = _pre_process_real_train(self.timestamps_name, self.datatype)
        elif self.datatype in [Datatype.WINDOPEN, Datatype.PVOPEN]:
            return _pre_process_open_train(self.timestamps_name, self.datatype)
        else:
            raise NotImplementedError
        # for MTL we have enough data, so we will not interpolate the data
        if self.model_architecture == phd_utils_models.ModelArchitecture.MTL:
            new_pre_procs = []
            for pp in pre_procs:
                ppt = pp
                # in case it is a class, we need to create an object so that isinstance is working
                if inspect.isclass(ppt):
                    ppt = ppt()
                if not isinstance(ppt, Interpolate):
                    new_pre_procs.append(pp)
            pre_procs = new_pre_procs

        return pre_procs

    @property
    def pre_process_test(self):
        if self.datatype in [Datatype.WINDSYN, Datatype.PVSYN]:
            return _pre_process_syn_test()
        elif self.datatype in [Datatype.WINDREAL, Datatype.PVREAL]:
            return _pre_process_real_test()
        elif self.datatype in [Datatype.WINDOPEN, Datatype.PVOPEN]:
            return _pre_process_open_test()
        else:
            raise NotImplementedError


def _pre_process_real_train(timestamps_name: str, data_type: Datatype):
    precs = [
        CreateTimeStampIndex(timestamps_name),
        AddSeasonalFeatures,
        Interpolate,
        FilterInconsistentSamplesPerDay,
        DropCols(
            [
                "assetName",
                "loc_id",
                "input_file_name",
                "num_train_samples",
                "num_test_samples",
            ]
        ),
    ]

    if data_type == Datatype.WINDREAL:
        precs = precs + [
            FilterEinsmanWind(
                column_wind="WindSpeed58m", column_target="PowerGeneration"
            )
        ]

    return precs


def _pre_process_open_train(timestamps_name: str, data_type: Datatype):

    sample_time = "1H"
    offset_correction = None
    if data_type == Datatype.WINDOPEN:
        sample_time = "15min"
    else:
        offset_correction = 2
    return [
        CreateTimeStampIndex(timestamps_name, offset_correction=offset_correction),
        AddSeasonalFeatures,
        Interpolate(sample_time=sample_time),
        FilterInconsistentSamplesPerDay,
        DropCols(
            [
                "Season",
                "SeasonCos",
                "SeasonSin",
                "End",
                "Name",
                "Start",
                "Timezone",
            ]
        ),
        FilterYear(year=2015, drop=False),
        FilterOutliers(),
    ]


def _pre_process_open_test():
    return [FilterYear(year=2015, drop=True)]


def _pre_process_real_test():
    precs = []

    return precs


def _pre_process_syn_train(timestamps_name: str):
    return [
        CreateTimeStampIndex(timestamps_name),
        AddSeasonalFeatures,
        Interpolate,
        FilterByCol("TestFlag", drop=True, drop_col_after_filter=True),
        FilterInconsistentSamplesPerDay,
        DropCols(
            [
                "loc_id",
                # "long",
                # "lat",
                "input_file_name",
                # "target_file_name",
                "num_train_samples",
                "num_test_samples",
                "module_name",
                "inverter_name",
                "modules_per_string",
                "strings_per_inverter",
            ]
        ),
    ]


def _pre_process_syn_test():
    return [
        FilterByCol("TestFlag", drop=False, drop_col_after_filter=True),
    ]


def days_date_to_mask(selected_days: list, all_days: list):
    return [True if d in selected_days else False for d in all_days]


def get_train_test_split_by_day(dfs: list):
    train_masks, test_masks = [], []
    dfs = dfs.sort_values("TaskID")

    for _, df in dfs.groupby("TaskID"):
        df.sort_index()
        unique_days = np.unique(df.index.date)
        train_days, test_days = train_test_split(
            unique_days, random_state=42, test_size=0.25
        )
        train_mask = days_date_to_mask(train_days, df.index.date)
        test_mask = days_date_to_mask(test_days, df.index.date)

        train_masks += train_mask
        test_masks += test_mask

    if len(dfs) != (np.sum(train_masks) + np.sum(test_masks)):
        raise ValueError("All elements should be selected by splitting.")

    return dfs[train_masks], dfs[test_masks]


def source_data(
    files: list,
    features_config: FeaturesConfig,
    y_block,
    is_mtl_dataset: bool,
    limit_data=True,
):

    files = listify(files)
    dfs = read_files(files, key_metadata="metadata")

    dfs = pd.concat(dfs, axis=0)

    if is_mtl_dataset and limit_data:
        # in case of mtl we have a lot of data, so it is sufficient to utilize an hourly resolution.
        # futher this will save us computation time
        mask = dfs.index.minute == 0
        dfs = dfs[mask]

    # save us some memory
    dfs = df_shrink(dfs)

    if features_config.datatype in [Datatype.WINDREAL, Datatype.PVREAL]:
        # in case of the real data we have various lengths in the date,
        # therefore we a traditional train test split, however we split by specific dates
        df_train, df_test = get_train_test_split_by_day(dfs)
        splits = TrainTestSplitByDays(test_size=0.25)
    else:
        # in case of the synthetic data, the data is splitted by a TestFlag to separate between train and test
        # in case of the open data, the data is splitted by the year
        # split between train and test is achieved through FilterByCol
        df_train = dfs
        df_test = dfs
        splits = TrainTestSplitByDays(test_size=0.25)

    procs = [
        Categorify,
        NormalizePerTask(ignore_cont_cols=_temporal_features, norm_type="minmaxnorm"),
        VerifyAndNormalizeTarget,
    ] + features_config.procs

    pd.options.mode.chained_assignment = None
    to_train = TabularRenewables(
        df_train,
        cat_names=features_config.cat_names,
        cont_names=features_config.cont_names,
        y_names=features_config.y_names,
        pre_process=features_config.pre_process_train,
        procs=procs,
        splits=splits,
        y_block=y_block,
    )

    to_test = to_train.new(
        df_test, pre_process=features_config.pre_process_test, include_preprocess=True
    )
    to_test.process()

    if len(files) != len(to_train.items.TaskID.unique()):
        raise ValueError("Should be of equal length.")

    task_id_to_name = dict(zip(range(len(files)), files))

    return to_train, to_test, task_id_to_name


def read_data_as_dl(
    train_file,
    model_config,
    one_batch: bool = False,
    drop_last: bool = True,
):
    to_train = read_renewables_data(train_file, model_config)

    if "batch_size" in model_config.config.keys():
        bs = model_config.config["batch_size"]
    else:
        bs = 64

    if one_batch:
        bs = len(to_train.train)

    dl = to_train.dataloaders(bs=bs, drop_last=drop_last, num_workers=0)
    return dl


def read_renewables_data(
    train_file, model_config, is_test_data=False, convert_to_timeseries=True
):
    to_train = pkl.load(open(train_file, "rb"))

    if model_config.is_ae:
        # TODO: this is actually a workaround as the current data has been created with an older version
        to_train.original_y_block = RegressionBlock().type_tfms
        relevant_cols = [
            c
            for c in to_train.cont_names
            if c
            not in ["MonthSin", "MonthCos", "DaySin", "DayCos", "HourSin", "HourCos"]
        ]
        to_train.update_ys(relevant_cols, RegressionBlock())

    to_train = update_categoricals(model_config, to_train)

    if model_config.is_timeseries_model and convert_to_timeseries:
        if not is_test_data:
            splits = RandomSplitter()
        else:
            splits = None
        to_train = Timeseries(to_train, splits=splits)

    return to_train


def update_categoricals(model_config, to_train):
    if (
        model_config.mtl_type == phd_utils_models.MTLType.TASK
        and len(to_train.cat_names) > 1
        and to_train.cat_names[0] != "TaskID"
    ):
        to_train.update_cats(["TaskID"])
    elif (
        model_config.mtl_type == phd_utils_models.MTLType.METATASK
        and len(to_train.cat_names) > 0
    ):
        to_train.update_cats(to_train.cat_names + ["TaskID"])
    elif (
        model_config.mtl_type == phd_utils_models.MTLType.UNIFIED
        and len(to_train.cat_names) > 0
    ):
        to_train.update_cats([])

    return to_train


def get_ts_length(dl: DataLoader):
    ts_length = 1
    if isinstance(dl.train_ds, TimeseriesDataset):
        ts_length = dl.train_ds.input_sequence_length

    return ts_length


class TLType(Enum):
    SOURCE = auto()
    TARGET = auto()


class DataConfig:
    def __init__(
        self,
        fold_id: int,
        result_folder: Path,
        model_config,
        SERVER: bool,
        tl_type: TLType,
        data_file_name: Path = None,
        create_forecasts=False,
    ):
        """[The data config containing all relevant information to read the input data.]

        Args:
            fold_id (int): [The fold id of the dataset.]
            result_folder (Path): [The folder of the preprocessed data]
            model_config ([type]): [The model config for the model to be trained.]
            SERVER (bool): [Whether it is running on the server or not.]
            tl_type (TLType): [Whether we doing an experiment for the source or the target.]
            data_file_name (Path, optional): [An explicit name of a park.]. Defaults to None.
        """
        self.fold_id = int(fold_id)
        self.result_folder = str_to_path(result_folder)
        self._model_config = model_config
        self.SERVER = SERVER
        self.data_file_name = data_file_name
        self.tl_type = tl_type
        self.data_type = Datatype.datafolder_to_datatype(self.result_folder)
        self._model_config.data_type = self.data_type
        self.splits = None
        self._task_id_to_name = None
        self._name_to_task_id = None
        self.create_forecasts = create_forecasts

    def _get_data_file(
        self,
        model_architecture,
        suffix,
        data_file_name: str = None,
    ):
        if model_architecture == phd_utils_models.ModelArchitecture.MTL:
            data_file = (
                self.result_folder
                / f"mtl_files/fold_{int(self.fold_id):02d}_{suffix}.pkl"
            )
        elif data_file_name is not None:
            data_file = self.result_folder / f"stl_files/{data_file_name}_{suffix}.pkl"
        else:
            raise ValueError("Missing value of data_file_name")

        return data_file

    def get_all_files(self, data_filter: str = "train") -> list:
        """[Return all the path of all file based on the config]

        Args:
            data_filter (str, optional): [The kind of files to return (train,test, or all)]. Defaults to "train".

        Returns:
            list: [Return list with all paths of the files]
        """
        files = []
        if (
            self._model_config.model_architecture
            == phd_utils_models.ModelArchitecture.MTL
        ):
            folder = self.result_folder / f"mtl_files"
        else:
            folder = self.result_folder / f"stl_files"

        files = folder.ls()

        if data_filter != "all":
            files = [f for f in files if data_filter in f.stem]

        return files

    def as_short_name(self, file_name):
        return Path(file_name).stem.replace("_train", "").replace("_test", "")

    def set_file_name(self, data_file_name):
        data_file_name = self.as_short_name(data_file_name)
        self.data_file_name = Path(data_file_name).stem

    @property
    def experiment_name(self):
        data_file = ""
        if self.data_file_name is not None:
            data_file = f"_{self.data_file_name}"

        return f"{self.result_folder.stem}{data_file}_{int(self.fold_id):02d}_{self.tl_type.name}_{self._model_config.model_type.name}"

    @property
    def tmp_dir(self):
        return utils.get_tmp_dir(self.SERVER)

    @property
    def train_file(self):
        return self._get_data_file(
            self._model_config.model_architecture, "train", self.data_file_name
        )

    @property
    def test_file(self):
        return self._get_data_file(
            self._model_config.model_architecture, "test", self.data_file_name
        )

    @property
    def source_splits(self):
        return self._get_splits()["sources"]

    @property
    def target_splits(self):
        return self._get_splits()["targets"]

    @property
    def task_id_to_name_file(self):
        return (
            self.result_folder
            / f"mtl_files/fold_{int(self.fold_id):02d}_taskIdToName.pkl"
        )

    @property
    def task_id_to_name_dict(self):
        if self._task_id_to_name is None:
            self._task_id_to_name = json.loads(
                str(json.load(open(self.task_id_to_name_file, "rb")))
            )
            # remove folder
            for k in self._task_id_to_name:
                self._task_id_to_name[k] = Path(self._task_id_to_name[k]).stem

        return self._task_id_to_name

    @property
    def name_to_task_id_dict(self):
        if self._name_to_task_id is None:
            # get task_id_to_name
            self._name_to_task_id = self.task_id_to_name_dict
            # revert previous dict to name_to_task_id
            self._name_to_task_id = {v: k for k, v in self._name_to_task_id.items()}

        return self._name_to_task_id

    def _get_splits(self):
        if self.splits is None:
            self.splits = json.load(open(self.result_folder / "splits.json", "r"))[
                str(self.fold_id)
            ]

        return self.splits

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"""DataConfig(fold_id={self.fold_id}, result_folder={str(self.result_folder)}, model_config={self._model_config})"""

    def is_in_source_files(self, data_file_name):
        data_file_name = self.as_short_name(data_file_name)

        res = False

        for cur_source_file in self.source_splits:
            if Path(cur_source_file).stem == data_file_name:
                res = True
                break

        return res

    def get_task_dicts_updated(self):
        """Return task id dicts updated based on the MTL Type and convert to int"""
        task_id_to_name_dict = self.task_id_to_name_dict
        name_to_task_id_dict = self.name_to_task_id_dict

        offset = 0
        # due to the categorization the task id is plus one when the MTL data has been updated
        if self._model_config.mtl_type in [
            phd_utils_models.MTLType.TASK,
            phd_utils_models.MTLType.METATASK,
        ]:
            offset = 1

        for name in name_to_task_id_dict.keys():
            name_to_task_id_dict[name] = int(name_to_task_id_dict[name]) + offset

        task_id_to_name_new = {}
        for task_id in task_id_to_name_dict.keys():
            task_id_to_name_new[int(task_id) + offset] = task_id_to_name_dict[task_id]

        task_id_to_name_dict = task_id_to_name_new

        return task_id_to_name_dict, name_to_task_id_dict
