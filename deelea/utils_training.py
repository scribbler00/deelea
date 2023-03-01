from logging import warn
from pathlib import Path
from time import sleep
import traceback
import warnings
from fastrenewables.timeseries.model import TemporalCNN
from optuna.study import study
import torch
from torch import nn

from sacred.experiment import Experiment
from optuna.study.study import Study

from fastcore.foundation import L
from fastai.losses import MSELossFlat
from fastai.metrics import rmse
from fastai.tabular.model import get_emb_sz
from fastai.callback.core import TrainEvalCallback
from fastai.data.load import DataLoader
from fastai.learner import Learner
from fastai.torch_core import set_seed

from fastrenewables.losses import (
    GaussianNegativeLogLikelihoodLoss,
    VAEReconstructionLoss,
)
from fastrenewables.tabular.model import EmbeddingModule, MultiLayerPerceptron
from fastrenewables.baselines import ELM, BayesLinReg
from fastrenewables.tabular.core import TabularRenewables
from fastrenewables.tabular.learner import RenewableLearner
from fastrenewables.models.autoencoders import *
from fastrenewables.timeseries.learner import RenewableTimeseriesLearner
from fastrenewables.metrics import rmse_nll
from deelea.mongo import (
    filter_results_by_config,
    filter_results_by_config_target,
    get_results_of_db,
    is_configuration_running_or_completed,
    is_experiment_done,
)
from deelea.utils_eval import get_all_errors_dl_model, get_error_dict, get_errors
from deelea.utils_models import ModelConfig, ModelConfigTarget, ModelType, get_layer_sizes
from deelea.utils_data import (
    DataConfig,
    get_ts_length,
    read_data_as_dl,
    read_renewables_data,
)
from deelea.utils import get_tmp_dir
from deelea.utils_optuna import setup_log_handler, setup_optuna_backend
from deelea.utils_sacred import (
    SacredConfig,
    create_sacred_experiment,
    save_and_log_dl_results,
    store_hyperparameter_results,
)
import copy
from optuna.exceptions import StorageInternalError

# global variables for config of sacred experiment, will be replaced by the function of an experiment,
data_config, model_config, resources_per_trial, sacred_experiment, optuna_study = (
    None,
    None,
    None,
    None,
    None,
)


class TargetTraining:

    """Iterator for target training."""

    def __init__(self, debug: bool = False):
        self.months_to_train_on = [
            [
                12,
                1,
                2,
            ],
            [
                3,
                4,
                5,
            ],
            [6, 7, 8],
            [9, 10, 11],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ]

        self.num_days_training = [7, 14, 30, 60, 90, 365]
        if debug:
            self.months_to_train_on = [self.months_to_train_on[0]]
            self.num_days_training = [7]

    def params(self):
        for cur_season, months_to_train_on in enumerate(self.months_to_train_on):
            for num_days_training in self.num_days_training:
                if (cur_season == 4 and num_days_training != 365) or (
                    num_days_training == 365 and cur_season != 4
                ):
                    continue
                else:
                    yield (cur_season, months_to_train_on, num_days_training)


def get_model_and_loss(
    dl: DataLoader, config: dict, model_config: ModelConfig, ts_length: int
):

    model = get_model(dl, config, model_config, ts_length)

    loss_func = get_loss(config, model_config, model)

    return model, loss_func


def get_embedding(dl: DataLoader, config: dict, model_config: ModelConfig):
    emb_module = None

    if len(dl.train_ds.cat_names) > 0 and config["use_embedding"]:
        if model_config.full_bayes:
            embed_p = config["dropout"]
        else:
            embed_p = config.get("embed_dropout", 0)

        emb_szs = get_emb_sz(dl.train_ds)

        emb_module = EmbeddingModule(
            None,
            embedding_dropout=embed_p,
            embedding_dimensions=emb_szs,
            embedding_type=model_config.embedding_type,
        )

    return emb_module


def get_model(dl: DataLoader, config: dict, model_config: ModelConfig, ts_length: int):

    emb_module = get_embedding(dl, config, model_config)

    ann_structure = get_layer_sizes(dl, config)

    if model_config.full_bayes:
        # times two for mean and std
        ann_structure[-1] = ann_structure[-1] * 2

    if model_config.model_type == ModelType.MLP:
        model = MultiLayerPerceptron(
            ann_structure, embedding_module=emb_module, ps=config.get("dropout", 0)
        )
    elif model_config.model_type in [
        ModelType.AE,
        ModelType.AETCN,
        ModelType.VAE,
        ModelType.VAETCN,
    ]:
        model = get_autoencoder(
            model_config.model_type, config, emb_module, ann_structure, ts_length
        )
    elif model_config.model_type in [ModelType.TCN]:

        if config["emb_at_ith_layer"] == "first":
            combined_embds_at_layers = L(0)
        elif config["emb_at_ith_layer"] == "all":
            combined_embds_at_layers = L(range(len(ann_structure) - 2))
        elif model_config.is_mtl:
            raise Exception
        else:
            combined_embds_at_layers = L()

        model = TemporalCNN(
            ann_structure,
            embedding_module=emb_module,
            dropout=config.get("dropout", 0),
            add_embedding_at_layer=combined_embds_at_layers,
        )
    else:
        raise NotImplemented("Unsuported model type.")

    return model


def get_autoencoder(
    model_type: ModelType,
    config: dict,
    emb_module: EmbeddingModule,
    layers: list,
    ts_length: int,
):
    n_outputs = layers[-1]
    layers = [int(l) for l in layers[:-1]]
    encoder_layer_structure = layers
    decoder_layer_structure = layers[::-1]

    if decoder_layer_structure[-1] != n_outputs:
        decoder_layer_structure += [n_outputs]

    if "tcn" in model_type.name.lower():
        encoder = TemporalCNN(
            encoder_layer_structure,
            embedding_module=emb_module,
            dropout=config.get("dropout", 0),
        )
        decoder = TemporalCNN(
            decoder_layer_structure,
            embedding_module=emb_module,
            dropout=config.get("dropout", 0),
        )
    else:
        encoder = MultiLayerPerceptron(
            encoder_layer_structure,
            embedding_module=emb_module,
            ps=config.get("dropout", 0),
        )
        decoder = MultiLayerPerceptron(
            decoder_layer_structure,
            embedding_module=emb_module,
            ps=config.get("dropout", 0),
        )

    if model_type in [ModelType.AE, ModelType.AETCN]:
        model = Autoencoder(encoder, decoder)
    elif model_type in [ModelType.VAE, ModelType.VAETCN]:
        if model_type != ModelType.VAETCN:
            ts_length = 1
        model = VariationalAutoencoder(
            encoder,
            decoder,
            encoder_layer_structure[-1],  # * ts_length,
            encoder_layer_structure[-1],  # * ts_length,
            # encoder_layer_structure[-1] * ts_length,
            # encoder_layer_structure[-1] * ts_length,
            is_ts="tcn" in model_type.name.lower(),
        )
    else:
        raise ValueError("Unsupported Autoencoder Model Type.")

    return model


def get_loss(config: dict, model_config: ModelConfig, model: nn.Module):
    loss_func = MSELossFlat()

    if model_config.is_bayes:
        loss_func = GaussianNegativeLogLikelihoodLoss(model)
    elif model_config.is_vae:
        loss_func = VAEReconstructionLoss(model, reconstruction_cost_function=loss_func)

    return loss_func


FIT_DL_SEED = 46546


def fit_dl_model(
    config: dict,
    train_file: Path,
    model_config: ModelConfig,
    silent: bool = False,
    pre_loaded_train_file=None,
    get_model_and_loss=get_model_and_loss,
    seed=FIT_DL_SEED,
) -> Learner:
    # to assure that we have the same results of the hyperparameter training we need to have
    # a random seed when creating the model
    set_seed(seed, reproducible=True)

    if pre_loaded_train_file is None:
        dl = read_data_as_dl(
            train_file,
            config,
            one_batch=False,
            as_autoencoder_data=model_config.is_ae,
            as_timeseries=model_config.is_timeseries_model,
            drop_last=False,
        )
    else:
        dl = pre_loaded_train_file.dataloaders(
            bs=config["batch_size"], drop_last=False, num_workers=0
        )

    if torch.allclose(dl.one_batch()[1], dl.one_batch()[1]):
        raise ValueError(
            "Dataset does not to be seemed shuffle, which is required for training."
        )
    cats, conts, ys = dl.one_batch()

    debug_print_data(config, model_config, dl, cats, conts, ys)

    ts_length = get_ts_length(dl)
    set_seed(seed, reproducible=True)
    model, loss_func = get_model_and_loss(
        dl, config, model_config=model_config, ts_length=ts_length
    )
    metric = rmse
    if model_config.full_bayes:
        metric = rmse_nll

    if model_config.is_timeseries_model:
        learner = RenewableTimeseriesLearner(
            dl, model, loss_func=loss_func, metrics=metric
        )
    else:
        learner = RenewableLearner(dl, model, loss_func=loss_func, metrics=metric)

    if silent and not model_config.smoke_test:
        learner.cbs = L(c for c in learner.cbs if type(c) == TrainEvalCallback)

    learner.fit(int(config["epochs"]), lr=config["lr"], wd=config["wd"])

    if silent:
        val_errors = get_errors(
            learner,
            ds_idx=1,
            dl=None,
            is_bayes=model_config.full_bayes,
            filter_preds=config["filter_preds"],
        )

        return val_errors[f"{model_config.metric_to_optimize}"]
    else:
        return learner


def debug_print_data(config, model_config, dl, cats, conts, ys):
    if model_config.smoke_test:
        print("***************")
        print(
            f"cats lengths: {len(cats)}, conts.shape: {conts.shape}, ys.shape: {ys.shape}"
        )
        if cats is not None and len(cats) > 0 and cats.shape[1] > 0:
            print(
                f"cats.shape: {cats.shape} \n cats samples {cats[0,:]} \n {cats[0:1,0]}"
            )
        print(
            "iters per epoch: ",
            len(dl[0]),
            "with a batch size of ",
            config["batch_size"],
        )
        print("***************")


def fit_elm_model(
    config: dict,
    train_file: Path,
    model_config: ModelConfig,
    silent: bool = False,
    pre_loaded_train_file: TabularRenewables = None,
):
    set_seed(46546)
    if pre_loaded_train_file is None:
        dl = read_data_as_dl(
            train_file,
            config,
            one_batch=False,
            as_autoencoder_data=model_config.is_ae,
            as_timeseries=model_config.is_timeseries_model,
            drop_last=False,
        )
    else:
        dl = pre_loaded_train_file.dataloaders(bs=64, drop_last=False)

    X_train, y_train = dl.train_ds.conts.values, dl.train_ds.ys.values.ravel()
    linear_model = BayesLinReg(
        alpha=config["alpha"],
        beta=config["beta"],
        empirical_bayes=config["empirical_bayes"],
        n_iter=int(config["n_iter"]),
    )
    model = ELM(
        n_hidden=int(config["n_hidden"]),
        activations=config["activations"],
        prediction_model=linear_model,
        include_original_features=True,
    )

    model = model.fit(X_train, y_train)

    preds = model.predict(X_train)
    error_dict_train = get_error_dict(y_train, preds, prefix="train_")
    error_dict_train["train_log_evidence"] = model.log_evidence(X_train, y_train)

    X_valid, y_valid = dl.valid_ds.conts.values, dl.valid_ds.ys.values.ravel()
    preds = model.predict(X_valid)
    error_dict_valid = get_error_dict(y_valid, preds, prefix="valid_")
    error_dict_valid["valid_log_evidence"] = model.log_evidence(X_valid, y_valid)

    if silent:
        return error_dict_valid["valid_log_evidence"]
    else:
        model.co = {**error_dict_train, **error_dict_valid}
        return model


def hyperparameter_optimization_optuna(
    data_config: DataConfig,
    model_config: ModelConfig,
    resources_per_trial: int,
    optuna_study: Study,
    sacred_experiment: Experiment = None,
    fit_function=fit_dl_model,
    save_and_log_func=save_and_log_dl_results,
    n_jobs: int = 2,
    max_re_trials_after_optimization=3,  #
    **kwargs,
):

    tmp_dir = get_tmp_dir(data_config.SERVER)

    train_file = data_config.train_file

    to_train = read_renewables_data(train_file, model_config)

    set_seed(16436514)

    def fit_model_wrapper(trial):
        config = model_config.get_optimization_config(trial)
        try:
            return fit_function(
                config=config,
                train_file=train_file,
                silent=True,
                model_config=model_config,
                pre_loaded_train_file=to_train,
            )
            #
        except StorageInternalError:
            print(f"StorageInternalError occurred. Ignoring this run...")
            # return value that is really bad depending on the optimization mode
            res = 1e18
            if model_config.optimization_mode == "maximize":
                res = -res
            return res

    set_seed(16436514)
    optuna_study.optimize(
        fit_model_wrapper,
        n_trials=model_config.num_hyperopt_samples,
        n_jobs=max(1, n_jobs),
        gc_after_trial=True,
    )

    best_params = optuna_study.best_params
    best_params["emb_at_ith_layer"] = model_config.emb_at_ith_layer
    model_config.set_best_config(best_params)

    min_error_hyper_opt = optuna_study.trials_dataframe().value.min()

    if model_config.optimization_mode == "maximize":
        min_error_hyper_opt = -1 * min_error_hyper_opt

    store_hyperparameter_results(
        sacred_experiment,
        optuna_study.trials_dataframe(),
        model_config.best_config,
        tmp_dir,
    )

    set_seed(16436514)
    metric_name = f"valid_{model_config.metric_to_optimize}"
    best_model, best_error = None, 1e12
    for i_trial in range(max_re_trials_after_optimization):
        learner = fit_function(
            config=model_config.best_config,
            train_file=data_config.train_file,
            model_config=model_config,
            silent=False,
            pre_loaded_train_file=to_train,
            # in the first run use the same seed, afterwards multiply it to get a different one in case there was
            # some random effect that caused a collapsion of the training
            seed=FIT_DL_SEED * (i_trial + 1),
        )

        result_dict = get_all_errors_dl_model(
            model_config, data_config.test_file, learner
        )
        # the difference should not exceed 2.5 percent
        if min_error_hyper_opt - result_dict[metric_name] > -0.025:
            best_model = learner
            break
        elif i_trial + 1 < max_re_trials_after_optimization:
            # if there is some unforeseen non deterministic behaviour lets at least take the best one by the validation error
            if result_dict[metric_name] < best_error:
                best_model = learner
                best_error = result_dict[metric_name]
        else:
            warnings.warn(
                "Model has not been converged as in the hyperparameter optimization."
            )
    learner = best_model
    errors = save_and_log_func(
        model_config,
        data_config,
        learner,
        sacred_experiment,
    )

    return errors


def run_mtl_experiment(
    sacred_config: SacredConfig,
    new_data_config: DataConfig,
    new_model_config: ModelConfig,
    new_resources_per_trial: dict,
    train_function,
    use_optuna: bool = True,  # can be deactivated e.g. for sklearn like hyperparameter optimization
    delete_optuna_db=False,
    check_db_entries=False,
):
    global data_config, model_config, resources_per_trial, sacred_experiment
    data_config, model_config, resources_per_trial = (
        new_data_config,
        new_model_config,
        new_resources_per_trial,
    )

    if check_db_entries and is_configuration_running_or_completed(
        model_config, data_config, sacred_config
    ):
        print("Experiment is running or completed.")
        return

    sacred_experiment = create_sacred_experiment(sacred_config)

    if use_optuna:
        study_name = (
            data_config.experiment_name
            + model_config.model_type.name
            + model_config.mtl_type.name
            + str(data_config.result_folder.stem)
            + model_config.emb_at_ith_layer
        )
        optuna_study = setup_optuna_backend(
            data_config.SERVER,
            study_name,
            model_config.optimization_mode,
            delete_if_exists=delete_optuna_db,
        )
    else:
        optuna_study = None

    @sacred_experiment.config
    def mtl_config():
        data_config = data_config
        model_config = model_config
        resources_per_trial = resources_per_trial

    @sacred_experiment.main
    def wrapper_sacred_training(
        data_config: DataConfig,
        model_config: ModelConfig,
        resources_per_trial: dict,
    ):
        return train_function(
            data_config,
            model_config,
            resources_per_trial,
            optuna_study,
            sacred_experiment,
        )

    sacred_experiment.run()


def run_stl_experiment(
    sacred_config: SacredConfig,
    new_data_config: DataConfig,
    new_model_config: ModelConfig,
    new_resources_per_trial: dict,
    train_function,
    use_optuna: bool = True,  # optuna can be deactivated e.g. for sklearn like hyperparameter optimization
    delete_optuna_db=False,
    check_db_entries=False,
):
    global data_config, model_config, resources_per_trial, sacred_experiment, optuna_study
    data_config, model_config, resources_per_trial = (
        new_data_config,
        new_model_config,
        new_resources_per_trial,
    )
    setup_log_handler()
    files = data_config.target_splits

    logging_file_failed_experiments = Path(
        f"log/failed_files_fold_{data_config.result_folder.stem}_{data_config.fold_id}_stl.log"
    )
    if logging_file_failed_experiments.exists():
        logging_file_failed_experiments.unlink()

    if check_db_entries:
        additional_keys = []
        if isinstance(model_config, ModelConfigTarget):
            additional_keys = [
                "similarity_measure_type",
                "adaption_strategy",
                "ensemble_type",
            ]
            if model_config.train_n_epochs not in [None, 1]:
                additional_keys += ["train_n_epochs"]

        df_res_TMP, db = get_results_of_db(
            model_config,
            data_config,
            sacred_config,
            set_db_index=False,
            additional_keys=additional_keys,
        )

    for data_file_name in files:
        data_config.set_file_name(data_file_name)
        model_config.reset()

        if check_db_entries:
            if len(df_res_TMP) == 0:
                warnings.warn(f"No database entries found.")

            df_res = filter_results_by_config(
                model_config, data_config, copy.copy(df_res_TMP)
            )
            if isinstance(model_config, ModelConfigTarget):
                df_res = filter_results_by_config_target(
                    model_config, data_config, copy.copy(df_res)
                )

            if is_experiment_done(df_res):
                print(f"Nothing to do here for {data_file_name}, {len(df_res)}.")
                continue
            else:
                print(f"Missing experiment for {data_file_name}")

        if data_config.train_file.exists():
            succeded = False
            # in case it fails lets try two more times
            for n_trials in range(3):
                if succeded:
                    break
                try:
                    set_seed(1123131231)
                    sacred_experiment = create_sacred_experiment(sacred_config)

                    if use_optuna:
                        study_name = (
                            data_config.experiment_name
                            + model_config.model_type.name
                            + model_config.mtl_type.name
                            + str(data_config.result_folder.stem)
                        )
                        optuna_study = setup_optuna_backend(
                            data_config.SERVER,
                            study_name,
                            model_config.optimization_mode,
                            delete_if_exists=delete_optuna_db,
                        )
                    else:
                        optuna_study = None

                    @sacred_experiment.config
                    def stl_config():
                        data_config = data_config
                        model_config = model_config
                        resources_per_trial = resources_per_trial

                    @sacred_experiment.main
                    def wrapper_sacred_training(
                        data_config: DataConfig,
                        model_config: ModelConfig,
                        resources_per_trial: dict,
                    ):
                        return train_function(
                            data_config,
                            model_config,
                            resources_per_trial,
                            optuna_study,
                            sacred_experiment,
                        )

                    set_seed(9891823123)
                    sacred_experiment.run()

                    sleep(3)
                    succeded = True
                except:
                    print(traceback.format_exc())

                    # something went wrong, lets store the file that failed
                    file_object = open(logging_file_failed_experiments, "a")
                    log_string = f"Failed: {str(data_file_name)} \n"
                    file_object.write(log_string)
                    log_string = traceback.format_exc()
                    error_log = f"Unexpected error:{log_string} \n"
                    file_object.write(error_log)

        else:
            file_object = open(logging_file_failed_experiments, "a")
            log_string = f"Preprocessed file {data_file_name} not found.\n"
            file_object.write(log_string)
