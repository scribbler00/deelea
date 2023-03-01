import os

SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")

if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = 4
SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)

from deelea.config_environment import SETUP_ENVIRONMENT

# SHOULD ALWAYS BE THE FIRST TO INCLUDE
SLURM_CPUS_PER_TASK, RESSOURCES_PER_TRIAL, SERVER = SETUP_ENVIRONMENT(
    RESSOURCES_PER_TRIAL=SLURM_CPUS_PER_TASK - 1
)


from functools import partial
from pathlib import Path
import time

from deelea.utils_sacred import (
    SacredConfig,
    delete_table,
    save_and_log_sklearn_bayes_results,
)

import argparse

from deelea.utils_training import (
    fit_elm_model,
    hyperparameter_optimization_optuna,
    run_stl_experiment,
)

from deelea.utils_data import DataConfig, TLType
from deelea.utils_models import ModelConfig
from deelea.utils import (
    add_args,
    get_tmp_dir,
)


if __name__ == "__main__":
    DEBUG = True
    parser = argparse.ArgumentParser()

    add_args(parser)
    tmp_dir = get_tmp_dir(SERVER)

    parser.add_argument(
        "--model_architecture",
        help="Single-task learning [stl] or multi-task learning [mtl]",
        default="stl",
    )

    parser.add_argument(
        "--model_type", help="elm or linear", default="elm",
    )

    args = parser.parse_args()

    model_config = ModelConfig(
        args.embedding_type,
        args.full_bayes,
        args.model_type,
        args.model_architecture,
        debug=DEBUG,
        smoke_test=args.smoke_test,
    )

    if args.fold_id == "all":
        folds_ids = ["0", "1", "2", "3", "4"]
    else:
        folds_ids = [args.fold_id]

    for fold_id in folds_ids:

        data_config = DataConfig(
            fold_id, args.result_folder, model_config, SERVER, tl_type=TLType.SOURCE
        )

        sacred_config = SacredConfig(
            experiment_name="ELMSource",
            db_name="sourceelm",
            mongo_observer=True,
            hostname="alatar",
            # hostname="localhost",
            # port="27029",
            port="27029",
        )

        Path("log").mkdir(exist_ok=True)

        run_stl_experiment(
            sacred_config,
            data_config,
            model_config,
            RESSOURCES_PER_TRIAL,
            train_function=partial(
                hyperparameter_optimization_optuna,
                fit_function=fit_elm_model,
                save_and_log_func=save_and_log_sklearn_bayes_results,
                # n_jobs=(SLURM_CPUS_PER_TASK - 1) // RESSOURCES_PER_TRIAL,
                n_jobs=1,
            ),
            delete_optuna_db=True,
        )

        time.sleep(5)
