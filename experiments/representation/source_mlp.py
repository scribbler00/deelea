from deelea.config_environment import SETUP_ENVIRONMENT

# SHOULD ALWAYS BE THE FIRST TO INCLUDE
SLURM_CPUS_PER_TASK, RESSOURCES_PER_TRIAL, SERVER = SETUP_ENVIRONMENT(10)
from deelea.mongo import convert_mongo_df, get_mongo_db_connection, sacred_db_to_df


from functools import partial
import os
import warnings

import numpy as np
import argparse
from pathlib import Path
from deelea.utils_training import (
    hyperparameter_optimization_optuna,
    run_mtl_experiment,
    run_stl_experiment,
)
from deelea.utils_sacred import SacredConfig, delete_table
from deelea.utils_data import DataConfig, TLType
from deelea.utils_models import (
    ModelArchitecture,
    ModelConfig,
)
from deelea.utils import (
    add_args,
    get_tmp_dir,
)


if __name__ == "__main__":
    DEBUG = False
    parser = argparse.ArgumentParser()

    add_args(parser)
    tmp_dir = get_tmp_dir(SERVER)

    parser.add_argument(
        "--model_architecture",
        help="Single-task learning [stl] or multi-task learning [mtl]",
        default="stl",
    )

    parser.add_argument(
        "--model_type", help="tcn, cnn, or mlp.", default="tcn",
    )

    args = parser.parse_args()

    model_config = ModelConfig(
        args.embedding_type,
        args.full_bayes,
        args.model_type,
        args.model_architecture,
        debug=DEBUG,
        smoke_test=args.smoke_test,
        mtl_type=args.mtl_type,
        emb_at_ith_layer=args.emb_at_ith_layer,
    )

    experiment_name = f"source{args.model_architecture}{args.model_type}"
    db_name = f"source{args.model_type}"
    port = "27031"
    if not SERVER and model_config.smoke_test:
        port = "27029"
    sacred_config = SacredConfig(
        experiment_name=experiment_name,
        db_name=db_name,
        mongo_observer=True,
        hostname="alatar",
        port=port,
    )

    # %%

    Path("log").mkdir(exist_ok=True)

    if args.fold_id == "all":
        folds_ids = ["0", "1", "2", "3", "4"]
    else:
        folds_ids = [args.fold_id]

    for fold_id in folds_ids:
        data_config = DataConfig(
            fold_id, args.result_folder, model_config, SERVER, tl_type=TLType.SOURCE
        )

        if model_config.model_architecture == ModelArchitecture.MTL:
            run_mtl_experiment(
                sacred_config,
                data_config,
                model_config,
                RESSOURCES_PER_TRIAL,
                train_function=partial(
                    hyperparameter_optimization_optuna,
                    n_jobs=(SLURM_CPUS_PER_TASK - 1) // RESSOURCES_PER_TRIAL,
                ),
                delete_optuna_db=True,
                check_db_entries=True,
            )
        else:
            run_stl_experiment(
                sacred_config,
                data_config,
                model_config,
                RESSOURCES_PER_TRIAL,
                train_function=partial(
                    hyperparameter_optimization_optuna,
                    n_jobs=(SLURM_CPUS_PER_TASK - 1) // RESSOURCES_PER_TRIAL,
                ),
                delete_optuna_db=True,
                check_db_entries=True,
            )
