import logging
from pathlib import Path
import sys
import optuna
from optuna.study.study import Study


def setup_log_handler():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


def setup_optuna_backend(
    SERVER,
    study_name,
    optimization_mode: str,
    delete_if_exists=False,
    load_if_exists=False,
) -> Study:

    if SERVER:
        storage_location = "/mnt/work/transfer/optuna_results/"
    else:
        storage_location = "/tmp/optuna_results/"

    # create destination path in case it does not exists
    Path(storage_location).mkdir(exist_ok=True, parents=True)

    db_path = Path(f"{storage_location}{study_name}.db")

    storage_name = f"sqlite:///{storage_location}{study_name}.db"

    # in case we don't want to resume, lets delete the db
    # TODO: this causes experiments to fail in case of the same name
    if delete_if_exists and db_path.exists():
        print(f"Deleted optuna db at path {db_path}")
        db_path.unlink()

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=123123),
        study_name=study_name,
        storage=storage_name,
        load_if_exists=load_if_exists,
        direction=optimization_mode,
    )

    return study
