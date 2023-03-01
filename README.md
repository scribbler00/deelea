# Model Selection, Adaption, and Combination for Deep Transfer Learning through Neural Networks in Renewables


Impressum can be found [here](https://www.uni-kassel.de/uni/index.php?id=372).

Authors: Jens Schreiber and Bernhard Sick

[Supplementary Material](supplementary-material.pdf) contains further details on the experimental setup (i.e., information about data), a list of symbolds, and additional results as well as definitions.


## Project Structure


- `experiments` contains source code to recreate the experiments.
    - `baselines` contains the GBRT baseline.
    - `ensembles` contains scripts to train the target ensembles.
    - `forecasts` contains scripts to create forecasts for plots.
    - `preparation` contains scripts to prepare the data.
    - `representation` contains scripts to train the source models.
    - `target` contains scripts to train the target models.
- `phd` contains all helper sripts.
    


## Execute Experiments


Due to the size of the experiment it has been executed on a compute cluster. Regardless, the experiments can be executed on a PC.

1. Create a python environment with version 3.8.
1. Execute `pip install -R requirements.txt`.
1. Create and start `mongo db`.
1. Adapt `HOSTNAME`, in script you want to start, with mongo db address.
1. Execute `preparation\create_splits.py` (adapt folders beforehand).
1. Execute `preparation\create_processed_data.py` (adapt folders beforehand).
1. Execute, e.g., `baselines\gbrt.py --data_folder DATA_FOLDER --result_folder RESULT_FOLDER --fold_id FOLD_ID -model_architecture STL`

Note, to get help on the parameters of a script just execute `python SCRIPT_NAME --help`
If you are just interested in the utilized models and methods you can directly use the [fastrenewables](https://scribbler00.github.io/fastrenewables/) library.