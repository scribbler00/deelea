from enum import Enum, auto, IntEnum
from pathlib import Path
import warnings

from optuna import trial


from fastai.data.load import DataLoader
from fastai.data.transforms import get_c
from fastai.learner import Learner
from fastrenewables.tabular.model import EmbeddingType, get_structure

import deelea.configs as configs
import deelea.utils as utils


class ModelType(Enum):
    MLP = auto()  # Multi Layer Preceptron
    CNN = auto()  # normal cnn without residual blocks
    TCN = auto()  # temporal convolution network with residual blocks
    GBRT = auto()
    ELM = auto()  # extreme learning machine
    Linear = auto()
    AE = auto()  # autoencoder with mlp layers
    AETCN = auto()  # autoencoder with residual layers based on a tcn
    AEMLP = (
        auto()
    )  # MLP that uses the latent features from the AE as input to predict the power
    AETCNTCN = (
        auto()
    )  # TCN that uses the latent features from the AETCN as input to predict the power
    BMA = auto()  # bayesian model averaging
    CSGE = auto()  # coopetetive soft gating ensemble
    VAE = auto()
    VAETCN = auto()
    VAETCNTCN = auto()
    VAEMLP = auto()


class ModelArchitecture(Enum):
    STL = auto()
    MTL = auto()
    AE = auto()


class MTLType(Enum):
    TASK = auto()  # uses task id to differentiate between tasks
    META = auto()  # uses meta information to differentiate between tasks
    METATASK = auto()
    UNIFIED = auto()
    # UNIFIED


class ModelConfig:
    def __init__(
        self,
        embedding_type: str,
        full_bayes: str,
        model_type: str,
        model_architecture: str,
        debug: bool = False,
        smoke_test: bool = False,
        mtl_type="meta",
        emb_at_ith_layer="first",
    ):
        """[The model config containing all relevant information to set-up a model.]

        Args:
            embedding_type (str): [The kind of embedding, such as task normal or Bayesian.]
            full_bayes (str): [Whether or not to train a full Bayesian model.]
            model_type (str): [The model type, such as MLP, TCN, etc.]
            model_architecture (str): [The kind of architecture stl/mtl/ae.]
            debug (bool, optional): [Flag for debugging.]. Defaults to False.
            smoke_test (bool, optional): [Whether or not it is a smoke test.]. Defaults to False.
            mtl_type (str, optional): [The kind of task embedding]. Defaults to "meta".
            emb_at_ith_layer (str, optional): [Position of embeding layer in MTL]. Defaults to "first".
        """
        self.embedding_type = utils.str_to_enum_type(embedding_type, EmbeddingType)

        self.full_bayes = utils.str_to_boolean(full_bayes)

        self.model_type = utils.str_to_enum_type(model_type, ModelType)

        self.model_architecture = utils.str_to_enum_type(
            model_architecture, ModelArchitecture
        )

        self.mtl_type = utils.str_to_enum_type(mtl_type, MTLType)

        self.config_optimization, self.config_debug = None, None
        self.debug = debug
        self.smoke_test = utils.str_to_boolean(smoke_test)
        self.best_config = None

        self._init_config()
        self._set_error_info()
        self.data_type = None
        self.emb_at_ith_layer = emb_at_ith_layer

    def _set_error_info(self):
        self.metric_to_optimize = "rmse"
        self.optimization_mode = "minimize"

        if self.full_bayes:
            self.metric_to_optimize = "distance_ideal_curve"

        if self.model_type in [ModelType.ELM, ModelType.Linear]:
            self.metric_to_optimize = "log_evidence"
            self.optimization_mode = "maximize"

    @property
    def num_hyperopt_samples(self) -> int:
        num = 200
        if self.full_bayes:
            num = 100
        elif self.is_mtl and self.is_timeseries_model:
            num = 100
        if self.smoke_test:
            num = 1
        return num

    def reset(self):
        self.best_config = None

    def _init_config(self):
        empty_config = lambda x: {}
        empty_config_optimization = lambda x, y: {}

        self._configs = {
            ModelType.MLP: (
                configs.get_mlp_config,
                configs.get_mlp_config_optimization,
            ),
            ModelType.TCN: (
                configs.get_tcn_config,
                configs.get_tcn_config_optimization,
            ),
            ModelType.ELM: (
                configs.get_elm_config,
                configs.get_elm_config_optimization,
            ),
            ModelType.GBRT: (
                configs.get_gbrt_config,
                configs.get_gbrt_config_optimization,
            ),
            ModelType.AE: (
                configs.get_ae_config,
                configs.get_ae_config_optimization,
            ),
            ModelType.AETCN: (
                configs.get_ae_config,
                configs.get_ae_config_optimization,
            ),
            ModelType.VAE: (
                configs.get_ae_config,
                configs.get_ae_config_optimization,
            ),
            ModelType.VAETCN: (
                configs.get_ae_config,
                configs.get_ae_config_optimization,
            ),
            ModelType.VAEMLP: (
                configs.get_aemlp_config,
                configs.get_aemlp_config_optimization,
            ),
            ModelType.AEMLP: (
                configs.get_aemlp_config,
                configs.get_aemlp_config_optimization,
            ),
            ModelType.VAETCNTCN: (
                configs.get_aetcntcn_config,
                configs.get_aetcntcn_config_optimization,
            ),
            ModelType.AETCNTCN: (
                configs.get_aetcntcn_config,
                configs.get_aetcntcn_config_optimization,
            ),
            ModelType.BMA: (
                empty_config,
                empty_config_optimization,
            ),
        }

        if self.model_type not in list(self._configs.keys()):
            raise ValueError("Model Type not supported.")

        self.base_config

    def __str__(self):
        return f"{self.model_type.name}_embeddingType_{self.embedding_type.name}_fullBayes_{self.full_bayes}"

    @property
    def base_config(self):
        return self._configs[self.model_type][0](self)

    def get_optimization_config(self, trial: trial = None):
        if self.data_type is None:
            warnings.warn("Datatype not yet set.")
        return self._configs[self.model_type][1](trial, self)

    def set_best_config(self, config):
        self.best_config = self.base_config
        for config_name in config.keys():
            self.best_config[config_name] = config[config_name]

    @property
    def config(self) -> dict:
        if self.best_config is not None:
            return self.best_config
        if self.debug:
            return self.config_debug
        else:
            _config = self.config_optimization

            return _config

    def set_debug(self, debug: bool):
        self.debug = debug
        self._set_config_attr()

    @property
    def is_mtl(self):
        return self.model_architecture == ModelArchitecture.MTL

    @property
    def is_timeseries_model(self) -> bool:

        model_type_as_name = self.model_type.name.lower()

        return (
            ("cnn" in model_type_as_name)
            or ("tcn" in model_type_as_name)
            or ("lstm" in model_type_as_name)
        )

    @property
    def is_bayes(self) -> bool:
        return (self.embedding_type == EmbeddingType.Bayes) or self.full_bayes

    @property
    def is_vae(self) -> bool:
        return self.model_type in [
            ModelType.VAE,
            ModelType.VAETCN,
        ]

    @property
    def is_ae(self) -> bool:
        return self.model_type in [
            ModelType.AE,
            ModelType.AETCN,
            ModelType.VAE,
            ModelType.VAETCN,
        ]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"""ModelConfig(embedding_type="{self.embedding_type.name}", full_bayes={self.full_bayes}, model_type="{self.model_type.name}", model_architecture="{self.model_architecture.name}", mtl_type="{self.mtl_type.name}")"""


class SimilarityMeasure(Enum):
    RANDOM = auto()
    RMSE = auto()
    LOGEVIDENCE = auto()
    SAMPLERMSE = auto()
    SAMPLEBAYESRMSE = auto()
    POSTERIOR = auto()
    BAYESFACTOR = auto()
    LOGPOSTERIOR = auto()
    LIKELIHOOD = auto()
    UNCERTAINTY = auto()


class AdaptionStrategy(Enum):
    DIRECT = auto()
    DIRECTLINEAR = auto()
    WEIGHTDECAY = auto()
    BAYESTUNE = auto()
    DIRECTLINEARAUTO = auto()
    WEIGHTDECAYSOURCE = auto()
    WEIGHTDECAYSOURCEALL = auto()
    WEIGHTDECAYALL = auto()


class ModelConfigTarget(ModelConfig):
    def __init__(
        self,
        embedding_name: str,
        full_bayes: str,
        model_name: str,
        model_architecture: str,
        debug: bool = False,
        smoke_test: bool = False,
        mtl_type="meta",
        similarity_measure: SimilarityMeasure = SimilarityMeasure.RMSE,
        ensemble_type="",
        adaption_strategy: AdaptionStrategy = AdaptionStrategy.DIRECT,
        train_n_epochs=None,
    ):
        super().__init__(
            embedding_type=embedding_name,
            full_bayes=full_bayes,
            model_type=model_name,
            model_architecture=model_architecture,
            debug=debug,
            smoke_test=smoke_test,
            mtl_type=mtl_type,
        )
        self.ensemble_type = ensemble_type
        self.similarity_measure_type = utils.str_to_enum_type(
            similarity_measure, SimilarityMeasure
        )
        self.adaption_strategy = utils.str_to_enum_type(
            adaption_strategy, AdaptionStrategy
        )
        self.train_n_epochs = train_n_epochs


def get_layer_sizes(dl: DataLoader, config: dict) -> list:

    n_input_features = len(dl.cont_names)

    ann_structure = [n_input_features] + get_structure(
        n_input_features * config["input_size_multiplier"],
        config["percental_reduce"],
        config["minimum_hidden_size"],
        final_outputs=[config["second_last_layer_size"], get_c(dl)],
    )

    return ann_structure


def save_dl_model(
    learner: Learner, train_file: Path, model_config: ModelConfig, result_folder: Path
):
    model_folder = result_folder / f"trained_models"
    model_folder.mkdir(exist_ok=True, parents=True)
    learner.path = model_folder
    file_name = str(train_file.stem).replace("train", "")
    learner.export(f"{file_name}_{str(model_config)}.pkl")
