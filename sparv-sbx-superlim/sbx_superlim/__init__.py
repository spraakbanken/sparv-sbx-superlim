"""A Sparv plugin for classifying text using the Superlim baseline models."""
from sparv.api import Config
from . import superlim

__config__ = [
    # TODO: Split this variable into many different variables according to superlim task
    # TODO: Make it configurable from config.yaml in corpus directory
    Config("sbx_superlim.hf_model_path", "sbx/bert-base-swedish-cased-argumentation_sent", description="HuggingFace model to use for classification")
]

__description__ = "A Sparv plugin for classifying text using the Superlim baseline models."
