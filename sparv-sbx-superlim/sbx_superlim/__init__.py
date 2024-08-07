"""A Sparv plugin for classifying text using the Superlim baseline models."""
from sparv.api import Config
from . import superlim

__config__ = [
    # TODO: Split this variable into many different variables according to superlim task
    # TODO: Make it configurable from config.yaml in corpus directory
    Config("sbx_superlim.hf_model_path.argumentation", "sbx/bert-base-swedish-cased_argumentation", description="HuggingFace model fine-tuned on argumentation_sent"),
    Config("sbx_superlim.hf_model_path.absabank-imm", "sbx/bert-base-swedish-cased_absabank-imm", description="HuggingFace model fine-tuned on absabank-imm"),
    Config("sbx_superlim.hf_model_path.dalaj-ged", "sbx/bert-base-swedish-cased_dalaj-ged", description="HuggingFace model fine-tuned on dalaj-ged"),
    Config("sbx_superlim.hf_model_path.swenli", "sbx/bert-base-swedish-cased_swenli", description="HuggingFace model fine-tuned on swenli")
]

__description__ = "A Sparv plugin for classifying text using the Superlim baseline models."
