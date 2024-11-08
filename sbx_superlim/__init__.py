"""A Sparv plugin for classifying text using the Superlim baseline models."""
from sparv.api import Config
from .annotators import (
    absabank_imm,
    argumentation_sent,
    dalaj_ged,
    swenli
    )
from . import exporters

__config__ = [
    # TODO: Split this variable into many different variables according to superlim task
    # TODO: Make it configurable from config.yaml in corpus directory
    Config("sbx_superlim.hf_model_path.argumentation", "sbx/bert-base-swedish-cased_argumentation_sent", description="HuggingFace model fine-tuned on argumentation_sent"),
    Config("sbx_superlim.hf_model_path.absabank-imm", "sbx/bert-base-swedish-cased_absabank-imm", description="HuggingFace model fine-tuned on absabank-imm"),
    Config("sbx_superlim.hf_model_path.dalaj-ged", "sbx/bert-base-swedish-cased_dalaj-ged", description="HuggingFace model fine-tuned on dalaj-ged"),
    Config("sbx_superlim.hf_model_path.swenli", "sbx/bert-base-swedish-cased_swenli", description="HuggingFace model fine-tuned on swenli"),
    Config("sbx_superlim.hf_model_path.swepar", "sbx/bert-base-swedish-cased_swepar", description="HuggingFace model fine-tuned on swepar"),
    Config("sbx_superlim.hf_inference_args.batch_size", None, description="Batch size for inference. Required with large files and limited CPU/GPU memory."),
    Config("sbx_superlim.predictions.contains_words", [], description="Batch size for inference. Required with large files and limited CPU/GPU memory."),
    Config("sbx_superlim.predictions.source_files_spans", {}, description="Spans of the source files to analyze.")
]

__description__ = "A Sparv plugin for classifying text using the Superlim baseline models."
