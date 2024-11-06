"""Example for a custom annotator."""

from typing import List, Literal

from datasets import get_dataset_config_info
from sparv.api import (
    Annotation,
    Config,
    Output,
    Text,
    annotator
    )

from transformers import pipeline
    

from ..common import prepare_inputs
from ..helpers import get_label_mapper


def argumentation_sentences_stance(
    out_stance,
    out_stance_certainty,
    sentence,
    text,
    topic,
    hf_model_path,
    hf_batch_size
):
    ds_config = get_dataset_config_info('sbx/superlim-2', 'argumentation_sent')
    inputs = prepare_inputs(text, sentence, f" [SEP] {topic}") # TODO: Change [SEP] token depending on model.
    pipe = pipeline("text-classification", model=hf_model_path, batch_size=hf_batch_size)
    output = pipe(inputs)
    label_mapper = get_label_mapper(ds_config, pipe.model.config)
    labels = [label_mapper[o['label']] for o in output]
    out_stance.write([l for l in labels])
    out_stance_certainty.write([str(o['score']) for o in output])


@annotator(
    "Identify the stance towards abortion",
    language="swe"
)
def abortion_stance(
    out_stance: Output = Output("<sentence>:sbx_superlim.abortion_stance"),
    out_stance_certainty: Output = Output("<sentence>:sbx_superlim.abortion_stance.certainty"),
    sentence: Annotation = Annotation("<sentence>"),
    text = Text(),
    hf_model_path = Config("sbx_superlim.hf_model_path.argumentation"),
    hf_batch_size = Config("sbx_superlim.hf_inference_args.batch_size")

):
    argumentation_sentences_stance(
        out_stance,
        out_stance_certainty,
        sentence,
        text,
        'abort',
        hf_model_path,
        hf_batch_size
    )


@annotator(
    "Identify the stance towards minimum wage",
    language="swe"
)
def minimum_wage_stance(
    out_stance: Output = Output("<sentence>:sbx_superlim.minimum wage_stance"),
    out_stance_certainty: Output = Output("<sentence>:sbx_superlim.minimum wage_stance.certainty"),
    sentence: Annotation = Annotation("<sentence>"),
    text = Text(),
    hf_model_path = Config("sbx_superlim.hf_model_path.argumentation"),
    hf_batch_size = Config("sbx_superlim.hf_inference_args.batch_size")

):
    argumentation_sentences_stance(
        out_stance,
        out_stance_certainty,
        sentence,
        text,
        'minimilön',
        hf_model_path,
        hf_batch_size
    )


@annotator(
    "Identify the stance towards marijuana legalization",
    language="swe"
)
def marijuana_legalization_stance(
    out_stance: Output = Output("<sentence>:sbx_superlim.marijuana_legalization_stance"),
    out_stance_certainty: Output = Output("<sentence>:sbx_superlim.marijuana_legalization_stance.certainty"),
    sentence: Annotation = Annotation("<sentence>"),
    text = Text(),
    hf_model_path = Config("sbx_superlim.hf_model_path.argumentation"),
    hf_batch_size = Config("sbx_superlim.hf_inference_args.batch_size")

):
    argumentation_sentences_stance(
        out_stance,
        out_stance_certainty,
        sentence,
        text,
        'marijuanalegalisering',
        hf_model_path,
        hf_batch_size
    )


@annotator(
    "Identify the stance towards death penalty",
    language="swe"
)
def death_penalty_stance(
    out_stance: Output = Output("<sentence>:sbx_superlim.death_penalty_stance"),
    out_stance_certainty: Output = Output("<sentence>:sbx_superlim.death_penalty_stance.certainty"),
    sentence: Annotation = Annotation("<sentence>"),
    text = Text(),
    hf_model_path = Config("sbx_superlim.hf_model_path.argumentation"),
    hf_batch_size = Config("sbx_superlim.hf_inference_args.batch_size")

):
    argumentation_sentences_stance(
        out_stance,
        out_stance_certainty,
        sentence,
        text,
        'dödstraff',
        hf_model_path,
        hf_batch_size
    )


@annotator(
    "Identify the stance towards nuclear power",
    language="swe"
)
def nuclear_stance(
    out_stance: Output = Output("<sentence>:sbx_superlim.nuclear_stance"),
    out_stance_certainty: Output = Output("<sentence>:sbx_superlim.nuclear_stance.certainty"),
    sentence: Annotation = Annotation("<sentence>"),
    text = Text(),
    hf_model_path = Config("sbx_superlim.hf_model_path.argumentation"),
    hf_batch_size = Config("sbx_superlim.hf_inference_args.batch_size")

):
    argumentation_sentences_stance(
        out_stance,
        out_stance_certainty,
        sentence,
        text,
        'kärnkraft',
        hf_model_path,
        hf_batch_size
    )


@annotator(
    "Identify the stance towards cloning",
    language="swe"
)
def cloning_stance(
    out_stance: Output = Output("<sentence>:sbx_superlim.cloning_stance"),
    out_stance_certainty: Output = Output("<sentence>:sbx_superlim.cloning_stance.certainty"),
    sentence: Annotation = Annotation("<sentence>"),
    text = Text(),
    hf_model_path = Config("sbx_superlim.hf_model_path.argumentation"),
    hf_batch_size = Config("sbx_superlim.hf_inference_args.batch_size")

):
    argumentation_sentences_stance(
        out_stance,
        out_stance_certainty,
        sentence,
        text,
        'kloning',
        hf_model_path,
        hf_batch_size
    )