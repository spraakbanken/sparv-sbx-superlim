"""Example for a custom annotator."""

import os

from typing import List, Literal

from datasets import get_dataset_config_info
from sparv.api import (
    Annotation,
    AllSourceFilenames,
    Config,
    Export,
    Output,
    Text,
    annotator,
    exporter
    )

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from .common import prepare_inputs
from .helpers import get_label_mapper


@annotator(
    "Identify the stance towards a given topic", 
    language="swe"
)
def argumentation(
    text : Text = Text(),
    sentence: Annotation = Annotation("<sentence>"),
    out_stance: Output = Output("<sentence>:sbx_superlim.argumentation.stance"),
    out_stance_certainty: Output = Output("<sentence>:sbx_superlim.argumentation.certainty"),
    hf_model_path: str = Config("sbx_superlim.hf_model_path.argumentation")
):
    # TODO: figure out how to pass this as an argument
    topic : List[Literal['abort', 'minimilön', 'marijuanalegalisering', 'dödsstraff', 'kärnkraft', 'kloning']] = 'abort'
    ds_config = get_dataset_config_info('sbx/superlim-2', 'argumentation_sent')
    inputs = prepare_inputs(text, sentence, f" $ {topic}")
    pipe = pipeline("text-classification", model=hf_model_path)
    output = pipe(inputs)
    label_mapper = get_label_mapper(ds_config, pipe.model.config)
    labels = [label_mapper[o['label']] for o in output]
    out_stance.write([f"{topic}.{l}" for l in labels])
    out_stance_certainty.write([str(o['score']) for o in output])


@annotator("Label the sentiment towards immigration on a continuous 1--5 scale", language="swe")
def absabank_imm(
    text : Text = Text(),
    sentence: Annotation = Annotation("<sentence>"),
    out_score: Output = Output("<sentence>:sbx_superlim.absabank-imm.score"),
    hf_model_path: str = Config("sbx_superlim.hf_model_path.absabank-imm")
):
    inputs = prepare_inputs(text, sentence)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
    # TODO: Decide padding policy
    model_outputs = model(**tokenizer(inputs, return_tensors='pt', truncation=True, padding=True))
    scores = model_outputs.logits[:,0].tolist()
    out_score.write([str(s) for s in scores])


@annotator(
        "Determine whether a sentence is correct Swedish or not", 
        language="swe"
)
def dalaj_ged(
    text : Text = Text(),
    sentence: Annotation = Annotation("<sentence>"),
    out_label : Output = Output("<sentence>:sbx_superlim.dalaj-ged.label"),
    out_certainty: Output = Output("<sentence>:sbx_superlim.dalaj-ged.certainty"),
    hf_model_path: str = Config("sbx_superlim.hf_model_path.dalaj-ged")
):
    ds_config = get_dataset_config_info('sbx/superlim-2', 'dalaj-ged')
    inputs = prepare_inputs(text, sentence)
    pipe = pipeline("text-classification", model=hf_model_path)
    output = pipe(inputs)
    label_mapper = get_label_mapper(ds_config, pipe.model.config)
    labels = [label_mapper[o['label']] for o in output]
    out_label.write(labels)
    out_certainty.write([str(o['score']) for o in output])


@annotator("Determine how related two words are on a continuous scale from 0 to 10")
def supersim_relatedness(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Determine how similar two words are on a continuous scale from 0 to 10")
def supersim_similarity(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Given a word pair A:B and a word C, find a word D such that A:B = C:D")
def sweanalogy(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Given a newspaper article, provide its summary")
def swedn(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Given the question, find the suitable answer within the same category")
def swefaq(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Select the correct synonym")
def swesat_synonyms(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Determine if the target word in two contexts expresses the same sense.")
def swewic(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


def pair_files(filenames : AllSourceFilenames) -> List[tuple]:
    pairs : dict = {}
    for fn in filenames:
        stem, _, = fn.split('.')
        pairs.setdefault(stem, []).append(fn)
    return [tuple(values) for values in pairs.values()]


@exporter("Determine the logical relation between two sentences", language="swe")
def swenli(
    source_files: AllSourceFilenames = AllSourceFilenames(),
    hf_model_path: str = Config("sbx_superlim.hf_model_path.swenli"),
    out: Export = Export("<sentence>:sbx_superlim.swenli/labels.csv"),
):
    pairs = pair_files(source_files)
    for sf_sv, sf_en in pairs:
        prefix = 'source'
        pair_inputs = []
        with open(f'{prefix}/{sf_sv}.txt') as f1, open(f'{prefix}/{sf_en}.txt') as f2:
            for line_sv, line_en in zip(f1.readlines(), f2.readlines()):
                pair_inputs.append(" ".join(line_sv + line_en))
    # Temp
    inputs = pair_inputs
    ds_config = get_dataset_config_info('sbx/superlim-2', 'swenli')
    pipe = pipeline("text-classification", model=hf_model_path)
    output = pipe(inputs)
    label_mapper = get_label_mapper(ds_config, pipe.model.config)
    labels = [label_mapper[o['label']] for o in output]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.writelines([f"{l}\n" for l in labels])
