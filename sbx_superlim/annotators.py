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
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TextClassificationPipeline, 
    pipeline
    )

from .common import prepare_inputs
from .helpers import get_label_mapper


@annotator(
    "Identify the stance towards a given topic", 
    language="swe"
)
def argumentation_sentences_stance(
    text : Text = Text(),
    sentence: Annotation = Annotation("<sentence>"),
    out_stance: Output = Output("<sentence>:sbx_superlim.argumentation.stance"),
    out_stance_certainty: Output = Output("<sentence>:sbx_superlim.argumentation.certainty"),
    hf_model_path: str = Config("sbx_superlim.hf_model_path.argumentation"),
    hf_batch_size: int = Config("sbx_superlim.hf_inference_args.batch_size")
):
    # TODO: figure out how to pass this as an argument
    topic : List[Literal['abort', 'minimilön', 'marijuanalegalisering', 'dödsstraff', 'kärnkraft', 'kloning']] = 'kärnkraft'
    ds_config = get_dataset_config_info('sbx/superlim-2', 'argumentation_sent')
    inputs = prepare_inputs(text, sentence, f" [SEP] {topic}") # TODO: Change [SEP] token depending on model.
    pipe = pipeline("text-classification", model=hf_model_path)
    output = pipe(inputs)
    label_mapper = get_label_mapper(ds_config, pipe.model.config)
    labels = [label_mapper[o['label']] for o in output]
    out_stance.write([f"{topic}.{l}" for l in labels])
    out_stance_certainty.write([str(o['score']) for o in output])



class ABSAbankPipeline(TextClassificationPipeline):
    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"]
        return best_class


@annotator("Label the sentiment towards immigration on a continuous 1--5 scale", language="swe")
def absabank_imm(
    text : Text = Text(),
    sentence: Annotation = Annotation("<sentence>"),
    out_score: Output = Output("<sentence>:sbx_superlim.absabank-imm.score"),
    hf_model_path: str = Config("sbx_superlim.hf_model_path.absabank-imm"),
    hf_batch_size: int = Config("sbx_superlim.hf_inference_args.batch_size")
):
    inputs = prepare_inputs(text, sentence)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
    # TODO: Decide padding policy
    pipe = ABSAbankPipeline(model=model, tokenizer=tokenizer)
    model_outputs = pipe(inputs, batch_size=hf_batch_size)
    scores = [str(float(o)) for o in model_outputs]
    out_score.write(scores)


@annotator(
        "Determine whether a sentence is correct Swedish or not", 
        language="swe"
)
def dalaj_ged(
    text : Text = Text(),
    sentence: Annotation = Annotation("<sentence>"),
    out_label : Output = Output("<sentence>:sbx_superlim.dalaj-ged.label"),
    out_certainty: Output = Output("<sentence>:sbx_superlim.dalaj-ged.certainty"),
    hf_model_path: str = Config("sbx_superlim.hf_model_path.dalaj-ged"),
    hf_batch_size: int = Config("sbx_superlim.hf_inference_args.batch_size")
):
    ds_config = get_dataset_config_info('sbx/superlim-2', 'dalaj-ged')
    inputs = prepare_inputs(text, sentence)
    pipe = pipeline("text-classification", model=hf_model_path)
    output = pipe(inputs, batch_size = hf_batch_size)
    label_mapper = get_label_mapper(ds_config, pipe.model.config)
    labels = [label_mapper[o['label']] for o in output]
    out_label.write(labels)
    out_certainty.write([str(o['score']) for o in output])


@annotator("Determine the logical relation between two sentences", language="swe")
def swenli(
    text : Text = Text(),
    sentence: Annotation = Annotation("<sentence>"),
    out_label : Output = Output("<sentence>:sbx_superlim.swenli.label"),
    out_certainty: Output = Output("<sentence>:sbx_superlim.swenli.certainty"),
    hf_model_path: str = Config("sbx_superlim.hf_model_path.swenli"),
    hf_batch_size: int = Config("sbx_superlim.hf_inference_args.batch_size")
):
    ds_config = get_dataset_config_info('sbx/superlim-2', 'swenli')
    inputs = prepare_inputs(text, sentence, input_type = 'sentence_pair')
    pipe = pipeline("text-classification", model=hf_model_path)
    output = pipe(inputs, batch_size=hf_batch_size)
    label_mapper = get_label_mapper(ds_config, pipe.model.config)
    labels = ["N/A"] + [label_mapper[o['label']] for o in output]
    out_label.write(labels)
    out_certainty.write(["N/A"] + [str(o['score']) for o in output])


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


