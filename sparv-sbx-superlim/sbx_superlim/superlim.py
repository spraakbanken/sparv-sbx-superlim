"""Example for a custom annotator."""

from typing import List, Literal

from datasets import get_dataset_config_info
from sparv.api import Annotation, Output, annotator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from .helpers import get_label_mapper


@annotator(
    "Identify the stance towards a given topic", 
    language="swe"
)
def argumentation(
    sentence: Annotation = Annotation("<sentence>"),
    word: Annotation = Annotation("<token:word>"),
    out_stance: Output = Output("<sentence>:sbx_superlim.argumentation.stance"),
    out_stance_certainty: Output = Output("<sentence>:sbx_superlim.argumentation.certainty"),
    hf_model_path: str = "sbx/bert-base-swedish-cased_argumentation_sent"
):
    # TODO: figure out how to pass this as an argument
    topic : List[Literal['abort', 'minimilön', 'marijuanalegalisering', 'dödsstraff', 'kärnkraft', 'kloning']] = 'abort'
    ds_config = get_dataset_config_info('sbx/superlim-2', 'argumentation_sent')
    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())
    inputs : List[str] = []
    for s in sentences:
        s_words = []
        for w_idx in s:
            s_words.append(token_word[w_idx])
        sentence_string = " ".join(s_words)
        # TODO: Temporary string formatting hack until training scripts and models are fixed
        input_string = f"{sentence_string} $ {topic}"
        inputs.append(input_string)
    pipe = pipeline("text-classification", model=hf_model_path)
    output = pipe(inputs)
    label_mapper = get_label_mapper(ds_config, pipe.model.config)
    labels = [label_mapper[o['label']] for o in output]
    out_stance.write([f"{topic}.{l}" for l in labels])
    out_stance_certainty.write([str(o['score']) for o in output])


@annotator("Label the sentiment towards immigration on a continuous 1--5 scale", language="swe")
def absabank_imm(
    sentence: Annotation = Annotation("<sentence>"),
    word: Annotation = Annotation("<token:word>"),
    out_score: Output = Output("<sentence>:sbx_superlim.absabank-imm.score"),
    hf_model_path: str = "sbx/bert-base-swedish-cased_absabank-imm"
):
    ds_config = get_dataset_config_info('sbx/superlim-2', 'absabank-imm')
    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())
    inputs : List[str] = []
    for s in sentences:
        s_words = []
        for w_idx in s:
            s_words.append(token_word[w_idx])
        sentence_string = " ".join(s_words)
        inputs.append(sentence_string)
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
    sentence: Annotation = Annotation("<sentence>"),
    word: Annotation = Annotation("<token:word>"),
    out_label : Output = Output("<sentence>:sbx_superlim.dalaj-ged.label"),
    out_certainty: Output = Output("<sentence>:sbx_superlim.dalaj-ged.certainty"),
    hf_model_path: str = "sbx/bert-base-swedish-cased_dalaj-ged"
):  
    ds_config = get_dataset_config_info('sbx/superlim-2', 'dalaj-ged')
    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())
    inputs : List[str] = []
    for s in sentences:
        s_words = []
        for w_idx in s:
            s_words.append(token_word[w_idx])
        sentence_string = " ".join(s_words)
        inputs.append(sentence_string)
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


@annotator("Determine the logical relation between two sentences", language="swe")
def swenli(
    sentence: Annotation = Annotation("<sentence>")
):
    raise NotImplementedError("Sparv use case not yet defined.")
