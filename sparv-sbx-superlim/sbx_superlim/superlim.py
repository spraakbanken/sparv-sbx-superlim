"""Example for a custom annotator."""

from typing import List, Literal

from sparv.api import Annotation, Config, Output, annotator
from transformers import pipeline


@annotator("Determine if a block of text is for, against or unrelated to a topic", language="swe")
def argumentation(
    sentence: Annotation = Annotation("<sentence>"),
    out_stance: Output = Output("<sentence>:sbx_superlim.argumentation.stance"),
    out_stance_certainty: Output = Output("<sentence>:sbx_superlim.argumentation.certainty"),
    hf_model_path: str = Config("sbx_superlim.hf_model_path")
):
    # TODO: Add as argument and automatically fetch allowed topics from from HF
    topic : List[Literal['abort', 'minimilön', 'marijuanalegalisering', 'dödsstraff', 'kärnkraft', 'kloning']] = 'abort'
    # TODO: Run in batches
    pipe = pipeline("text-classification", model=hf_model_path)
    # TODO: Temporary string formatting hack until training scripts and models are fixed
    output = pipe([f"{s} $ {topic}" for s in sentence.read()])
    out_stance.write([f"{topic}.{o['label']}" for o in output])
    out_stance_certainty.write([str(o['score']) for o in output])
