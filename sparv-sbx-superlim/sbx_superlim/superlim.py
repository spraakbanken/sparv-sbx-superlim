"""Example for a custom annotator."""

from sparv.api import Annotation, Config, Output, annotator
from transformers import pipeline


# @annotator("Determine if a sentence is for, against or unrelated to a topic", language="swe")
# def argumentation(
#     sentence: Annotation = Annotation("<sentence>"),
#     out_stance: Output = Output("<sentence>:sbx_superlim.argumentation_stance"),
#     out_stance_certainty: Output = Output("<sentence>:sbx_superlim.argumentation_stance_certainty"),
#     hf_model_path: str = Config("sbx_superlim.hf_model_path")
# ):
#     pipe = pipeline("text-classification", model=hf_model_path)
#     output = pipe(sentence)
#     out_stance.write(output['label'])
#     out_stance_certainty.write(output['score'])


@annotator("Test annotation function", language="swe")
def test(
    sentence: Annotation = Annotation("<sentence>"),
    out: Output = Output("<sentence>:sbx_superlim.test")
):
    print("alskdjklasjdlkasjd")
    out.write("test")
