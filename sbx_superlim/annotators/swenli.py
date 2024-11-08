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