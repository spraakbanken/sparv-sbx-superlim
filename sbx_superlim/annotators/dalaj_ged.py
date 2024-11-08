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