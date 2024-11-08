"""Annotations for a model trained on argumentation_sentences."""


from datasets import get_dataset_config_info
from sparv.api import (
    Annotation,
    Config,
    Output,
    Text,
    annotator
    )

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from ..common import prepare_inputs
from ..helpers import get_label_mapper


TOPIC_EN_SV = {
    'abortion': 'abort',
    'cloning': 'kloning',
    'death_penalty': 'dödsstraff',
    'marijuana_legalization': 'marijuanalegalisering',
    'minimum_wage': 'minimilön',
    'nuclear': 'kärnkraft'
}

def create_argsent_annotator(topic: str):
    if topic not in TOPIC_EN_SV:
        raise ValueError(f"{t} is not a valid topic")
    @annotator(f"Identify the stance towards {topic}", topic)
    def argsent_func(
        out_stance: Output = Output(f"<sentence>:sbx_superlim.{topic}_stance"),
        out_stance_certainty: Output = Output(f"<sentence>:sbx_superlim.{topic}_stance.certainty"),
        sentence: Annotation = Annotation("<sentence>"),
        text = Text(),
        hf_model_path = Config("sbx_superlim.hf_model_path.argumentation"),
        hf_batch_size = Config("sbx_superlim.hf_inference_args.batch_size")
    ):
        ds_config = get_dataset_config_info('sbx/superlim-2', 'argumentation_sent')
        model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
        sep_token = tokenizer.special_tokens_map['sep_token']
        topic_sv = TOPIC_EN_SV[topic]
        inputs = prepare_inputs(text, sentence, f" {sep_token} {topic_sv}")
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, batch_size=hf_batch_size)
        output = pipe(inputs)
        label_mapper = get_label_mapper(ds_config, pipe.model.config)
        labels = [label_mapper[o['label']] for o in output]
        out_stance.write([l for l in labels])
        out_stance_certainty.write([str(o['score']) for o in output])

for t in TOPIC_EN_SV:
    create_argsent_annotator(t)