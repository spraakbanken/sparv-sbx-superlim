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
    TextClassificationPipeline
    )

from ..common import prepare_inputs


class ABSAbankPipeline(TextClassificationPipeline):
    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"]
        return best_class


@annotator("Label the sentiment towards immigration on a continuous 1--5 scale", language="swe")
def migration_stance(
    text : Text = Text(),
    sentence: Annotation = Annotation("<sentence>"),
    out_score: Output = Output("<sentence>:sbx_superlim.migration_stance"),
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