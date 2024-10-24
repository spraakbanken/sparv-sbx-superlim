import os
import pandas as pd

from pathlib import Path

from datasets import get_dataset_config_info

from sparv.api import (
    AllSourceFilenames,
    AnnotationAllSourceFiles,
    Config,
    Export,
    exporter,
    Text
    )

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
    )

from .common import pair_files
from .helpers import get_label_mapper


@exporter("Saves sentence-wise predictions to a .tsv for further analysis", language="swe")
def predictions(
    source_files: AllSourceFilenames = AllSourceFilenames(),
    annotation_source_sentences: AnnotationAllSourceFiles = AnnotationAllSourceFiles("<sentence>"),
    annotation_migration_stance: AnnotationAllSourceFiles = AnnotationAllSourceFiles("<sentence>:sbx_superlim.migration_stance"),
    annotation_nuclear_stance: AnnotationAllSourceFiles = AnnotationAllSourceFiles("<sentence>:sbx_superlim.nuclear_stance"),
    annotation_nuclear_certainty: AnnotationAllSourceFiles = AnnotationAllSourceFiles("<sentence>:sbx_superlim.nuclear_stance.certainty"),
    # Add these to label files too!
    out: Export = Export("sbx_superlim.predictions/summary.tsv")
):
    rows = []
    for sf in source_files:
        text = Text(sf).read()
        party, year, _ = sf.split('-', maxsplit=2)
        migration_stances = [float(s) for s in annotation_migration_stance.read(sf)]
        nuclear_stances = [s for s in annotation_nuclear_stance.read(sf)]
        nuclear_certainty = [c for c in annotation_nuclear_certainty.read(sf)]
        text = Text(sf).read()
        sentences = [text[start: end] for start, end in annotation_source_sentences.read_spans(sf)]
        pred_fn = f'export/sbx_superlim.predictions/{sf}.tsv'
        output_dir = Path(pred_fn).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'sentence': sentences, 
            'migration_stance': migration_stances, 
            'nuclear_stance': nuclear_stances,
            'nuclear_certainty': nuclear_certainty
        }).to_csv(pred_fn, sep='\t', index=False)
        n_sents = len(migration_stances)
        mean = sum(migration_stances) / len(migration_stances) if n_sents > 0 else 0
        row = [sf, party, year, mean, n_sents]
        rows.append(row)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame\
        .from_records(rows, columns = ['source_file', 'party', 'year', 'score', 'n_sents'])\
        .sort_values('source_file')\
        .to_csv(out, sep='\t', index=False)

@exporter("Determine the logical relation between two sentences", language="swe")
def swenli_parallel(
    source_files: AllSourceFilenames = AllSourceFilenames(),
    hf_model_path: str = Config("sbx_superlim.hf_model_path.swenli"),
    out: Export = Export("sbx_superlim.swenli/predictions.tsv"),
    hf_batch_size: int = Config("sbx_superlim.hf_inference_args.batch_size")
):
    pairs = pair_files(source_files)
    pair_predictions : dict = {}
    for sf_sv, sf_en in pairs:
        prefix = 'source'
        with open(f'{prefix}/{sf_sv}.txt') as f1, open(f'{prefix}/{sf_en}.txt') as f2:
            pair_inputs = []
            for line_sv, line_en in zip(f1.readlines(), f2.readlines()):
                pair_inputs.append(" ".join(line_sv + line_en))
            ds_config = get_dataset_config_info('sbx/superlim-2', 'swenli')
            pipe = pipeline("text-classification", model=hf_model_path, batch_size = hf_batch_size)
            output = pipe(pair_inputs, batch_size=hf_batch_size)
            label_mapper = get_label_mapper(ds_config, pipe.model.config)
            base = sf_sv.split(".")[0]
            pair_predictions[f"{base}.label"] = [label_mapper[o['label']] for o in output]
            pair_predictions[f"{base}.score"] = [o['score'] for o in output]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame.from_records(pair_predictions).to_csv(out, sep='\t')


@exporter("Determine the logical relation between two sentences", language="swe")
def swepar_parallel(
    source_files: AllSourceFilenames = AllSourceFilenames(),
    hf_model_path: str = Config("sbx_superlim.hf_model_path.swepar"),
    out: Export = Export("sbx_superlim.swepar/predictions.tsv"),
    hf_batch_size: int = Config("sbx_superlim.hf_inference_args.batch_size")
):
    pairs = pair_files(source_files)
    pair_predictions : dict = {}
    for sf_sv, sf_en in pairs:
        prefix = 'source'
        with open(f'{prefix}/{sf_sv}.txt') as f1, open(f'{prefix}/{sf_en}.txt') as f2:
            pair_inputs = []
            for line_sv, line_en in zip(f1.readlines(), f2.readlines()):
                pair_inputs.append(" ".join(line_sv + line_en))
            # TODO: implement as pipeline
            model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
            tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
            model_outputs = model(**tokenizer(pair_inputs, return_tensors='pt', truncation=True, padding=True))
            output = model_outputs.logits[:,0].tolist()
            base = sf_sv.split(".")[0]
            pair_predictions[f"{base}.score"] = [o for o in output]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame.from_records(pair_predictions).to_csv(out, sep='\t')