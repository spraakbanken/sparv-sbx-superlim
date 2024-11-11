# Sparv-Superlim: A plugin for classifying text using the models trained on tasks in Superlim

Sparv-Superlim is a [Sparv](https://github.com/spraakbanken/sparv-pipeline) plugin for classifying text using the Superlim baseline models. Superlim is a multi-task benchmark for Swedish, which includes baseline models.

# How to use?

Install Sparv-Superlim by injecting it into the Sparv Pipeline: 

```
pipx inject sparv-pipeline git@github.com:spraakbanken/sparv-sbx-superlim.git
```

See the [Sparv documentation](https://spraakbanken.gu.se/sparv/#/user-manual/installation-and-setup?id=plugins) for more details on how to install plugins.

Then make a config file and choose the relevant annotations:

```
metadata:
  id: corpora
  name:
    eng: corpora
    swe: korpora
  language: swe
  description:
    eng: Swedish political manifestos with Superlim annotations
    swe: Svenska valmanifest med Superlimannoteringar
import:
  source_dir: source_small
  importer: text_import:parse
export:
  default:
    - xml_export:pretty
    - sbx_superlim:predictions
  annotations:
    - <token>
    - <sentence>
    - <sentence>:sbx_superlim.migration_stance
    - <sentence>:sbx_superlim.nuclear_stance
sbx_superlim:
  hf_model_path:
    absabank-imm: 'sbx/bert-base-swedish-cased_absabank-imm'
    argumentation: 'sbx/bert-base-swedish-cased-argumentation_sent'
  hf_inference_args:
    batch_size: 32
```

Plugin-specicific variables which start with ```hf``` are HuggingFace parameters. The most important one is the ```hf_model_path``` which tells which fine-tuned model to use for each task.

Full working examples can be found in the ```examples``` folder.

# Available annotations

So far, Sparv-Superlim provides 10 different annotations. These are summarized in the table below:


|       Superlim task      | Sparv-Superlim Annotation |                  Annotation                 |                  Label                 |     Segment    |
|:------------------------:|---------------------------|:-------------------------------------------:|:--------------------------------------:|:--------------:|
| absabank-imm             | migration_stance          | Attitude towards  immigration               | float  between 1-5                     | sentence       |
| argumentation- sentences | [*topic*]_stance          | Stance to a  given *topic*                | pro, con or  neutral                   | sentence       |
| dalaj-ged                | correct_swedish           | Correct Swedish                             | correct or  incorrect                  | sentence       |
| swenli                   | previous_entailment       | The logical  relationship  of two sentences | entailment,  contradiction  or neutral | sentence pair  |
| sweparaphrase            | similarity                | Similarity  between  two sentences          | float  between 1-5                     | sentence  pair |


# Wish to contribute?

Do you have new, innovative ways of incorporating models trained on Superlim into Sparv-Superlim? Make a feature request or even better a pull request!

# How to cite?

Please cite the following technical report: *Felix Morger. 2024. When Sparv met Superlim…A Sparv plugin for natural
language understanding analysis of Swedish. Tech. rep. University of Gothenburg*. You can also use the bibtex entry below.


```
@techreport{sparv-superlim,
  title =	 {When {S}parv met {S}uperlim\ldots {A} {S}parv Plugin for Natural Language Understanding Analysis of {S}wedish},
  author =	 {Morger, Felix},
  url = {https://hdl.handle.net/2077/83664},
  year =	 {2024},
  publisher = {Språkbanken Text},
  institution =	 {University of Gothenburg},
}
```
