# MT Translation Framework

This framework is designed to evaluate MT translation. This Framework validates the translation quality of the models

## Usage Guide
To start working with the framework need to complete the setup steps:
1. Execute `poetry install`
2. Prepare model to run
3. Execute `poetry run mt-evaluation evaluate-model --source-file-path=flores_eng_ukr_minor.csv --target-file-path=target --translation-model-path=m2m100_418m/ --src-prefix=__en__ --target-prefix=__uk__ --tokenizer-model-path=./m2m100_418m/sentencepiece.model --validation-field-name=sentence_ukr_Cyrl --source-field-name=sentence_eng_Latn --metrics bleu --metrics meteor --metrics ter`
4. Execute to evaluate csv file with translations  `poetry run mt-evaluation evaluate-сsv --source-file-path=source.csv --target-file-path=target.csv --validation-field-name=original_translation  --translation-field-name=mt_translation --source-field-name=source`

Arguments Description
* `source-file-path` - csv file with data(example file `example/flores_eng_ukr_minor.csv`)
* `target-file-path` - path to store the validation scores
* `translation-model-path` - path to the translation model
* `src-prefix` - source language https://github.com/huggingface/transformers/blob/main/src/transformers/models/m2m_100/tokenization_m2m_100.py#L58
* `target-prefix` - target language https://github.com/huggingface/transformers/blob/main/src/transformers/models/m2m_100/tokenization_m2m_100.py#L58
* `tokenizer-model-path` - path to tokenizer
* `validation-field-name` - field name in the csv file that will be used in validation purposes
* `source-field-name` - field name in the csv file on which the translation model will be applied
* `metrics` - metrics to be validated on the data. Available options: BLEU, METEOR, chrf, TER, NIST(to be fixed)
* `translation-field-name` - field name with mt translation


## Current status

### High priority
- [ ] Add tests
- [ ] Improve CI
- [ ] Add Dockerfile
- [ ] Update Readme with more usage & inner libraries description
- [ ] Add ML-based metrics https://machinetranslate.org/metrics#machine-learning-based-metrics
- [ ] Fix NIST metric (error for example test set: `Mismatch in the number of predictions (305) and references (1)`)
- [x] Fix linting issues
- [x] Fix mypy issues
- [x] Add docstrings
- [ ] Add batch processing options with data(batch evaluation, etc.)
- [ ] Add progress bar



### Low priority
- [ ] Better set up pre-commit

### Done
- [x] MVP
- [x] Added evaluation for metrics: BLEU, METEOR, chrf, TER
