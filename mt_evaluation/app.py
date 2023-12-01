import time
from typing import Callable, Dict, List

import sentencepiece as spm
import typer
from typing_extensions import Annotated

from mt_evaluation.file_utils import write_to_file
from mt_evaluation.logger import logger
from mt_evaluation.metrics_utils import (
    evaluate_bleu_score_per_sentence,
    evaluate_meteor_score_per_sentence,
    evaluate_nist_score_per_sentence,
    evaluate_ter_score_per_sentence,
)
from mt_evaluation.sentence_utils import (
    decode_source_sentences,
    encode_source_sentences,
    extract_sentences,
    translate_source_sentences,
)

app = typer.Typer()

METRIC_TO_FUNCTION: Dict[str, Callable] = {
    "bleu": evaluate_bleu_score_per_sentence,
    "meteor": evaluate_meteor_score_per_sentence,
    "nist": evaluate_nist_score_per_sentence,
    "ter": evaluate_ter_score_per_sentence,
}


@app.command()
def evaluate_model(
    source_file_path: Annotated[str, typer.Option()],
    target_file_path: Annotated[str, typer.Option()],
    translation_model_path: Annotated[str, typer.Option()],
    src_prefix: Annotated[str, typer.Option()],
    target_prefix: Annotated[str, typer.Option()],
    tokenizer_model_path: Annotated[str, typer.Option()],
    validation_field_name: Annotated[str, typer.Option()],
    source_field_name: Annotated[str, typer.Option()],
    metrics: Annotated[List[str], typer.Option()] = ["bleu", "meteor"],
    device: str = "cpu",
    beam_size: int = 5,
    max_batch_size: int = 2024,
    batch_type: str = "tokens",
):
    start = time.time()
    logger.info("Creating SentencePieceProcessor from %s...", tokenizer_model_path)
    sp_processor = spm.SentencePieceProcessor()
    sp_processor.load(tokenizer_model_path)
    logger.info("SentencePieceProcessor was created!")
    logger.info("*" * 50)
    logger.info(
        "Extracting source & validation sentences from file %s...", source_file_path
    )
    source_sentences, validation_sentences = extract_sentences(
        source_file_path=source_file_path,
        validation_field_name=validation_field_name,
        source_field_name=source_field_name,
    )
    logger.info("Senteces extracted!")
    logger.info("*" * 50)
    logger.info("Tokenizing sentences...")
    source_sentences_tokenized = encode_source_sentences(
        source_sentences=source_sentences,
        src_prefix=src_prefix,
        sp_processor=sp_processor,
    )
    logger.info("Tokenization completed!")
    logger.info("*" * 50)
    logger.info("Translating sentences...")
    translated_sentences_tokenized = translate_source_sentences(
        source_sentences_tokenized=source_sentences_tokenized,
        translation_model_path=translation_model_path,
        target_prefix=target_prefix,
        device=device,
        beam_size=beam_size,
        max_batch_size=max_batch_size,
        batch_type=batch_type,
    )
    logger.info("Translation completed")
    logger.info("*" * 50)
    logger.info("Decoding translation...")
    translations_decoded = decode_source_sentences(
        translated_sentences_tokenized=translated_sentences_tokenized,
        target_prefix=target_prefix,
        sp_processor=sp_processor,
    )
    logger.info("Decoding completed")
    logger.info("*" * 50)
    logger.info("Evaluating the model...")
    metrics_evaluated: Dict[str, List[float]] = {}
    for metric in metrics:
        if metric not in METRIC_TO_FUNCTION.keys():
            logger.error(
                "Metric %s is not available. Please select from %s",
                metric,
                METRIC_TO_FUNCTION.keys(),
            )
        else:
            evaluation_results = METRIC_TO_FUNCTION[metric](
                translation_sentences=translations_decoded,
                validation_sentences=validation_sentences,
            )
        metrics_evaluated[metric] = evaluation_results
    logger.info("Evaluation completed")
    logger.info("*" * 50)
    logger.info("Wrting results to file %s...", target_file_path)
    write_to_file(
        target_file_path=target_file_path,
        source_sentences=source_sentences,
        translation_sentences=translations_decoded,
        validation_sentences=validation_sentences,
        metrics_evaluated=metrics_evaluated,
    )
    logger.info("Results saved")
    logger.info("*" * 50)
    logger.info("Execution lasted for %s", time.time() - start)
