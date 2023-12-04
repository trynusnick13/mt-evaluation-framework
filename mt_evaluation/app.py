"""
The main module of the application
"""

import time
from typing import Callable, Dict, List

import sentencepiece as spm  # type: ignore
import typer  # type: ignore
from typing_extensions import Annotated

from mt_evaluation.file_utils import write_to_file
from mt_evaluation.logger import logger, log_hr
from mt_evaluation.metrics_utils import (
    evaluate_bleu_score_per_sentence,
    evaluate_meteor_score_per_sentence,
    evaluate_nist_score_per_sentence,
    evaluate_ter_score_per_sentence,
    evaluate_comet_score_per_sentence,
)
from mt_evaluation.sentence_utils import (
    decode_sentences,
    encode_source_sentences,
    extract_sentences,
    translate_source_sentences,
)

app = typer.Typer()

METRIC_TO_FUNCTION: Dict[str, Callable] = {
    "bleu": evaluate_bleu_score_per_sentence,
    "meteor": evaluate_meteor_score_per_sentence,
    "comet": evaluate_comet_score_per_sentence,
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
) -> None:
    """
    Evaluate the translation model
    Args:
        source_file_path: path to the source file
        target_file_path: path to the target file
        translation_model_path: path to the translation model
        src_prefix: source prefix
        target_prefix: target prefix
        tokenizer_model_path: path to the tokenizer model
        validation_field_name: validation field name
        source_field_name: source field name
        metrics: list of metrics to evaluate
        device: device to use (cpu or cuda)
        beam_size: beam size
        max_batch_size: max batch size
        batch_type: batch type
    Returns:
        None
    """
    start = time.time()
    logger.info("Creating SentencePieceProcessor from %s...", tokenizer_model_path)

    # Why not tokenizer = Tokenizer.from_file?
    sp_processor = spm.SentencePieceProcessor()
    sp_processor.Load(
        tokenizer_model_path
    )  # I don't know why the linter wants Load not load
    logger.info("SentencePieceProcessor was created!")
    log_hr()
    logger.info(
        "Extracting source & validation sentences from file %s...", source_file_path
    )
    source_sentences, validation_sentences = extract_sentences(
        source_file_path=source_file_path,
        validation_field_name=validation_field_name,
        source_field_name=source_field_name,
    )
    logger.info("Sentences extracted!")
    log_hr()
    logger.info("Tokenizing sentences...")
    source_sentences_tokenized = encode_source_sentences(
        source_sentences=source_sentences,
        src_prefix=src_prefix,
        sp_processor=sp_processor,
    )
    logger.info("Tokenization completed!")
    log_hr()
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
    log_hr()
    logger.info("Decoding translation...")

    # TODO: here you receive the list of sentences (lists of tokens)
    translations_decoded = decode_sentences(
        translated_sentences_tokenized=translated_sentences_tokenized,
        target_prefix=target_prefix,
        sp_processor=sp_processor,
    )
    logger.info("Decoding completed")
    log_hr()
    logger.info("Evaluating the model...")
    metrics_evaluated: Dict[str, List[float]] = {}
    for metric in metrics:
        # TODO: it's probably better to validate the metric before evaluating
        if metric not in METRIC_TO_FUNCTION:
            logger.error(
                f"Metric {metric} is not available. Please select from {METRIC_TO_FUNCTION.keys()}",
            )
        else:
            evaluation_results = METRIC_TO_FUNCTION[metric](
                translation_sentences=translations_decoded,
                validation_sentences=validation_sentences,
            )
        metrics_evaluated[metric] = evaluation_results

    logger.info("Evaluation completed")
    log_hr()
    logger.info(f"Writing results to file {target_file_path}...")
    write_to_file(
        target_file_path=target_file_path,
        source_sentences=source_sentences,
        # TODO: here you want to provide the list of sentences List[str]
        translation_sentences=translations_decoded,
        validation_sentences=validation_sentences,
        metrics_evaluated=metrics_evaluated,
    )
    logger.info("Results saved")
    log_hr()
    logger.info(f"Execution lasted for {time.time() - start}")
