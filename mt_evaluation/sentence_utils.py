"""
A module to work with sentences: read, translate, encode, decode
"""

from typing import List, Tuple
import csv

import ctranslate2  # type: ignore
import sentencepiece as spm  # type: ignore

from mt_evaluation.logger import logger, log_hr


def encode_source_sentences(
    source_sentences: List[str],
    src_prefix: str,
    sp_processor: spm.SentencePieceProcessor,
) -> List[List[str]]:
    """
    Encode the source sentences using the SentencePieceProcessor
    Args:
        source_sentences: list of source sentences
        src_prefix: source prefix
        sp_processor: SentencePieceProcessor
    Returns:
        source_sentences_tokenized: list of source sentences tokenized
    """
    source_sentences_tokenized = sp_processor.encode(source_sentences, out_type=str)
    source_sentences_tokenized = [
        [src_prefix] + sentence for sentence in source_sentences_tokenized
    ]
    logger.debug(f"First sentence tokenized: {source_sentences_tokenized[0]}")
    log_hr()

    return source_sentences_tokenized


def decode_sentences(
    translated_sentences_tokenized: List[List[str]],
    target_prefix: str,
    sp_processor: spm.SentencePieceProcessor,
) -> List[List[str]]:
    """
    Decode the translated sentences using the SentencePieceProcessor
    Args:
        translated_sentences_tokenized: list of translated sentences tokenized
        target_prefix: target prefix
        sp_processor: SentencePieceProcessor
    Returns:
        translations_decoded: list of translated sentences decoded
    """

    translations_decoded = sp_processor.decode(translated_sentences_tokenized)
    translations_decoded = [sent[len(target_prefix) :] for sent in translations_decoded]
    log_hr()

    return translations_decoded


def translate_source_sentences(
    source_sentences_tokenized: List[List[str]],
    translation_model_path: str,
    target_prefix: str,
    device: str,
    beam_size: int,
    max_batch_size: int,
    batch_type: str,
) -> List[List[str]]:
    """
    Translate the source sentences
    Args:
        source_sentences_tokenized: list of source sentences tokenized
        translation_model_path: path to the translation model
        target_prefix: target prefix
        device: device to use (cpu or cuda)
        beam_size: beam size
        max_batch_size: max batch size
        batch_type: batch type
    Returns:
        translations: list of translated sentences
    """
    translator = ctranslate2.Translator(translation_model_path, device=device)
    target_prefixes = [[target_prefix]] * len(source_sentences_tokenized)
    translations = translator.translate_batch(
        source_sentences_tokenized,
        batch_type=batch_type,
        max_batch_size=max_batch_size,
        beam_size=beam_size,
        target_prefix=target_prefixes,
    )
    translations = [translation[0]["tokens"] for translation in translations]
    logger.debug(f"First sentence translated: {translations[0]}")
    log_hr()

    return translations


def extract_sentences(
    source_file_path: str,
    validation_field_name: str,
    source_field_name: str,
) -> Tuple[List[str], List[str]]:
    """
    Extract source & validation sentences from a file
    Args:
        source_file_path: path to the source file
        validation_field_name: validation field name
        source_field_name: source field name
    Returns:
        source_sentences: list of source sentences
        validation_sentences: list of validation sentences
    """
    source_sentences: List[str] = []
    validation_sentences: List[str] = []

    with open(source_file_path, "r", encoding="utf-8") as fp_in:
        reader = csv.DictReader(fp_in)
        for row in reader:
            source_sentences.append(row[source_field_name])
            validation_sentences.append(row[validation_field_name])

    logger.debug(f"First source sentence: {source_sentences[0]} \n")
    logger.debug(f"First validation sentence: {validation_sentences[0]}")
    log_hr()

    return source_sentences, validation_sentences
