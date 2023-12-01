from typing import List, Tuple

import ctranslate2
import pandas as pd
import sentencepiece as spm

from mt_evaluation.logger import logger


def encode_source_sentences(
    source_sentences: List[str],
    src_prefix: str,
    sp_processor: spm.SentencePieceProcessor,
) -> List[List[str]]:
    source_sentences_tokenized = sp_processor.encode(source_sentences, out_type=str)
    source_sentences_tokenized = [
        [src_prefix] + sentence for sentence in source_sentences_tokenized
    ]
    logger.debug("First sentence tokenized: %s", source_sentences_tokenized[0])
    logger.debug("*" * 50)

    return source_sentences_tokenized


def decode_source_sentences(
    translated_sentences_tokenized: List[List[str]],
    target_prefix: str,
    sp_processor: spm.SentencePieceProcessor,
) -> List[List[str]]:
    translations_decoded = sp_processor.decode(translated_sentences_tokenized)
    translations_decoded = [sent[len(target_prefix) :] for sent in translations_decoded]
    logger.debug("*" * 50)

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
    logger.debug("First sentence translated: %s", translations[0])
    logger.debug("*" * 50)

    return translations


def extract_sentences(
    source_file_path: str,
    validation_field_name: str,
    source_field_name: str,
) -> Tuple[List[str], List[str]]:
    dataset = pd.read_csv(source_file_path)
    source_sentences = dataset[source_field_name].to_list()
    validation_sentences = dataset[validation_field_name].to_list()
    logger.debug("First source sentence: %s /n", source_sentences[0])
    logger.debug("First validation sentence: %s", validation_sentences[0])
    logger.debug("*" * 50)

    return source_sentences, validation_sentences
