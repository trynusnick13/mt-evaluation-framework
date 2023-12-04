"""
This module contains functions to compute the scores of the different metrics
using huggingface's evaluate library.
"""
from typing import List

import evaluate # type: ignore


def evaluate_bleu_score_per_sentence(
    translation_sentences: List[str], validation_sentences: List[str]
) -> List[float]:
    bleu = evaluate.load("bleu")
    results = []
    for translation, validation in zip(translation_sentences, validation_sentences):
        predictions: List[str] = [translation]
        references: List[List[str]] = [[validation]]
        result = bleu.compute(predictions=predictions, references=references)
        results.append(result)
    scores = [result["bleu"] for result in results]

    return scores


def evaluate_comet_score_per_sentence(
    translation_sentences: List[str], validation_sentences: List[str]
) -> List[float]:
    """
    Computes the COMET score for each sentence in the translation_sentences list.
    Args:
        translation_sentences: List of translated sentences.
        validation_sentences: List of validation sentences.
    Returns:
        List of COMET scores.
    """
    comet = evaluate.load("comet")
    results = []
    for translation, validation in zip(translation_sentences, validation_sentences):
        predictions: List[str] = [translation]
        references: List[List[str]] = [[validation]]
        result = comet.compute(predictions=predictions, references=references)
        results.append(result)
    scores = [result["comet"] for result in results]

    return scores


def evaluate_meteor_score_per_sentence(
    translation_sentences: List[str], validation_sentences: List[str]
) -> List[float]:
    meteor = evaluate.load("meteor")
    results = []
    for translation, validation in zip(translation_sentences, validation_sentences):
        predictions: List[str] = [translation]
        references: List[str] = [validation]
        result = meteor.compute(predictions=predictions, references=references)
        results.append(result)
    scores = [result["meteor"] for result in results]

    return scores


def evaluate_nist_score_per_sentence(
    translation_sentences: List[str], validation_sentences: List[str]
) -> List[float]:
    nist = evaluate.load("nist_mt")
    results = []
    for translation, validation in zip(translation_sentences, validation_sentences):
        predictions: str = translation
        references: List[str] = [validation]
        result = nist.compute(predictions=predictions, references=references)
        results.append(result)
    scores = [result["nist_mt"] for result in results]

    return scores


def evaluate_ter_score_per_sentence(
    translation_sentences: List[str], validation_sentences: List[str]
) -> List[float]:
    ter = evaluate.load("ter")
    results = []
    for translation, validation in zip(translation_sentences, validation_sentences):
        predictions: List[str] = [translation]
        references: List[List[str]] = [[validation]]
        result = ter.compute(predictions=predictions, references=references)
        results.append(result)
    scores = [result["score"] for result in results]

    return scores


def evaluate_chrf_score_per_sentence(
    translation_sentences: List[str], validation_sentences: List[str]
) -> List[float]:
    chrf = evaluate.load("chrf")
    results = []
    for translation, validation in zip(translation_sentences, validation_sentences):
        predictions: List[str] = [translation]
        references: List[List[str]] = [[validation]]
        result = chrf.compute(predictions=predictions, references=references)
        results.append(result)
    scores = [result["score"] for result in results]

    return scores
