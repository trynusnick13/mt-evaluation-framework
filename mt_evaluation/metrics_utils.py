"""
This module contains functions to compute the scores of the different metrics
using huggingface's evaluate library.
"""

from typing import List

from comet import download_model, load_from_checkpoint
import evaluate  # type: ignore


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


def evaluate_bleu(
    translation_sentences: List[str], validation_sentences: List[str]
) -> List[float]:
    sacrebleu = evaluate.load("sacrebleu")
    score_bleu = sacrebleu.compute(
        predictions=translation_sentences, references=validation_sentences
    )

    return score_bleu


def evaluate_spbleu_101(
    translation_sentences: List[str], validation_sentences: List[str]
) -> List[float]:
    sacrebleu = evaluate.load("sacrebleu")
    score_spbleu_101 = sacrebleu.compute(
        predictions=translation_sentences,
        references=validation_sentences,
        tokenize="flores101",
    )

    return score_spbleu_101


def evaluate_spbleu_200(
    translation_sentences: List[str], validation_sentences: List[str]
) -> List[float]:
    sacrebleu = evaluate.load("sacrebleu")
    score_spbleu_200 = sacrebleu.compute(
        predictions=translation_sentences,
        references=validation_sentences,
        tokenize="flores200",
    )

    return score_spbleu_200


def evaluate_chrf(
    translation_sentences: List[str], validation_sentences: List[str]
) -> List[float]:
    chrf = evaluate.load("chrf")
    score_chrf = chrf.compute(
        predictions=translation_sentences, references=validation_sentences
    )

    return score_chrf


def evaluate_comet(
    translation_sentences: List[str],
    validation_sentences: List[str],
    source_sentences: List[str],
) -> List[float]:
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(
            source_sentences, translation_sentences, validation_sentences
        )
    ]
    model_output = model.predict(
        data,
        batch_size=16,
    )

    return model_output


def evaluate_cometkiwi_da_xl(
    translation_sentences: List[str],
    source_sentences: List[str],
) -> List[float]:
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    model = load_from_checkpoint(model_path)
    data = [
        {"src": src, "mt": mt}
        for src, mt in zip(source_sentences, translation_sentences)
    ]
    model_output = model.predict(
        data,
        batch_size=16,
    )
    return model_output


def evaluate_cometkiwi_da_xxl(
    translation_sentences: List[str],
    source_sentences: List[str],
) -> List[float]:
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
    model = load_from_checkpoint(model_path)
    data = [
        {"src": src, "mt": mt}
        for src, mt in zip(source_sentences, translation_sentences)
    ]
    model_output = model.predict(
        data,
        batch_size=16,
    )
    return model_output


def evaluate_xcomet_xxl(
    translation_sentences: List[str],
    validation_sentences: List[str],
    source_sentences: List[str],
) -> List[float]:
    model_path = download_model("Unbabel/XCOMET-XXL")
    model = load_from_checkpoint(model_path)
    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(
            source_sentences, translation_sentences, validation_sentences
        )
    ]
    model_output = model.predict(
        data,
        batch_size=16,
    )
    return model_output


def evaluate_xcomet_xl(
    translation_sentences: List[str],
    validation_sentences: List[str],
    source_sentences: List[str],
) -> List[float]:
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(
            source_sentences, translation_sentences, validation_sentences
        )
    ]
    model_output = model.predict(
        data,
        batch_size=16,
    )
    return model_output



def evaluate_bert_score(
    translation_sentences: List[str], validation_sentences: List[str]
) -> List[float]:
    bertscore = evaluate.load("bertscore")
    score_bert = bertscore.compute(
        predictions=translation_sentences,
        references=validation_sentences,
        lang="uk",
        model_type="distilbert-base-uncased",
    )

    return score_bert
