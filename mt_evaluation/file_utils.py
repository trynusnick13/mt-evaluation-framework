"""
This module contains the functions to write the evaluation results to a file
(and maybe more in the future)
"""
from typing import Dict, List, Any
import csv


def write_to_file(
    target_file_path: str,
    source_sentences: List[str],
    translation_sentences: List[str],
    validation_sentences: List[str],
    metrics_evaluated: Dict[str, List[float]],
) -> None:
    """
    Write the evaluation results to a file
    Args:
        target_file_path: path to the target file
        source_sentences: list of source sentences
        translation_sentences: list of translated sentences
        validation_sentences: list of validation sentences
        metrics_evaluated: dictionary of metrics evaluated
    Returns:
        None
    """
    evaluation_entity: Dict[str, List[Any]] = {
        "source": source_sentences,
        "original_translation": validation_sentences,
        "mt_translation": translation_sentences,
    }
    evaluation_entity.update(metrics_evaluated)

    evaluation_entity_list: List[Dict[str, Any]] = [
        dict(zip(evaluation_entity, t)) for t in zip(*evaluation_entity.values())
    ]

    with open(target_file_path, "w", encoding="utf-8") as fp_out:
        writer = csv.DictWriter(fp_out, fieldnames=evaluation_entity_list[0].keys())
        writer.writeheader()
        writer.writerows(evaluation_entity_list)
