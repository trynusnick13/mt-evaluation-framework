"""
This module contains the functions to write the evaluation results to a file
(and maybe more in the future)
"""

from typing import Dict, List, Any, Optional
import csv
import json
import os


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


def get_sentences_from_file(
    source_file_path: str, field_name: str, file_type: Optional[str] = None
) -> List[str]:
    """
    Get all sentences from a single column from csv or jsonl file
    Args:
        source_file_path: path to the source file
        field_name: column/field with sentences to be extracted
        file_type: type of file ('csv' or 'jsonl'). If not provided, inferred from file extension.
    Returns:
        List[str]
    """
    if file_type is None:
        _, extension = os.path.splitext(source_file_path)
        file_type = extension.lstrip(".").lower()

    all_sentences: List[str] = []

    if file_type == "csv":
        with open(source_file_path, "r", encoding="utf-8") as fp_in:
            reader = csv.DictReader(fp_in)
            for row in reader:
                all_sentences.append(row[field_name])
    elif file_type == "jsonl":
        with open(source_file_path, "r", encoding="utf-8") as fp_in:
            for line in fp_in:
                json_obj = json.loads(line)
                all_sentences.append(json_obj[field_name])
    else:
        raise ValueError(
            f"Unsupported file type: {file_type}. Supported types are 'csv' and 'jsonl'."
        )

    return all_sentences
