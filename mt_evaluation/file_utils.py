from typing import Dict, List

import pandas as pd


def write_to_file(
    target_file_path: str,
    source_sentences: List[str],
    translation_sentences: List[str],
    validation_sentences: List[str],
    metrics_evaluated: Dict[str, List[float]],
) -> None:
    evaluation_entity = {
        "source": source_sentences,
        "original_translation": validation_sentences,
        "mt_translation": translation_sentences,
    }
    for metric, scores in metrics_evaluated.items():
        evaluation_entity[metric] = scores
    df = pd.DataFrame.from_dict(evaluation_entity)

    df.to_csv(target_file_path)
