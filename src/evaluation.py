"""
Compare predicted answer to gold answer and alt_ans; compute accuracy.
Do not change this module's location in the project.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


def normalize_text(text: Any) -> str:
	if text is None:
		text = ""
	elif not isinstance(text, str):
		text = str(text)
	text = text.strip().lower()
	text = re.sub(r"\s+", " ", text)
	text = re.sub(r"[^a-z0-9\s]", "", text)
	return text.strip()


def is_correct(prediction: Any, answer: Any, alt_ans: List[Any]) -> bool:
	pred = normalize_text(prediction)
	candidates = [answer] + list(alt_ans or [])
	normalized_candidates = [normalize_text(c) for c in candidates if c is not None]

	if pred in normalized_candidates:
		return True

	# Soft containment handles short factual variants.
	for gold in normalized_candidates:
		if gold and (gold in pred or pred in gold):
			return True

	return False


def accuracy_from_flags(flags: Iterable[bool]) -> float:
	flags = list(flags)
	if not flags:
		return 0.0
	return sum(1 for f in flags if f) / len(flags)


def evaluate_pipeline_predictions(records: List[Dict]) -> Dict[str, float]:
	# records rows: {pipeline, prediction, answer, alt_ans}
	grouped: Dict[str, List[bool]] = {}
	for row in records:
		pipeline = row["pipeline"]
		grouped.setdefault(pipeline, []).append(
			is_correct(row.get("prediction", ""), row.get("answer", ""), row.get("alt_ans", []))
		)

	return {name: accuracy_from_flags(flags) for name, flags in grouped.items()}
