import re
from collections import Counter


ID_SPLIT_RE = re.compile(r"[\s\u3000\-;]+")
MISSING = "<missing>"
EXTRA = "<extra>"
IGNORED_CONTROL_IDS = {"r"}


def annotation_ids(annotation):
	"""Flatten an NRDB annotation into ordered linguistic IDs, excluding non-evaluative controls."""
	return [
		value for value in ID_SPLIT_RE.split(str(annotation or "").strip())
		if value and value not in IGNORED_CONTROL_IDS
	]


def align_ids(predicted, gold):
	"""Levenshtein-align ID sequences and return operation counts and error pairs."""
	predicted = list(predicted)
	gold = list(gold)
	rows = len(gold) + 1
	cols = len(predicted) + 1
	dp = [[0] * cols for _ in range(rows)]
	back = [[None] * cols for _ in range(rows)]
	for i in range(1, rows):
		dp[i][0] = i
		back[i][0] = "delete"
	for j in range(1, cols):
		dp[0][j] = j
		back[0][j] = "insert"

	for i in range(1, rows):
		for j in range(1, cols):
			if gold[i - 1] == predicted[j - 1]:
				dp[i][j] = dp[i - 1][j - 1]
				back[i][j] = "match"
				continue
			choices = [
				(dp[i - 1][j - 1] + 1, "substitute"),
				(dp[i - 1][j] + 1, "delete"),
				(dp[i][j - 1] + 1, "insert"),
			]
			dp[i][j], back[i][j] = min(choices, key=lambda value: value[0])

	counts = {"matches": 0, "substitutions": 0, "insertions": 0, "deletions": 0}
	confusions = []
	i = len(gold)
	j = len(predicted)
	while i > 0 or j > 0:
		op = back[i][j]
		if op == "match":
			counts["matches"] += 1
			i -= 1
			j -= 1
		elif op == "substitute":
			counts["substitutions"] += 1
			confusions.append((gold[i - 1], predicted[j - 1]))
			i -= 1
			j -= 1
		elif op == "delete":
			counts["deletions"] += 1
			confusions.append((gold[i - 1], MISSING))
			i -= 1
		elif op == "insert":
			counts["insertions"] += 1
			confusions.append((EXTRA, predicted[j - 1]))
			j -= 1
		else:
			raise RuntimeError("invalid alignment state")

	confusions.reverse()
	gold_count = len(gold)
	predicted_count = len(predicted)
	edits = counts["substitutions"] + counts["insertions"] + counts["deletions"]
	denominator = max(gold_count, predicted_count)
	counts.update({
		"gold_ids": gold_count,
		"predicted_ids": predicted_count,
		"edits": edits,
		"linguistic_exact": edits == 0,
		"confusions": confusions,
		"id_match_rate": counts["matches"] / denominator if denominator else 1.0,
		"id_error_rate": edits / gold_count if gold_count else (0.0 if predicted_count == 0 else 1.0),
	})
	return counts


def annotation_metrics(predicted_annotation, gold_annotation):
	return align_ids(annotation_ids(predicted_annotation), annotation_ids(gold_annotation))


def _segmentation_signature(segmented):
	"""Return unsegmented surface and hyphen-boundary offsets, ignoring whitespace differences."""
	text = str(segmented or "").strip()
	surface = []
	boundaries = set()
	offset = 0
	for char in text:
		if char == "-":
			if offset > 0:
				boundaries.add(offset)
			continue
		if char.isspace() or char == "\u3000":
			continue
		surface.append(char)
		offset += 1
	return "".join(surface), boundaries


def segmentation_metrics(predicted_segmented, gold_segmented):
	"""Score morpheme boundaries by character offset in the unsegmented transcription."""
	predicted_surface, predicted_boundaries = _segmentation_signature(predicted_segmented)
	gold_surface, gold_boundaries = _segmentation_signature(gold_segmented)
	surface_match = predicted_surface == gold_surface
	if surface_match:
		correct = len(predicted_boundaries & gold_boundaries)
		false_positive = len(predicted_boundaries - gold_boundaries)
		false_negative = len(gold_boundaries - predicted_boundaries)
	else:
		# A surface mismatch means offsets are not comparable. Count all boundaries as errors.
		correct = 0
		false_positive = len(predicted_boundaries)
		false_negative = len(gold_boundaries)
	precision = correct / (correct + false_positive) if (correct + false_positive) else (1.0 if not gold_boundaries else 0.0)
	recall = correct / (correct + false_negative) if (correct + false_negative) else 1.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
	return {
		"surface_match": surface_match,
		"exact": surface_match and predicted_boundaries == gold_boundaries,
		"gold_boundaries": len(gold_boundaries),
		"predicted_boundaries": len(predicted_boundaries),
		"correct_boundaries": correct,
		"false_positive_boundaries": false_positive,
		"false_negative_boundaries": false_negative,
		"boundary_precision": precision,
		"boundary_recall": recall,
		"boundary_f1": f1,
	}


def job_segmentation_metrics(rows):
	totals = {
		"sentences_scored": 0,
		"exact_matches": 0,
		"surface_mismatches": 0,
		"gold_boundaries": 0,
		"predicted_boundaries": 0,
		"correct_boundaries": 0,
		"false_positive_boundaries": 0,
		"false_negative_boundaries": 0,
	}
	for row in rows:
		if not row.get("gold_segmented"):
			continue
		metrics = segmentation_metrics(row.get("ai_segmented"), row.get("gold_segmented"))
		totals["sentences_scored"] += 1
		if metrics["exact"]:
			totals["exact_matches"] += 1
		if not metrics["surface_match"]:
			totals["surface_mismatches"] += 1
		for key in ("gold_boundaries", "predicted_boundaries", "correct_boundaries", "false_positive_boundaries", "false_negative_boundaries"):
			totals[key] += metrics[key]
	precision_denominator = totals["correct_boundaries"] + totals["false_positive_boundaries"]
	recall_denominator = totals["correct_boundaries"] + totals["false_negative_boundaries"]
	totals["exact_accuracy"] = totals["exact_matches"] / totals["sentences_scored"] if totals["sentences_scored"] else None
	totals["boundary_precision"] = totals["correct_boundaries"] / precision_denominator if precision_denominator else None
	totals["boundary_recall"] = totals["correct_boundaries"] / recall_denominator if recall_denominator else None
	precision = totals["boundary_precision"]
	recall = totals["boundary_recall"]
	totals["boundary_f1"] = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and (precision + recall) else None
	return totals


def job_annotation_metrics(rows):
	totals = {
		"sentences_scored": 0,
		"linguistic_exact_matches": 0,
		"gold_ids": 0,
		"predicted_ids": 0,
		"matches": 0,
		"substitutions": 0,
		"insertions": 0,
		"deletions": 0,
		"edits": 0,
	}
	confusions = Counter()
	for row in rows:
		if not row.get("gold_annotation"):
			continue
		metrics = annotation_metrics(row.get("ai_annotation"), row.get("gold_annotation"))
		totals["sentences_scored"] += 1
		if metrics["linguistic_exact"]:
			totals["linguistic_exact_matches"] += 1
		for key in ("gold_ids", "predicted_ids", "matches", "substitutions", "insertions", "deletions", "edits"):
			totals[key] += metrics[key]
		confusions.update(metrics["confusions"])
	denominator = max(totals["gold_ids"], totals["predicted_ids"])
	totals["linguistic_exact_accuracy"] = totals["linguistic_exact_matches"] / totals["sentences_scored"] if totals["sentences_scored"] else None
	totals["id_match_rate"] = totals["matches"] / denominator if denominator else None
	totals["id_error_rate"] = totals["edits"] / totals["gold_ids"] if totals["gold_ids"] else None
	totals["confusions"] = [
		{"gold": gold, "predicted": predicted, "count": count}
		for (gold, predicted), count in confusions.most_common()
	]
	return totals
