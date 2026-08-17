import re
from collections import Counter


ID_SPLIT_RE = re.compile(r"[\s\u3000\-;]+")
MISSING = "<missing>"
EXTRA = "<extra>"


def annotation_ids(annotation):
	"""Flatten an NRDB annotation into its ordered atomic ID labels."""
	return [value for value in ID_SPLIT_RE.split(str(annotation or "").strip()) if value]


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
		"confusions": confusions,
		"id_match_rate": counts["matches"] / denominator if denominator else 1.0,
		"id_error_rate": edits / gold_count if gold_count else (0.0 if predicted_count == 0 else 1.0),
	})
	return counts


def annotation_metrics(predicted_annotation, gold_annotation):
	return align_ids(annotation_ids(predicted_annotation), annotation_ids(gold_annotation))


def job_annotation_metrics(rows):
	totals = {
		"sentences_scored": 0,
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
		for key in ("gold_ids", "predicted_ids", "matches", "substitutions", "insertions", "deletions", "edits"):
			totals[key] += metrics[key]
		confusions.update(metrics["confusions"])
	denominator = max(totals["gold_ids"], totals["predicted_ids"])
	totals["id_match_rate"] = totals["matches"] / denominator if denominator else None
	totals["id_error_rate"] = totals["edits"] / totals["gold_ids"] if totals["gold_ids"] else None
	totals["confusions"] = [
		{"gold": gold, "predicted": predicted, "count": count}
		for (gold, predicted), count in confusions.most_common()
	]
	return totals
