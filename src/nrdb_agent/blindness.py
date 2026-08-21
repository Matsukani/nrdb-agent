BLIND_POLICIES = {"row", "cohort"}


def normalize_blind_policy(value):
	policy = str(value or "row").strip().lower()
	if policy not in BLIND_POLICIES:
		raise ValueError("blind policy must be one of: {}".format(", ".join(sorted(BLIND_POLICIES))))
	return policy


def cohort_sentence_ranges(rows):
	by_dataset = {}
	for row in rows or []:
		dataset_id = int(row.get("dataset_id") or 0)
		sentence_id = int(row.get("sentence_id") or 0)
		if dataset_id > 0 and sentence_id > 0:
			by_dataset.setdefault(dataset_id, set()).add(sentence_id)
	ranges = []
	for dataset_id in sorted(by_dataset):
		start = None
		end = None
		for sentence_id in sorted(by_dataset[dataset_id]):
			if start is None:
				start = end = sentence_id
			elif sentence_id == end + 1:
				end = sentence_id
			else:
				ranges.append((dataset_id, start, end))
				start = end = sentence_id
		if start is not None:
			ranges.append((dataset_id, start, end))
	return ranges


def normalize_evidence_scope(datasets=None, texts=None, sentence_ranges=None, auto_text=None,
	blind_policy="row", cohort_rows=None):
	policy = normalize_blind_policy(blind_policy)
	dataset_values = sorted({int(value) for value in (datasets or []) if int(value) > 0})
	text_values = {(int(dataset_id), int(text_id)) for dataset_id, text_id in (texts or []) if int(dataset_id) > 0 and int(text_id) > 0}
	if auto_text is not None:
		text_values.add((int(auto_text[0]), int(auto_text[1])))
	raw_ranges = list(sentence_ranges or [])
	if policy == "cohort":
		raw_ranges.extend(cohort_sentence_ranges(cohort_rows))
	range_values = {
		(int(dataset_id), int(start), int(end))
		for dataset_id, start, end in raw_ranges
		if int(dataset_id) > 0 and int(start) > 0 and int(end) >= int(start)
	}
	return {
		"datasets": dataset_values,
		"texts": [list(value) for value in sorted(text_values)],
		"sentence_ranges": [list(value) for value in sorted(range_values)],
	}
