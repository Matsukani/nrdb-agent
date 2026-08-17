from nrdb_agent.metrics import align_ids, annotation_ids, job_annotation_metrics


def test_annotation_ids_flattens_phrase_segment_and_conflation_boundaries():
	assert annotation_ids("A;cvb-B C-dat") == ["A", "cvb", "B", "C", "dat"]


def test_align_ids_counts_substitution_insertion_and_deletion():
	metrics = align_ids(["A", "X", "C", "D"], ["A", "B", "C", "E"])
	assert metrics["matches"] == 2
	assert metrics["substitutions"] == 2
	assert metrics["insertions"] == 0
	assert metrics["deletions"] == 0
	assert metrics["id_match_rate"] == 0.5


def test_job_metrics_aggregate_only_gold_rows():
	metrics = job_annotation_metrics([
		{"ai_annotation": "A-B-C", "gold_annotation": "A-B-C"},
		{"ai_annotation": "A-X-C", "gold_annotation": "A-B-C"},
		{"ai_annotation": "A", "gold_annotation": None},
	])
	assert metrics["sentences_scored"] == 2
	assert metrics["gold_ids"] == 6
	assert metrics["matches"] == 5
	assert metrics["substitutions"] == 1
	assert metrics["id_match_rate"] == 5 / 6
