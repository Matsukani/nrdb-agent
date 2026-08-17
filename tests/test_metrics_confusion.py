from nrdb_agent.metrics import annotation_metrics, job_annotation_metrics


def test_confusion_matrix_tracks_substitution_and_missing_id():
	metrics = annotation_metrics("A-B-D", "A-C-D-E")
	assert metrics["matches"] == 2
	assert metrics["substitutions"] == 1
	assert metrics["deletions"] == 1
	assert ("C", "B") in metrics["confusions"]
	assert ("E", "<missing>") in metrics["confusions"]


def test_job_confusions_are_counted():
	metrics = job_annotation_metrics([
		{"ai_annotation": "A-X", "gold_annotation": "A-B"},
		{"ai_annotation": "A-X", "gold_annotation": "A-B"},
	])
	assert metrics["confusions"][0] == {"gold": "B", "predicted": "X", "count": 2}
