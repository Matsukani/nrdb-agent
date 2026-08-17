from nrdb_agent.metrics import annotation_ids, annotation_metrics


def test_r_control_is_ignored_in_linguistic_id_metrics():
	assert annotation_ids("A-r B;r C") == ["A", "B", "C"]
	metrics = annotation_metrics("A B C", "A-r B;r C")
	assert metrics["matches"] == 3
	assert metrics["edits"] == 0
	assert metrics["id_match_rate"] == 1.0
