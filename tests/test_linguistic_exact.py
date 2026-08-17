from nrdb_agent.metrics import annotation_metrics, job_annotation_metrics


def test_linguistic_exact_ignores_r_control_atom():
	metrics = annotation_metrics(
		"dm:2md l:雲舟nn-quot-top:1 知sv;cvb-foc ipf-dub",
		"dm:2md l:雲舟nn-quot-top:1 知sv;cvb-foc ipf-dub;r",
	)
	assert metrics["linguistic_exact"] is True
	assert metrics["edits"] == 0
	assert metrics["id_match_rate"] == 1.0


def test_job_linguistic_exact_counts_control_only_difference_as_exact():
	rows = [{
		"ai_annotation": "A B-cvb",
		"gold_annotation": "A B-cvb;r",
	}]
	metrics = job_annotation_metrics(rows)
	assert metrics["linguistic_exact_matches"] == 1
	assert metrics["linguistic_exact_accuracy"] == 1.0
