import json

from nrdb_agent.morph_eval import _agreement_calibration, _paired_summary, _run_contract


def test_run_contract_uses_dataset_and_text_identity(tmp_path):
	run = tmp_path / "run"
	run.mkdir()
	rows = [
		{"dataset_id": 30, "text_id": "642"},
		{"dataset_id": 30, "text_id": "643"},
		{"dataset_id": 21, "text_id": "12781"},
	]
	with (run / "train.jsonl").open("w", encoding="utf-8") as handle:
		for row in rows:
			handle.write(json.dumps(row) + "\n")
	contract = _run_contract(run)
	assert contract["dataset_ids"] == [21, 30]
	assert contract["train_identities"] == {(30, "642"), (30, "643"), (21, "12781")}
	assert contract["train_rows"] == 3
	assert contract["train_rows_missing_identity"] == 0


def test_paired_summary_counts_ceiling_break_and_damage():
	rows = [
		{
			"gold_segmented": "a-b", "gold_annotation": "A-B",
			"baseline_segmented": "ab", "baseline_annotation": "A-C",
			"agent_segmented": "a-b", "agent_annotation": "A-B",
			"baseline_id_exact": 0, "agent_id_exact": 1,
			"baseline_id_edits": 1, "agent_id_edits": 0,
			"baseline_seg_exact": 0, "agent_seg_exact": 1,
			"baseline_agent_id_agree": 0, "baseline_agent_seg_agree": 0,
			"baseline_agent_full_agree": 0,
		},
		{
			"gold_segmented": "c-d", "gold_annotation": "C-D",
			"baseline_segmented": "c-d", "baseline_annotation": "C-D",
			"agent_segmented": "cd", "agent_annotation": "C-E",
			"baseline_id_exact": 1, "agent_id_exact": 0,
			"baseline_id_edits": 0, "agent_id_edits": 1,
			"baseline_seg_exact": 1, "agent_seg_exact": 0,
			"baseline_agent_id_agree": 0, "baseline_agent_seg_agree": 0,
			"baseline_agent_full_agree": 0,
		},
	]
	summary = _paired_summary(rows)
	assert summary["paired"]["baseline_id_errors_corrected"] == 1
	assert summary["paired"]["baseline_id_error_recovery_rate"] == 1.0
	assert summary["paired"]["baseline_id_correct_damaged"] == 1
	assert summary["paired"]["baseline_id_damage_rate"] == 1.0
	assert summary["paired"]["rows_with_fewer_id_edits"] == 1
	assert summary["paired"]["rows_with_more_id_edits"] == 1
	assert summary["paired"]["baseline_seg_errors_corrected"] == 1
	assert summary["paired"]["baseline_seg_correct_damaged"] == 1


def test_agreement_calibration_reports_coverage_and_gold_precision():
	rows = [
		{
			"gold_segmented": "a-b", "gold_annotation": "A-B",
			"baseline_segmented": "a-b", "baseline_annotation": "A-B",
			"agent_segmented": "a-b", "agent_annotation": "A-B",
			"baseline_id_exact": 1, "agent_id_exact": 1,
			"baseline_seg_exact": 1, "agent_seg_exact": 1,
			"baseline_agent_id_agree": 1, "baseline_agent_seg_agree": 1,
			"baseline_agent_full_agree": 1,
		},
		{
			"gold_segmented": "c-d", "gold_annotation": "C-D",
			"baseline_segmented": "c-d", "baseline_annotation": "C-D",
			"agent_segmented": "c-d", "agent_annotation": "C-D",
			"baseline_id_exact": 1, "agent_id_exact": 1,
			"baseline_seg_exact": 1, "agent_seg_exact": 1,
			"baseline_agent_id_agree": 1, "baseline_agent_seg_agree": 1,
			"baseline_agent_full_agree": 1,
		},
		{
			"gold_segmented": "e-f", "gold_annotation": "E-F",
			"baseline_segmented": "e-f", "baseline_annotation": "E-X",
			"agent_segmented": "ef", "agent_annotation": "E-F",
			"baseline_id_exact": 0, "agent_id_exact": 1,
			"baseline_seg_exact": 1, "agent_seg_exact": 0,
			"baseline_agent_id_agree": 0, "baseline_agent_seg_agree": 0,
			"baseline_agent_full_agree": 0,
		},
	]
	calibration = _agreement_calibration(rows)
	id_agree = calibration["id"]["agreement"]
	id_disagree = calibration["id"]["disagreement"]
	assert id_agree["rows"] == 2
	assert id_agree["coverage"] == 2 / 3
	assert id_agree["baseline"]["id_exact_accuracy"] == 1.0
	assert id_agree["agent"]["id_exact_accuracy"] == 1.0
	assert id_disagree["rows"] == 1
	assert id_disagree["coverage"] == 1 / 3
	assert id_disagree["baseline"]["id_exact_accuracy"] == 0.0
	assert id_disagree["agent"]["id_exact_accuracy"] == 1.0
	full_agree = calibration["full_analysis"]["agreement"]
	assert full_agree["coverage"] == 2 / 3
	assert full_agree["baseline"]["full_analysis_exact_accuracy"] == 1.0
