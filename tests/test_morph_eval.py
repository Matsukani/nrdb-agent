import json

from nrdb_agent.morph_eval import _paired_summary, _run_contract


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
		},
		{
			"gold_segmented": "c-d", "gold_annotation": "C-D",
			"baseline_segmented": "c-d", "baseline_annotation": "C-D",
			"agent_segmented": "cd", "agent_annotation": "C-E",
			"baseline_id_exact": 1, "agent_id_exact": 0,
			"baseline_id_edits": 0, "agent_id_edits": 1,
			"baseline_seg_exact": 1, "agent_seg_exact": 0,
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
