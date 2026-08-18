import csv
import json

from nrdb_agent.asr_review import compact_from_units, review_asr_predictions, select_baseline_candidate


def test_compact_from_asr_units_expands_geminate_class_from_next_onset():
	assert compact_from_units(["u", "Q_stop", "t", "a"]) == "utta"
	assert compact_from_units(["a", "Q_fricative_s", "ʃ", "i"]) == "aʃʃi"


def test_deterministic_baseline_prefers_fewer_strong_linguistic_alerts():
	candidates = [
		{
			"rank": 1, "morph_confidence": 0.99,
			"id_review": {"strong_surprises": 1, "mean_log_probability": -1.0},
			"surface_review": {"strong_disagreements": 1, "phonotactic_mean_log_probability": -1.0},
		},
		{
			"rank": 2, "morph_confidence": 0.80,
			"id_review": {"strong_surprises": 0, "mean_log_probability": -1.5},
			"surface_review": {"strong_disagreements": 0, "phonotactic_mean_log_probability": -1.5},
		},
	]
	assert select_baseline_candidate(candidates)["rank"] == 2


class FakeNrdb:
	def morph_analyze(self, text, dialect_id, annotation_schema_id):
		if text == "ab":
			return {"segmented": "a-b", "annotation": "A-B", "confidence": 0.5}
		return {"segmented": "a-c", "annotation": "A-C", "confidence": 0.9}


def test_no_llm_asr_review_scores_selection_privately_after_reranking(tmp_path):
	predictions = tmp_path / "predictions.tsv"
	hypotheses = [
		{"rank": 1, "units": ["a", "b"], "combined_score": -1.0, "score_delta": 0.0},
		{"rank": 2, "units": ["a", "c"], "combined_score": -1.2, "score_delta": -0.2},
	]
	with predictions.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=["row_id", "ref_units", "nbest_json"], delimiter="\t")
		writer.writeheader()
		writer.writerow({"row_id": "u1", "ref_units": "a c", "nbest_json": json.dumps(hypotheses)})

	summary = review_asr_predictions(
		FakeNrdb(), predictions, tmp_path / "review", 2, "宮古", 19,
		use_llm=False,
	)
	assert summary["rows_scored"] == 1
	assert summary["top1_UER"] == 0.5
	assert summary["baseline_UER"] == 0.0
	assert summary["agent_UER"] == 0.0
	assert summary["oracle_UER"] == 0.0
	assert summary["agent_headroom_recovered"] == 1.0
	assert summary["improved_rows"] == 1
	assert (tmp_path / "review" / "asr_review.tsv").exists()
	assert (tmp_path / "review" / "summary.json").exists()
