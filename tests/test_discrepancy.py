import json

import pytest

from nrdb_agent import discrepancy


class FakeNrdb:
	def morph_eval_rows(self, dataset_ids, annotation_schema_id=None, region=None):
		assert dataset_ids in ([], [30])
		assert annotation_schema_id == 2
		assert region == "宮古"
		return [
			{
				"sentence_id": 1, "dataset_id": 30, "example_id": "a", "dialect_id": 19,
				"dialect_region": "宮古", "annotation_schema_id": 2, "text": "raw1",
				"gold_segmented": "a-b c", "gold_annotation": "A-adv C", "translation_jp": "金訳1",
			},
			{
				"sentence_id": 2, "dataset_id": 30, "example_id": "b", "dialect_id": 19,
				"dialect_region": "宮古", "annotation_schema_id": 2, "text": "raw2",
				"gold_segmented": "a-b", "gold_annotation": "A-foc", "translation_jp": "金訳2",
			},
			{
				"sentence_id": 3, "dataset_id": 30, "example_id": "c", "dialect_id": 19,
				"dialect_region": "宮古", "annotation_schema_id": 2, "text": "raw3",
				"gold_segmented": "a-b c", "gold_annotation": "A-adv C", "translation_jp": "",
			},
		]


class FakeJudge:
	def __init__(self, nrdb, model_name, client=None, progress=print):
		self.model_name = model_name

	def discrepancy(self, row, generated):
		return {
			"relation": "substantive_difference", "severity": 2,
			"target_morpheme_error": True, "likely_responsible_ids": ["adv"],
			"candidate_pattern": "V-neg-adv", "discrepancy_type": "clause_relation", "summary_jp": "逆接の誤り",
		}

	def repair(self, row, baseline, construction):
		return {"outcome": "repaired", "baseline_relation": "substantive_difference", "construction_relation": "equivalent"}


class OverlapNrdb:
	def morph_eval_rows(self, dataset_ids, annotation_schema_id=None, region=None):
		return [{
			"sentence_id": 9, "dataset_id": 30, "example_id": "overlap", "dialect_id": 19,
			"dialect_region": "宮古", "annotation_schema_id": 2, "text": "raw",
			"gold_segmented": "a-b-c", "gold_annotation": "A-adv-foc", "translation_jp": "金訳",
		}]


def test_create_discovery_filters_gold_translations_ids_and_morpheme_count(tmp_path):
	output = tmp_path / "cohort.json"
	result = discrepancy.create_discovery(
		FakeNrdb(), ["adv", "foc"], [30], 2, "宮古", limit=10,
		seed=4, min_morphemes=3, output=output,
	)
	assert [row["sentence_id"] for row in result["rows"]] == [1]
	assert result["rows"][0]["matched_target_ids"] == ["adv"]
	assert result["selection"]["limit_per_id"] == 10
	assert json.loads(output.read_text(encoding="utf-8"))["format"] == discrepancy.DISCOVERY_FORMAT


def test_create_discovery_pools_region_and_schema_when_datasets_are_omitted(tmp_path):
	result = discrepancy.create_discovery(
		FakeNrdb(), ["adv"], None, 2, "宮古", limit=10,
		output=tmp_path / "cohort.json",
	)
	assert result["selection"]["dataset_ids"] == []
	assert [row["sentence_id"] for row in result["rows"]] == [1]


def test_create_discovery_samples_each_target_independently_and_keeps_overlap_assignments(tmp_path):
	result = discrepancy.create_discovery(
		OverlapNrdb(), ["adv", "foc"], None, 2, "宮古", limit=1,
		output=tmp_path / "cohort.json",
	)
	assert len(result["rows"]) == 2
	assert [row["target_id"] for row in result["rows"]] == ["adv", "foc"]
	assert [row["matched_target_ids"] for row in result["rows"]] == [["adv"], ["foc"]]
	assert result["selection"]["eligible_pool_size_by_id"] == {"adv": 1, "foc": 1}
	assert result["selection"]["sampled_rows_by_id"] == {"adv": 1, "foc": 1}


def test_run_and_check_keep_translation_and_judge_models_separate(tmp_path, monkeypatch):
	cohort_path = tmp_path / "cohort.json"
	baseline_path = tmp_path / "baseline.json"
	check_path = tmp_path / "check.json"
	discrepancy.create_discovery(FakeNrdb(), ["adv"], [30], 2, "宮古", limit=1, output=cohort_path)
	calls = []

	def fake_translate(nrdb, text, target, schema, region, **kwargs):
		calls.append((kwargs["model_name"], kwargs["use_constructions"], kwargs.get("id_model")))
		return {"annotation": "A-adv C", "translation": "生成訳", "api_usage": {}}

	monkeypatch.setattr(discrepancy, "translate_text", fake_translate)
	monkeypatch.setattr(discrepancy, "DiscrepancyJudge", FakeJudge)
	baseline = discrepancy.run_discovery(
		FakeNrdb(), cohort_path, baseline_path,
		translation_model="gpt-5.6-luna", discrepancy_model="gpt-5.6-sol",
	)
	assert baseline["models"] == {"translation": "gpt-5.6-luna", "discrepancy": "gpt-5.6-sol", "id_critic": "nrdb_agent_default"}
	assert calls == [("gpt-5.6-luna", False, None)]
	assert baseline["summary"]["morphemes_to_analyse"][0]["morph_id"] == "adv"
	assert baseline["summary"]["morphemes_to_analyse"][0]["candidate_patterns"] == [{"pattern": "V-neg-adv", "count": 1}]

	checked = discrepancy.check_discovery(
		FakeNrdb(), baseline_path, check_path,
		discrepancy_model="gpt-5.6-terra",
	)
	assert checked["models"]["translation"] == "gpt-5.6-luna"
	assert checked["models"]["discrepancy"] == "gpt-5.6-terra"
	assert calls[-1] == ("gpt-5.6-luna", True, None)
	assert checked["summary"]["counts"]["repaired"] == 1


def test_check_rejects_translation_model_change(tmp_path):
	path = tmp_path / "baseline.json"
	path.write_text(json.dumps({
		"format": discrepancy.BASELINE_FORMAT, "selection": {},
		"models": {"translation": "gpt-5.6-luna"}, "rows": [],
	}), encoding="utf-8")
	with pytest.raises(ValueError, match="must use the baseline translation model"):
		discrepancy.check_discovery(FakeNrdb(), path, tmp_path / "out.json", translation_model="gpt-5.6-terra")
