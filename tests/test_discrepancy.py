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


class MissingGoldMorphNrdb:
	def morph_eval_rows(self, dataset_ids, annotation_schema_id=None, region=None):
		return [{
			"sentence_id": 10, "dataset_id": 30, "example_id": "missing-seg", "dialect_id": 19,
			"dialect_region": "宮古", "annotation_schema_id": 2, "text": "raw",
			"gold_segmented": "", "gold_annotation": "A-adv", "translation_jp": "金訳",
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
	assert result["selection"]["baseline_use_constructions"] is False
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


def test_create_discovery_gold_morph_requires_segmentation_and_annotation(tmp_path):
	with pytest.raises(ValueError, match="no translated gold rows"):
		discrepancy.create_discovery(
			MissingGoldMorphNrdb(), ["adv"], None, 2, "宮古", require_gold_morph=True,
			output=tmp_path / "cohort.json",
		)


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
	assert baseline["models"] == {"translation": "gpt-5.6-luna", "discrepancy": "gpt-5.6-sol", "morphology": "predicted", "id_critic": "nrdb_agent_default", "constructions": "disabled"}
	assert calls == [("gpt-5.6-luna", False, None)]
	assert baseline["summary"]["morphemes_to_analyse"][0]["morph_id"] == "adv"
	assert baseline["summary"]["morphemes_to_analyse"][0]["candidate_patterns"] == [{"pattern": "V-neg-adv", "count": 1}]

	checked = discrepancy.check_discovery(
		FakeNrdb(), baseline_path, check_path,
		discrepancy_model="gpt-5.6-terra",
	)
	assert checked["models"]["translation"] == "gpt-5.6-luna"
	assert checked["models"]["discrepancy"] == "gpt-5.6-terra"
	assert checked["models"]["constructions"] == "enabled_current"
	assert checked["baseline_models"]["constructions"] == "disabled"
	assert calls[-1] == ("gpt-5.6-luna", True, None)
	assert checked["summary"]["counts"]["repaired"] == 1


def test_run_uses_gold_morph_and_check_inherits_it(tmp_path, monkeypatch):
	cohort_path = tmp_path / "cohort.json"
	baseline_path = tmp_path / "baseline.json"
	check_path = tmp_path / "check.json"
	discrepancy.create_discovery(FakeNrdb(), ["adv"], [30], 2, "宮古", limit=1, require_gold_morph=True, output=cohort_path)
	calls = []

	def fake_translate(nrdb, text, target, schema, region, **kwargs):
		calls.append(kwargs)
		return {"annotation": kwargs["fixed_annotation"], "translation": "生成訳", "api_usage": {}}

	monkeypatch.setattr(discrepancy, "translate_text", fake_translate)
	monkeypatch.setattr(discrepancy, "DiscrepancyJudge", FakeJudge)
	baseline = discrepancy.run_discovery(FakeNrdb(), cohort_path, baseline_path, use_gold_morph=True)
	assert baseline["models"]["morphology"] == "gold"
	assert baseline["models"]["id_critic"] == "not_used"
	assert calls[0]["fixed_segmented"] == "a-b c"
	assert calls[0]["fixed_annotation"] == "A-adv C"
	discrepancy.check_discovery(FakeNrdb(), baseline_path, check_path)
	assert calls[1]["use_constructions"] is True
	assert calls[1]["fixed_segmented"] == "a-b c"
	assert calls[1]["fixed_annotation"] == "A-adv C"


def test_run_rejects_gold_morph_for_unrestricted_cohort(tmp_path):
	cohort_path = tmp_path / "cohort.json"
	discrepancy.create_discovery(FakeNrdb(), ["adv"], [30], 2, "宮古", limit=1, output=cohort_path)
	with pytest.raises(ValueError, match="created with discrepancy-create --gold-morph"):
		discrepancy.run_discovery(FakeNrdb(), cohort_path, tmp_path / "baseline.json", use_gold_morph=True)


def test_run_uses_declared_current_constructions(tmp_path, monkeypatch):
	cohort_path = tmp_path / "cohort.json"
	discrepancy.create_discovery(
		FakeNrdb(), ["adv"], [30], 2, "宮古", limit=1,
		use_constructions=True, output=cohort_path,
	)
	calls = []

	def fake_translate(nrdb, text, target, schema, region, **kwargs):
		calls.append(kwargs)
		return {"annotation": "A-adv C", "translation": "生成訳", "api_usage": {}}

	monkeypatch.setattr(discrepancy, "translate_text", fake_translate)
	monkeypatch.setattr(discrepancy, "DiscrepancyJudge", FakeJudge)
	baseline = discrepancy.run_discovery(
		FakeNrdb(), cohort_path, tmp_path / "baseline.json", use_constructions=True,
	)
	assert calls[0]["use_constructions"] is True
	assert baseline["models"]["constructions"] == "enabled_current"


@pytest.mark.parametrize("created_with,run_with", [(True, False), (False, True)])
def test_run_rejects_construction_condition_mismatch(tmp_path, created_with, run_with):
	cohort_path = tmp_path / "cohort.json"
	discrepancy.create_discovery(
		FakeNrdb(), ["adv"], [30], 2, "宮古", limit=1,
		use_constructions=created_with, output=cohort_path,
	)
	with pytest.raises(ValueError, match="options must match"):
		discrepancy.run_discovery(
			FakeNrdb(), cohort_path, tmp_path / "baseline.json", use_constructions=run_with,
		)


def test_check_rejects_translation_model_change(tmp_path):
	path = tmp_path / "baseline.json"
	path.write_text(json.dumps({
		"format": discrepancy.BASELINE_FORMAT, "selection": {},
		"models": {"translation": "gpt-5.6-luna"}, "rows": [],
	}), encoding="utf-8")
	with pytest.raises(ValueError, match="must use the baseline translation model"):
		discrepancy.check_discovery(FakeNrdb(), path, tmp_path / "out.json", translation_model="gpt-5.6-terra")


def test_list_discoveries_summarizes_recognized_local_artifacts(tmp_path):
	cohort = tmp_path / "discovery.json"
	baseline = tmp_path / "baseline.json"
	(tmp_path / "unrelated.json").write_text('{"hello":"world"}', encoding="utf-8")
	discrepancy.create_discovery(FakeNrdb(), ["adv"], None, 2, "宮古", limit=1, output=cohort)
	baseline.write_text(json.dumps({
		"format": discrepancy.BASELINE_FORMAT,
		"selection": {"target_ids": ["adv"], "annotation_schema_id": 2, "region": "宮古", "dataset_ids": [], "sampled_rows_by_id": {"adv": 1}},
		"models": {"translation": "gpt-5.6-luna", "discrepancy": "gpt-5.6-terra"},
		"rows": [{"sentence_id": 1}],
		"summary": {"failed": 0, "counts": {"equivalent": 1}, "estimated_cost_usd": 0.01, "pricing_complete": True},
	}), encoding="utf-8")
	rows = discrepancy.list_discoveries(tmp_path)
	assert {row["stage"] for row in rows} == {"discovery", "baseline"}
	baseline_row = next(row for row in rows if row["stage"] == "baseline")
	assert baseline_row["status"] == "completed"
	assert baseline_row["translation_model"] == "gpt-5.6-luna"
	assert baseline_row["morphology"] == "predicted"
	assert baseline_row["constructions"] == "disabled"
	assert baseline_row["counts"] == {"equivalent": 1}
	assert len(discrepancy.list_discoveries(tmp_path, latest=1)) == 1


def test_list_discoveries_rejects_missing_directory(tmp_path):
	with pytest.raises(ValueError, match="does not exist"):
		discrepancy.list_discoveries(tmp_path / "missing")
