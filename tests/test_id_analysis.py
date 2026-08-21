import csv
import io
import json
from types import SimpleNamespace

import pytest

from nrdb_agent.id_analysis import (
	licensed_candidate_tsv, normalize_analysis, result_tsv, run_combinatorial_probes,
	run_attested_probes, run_id_analysis_job,
)
from nrdb_agent.cli import _id_analysis_notes


def _evidence(target_id="advs", tokens=12):
	return {
		"target_id": target_id,
		"annotation_schema_id": 2,
		"scope": {"region": "宮古", "dialect_id": None, "source_kinds": ["txt", "sen"], "dataset_ids": []},
		"research_note": "逆接として検討する。",
		"summary": {"tokens": tokens, "sentences": 8, "datasets": 2, "dialects": 2},
		"local_id_metadata": {"local_id": target_id, "local_class": "p"},
		"existing_entries": [],
		"same_phrase_patterns": [{"phrase_annotation": "V-advs", "target_tokens": 8}],
		"anchored_phrase_ngrams": [{"token_1": "V", "token_2": target_id, "tokens": 8}],
		"examples": [{
			"example_key": "sen:7:21:31:41", "phrase_form2": "sii-ga", "phrase_form1": "",
			"sentence_trsc2": "sii-ga iks", "sentence_text": "",
			"phrase_annotation": "為sv-advs", "sentence_annotation": "為sv-advs 行iv",
			"translation_jp": "するが、行く。", "source_kind": "sen", "dataset_id": 7, "source_id": 21,
		}],
		"provider_pos_counts": [], "provider_lexemes": [],
		"method_notes": ["Translations are sentence-level evidence."],
	}


def _analysis_payload(target_id="advs"):
	return {
		"target_id": target_id,
		"linguistic_name_en": "adversative",
		"linguistic_name_jp": "逆接",
		"central_function_jp": "先行節と後続節の対立を表す。",
		"general_translation_policy_jp": "文脈に応じて「が」「けど」などで表す。",
		"candidates": [{
			"entry_type": "morpheme", "name": "adversative_advs", "trigger_id": target_id,
			"pattern": target_id, "meaning_jp": "逆接を表す。", "realization_jp": "「が」「けど」などで訳す。",
			"note": "述語に後続する。", "priority": 100, "confidence": 0.91,
			"example_references": [{"example_key": "sen:7:21:31:41", "mapping_note_jp": "逆接関係が明示される。"}],
		}],
		"combinatorial_patterns": [],
		"warnings": [],
	}


def test_normalize_analysis_materializes_only_attested_examples():
	payload = _analysis_payload()
	payload["candidates"][0]["example_references"].append({"example_key": "invented", "mapping_note_jp": "bad"})
	result = normalize_analysis(payload, _evidence())
	candidate = result["candidates"][0]
	assert candidate["trigger_id"] == "advs"
	assert candidate["pattern"] == "advs"
	assert candidate["name"] == "adversative_advs"
	assert [value["example_key"] for value in candidate["examples"]] == ["sen:7:21:31:41"]
	assert "sii-ga" in candidate["note"]
	assert "invented" in result["warnings"][0]


def test_morpheme_name_is_deterministic_english_key():
	payload = _analysis_payload("ppt>2")
	payload["linguistic_name_en"] = "Potential marker"
	payload["candidates"][0]["name"] = "可能標識"
	result = normalize_analysis(payload, _evidence("ppt>2"))
	assert result["candidates"][0]["name"] == "potential_marker_ppt_2"


def test_construction_name_is_normalized_and_scoped_to_target():
	payload = _analysis_payload()
	payload["candidates"].append({
		"entry_type": "construction", "name": "Counter-expectation Contrast", "trigger_id": "advs",
		"pattern": "V-advs foc", "meaning_jp": "反期待を表す。", "realization_jp": "「のに」などで訳す。",
		"note": "通常の合成より限定された用法。", "priority": 100, "confidence": 0.8,
		"example_references": [{"example_key": "sen:7:21:31:41", "mapping_note_jp": "反期待。"}],
	})
	result = normalize_analysis(payload, _evidence())
	assert result["candidates"][1]["name"] == "counter_expectation_contrast_advs"


def test_normalize_analysis_rejects_unknown_entry_type():
	payload = _analysis_payload()
	payload["candidates"][0]["entry_type"] = "guess"
	with pytest.raises(ValueError, match="invalid entry_type"):
		normalize_analysis(payload, _evidence())


def test_result_tsv_has_database_columns_and_disabled_candidates():
	analysis = normalize_analysis(_analysis_payload(), _evidence())
	analysis["evidence"] = _evidence()
	result = {
		"job": {"annotation_schema_id": 2, "region": "宮古", "dialect_id": None},
		"analyses": [analysis],
	}
	rows = list(csv.DictReader(io.StringIO(result_tsv(result)), delimiter="\t"))
	assert len(rows) == 1
	assert rows[0]["entry_type"] == "morpheme"
	assert rows[0]["enabled"] == "0"
	assert rows[0]["evidence_tokens"] == "12"
	assert json.loads(rows[0]["examples_json"])[0]["translation_jp"] == "するが、行く。"


class FakeResponses:
	def create(self, **kwargs):
		payload = json.loads(kwargs["input"][0]["content"])
		return SimpleNamespace(
			id="resp-{}".format(payload["target_id"]), status="completed", incomplete_details=None,
			output_text=json.dumps(_analysis_payload(payload["target_id"]), ensure_ascii=False), output=[], usage=None,
		)


class FakeClient:
	def __init__(self):
		self.responses = FakeResponses()


class FakeNrdb:
	def __init__(self):
		self.statuses = []
		self.saved = None

	def id_analysis_job(self, job_id):
		return {"job": {
			"id": job_id, "status": "queued", "annotation_schema_id": 2, "region": "宮古", "dialect_id": None,
			"target_ids": ["advs", "missing"], "source_kinds": ["txt", "sen"], "dataset_ids": [],
			"provider_dataset_ids": [], "minimum_ngram_count": 2, "example_limit": 30,
			"probe_limit": 24, "probe_seed": 1, "model_name": "gpt-test", "prompt_version": "id-analysis-v2",
		}}

	def set_id_analysis_status(self, job_id, status, error_message=None):
		self.statuses.append((job_id, status, error_message))

	def id_analysis_evidence(self, job_id, target_id):
		return {"evidence": _evidence(target_id, tokens=0 if target_id == "missing" else 12)}

	def save_id_analysis_result(self, job_id, result, tsv):
		self.saved = (job_id, result, tsv)


def test_run_job_handles_several_ids_and_skips_unattested_id():
	nrdb = FakeNrdb()
	bundle = run_id_analysis_job(nrdb, 9, openai_client=FakeClient(), progress=lambda _message: None)
	assert nrdb.statuses == [(9, "running", None)]
	assert nrdb.saved[0] == 9
	assert len(bundle["result"]["analyses"]) == 2
	assert bundle["result"]["analyses"][1]["candidates"] == []
	assert "adversative_advs" in bundle["tsv"]


def test_id_specific_and_shared_research_notes_are_parsed():
	assert _id_analysis_notes(["foc=May be omitted in Japanese."], ["foc", "advs"], "Compare translations.") == {
		"*": "Compare translations.", "foc": "May be omitted in Japanese.",
	}


def _productive_evidence():
	evidence = _evidence("fp:ra")
	evidence["same_phrase_patterns"] = [{"phrase_annotation": "N-fp:ra", "target_tokens": 30}]
	evidence["provider_pos_counts"] = [{"pos": "mn", "sampled_lexemes": 1}]
	evidence["provider_lexemes"] = [{
		"lexeme_id": 71, "dataset_id": 21, "dialect_id": 19, "region": "宮古", "pos": "mn",
		"form2": "munu", "form2_seg": "munu", "annotation": "物mn", "meaning_jp": "物",
	}]
	return evidence


def _productive_payload():
	payload = _analysis_payload("fp:ra")
	payload["combinatorial_patterns"] = [{
		"pattern_name_en": "nominal_final_particle", "target_position": "after", "join_type": "segment",
		"target_surface": "ra", "target_segmented": "ra", "host_pos": ["mn"],
		"host_morphological_scope_jp": "名詞に後接する。", "host_semantic_scope_jp": "特段の意味制限なし。",
		"productivity_evidence_jp": "複数の名詞語幹に付く。", "evidence_phrase_patterns": ["N-fp:ra"],
		"provider_lexeme_ids": [71], "confidence": 0.91,
	}]
	return payload


def test_combinatorial_patterns_are_grounded_in_exact_evidence_and_provider_rows():
	payload = _productive_payload()
	payload["combinatorial_patterns"][0]["evidence_phrase_patterns"].append("invented-fp:ra")
	payload["combinatorial_patterns"][0]["provider_lexeme_ids"].append(999)
	result = normalize_analysis(payload, _productive_evidence())
	pattern = result["combinatorial_patterns"][0]
	assert pattern["provider_lexeme_ids"] == [71]
	assert pattern["evidence_phrase_patterns"] == ["N-fp:ra"]
	assert any("unavailable provider lexeme 999" in value for value in result["warnings"])


class ProbeNrdb:
	def __init__(self, model_exact=False, licensed=False):
		self.model_exact = model_exact
		self.licensed = licensed

	def validate_analysis(self, text, segmented, annotation):
		return {"valid": True}

	def licensed_forms_in_text(self, text, annotation_schema_id, region, dialect_id, surfaces=None):
		matches = [{"id": 3, "form_romaji": "munura", "form_kana": ""}] if self.licensed else []
		return {"success": True, "matches": matches}

	def morph_analyze(self, text, dialect_id, annotation_schema_id):
		if self.model_exact:
			return {"segmented": "munu-ra", "annotation": "物mn-fp:ra"}
		return {"segmented": "munura", "annotation": "?"}


def test_productive_probe_emits_review_candidate_only_for_unlicensed_model_failure():
	evidence = _productive_evidence()
	analysis = normalize_analysis(_productive_payload(), evidence)
	job = {"annotation_schema_id": 2, "region": "宮古", "probe_limit": 24}
	probes, candidates = run_combinatorial_probes(ProbeNrdb(), job, analysis, evidence)
	assert probes[0]["failure_type"] == "model_failed_candidate"
	assert candidates[0]["form_romaji"] == "munura"
	assert candidates[0]["form_romaji_seg"] == "munu-ra"
	assert candidates[0]["annotation"] == "物mn-fp:ra"
	assert candidates[0]["license_status"] == "candidate"
	assert candidates[0]["generated_id"] == ""
	rows = list(csv.DictReader(io.StringIO(licensed_candidate_tsv({"analyses": [{"licensed_wordform_candidates": candidates}]})), delimiter="\t"))
	assert rows[0]["provider_lexeme_id"] == "71"


def test_productive_probe_does_not_duplicate_exact_or_already_licensed_forms():
	evidence = _productive_evidence()
	analysis = normalize_analysis(_productive_payload(), evidence)
	job = {"annotation_schema_id": 2, "region": "宮古", "probe_limit": 24}
	assert run_combinatorial_probes(ProbeNrdb(model_exact=True), job, analysis, evidence)[1] == []
	probes, candidates = run_combinatorial_probes(ProbeNrdb(licensed=True), job, analysis, evidence)
	assert probes[0]["failure_type"] == "model_failed_already_licensed"
	assert candidates == []


def test_attested_probe_uses_gold_phrase_surface_before_productive_generation():
	evidence = _productive_evidence()
	evidence["examples"][0].update({
		"example_key": "sen:21:1:2:3", "phrase_form2": "munu-ra",
		"phrase_annotation": "物mn-fp:ra", "dialect_id": 19, "region": "宮古", "dataset_id": 21,
	})
	analysis = normalize_analysis(_productive_payload(), evidence)
	probes, candidates = run_attested_probes(
		ProbeNrdb(), {"annotation_schema_id": 2, "region": "宮古"}, analysis, evidence, 4,
	)
	assert probes[0]["expected_text"] == "munura"
	assert candidates[0]["probe_source"] == "attested"
	assert candidates[0]["source_example_key"] == "sen:21:1:2:3"
