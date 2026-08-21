import csv
import io
import json
from types import SimpleNamespace

import pytest

from nrdb_agent.id_analysis import normalize_analysis, result_tsv, run_id_analysis_job
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
		"warnings": [],
	}


def test_normalize_analysis_materializes_only_attested_examples():
	payload = _analysis_payload()
	payload["candidates"][0]["example_references"].append({"example_key": "invented", "mapping_note_jp": "bad"})
	result = normalize_analysis(payload, _evidence())
	candidate = result["candidates"][0]
	assert candidate["trigger_id"] == "advs"
	assert candidate["pattern"] == "advs"
	assert [value["example_key"] for value in candidate["examples"]] == ["sen:7:21:31:41"]
	assert "sii-ga" in candidate["note"]
	assert "invented" in result["warnings"][0]


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
			"minimum_ngram_count": 2, "example_limit": 30, "model_name": "gpt-test", "prompt_version": "id-analysis-v1",
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
