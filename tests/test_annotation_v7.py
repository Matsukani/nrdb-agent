import pytest

from nrdb_agent.annotator_v7 import AnnotationAgentV7, V7_TRANSLATION_INSTRUCTIONS


class FakeNrdb:
	def __init__(self):
		self.construction_calls = []

	def lookup_id(self, label, schema_id):
		data = {
			"river_like_id": {"meaning_jp": "井戸"},
			"house_id": {"meaning_jp": "家"},
		}
		entry = data.get(label)
		return {
			"label": label,
			"lexical_entries": [] if entry is None else [{
				"form1": label,
				"form2": None,
				"meaning_jp": entry["meaning_jp"],
				"pos": "n",
				"dialect_name": "test",
			}],
			"local": None,
			"global": None,
		}

	def construction_candidates(self, annotation, schema_id, region=None, dialect_id=None):
		self.construction_calls.append((annotation, schema_id, region, dialect_id))
		return {
			"candidates": [{
				"id": 1,
				"name": "completive_irr_neg",
				"trigger_id": "irr",
				"pattern": "V;cvb irr-neg",
				"meaning_jp": "Vてしまった",
				"realization_jp": "Vてしまった",
				"priority": 100,
			}],
		}


def test_v7_translation_treats_id_spelling_as_opaque():
	assert "annotation labels are IDENTIFIERS, not glosses" in V7_TRANSLATION_INSTRUCTIONS
	assert "Dictionary meaning_jp/explanation_jp outrank" in V7_TRANSLATION_INSTRUCTIONS


def test_v7_translation_instructions_prioritize_matching_curated_constructions():
	assert "trigger hit alone does NOT prove" in V7_TRANSLATION_INSTRUCTIONS
	assert "interpret the WHOLE construction" in V7_TRANSLATION_INSTRUCTIONS
	assert "outranks a conflicting default atom-by-atom interpretation" in V7_TRANSLATION_INSTRUCTIONS


def test_v7_batch_grounding_uses_dictionary_meanings():
	agent = AnnotationAgentV7(FakeNrdb(), "test-model", client=object())
	result = agent._ground_lexical_ids(["river_like_id", "house_id", "unknown_id"], 2)
	assert len(result["labels"]) == 3
	assert result["labels"][0]["grounded"] is True
	assert result["labels"][0]["lexical_entries"][0]["meaning_jp"] == "井戸"
	assert result["labels"][1]["lexical_entries"][0]["meaning_jp"] == "家"
	assert result["labels"][2]["grounded"] is False


def test_v7_construction_lookup_is_strictly_opt_in():
	nrdb = FakeNrdb()
	agent = AnnotationAgentV7(nrdb, "test-model", client=object())
	item = {"dialect_region": "宮古", "dialect_id": 22}
	result = {"annotation": "剥mv;cvb irr-neg"}
	assert agent._construction_candidates(item, {"annotation_schema_id": 2, "use_constructions": False}, result) == []
	assert nrdb.construction_calls == []


def test_v7_construction_lookup_uses_frozen_annotation_and_scope():
	nrdb = FakeNrdb()
	agent = AnnotationAgentV7(nrdb, "test-model", client=object())
	item = {"dialect_region": "宮古", "dialect_id": 22}
	result = {"annotation": "剥mv;cvb irr-neg"}
	candidates = agent._construction_candidates(item, {"annotation_schema_id": 2, "use_constructions": True}, result)
	assert nrdb.construction_calls == [("剥mv;cvb irr-neg", 2, "宮古", 22)]
	assert candidates[0]["name"] == "completive_irr_neg"


def test_v7_finalization_requires_dictionary_grounding():
	agent = AnnotationAgentV7(FakeNrdb(), "test-model", client=object())
	assert agent._has_dictionary_grounding([]) is False
	assert agent._has_dictionary_grounding([{"tool": "corpus_examples"}]) is False
	assert agent._has_dictionary_grounding([{"tool": "ground_lexical_ids"}]) is True
	with pytest.raises(RuntimeError, match="cannot finalize without dictionary grounding"):
		agent._finalize_translation_v7([], [], "test")
