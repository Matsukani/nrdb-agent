from nrdb_agent.annotator_v9 import AnnotationAgentV9
from nrdb_agent.reverse_id_critic import IdSequenceCritic


class FakeIdModel:
	def score(self, annotation, annotation_schema_id):
		return {
			"segment": {"mean_log_probability": -1.0, "positions": []},
			"atom": {"mean_log_probability": -1.0, "positions": []},
			"strong_surprises": [{
				"representation": "atom",
				"index": 2,
				"threshold": -4.0,
				"token": "dat",
				"context": ["食kv"],
				"order": 2,
				"log_probability": -5.0,
				"alternatives": [["prp", 0.8]],
			}],
			"strong_surprise_count": 1,
			"combined_mean_log_probability": -1.0,
		}


class FakeNrdb:
	def __init__(self):
		self.form_calls = []
		self.lookup_calls = []

	def form_id_support(self, surface, candidate, region, schema_id, exclude_sentence_id=0):
		self.form_calls.append((surface, candidate, region, schema_id, exclude_sentence_id))
		return {
			"surface": surface, "candidate_id": candidate, "region": region,
			"combined": {"surface_total": 10, "candidate_count": 8, "candidate_rate": 0.8, "penalty": "none"},
			"corpus": {"surface_total": 10, "candidate_count": 8, "alternatives": []},
			"lexicon": {"surface_total": 0, "candidate_count": 0, "alternatives": []},
		}

	def lookup_id(self, label, schema_id):
		self.lookup_calls.append((label, schema_id))
		return {"label": label, "lexical_entries": [{"form1": "x", "meaning_jp": "X"}], "local": None, "global": None}


def _agent():
	agent = AnnotationAgentV9.__new__(AnnotationAgentV9)
	agent.id_critic = IdSequenceCritic(model=FakeIdModel())
	agent.id_model_path = "id.json"
	agent._shared_evidence = {"lookup": {}, "corpus": {}, "form": {}}
	return agent


def test_forward_v9_builds_id_and_surface_hotspots():
	agent = _agent()
	morph = {
		"annotation": "蕎麦sn-acc 食kv-dat 行iv-pst",
		"phrases": [{"segments": [
			{"surface": "soba", "label": "蕎麦sn", "confidence": 0.99, "alternatives": []},
			{"surface": "fo", "label": "食kv", "confidence": 0.72, "alternatives": [{"label": "食kv"}, {"label": "別kv"}]},
		]}],
	}
	context = agent._prepare_hotspots(morph, 2)
	assert "dat" in context["hotspot_ids"]
	assert "食kv" in context["hotspot_ids"]
	assert "fo" in context["uncertain_surfaces"]
	assert "soba" not in context["uncertain_surfaces"]


def test_forward_v9_missing_confidence_and_candidate_list_are_not_automatically_uncertain():
	agent = _agent()
	# Candidate existence/support is not a calibrated probability margin.
	morph = {
		"annotation": "東an-top:1 何処in-dat 有av-int:ba",
		"phrases": [{"segments": [
			{"surface": "aga", "label": "東an", "alternatives": [{"label": "東an", "support": 5}, {"label": "上av", "support": 4}]},
			{"surface": "ndza", "label": "何処in", "alternatives": [{"label": "何処in", "support": 60}]},
			{"surface": "arj", "label": "有av", "alternatives": [{"label": "有av", "support": 34}, {"label": "別av", "support": 12}]},
		]}],
	}
	context = agent._prepare_hotspots(morph, 2)
	assert "aga" not in context["uncertain_surfaces"]
	assert "ndza" not in context["uncertain_surfaces"]
	assert "arj" not in context["uncertain_surfaces"]


def test_forward_v9_suppresses_routine_queries_but_allows_hotspots():
	agent = _agent()
	agent._forward_hotspots = {
		"hotspot_ids": ["食kv", "dat"],
		"uncertain_surfaces": ["fo"],
		"surface_labels": {"soba": ["蕎麦sn"], "fo": ["食kv"]},
	}
	allowed, _ = agent._query_is_hotspot("corpus_examples", {"label": "食kv-dat 行iv"})
	assert allowed is True
	allowed, reason = agent._query_is_hotspot("corpus_examples", {"label": "蕎麦sn-acc"})
	assert allowed is False
	assert "routine grammar" in reason
	allowed, _ = agent._query_is_hotspot("form_id_support_batch", {"items": [{"surface": "fo", "candidate_ids": ["食kv", "別kv"]}]})
	assert allowed is True
	allowed, _ = agent._query_is_hotspot("form_id_support_batch", {"items": [{"surface": "soba", "candidate_ids": ["蕎麦sn"]}]})
	assert allowed is False
	allowed, _ = agent._query_is_hotspot("form_id_support_batch", {"items": [{"surface": "soba", "candidate_ids": ["別sn"]}]})
	assert allowed is True


def test_forward_v9_batches_multiple_form_pairs_into_one_tool_result():
	agent = _agent()
	agent.nrdb = FakeNrdb()
	agent._forward_hotspots = {
		"hotspot_ids": [],
		"uncertain_surfaces": ["aga", "ga"],
		"surface_labels": {"aga": ["東an"], "ga": ["itf:ga"]},
	}
	result = agent._tool_result_v9(
		"form_id_support_batch",
		{"items": [
			{"surface": "aga", "candidate_ids": ["東an", "上av"]},
			{"surface": "ga", "candidate_ids": ["itf:ga", "int:ga"]},
		]},
		{"sentence_id": 77, "dialect_region": "宮古"}, 2,
	)
	assert len(result["items"]) == 2
	assert len(agent.nrdb.form_calls) == 4
	assert all(call[-1] == 77 for call in agent.nrdb.form_calls)
	assert set(agent._shared_evidence["form"]) == {"aga\t東an", "aga\t上av", "ga\titf:ga", "ga\tint:ga"}


def test_forward_v9_reuses_cached_lexical_grounding():
	agent = _agent()
	agent.nrdb = FakeNrdb()
	first = agent._ground_lexical_ids(["東an", "何処in"], 2)
	second = agent._ground_lexical_ids(["東an", "何処in"], 2)
	assert len(first["labels"]) == 2
	assert len(second["labels"]) == 2
	assert agent.nrdb.lookup_calls == [("東an", 2), ("何処in", 2)]
	assert agent._review_cache_hit("ground_lexical_ids", {"labels": ["東an", "何処in"]}) is True
