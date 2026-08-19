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


def _agent():
	agent = AnnotationAgentV9.__new__(AnnotationAgentV9)
	agent.id_critic = IdSequenceCritic(model=FakeIdModel())
	agent.id_model_path = "id.json"
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
	allowed, _ = agent._query_is_hotspot("form_id_support", {"surface": "fo", "candidate_id": "食kv"})
	assert allowed is True
	allowed, _ = agent._query_is_hotspot("form_id_support", {"surface": "soba", "candidate_id": "蕎麦sn"})
	assert allowed is False
	allowed, _ = agent._query_is_hotspot("form_id_support", {"surface": "soba", "candidate_id": "別sn"})
	assert allowed is True
