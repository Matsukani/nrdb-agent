import json
from types import SimpleNamespace

from nrdb_agent.forward_critic_agent import ForwardCriticAnnotationAgent


class FakeNrdb:
	def validate_analysis(self, text, segmented, annotation):
		return {"valid": True}


class FakeIdCritic:
	def review(self, annotation, annotation_schema_id):
		if "cmpz>2" in annotation:
			return {
				"strong_surprise_count": 1,
				"strong_surprises": [{"representation": "atom", "token": "cmpz>2", "context": ["有av"], "alternatives": [["cmpz>1", 0.8]]}],
				"combined_mean_log_probability": -5.0,
				"segment": {}, "atom": {},
			}
		return {
			"strong_surprise_count": 0,
			"strong_surprises": [],
			"combined_mean_log_probability": -1.0,
			"segment": {}, "atom": {},
		}

	def compact(self, review):
		return {
			"strong_surprises": int(review.get("strong_surprise_count", 0)),
			"mean_log_probability": float(review.get("combined_mean_log_probability", 0.0)),
			"representations": {},
		}


class DiffuseIdCritic(FakeIdCritic):
	def review(self, annotation, annotation_schema_id):
		return {
			"strong_surprise_count": 7,
			"strong_surprises": [],
			"combined_mean_log_probability": -5.0,
			"segment": {}, "atom": {},
		}


class FakeSurfaceCritic:
	def review(self, segmented, annotation, dialect_ids, annotation_schema_id):
		if "cmpz>2" in annotation:
			return {"valid_alignment": True, "strong_disagreements": 1, "phonotactic_mean_log_probability": -2.0, "diagnostics": []}
		return {"valid_alignment": True, "strong_disagreements": 0, "phonotactic_mean_log_probability": -1.5, "diagnostics": []}


def _agent(candidate_annotation):
	agent = ForwardCriticAnnotationAgent.__new__(ForwardCriticAnnotationAgent)
	agent.nrdb = FakeNrdb()
	agent.id_critic = FakeIdCritic()
	agent.id_model_path = "id.json"
	agent.surface_critic = FakeSurfaceCritic()
	agent.surface_model_path = "surface.json"
	agent.max_active_id_surprises = 3
	agent.progress = lambda _message: None
	agent._create_response = lambda *args, **kwargs: SimpleNamespace(output_text=json.dumps({
		"annotation": candidate_annotation, "confidence": 0.9, "note": "narrow repair",
	}))
	return agent


def _result(annotation="東an-top:1 有av-cmpz>2"):
	return {
		"segmented": "aga-za arj-aː",
		"annotation": annotation,
		"confidence": 0.95,
		"decision": "proposed",
		"evidence": {},
	}


def test_forward_active_critic_accepts_id_repair_supported_by_surface():
	agent = _agent("東an-top:1 有av-cmpz>1")
	item = {"text": "aga za arjaː", "dialect_id": 19}
	job = {"annotation_schema_id": 2}
	out = agent._active_forward_critics(item, job, _result())
	assert out["annotation"] == "東an-top:1 有av-cmpz>1"
	assert out["evidence"]["forward_active_id_review"]["revision_accepted"] is True


def test_forward_active_critic_rejects_surface_worsening():
	agent = _agent("東an-top:1 有av-cmpz>2")
	assert agent._surface_better_or_equal(
		{"valid_alignment": True, "strong_disagreements": 2, "phonotactic_mean_log_probability": -3.0},
		{"valid_alignment": True, "strong_disagreements": 0, "phonotactic_mean_log_probability": -1.0},
	) is False


def test_forward_active_critic_abstains_when_surprises_are_diffuse():
	agent = _agent("東an-top:1 有av-cmpz>1")
	agent.id_critic = DiffuseIdCritic()
	calls = []
	agent._create_response = lambda *args, **kwargs: calls.append((args, kwargs))
	item = {"text": "aga za arjaː", "dialect_id": 19}
	job = {"annotation_schema_id": 2}
	out = agent._active_forward_critics(item, job, _result())
	assert out["annotation"] == "東an-top:1 有av-cmpz>2"
	assert calls == []
	assert out["evidence"]["forward_active_id_review"]["abstained_reason"] == "diffuse_id_surprises"


def test_forward_active_critic_malformed_output_keeps_validated_analysis_locally():
	agent = _agent("東an-top:1 有av-cmpz>1")
	agent._create_response = lambda *args, **kwargs: SimpleNamespace(output_text="")
	item = {"text": "aga za arjaː", "dialect_id": 19}
	job = {"annotation_schema_id": 2}
	out = agent._active_forward_critics(item, job, _result())
	assert out["annotation"] == "東an-top:1 有av-cmpz>2"
	assert out["evidence"]["forward_active_id_review"]["revision_accepted"] is False
	assert out["evidence"]["forward_active_id_review"]["abstained_reason"] == "malformed_critic_output"
