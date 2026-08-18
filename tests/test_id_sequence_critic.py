from types import SimpleNamespace

from nrdb_agent.reverse_id_critic import IdCriticSyntaxAwareReverseSurfaceAgent, IdSequenceCritic


class FakeIdModel:
	def score(self, annotation, annotation_schema_id):
		bad = "-dat " in annotation or annotation.endswith("-dat")
		surprises = [{
			"representation": "segment",
			"index": 1,
			"threshold": -4.0,
			"token": "dat",
			"context": ["食kv"],
			"order": 2,
			"log_probability": -5.0,
			"alternatives": [["prp", 0.8], ["cvb", 0.1]],
		}] if bad else []
		return {
			"segment": {
				"mean_log_probability": -3.0 if bad else -1.0,
				"positions": [],
			},
			"atom": {
				"mean_log_probability": -2.0 if bad else -1.0,
				"positions": [],
			},
			"strong_surprises": surprises,
			"strong_surprise_count": len(surprises),
			"combined_mean_log_probability": -2.5 if bad else -1.0,
		}


def test_id_critic_compacts_real_model_shape():
	critic = IdSequenceCritic(model=FakeIdModel())
	review = critic.review("食kv-dat 行iv", 2)
	compact = critic.compact(review)
	assert compact["strong_surprises"] == 1
	assert compact["mean_log_probability"] == -2.5
	position = compact["representations"]["segment"]["surprising_positions"][0]
	assert position["token"] == "dat"
	assert position["top_observed"][0]["token"] == "prp"
	assert position["top_observed"][0]["probability"] == 0.8


def test_id_revision_compares_surprise_counts_not_diagnostic_dicts():
	agent = IdCriticSyntaxAwareReverseSurfaceAgent.__new__(IdCriticSyntaxAwareReverseSurfaceAgent)
	agent.id_critic = IdSequenceCritic(model=FakeIdModel())
	agent.id_model_path = "id_sequence_model.json"
	agent.progress = lambda message: None
	agent._create_response = lambda *args, **kwargs: SimpleNamespace(
		output_text='{"annotation":"食kv-prp 行iv","confidence":0.91,"note":"purposive"}'
	)
	result = agent._review_ids(
		{"translation_jp": "食べに行く", "dialect_region": "宮古"},
		{"annotation_schema_id": 2},
		{"annotation": "食kv-dat 行iv", "confidence": 0.95, "evidence": {}},
	)
	assert result["annotation"] == "食kv-prp 行iv"
	assert result["evidence"]["id_sequence_review"]["revision_accepted"] is True
	assert result["evidence"]["id_sequence_review"]["strong_surprises"] == 1
	assert result["evidence"]["id_sequence_review"]["candidate"]["strong_surprises"] == 0
