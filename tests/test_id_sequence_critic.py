from nrdb_agent.reverse_id_critic import IdSequenceCritic


class FakeIdModel:
	def score(self, annotation, annotation_schema_id):
		bad = "dat" in annotation
		return {
			"mean_log_probability": -3.0 if bad else -1.0,
			"strong_surprises": 1 if bad else 0,
			"representations": {
				"segment": {
					"mean_log_probability": -3.0 if bad else -1.0,
					"strong_surprises": 1 if bad else 0,
					"positions": [{
						"index": 1, "token": "dat", "context": ["食kv"], "order": 2,
						"log_probability": -5.0, "strong_surprise": bad,
						"top_observed": [{"token": "prp", "count": 20}],
					}] if bad else [],
				},
				"atom": {"mean_log_probability": -1.0, "strong_surprises": 0, "positions": []},
			},
		}


def test_id_critic_exposes_only_strong_local_surprises():
	critic = IdSequenceCritic(model=FakeIdModel())
	review = critic.review("食kv-dat 行iv", 2)
	compact = critic.compact(review)
	assert compact["strong_surprises"] == 1
	position = compact["representations"]["segment"]["surprising_positions"][0]
	assert position["token"] == "dat"
	assert position["top_observed"][0]["token"] == "prp"
