from nrdb_agent.reverse_agent import REVERSE_TOOLS, ReverseIdAgent
from nrdb_agent.reverse_id_critic import IdCriticSyntaxAwareReverseSurfaceAgent, IdSequenceCritic


class FakeNrdb:
	def __init__(self):
		self.searches = []
		self.example_calls = []

	def search_japanese_evidence(self, query, annotation_schema_id, exclude_sentence_id, **kwargs):
		self.searches.append((query, annotation_schema_id, exclude_sentence_id, kwargs))
		return {
			"success": True,
			"query": query,
			"region": kwargs.get("region"),
			"lexical_entries": [{"label": query + "ID", "meaning_jp": query}],
			"corpus_examples": [],
		}

	def examples(self, label, annotation_schema_id, exclude_sentence_id, limit=6, exclude_job_id=None):
		self.example_calls.append((label, annotation_schema_id, exclude_sentence_id, limit))
		return {"success": True, "label": label, "examples": []}


class NoSurpriseModel:
	def score(self, annotation, annotation_schema_id):
		return {
			"segment": {"mean_log_probability": -1.0, "positions": []},
			"atom": {"mean_log_probability": -1.0, "positions": []},
			"strong_surprises": [],
			"strong_surprise_count": 0,
			"combined_mean_log_probability": -1.0,
		}


def test_reverse_initial_tools_do_not_expose_routine_grammar_corpus_queries():
	assert "corpus_examples" not in {tool["name"] for tool in REVERSE_TOOLS}
	assert "search_japanese_batch" in {tool["name"] for tool in REVERSE_TOOLS}


def test_batch_search_covers_multiple_lexical_queries_in_one_tool_result():
	nrdb = FakeNrdb()
	agent = ReverseIdAgent.__new__(ReverseIdAgent)
	agent.nrdb = nrdb
	agent._id_pass_dialect_ids = [19, 22, 14]
	result = agent._tool_result_reverse(
		"search_japanese_batch",
		{"queries": ["蕎麦", "食べる", "行く"], "limit": 5},
		{"sentence_id": 123, "dialect_region": "宮古"},
		2,
	)
	assert result["queries"] == ["蕎麦", "食べる", "行く"]
	assert len(result["results"]) == 3
	assert [value[0] for value in nrdb.searches] == ["蕎麦", "食べる", "行く"]
	assert all(value[3]["dialect_ids"] == [19, 22, 14] for value in nrdb.searches)


def test_no_id_surprise_makes_zero_grammar_corpus_queries():
	nrdb = FakeNrdb()
	agent = IdCriticSyntaxAwareReverseSurfaceAgent.__new__(IdCriticSyntaxAwareReverseSurfaceAgent)
	agent.id_critic = IdSequenceCritic(model=NoSurpriseModel())
	agent.id_model_path = "id.json"
	agent.nrdb = nrdb
	agent.progress = lambda _message: None
	result = agent._review_ids(
		{"sentence_id": 123, "translation_jp": "昨日ゴミを出した", "dialect_region": "宮古"},
		{"annotation_schema_id": 2},
		{"annotation": "昨日kn ゴミgn-acc 出iv2-pst", "confidence": 0.9, "evidence": {}},
	)
	assert result["annotation"] == "昨日kn ゴミgn-acc 出iv2-pst"
	assert nrdb.example_calls == []
