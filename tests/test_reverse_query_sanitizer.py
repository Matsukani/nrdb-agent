from nrdb_agent.reverse_agent import REVERSE_INSTRUCTIONS, ReverseIdAgent, _sanitize_japanese_query


class FakeNrdb:
	def __init__(self):
		self.queries = []

	def search_japanese_evidence(self, query, schema_id, sentence_id, region=None, dialect_ids=None, limit=5):
		self.queries.append((query, region, tuple(dialect_ids or [])))
		return {"query": query, "region": region, "lexical_entries": [], "corpus_examples": []}


def _agent():
	agent = ReverseIdAgent.__new__(ReverseIdAgent)
	agent.nrdb = FakeNrdb()
	agent._id_pass_dialect_ids = [19, 22, 14]
	return agent


def test_reverse_batch_strips_region_helper_and_deduplicates():
	agent = _agent()
	item = {"sentence_id": 0, "dialect_region": "宮古"}
	result = agent._tool_result_reverse(
		"search_japanese_batch",
		{"queries": ["食べる 宮古", "山羊汁 宮古", "食べる 宮古", "今 宮古"], "limit": 6},
		item,
		2,
	)
	assert result["queries"] == ["食べる", "山羊汁", "今"]
	assert [row[0] for row in agent.nrdb.queries] == ["食べる", "山羊汁", "今"]
	assert all(row[1] == "宮古" for row in agent.nrdb.queries)


def test_reverse_followup_rejects_compound_pseudo_search_without_calling_nrdb():
	agent = _agent()
	item = {"sentence_id": 0, "dialect_region": "宮古"}
	result = agent._tool_result_reverse(
		"search_japanese_evidence",
		{"query": "何を食べている 山羊汁を食べている", "limit": 8},
		item,
		2,
	)
	assert result["rejected"] is True
	assert agent.nrdb.queries == []


def test_reverse_query_sanitizer_and_prompt_keep_scope_out_of_lexical_key():
	assert _sanitize_japanese_query("食べる 宮古方言", "宮古") == "食べる"
	assert _sanitize_japanese_query("Miyako 山羊汁", "宮古") == "山羊汁"
	assert "database lexical keys" in REVERSE_INSTRUCTIONS.lower()
	assert "never append" in REVERSE_INSTRUCTIONS.lower()
