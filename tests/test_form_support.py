from nrdb_agent.annotator import AnnotationAgent


class FakeNrdb:
	def __init__(self):
		self.calls = []

	def form_id_support(self, surface, candidate_id, region, annotation_schema_id):
		self.calls.append((surface, candidate_id, region, annotation_schema_id))
		return {
			"success": True,
			"surface": surface,
			"candidate_id": candidate_id,
			"region": region,
			"corpus": {"surface_total": 8, "candidate_count": 0, "alternatives": [{"id": "dat", "count": 8}]},
			"lexicon": {"surface_total": 2, "candidate_count": 0, "alternatives": [{"id": "dat", "count": 2}]},
			"combined": {"surface_total": 10, "candidate_count": 0, "candidate_rate": 0.0, "penalty": "strong"},
		}


def test_form_id_support_region_is_injected_from_current_item():
	nrdb = FakeNrdb()
	agent = AnnotationAgent(nrdb, "test", client=object())
	item = {"sentence_id": 1, "text": "dui", "dialect_region": "Miyako"}
	result = agent._tool_result("form_id_support", {"surface": "dui", "candidate_id": "foc"}, item, 2)
	assert nrdb.calls == [("dui", "foc", "Miyako", 2)]
	assert result["combined"]["penalty"] == "strong"


def test_form_id_support_without_region_applies_no_penalty():
	nrdb = FakeNrdb()
	agent = AnnotationAgent(nrdb, "test", client=object())
	item = {"sentence_id": 1, "text": "dui", "dialect_region": None}
	result = agent._tool_result("form_id_support", {"surface": "dui", "candidate_id": "foc"}, item, 2)
	assert nrdb.calls == []
	assert result["combined"]["penalty"] == "none"
