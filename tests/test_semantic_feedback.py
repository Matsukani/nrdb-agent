from nrdb_agent.task_agent import TaskAwareAnnotationAgent


class FakeNrdb:
	def validate_analysis(self, text, segmented, annotation):
		return {"valid": True}


class SemanticHarness(TaskAwareAnnotationAgent):
	def __init__(self, revise=False):
		self.nrdb = FakeNrdb()
		self.progress = lambda _message: None
		self.revise = revise
		self.generated_calls = 0
		self.review_calls = 0
		self._shared_evidence = {"lookup": {}, "corpus": {}, "form": {}}

	def _annotation_phase_v9(self, item, job, morph_result):
		return {
			"segmented": morph_result["segmented"],
			"annotation": morph_result["annotation"],
			"trsl_ai": "",
			"decision": "proposed",
			"confidence": 0.8,
			"evidence": {},
		}

	def _generate_translation(self, item, job, result):
		self.generated_calls += 1
		return {
			"trsl_ai": "訳{}".format(self.generated_calls),
			"confidence": 0.9,
			"translation_evidence": {"note": "generated"},
		}

	def _semantic_review(self, item, job, result):
		self.review_calls += 1
		if self.revise:
			return {
				"action": "revise", "segmented": "a-b", "annotation": "A-C",
				"confidence": 0.95, "changed_ids": ["C"], "note": "semantic mismatch",
			}
		return {
			"action": "keep", "segmented": result["segmented"], "annotation": result["annotation"],
			"confidence": 0.9, "changed_ids": [], "note": "coherent",
		}


def _item(translation=""):
	return {
		"sentence_id": 1, "text": "abc", "dialect_id": 19,
		"dialect_region": "宮古", "translation_jp": translation,
	}


def _job(task="morph", feedback="none", require=False):
	return {
		"annotation_schema_id": 2, "model_name": "gpt-5.6-terra",
		"task": task, "semantic_feedback": feedback,
		"require_semantic_feedback": require,
		"produce_translation": task in {"translate", "morph-translate"},
	}


def _morph():
	return {"segmented": "a-b", "annotation": "A-B"}


def test_generated_feedback_can_review_morph_without_translation_output():
	agent = SemanticHarness(revise=True)
	result = agent.annotate(_item(), _job(task="morph", feedback="generated"), _morph())
	assert result["annotation"] == "A-C"
	assert result["trsl_ai"] == ""
	assert agent.generated_calls == 1
	assert agent.review_calls == 1
	assert result["evidence"]["semantic_feedback"]["mode"] == "generated"


def test_translation_output_does_not_imply_semantic_feedback():
	agent = SemanticHarness(revise=True)
	result = agent.annotate(_item(), _job(task="morph-translate", feedback="none"), _morph())
	assert result["annotation"] == "A-B"
	assert result["trsl_ai"] == "訳1"
	assert agent.generated_calls == 1
	assert agent.review_calls == 0


def test_generated_feedback_translation_is_regenerated_after_revision():
	agent = SemanticHarness(revise=True)
	result = agent.annotate(_item(), _job(task="morph-translate", feedback="generated"), _morph())
	assert result["annotation"] == "A-C"
	assert result["trsl_ai"] == "訳2"
	assert agent.generated_calls == 2
	assert agent.review_calls == 1


def test_existing_feedback_uses_data_translation_without_generating_one_for_morph():
	agent = SemanticHarness(revise=False)
	result = agent.annotate(_item("人の話だ"), _job(task="morph", feedback="existing", require=True), _morph())
	assert result["annotation"] == "A-B"
	assert result["trsl_ai"] == ""
	assert agent.generated_calls == 0
	assert agent.review_calls == 1
	assert result["evidence"]["semantic_feedback"]["source"] == "human"


def test_v9_semantic_review_adapters_expose_cached_evidence_and_legacy_parser_signature():
	agent = SemanticHarness()
	agent._shared_evidence = {
		"lookup": {
			"人pn": {"success": True, "label": "人pn", "lexical_entries": [], "local": None, "global": None},
		},
		"corpus": {
			"人pn-top:1": {"success": True, "label": "人pn-top:1", "examples": []},
		},
		"form": {
			"pstu\t人pn": {"success": True, "surface": "pstu", "candidate_id": "人pn", "combined": {}},
		},
	}
	compact = agent._shared_evidence_compact()
	assert compact["lookup"][0]["label"] == "人pn"
	assert compact["corpus"][0]["label"] == "人pn-top:1"
	assert compact["form"][0]["surface"] == "pstu"
	assert compact["form"][0]["candidate_id"] == "人pn"
	assert agent._review_query_already_known("ground_lexical_ids", {"labels": ["人pn"]}) is True
	assert agent._review_query_already_known("corpus_examples", {"label": "人pn-top:1"}) is True
	assert agent._review_query_already_known("form_id_support", {"surface": "pstu", "candidate_id": "人pn"}) is True
	parsed = agent._parse_review(
		'{"action":"keep","segmented":"a-b","annotation":"A-B","confidence":0.9,"changed_ids":[],"note":"ok"}',
		{"segmented": "a-b", "annotation": "A-B"},
	)
	assert parsed["action"] == "keep"
	assert agent.max_review_rounds == 4


def test_legacy_translation_evidence_use_maps_to_auto_feedback():
	mode, active, required = TaskAwareAnnotationAgent._semantic_policy({"translation_evidence": "use"}, "")
	assert mode == "auto"
	assert active == "generated"
	assert required is False
