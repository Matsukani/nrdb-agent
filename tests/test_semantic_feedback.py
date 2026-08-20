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
				"action": "revise", "segmented": "a-b-c", "annotation": "A-B-C",
				"confidence": 0.95, "changed_ids": ["C"], "note": "semantic mismatch",
			}
		return {
			"action": "keep", "segmented": result["segmented"], "annotation": result["annotation"],
			"confidence": 0.9, "changed_ids": [], "note": "coherent",
		}

	def _shared_evidence_compact(self):
		return {}


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
	assert result["annotation"] == "A-B-C"
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
	assert result["annotation"] == "A-B-C"
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
