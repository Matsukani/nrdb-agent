from nrdb_agent.task_agent import TaskAwareAnnotationAgent


def test_v9_round_exhaustion_keeps_morph_baseline(monkeypatch):
	agent = object.__new__(TaskAwareAnnotationAgent)
	agent.progress = lambda _message: None

	def exhausted(*_args, **_kwargs):
		raise RuntimeError("annotation-v9 exceeded maximum tool rounds")

	monkeypatch.setattr(agent, "_annotation_phase_v9", exhausted)
	item = {"translation_jp": None}
	job = {"translation_evidence": "ignore", "produce_translation": False}
	morph = {"segmented": "foo-bar", "annotation": "A-B"}

	result = agent.annotate(item, job, morph)

	assert result["segmented"] == "foo-bar"
	assert result["annotation"] == "A-B"
	assert result["decision"] == "uncertain"
	assert result["evidence"]["round_exhaustion_fallback"]["kept_baseline"] is True
