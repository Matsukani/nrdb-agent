from nrdb_agent import execution
from nrdb_agent.execution import ExecutionRequest


class FakeNrdb:
	def __init__(self):
		self.morph_calls = 0

	def morph_analyze(self, text, dialect_id, annotation_schema_id):
		self.morph_calls += 1
		return {
			"segmented": "pred-seg", "annotation": "pred-ann", "phrases": [],
			"inference": {"model_id": "morph-v1", "model_label": "Morph V1"},
		}

	def validate_analysis(self, text, segmented, annotation):
		return {"valid": True}


class FakeAgent:
	last_job = None

	def __init__(self, nrdb, model_name, **kwargs):
		self.nrdb = nrdb
		self.model_name = model_name

	def translate_frozen(self, item, job, segmented, annotation, confidence=1.0):
		type(self).last_job = dict(job)
		return {
			"segmented": segmented, "annotation": annotation, "trsl_ai": "訳",
			"decision": "proposed", "confidence": 0.9, "evidence": {"frozen": True},
		}

	def annotate(self, item, job, morph):
		type(self).last_job = dict(job)
		return {
			"segmented": morph["segmented"], "annotation": morph["annotation"],
			"trsl_ai": "訳" if job.get("produce_translation") else "",
			"decision": "proposed", "confidence": 0.8, "evidence": {},
		}


def _item(**values):
	base = {
		"sentence_id": 1, "text": "aga", "dialect_id": 19, "dialect_region": "宮古",
		"translation_jp": "東だ", "existing_segmented": "gold-seg", "existing_annotation": "gold-ann",
	}
	base.update(values)
	return base


def _run(nrdb, task, **values):
	request = ExecutionRequest(
		item=_item(**values.pop("item", {})), task=task, annotation_schema_id=2, region="宮古",
		semantic_feedback=values.pop("semantic_feedback", "none"),
		require_semantic_feedback=values.pop("require_semantic_feedback", False),
		use_constructions=values.pop("use_constructions", False),
		use_licensed_forms=values.pop("use_licensed_forms", False),
		nrdb_evidence=values.pop("nrdb_evidence", "enabled"),
		morphology_source=values.pop("morphology_source", "predict"),
	)
	assert not values
	return execution.execute_request(nrdb, request)


def test_translate_existing_skips_morph_model(monkeypatch):
	monkeypatch.setattr(execution, "TaskAwareAnnotationAgent", FakeAgent)
	nrdb = FakeNrdb()
	result = _run(nrdb, "translate", morphology_source="existing")
	assert nrdb.morph_calls == 0
	assert result["annotation"] == "gold-ann"
	assert result["translation"] == "訳"
	assert result["morph_baseline"]["source"] == "existing"


def test_translate_threads_construction_mode_independently(monkeypatch):
	monkeypatch.setattr(execution, "TaskAwareAnnotationAgent", FakeAgent)
	FakeAgent.last_job = None
	nrdb = FakeNrdb()
	result = _run(nrdb, "translate", morphology_source="existing", semantic_feedback="none", use_constructions=True)
	assert FakeAgent.last_job["semantic_feedback"] == "none"
	assert FakeAgent.last_job["use_constructions"] is True
	assert result["use_constructions"] is True


def test_predict_threads_licensed_mode_and_selects_licensed_agent(monkeypatch):
	monkeypatch.setattr(execution, "LicensedTaskAwareAnnotationAgent", FakeAgent)
	FakeAgent.last_job = None
	nrdb = FakeNrdb()
	result = _run(nrdb, "morph", morphology_source="predict", semantic_feedback="none", use_licensed_forms=True)
	assert nrdb.morph_calls == 1
	assert FakeAgent.last_job["use_licensed_forms"] is True
	assert result["use_licensed_forms"] is True


def test_auto_uses_existing_but_predict_calls_morph(monkeypatch):
	monkeypatch.setattr(execution, "TaskAwareAnnotationAgent", FakeAgent)
	nrdb = FakeNrdb()
	_run(nrdb, "morph", morphology_source="auto")
	assert nrdb.morph_calls == 0
	result = _run(nrdb, "morph", morphology_source="predict")
	assert nrdb.morph_calls == 1
	assert result["morph_baseline"]["source"] == "nrdb-morph"
	assert result["morph_baseline"]["segmented"] == "pred-seg"
	assert result["morph_baseline"]["annotation"] == "pred-ann"
	assert result["morph_baseline"]["inference"]["model_id"] == "morph-v1"


def test_required_translation_is_enforced(monkeypatch):
	monkeypatch.setattr(execution, "TaskAwareAnnotationAgent", FakeAgent)
	nrdb = FakeNrdb()
	try:
		_run(nrdb, "morph", item={"translation_jp": ""}, semantic_feedback="existing", require_semantic_feedback=True)
	except ValueError as error:
		assert "required" in str(error)
	else:
		raise AssertionError("missing required human translation was accepted")


def test_translate_without_morphology_skips_morph_model(monkeypatch):
	monkeypatch.setattr(execution, "TaskAwareAnnotationAgent", FakeAgent)
	nrdb = FakeNrdb()
	result = _run(nrdb, "translate", morphology_source="none")
	assert nrdb.morph_calls == 0
	assert result["segmented"] == ""
	assert result["annotation"] == ""
	assert result["translation"] == "訳"
	assert result["morph_baseline"] is None
