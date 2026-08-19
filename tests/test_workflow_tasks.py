from nrdb_agent import workflow


class FakeNrdb:
	def __init__(self):
		self.morph_calls = 0

	def morph_analyze(self, text, dialect_id, annotation_schema_id):
		self.morph_calls += 1
		return {"segmented": "pred-seg", "annotation": "pred-ann", "phrases": []}

	def validate_analysis(self, text, segmented, annotation):
		return {"valid": True}


class FakeAgent:
	def __init__(self, nrdb, model_name, **kwargs):
		self.nrdb = nrdb
		self.model_name = model_name

	def translate_frozen(self, item, job, segmented, annotation, confidence=1.0):
		return {
			"segmented": segmented, "annotation": annotation, "trsl_ai": "訳",
			"decision": "proposed", "confidence": 0.9, "evidence": {"frozen": True},
		}

	def annotate(self, item, job, morph):
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


def test_translate_existing_skips_morph_model(monkeypatch):
	monkeypatch.setattr(workflow, "TaskAwareAnnotationAgent", FakeAgent)
	nrdb = FakeNrdb()
	result = workflow.execute_item(nrdb, _item(), "translate", 2, "宮古", morphology_source="existing")
	assert nrdb.morph_calls == 0
	assert result["annotation"] == "gold-ann"
	assert result["translation"] == "訳"


def test_auto_uses_existing_but_predict_calls_morph(monkeypatch):
	monkeypatch.setattr(workflow, "TaskAwareAnnotationAgent", FakeAgent)
	nrdb = FakeNrdb()
	workflow.execute_item(nrdb, _item(), "morph", 2, "宮古", morphology_source="auto")
	assert nrdb.morph_calls == 0
	workflow.execute_item(nrdb, _item(), "morph", 2, "宮古", morphology_source="predict")
	assert nrdb.morph_calls == 1


def test_required_translation_is_enforced(monkeypatch):
	monkeypatch.setattr(workflow, "TaskAwareAnnotationAgent", FakeAgent)
	nrdb = FakeNrdb()
	try:
		workflow.execute_item(nrdb, _item(translation_jp=""), "morph", 2, "宮古", translation_evidence="required")
	except ValueError as error:
		assert "required" in str(error)
	else:
		raise AssertionError("missing required human translation was accepted")
