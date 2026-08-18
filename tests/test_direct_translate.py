from nrdb_agent import translate as translate_module


class FakeNrdb:
	def __init__(self):
		self.exclude_job_id = None
		self.morph_calls = []

	def region_dialects(self, region, annotation_schema_id):
		assert region == "宮古"
		assert annotation_schema_id == 2
		return [{"id": 22}, {"id": 19}]

	def morph_analyze(self, text, dialect_id, annotation_schema_id):
		self.morph_calls.append((text, dialect_id, annotation_schema_id))
		return {"segmented": "mi-ja", "annotation": "見mv-top:1"}


class FakeForwardAgent:
	def __init__(self, nrdb, model_name, client=None, progress=print):
		self.nrdb = nrdb

	def annotate(self, item, job, morph):
		assert item["sentence_id"] == 0
		assert item["dialect_region"] == "宮古"
		assert job["prompt_version"] == "annotation-v8"
		assert job["produce_translation"] is True
		assert morph["annotation"] == "見mv-top:1"
		return {
			"segmented": "mi-ja", "annotation": "見mv-top:1", "trsl_ai": "見るよ。",
			"decision": "proposed", "confidence": 0.9, "evidence": {},
		}


class FakeReverseAgent:
	def __init__(self, nrdb, model_name, client=None, progress=print, surface_model_path=None):
		assert surface_model_path == "/tmp/surface.json"

	def annotate(self, item, job, morph):
		assert item["translation_jp"] == "魚を取りに行こう"
		assert job["target_dialect_ids"] == [19, 22, 14]
		return {
			"segmented": "zzu-tu-ga dzoː", "annotation": "魚in-取tv-prp intj:dzoo",
			"decision": "proposed", "confidence": 0.8, "evidence": {},
		}


def test_direct_miyako_to_japanese_uses_region_dialect_and_morph_service(monkeypatch):
	monkeypatch.setattr(translate_module, "AnnotationAgentV8", FakeForwardAgent)
	nrdb = FakeNrdb()
	result = translate_module.translate_text(nrdb, "mija", "japanese", 2, "宮古", progress=lambda _: None)
	assert nrdb.morph_calls == [("mija", 22, 2)]
	assert result["translation"] == "見るよ。"
	assert result["morph_dialect_id"] == 22
	assert nrdb.exclude_job_id == 0


def test_direct_japanese_to_miyako_uses_ordered_dialects_and_surface_critic(monkeypatch):
	monkeypatch.setattr(translate_module, "SurfaceCriticReverseAgent", FakeReverseAgent)
	nrdb = FakeNrdb()
	result = translate_module.translate_text(
		nrdb, "魚を取りに行こう", "miyako", 2, "宮古",
		dialect_ids=[19, 22, 14], surface_model="/tmp/surface.json", progress=lambda _: None,
	)
	assert result["translation"] == "zzu-tu-ga dzoː"
	assert result["target_dialect_ids"] == [19, 22, 14]
