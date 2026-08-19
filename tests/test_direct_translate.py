import json

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
		return {
			"segmented": "mi-ja",
			"annotation": "見mv-top:1",
			"inference": {
				"model_id": "miyako-65k-hybrid-shared-v002",
				"model_label": "Miyako 65k hybrid shared v002",
				"backend": "bilstm_crf_lookup_v1",
				"segmentation_mode": "joint",
				"segmentation_top_k": 5,
				"segmentation_id_weight": 1.0,
			},
		}


class FakeForwardAgent:
	def __init__(self, nrdb, model_name, client=None, progress=print, id_model_path=None, surface_model_path=None):
		self.nrdb = nrdb

	def annotate(self, item, job, morph):
		assert item["sentence_id"] == 0
		assert item["dialect_region"] == "宮古"
		assert job["prompt_version"] == "annotation-v9"
		assert job["produce_translation"] is True
		assert morph["annotation"] == "見mv-top:1"
		return {
			"segmented": "mi-ja", "annotation": "見mv-top:1", "trsl_ai": "見るよ。",
			"decision": "proposed", "confidence": 0.9, "evidence": {},
		}


class FlakyForwardAgent(FakeForwardAgent):
	attempts = 0

	def annotate(self, item, job, morph):
		type(self).attempts += 1
		if type(self).attempts == 1:
			raise json.JSONDecodeError("Unterminated string", '{"surface":"aga', 12)
		return super().annotate(item, job, morph)


class FakeReverseAgent:
	def __init__(self, nrdb, model_name, client=None, progress=print, surface_model_path=None, id_model_path=None):
		assert surface_model_path == "/tmp/surface.json"

	def annotate(self, item, job, morph):
		assert item["translation_jp"] == "魚を取りに行こう"
		assert job["target_dialect_ids"] == [19, 22, 14]
		return {
			"segmented": "zzu-tu-ga dzoː", "annotation": "魚in-取tv-prp intj:dzoo",
			"decision": "proposed", "confidence": 0.8, "evidence": {},
		}


def test_direct_miyako_to_japanese_uses_region_dialect_morph_service_and_provenance(monkeypatch):
	monkeypatch.setattr(translate_module, "ForwardCriticAnnotationAgent", FakeForwardAgent)
	nrdb = FakeNrdb()
	messages = []
	result = translate_module.translate_text(nrdb, "mija", "japanese", 2, "宮古", progress=messages.append)
	assert nrdb.morph_calls == [("mija", 22, 2)]
	assert result["translation"] == "見るよ。"
	assert result["morph_dialect_id"] == 22
	assert result["morph_inference"]["model_id"] == "miyako-65k-hybrid-shared-v002"
	assert any("miyako-65k-hybrid-shared-v002" in message and "top-k=5" in message for message in messages)
	assert nrdb.exclude_job_id == 0


def test_direct_translation_retries_malformed_tool_json(monkeypatch):
	FlakyForwardAgent.attempts = 0
	monkeypatch.setattr(translate_module, "ForwardCriticAnnotationAgent", FlakyForwardAgent)
	nrdb = FakeNrdb()
	messages = []
	result = translate_module.translate_text(nrdb, "mija", "japanese", 2, "宮古", progress=messages.append)
	assert result["translation"] == "見るよ。"
	assert FlakyForwardAgent.attempts == 2
	assert any("malformed/truncated tool or final JSON" in message for message in messages)


def test_direct_japanese_to_miyako_uses_ordered_dialects_and_surface_critic(monkeypatch):
	monkeypatch.setattr(translate_module, "SurfaceCriticReverseAgent", FakeReverseAgent)
	nrdb = FakeNrdb()
	result = translate_module.translate_text(
		nrdb, "魚を取りに行こう", "miyako", 2, "宮古",
		dialect_ids=[19, 22, 14], surface_model="/tmp/surface.json", progress=lambda _: None,
	)
	assert result["translation"] == "zzu-tu-ga dzoː"
	assert result["target_dialect_ids"] == [19, 22, 14]
