import json

from nrdb_agent.annotator import AnnotationAgent, parse_translation_json


def test_parse_translation_json_accepts_complete_contract():
	payload = {
		"trsl_ai": "自然な訳です。",
		"confidence": 0.9,
		"translation_evidence": {
			"dictionary_ids": ["行iv"],
			"example_sentence_ids": [123],
			"ungrounded_ids": [],
			"note": "grounded",
		},
	}
	result = parse_translation_json(json.dumps(payload, ensure_ascii=False))
	assert result["trsl_ai"] == "自然な訳です。"
	assert result["confidence"] == 0.9


def test_translation_finalizer_retries_malformed_json():
	class Response:
		status = "completed"
		incomplete_details = None
		output = []

		def __init__(self, text):
			self.output_text = text

	class Client:
		def __init__(self):
			self.calls = 0
			self.responses = self

		def create(self, **_kwargs):
			self.calls += 1
			if self.calls == 1:
				return Response('{"trsl_ai":"途中')
			return Response(json.dumps({
				"trsl_ai": "修復された訳です。",
				"confidence": 0.8,
				"translation_evidence": {
					"dictionary_ids": [],
					"example_sentence_ids": [],
					"ungrounded_ids": [],
					"note": "retry",
				},
			}, ensure_ascii=False))

	agent = AnnotationAgent(nrdb=None, model_name="test", client=Client())
	result = agent._finalize_translation([{"role": "user", "content": "x"}], [], "test")
	assert result["trsl_ai"] == "修復された訳です。"
	assert agent.client.calls == 2
