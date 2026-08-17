import json
from types import SimpleNamespace

import pytest

from nrdb_agent.annotator import AnnotationAgent, _compact_morph, _compact_tool_result, parse_final_json


def test_parse_final_json_accepts_proposal():
	result = parse_final_json('{"segmented":"an-na","annotation":"dm-top","trsl_ai":"頭だ。","decision":"proposed","confidence":0.91,"evidence":{}}')
	assert result["decision"] == "proposed"
	assert result["confidence"] == 0.91
	assert result["trsl_ai"] == "頭だ。"


def test_parse_final_json_defaults_translation_to_blank():
	result = parse_final_json('{"segmented":"an-na","annotation":"dm-top","decision":"proposed","confidence":0.91,"evidence":{}}')
	assert result["trsl_ai"] == ""


def test_parse_final_json_accepts_fenced_json():
	result = parse_final_json('```json\n{"segmented":"","annotation":"","decision":"failed","confidence":0,"evidence":{}}\n```')
	assert result["decision"] == "failed"


def test_parse_final_json_rejects_bad_confidence():
	with pytest.raises(ValueError, match="confidence"):
		parse_final_json('{"segmented":"x","annotation":"x","decision":"proposed","confidence":2,"evidence":{}}')


def test_compact_morph_bounds_alternatives():
	result = _compact_morph({
		"segmented": "anna", "annotation": "頭an", "unused_large_field": "x" * 10000,
		"phrases": [{"raw": "anna", "segments": [{
			"surface": "anna", "label": "頭an", "alternatives": [
				{"label": "a", "support": 1}, {"label": "b", "support": 2},
				{"label": "c", "support": 3}, {"label": "d", "support": 4},
			],
		}]}],
	})
	assert "unused_large_field" not in result
	assert len(result["phrases"][0]["segments"][0]["alternatives"]) == 3


def test_compact_examples_bounds_rows_and_text():
	result = _compact_tool_result("corpus_examples", {"label": "頭an", "examples": [
		{"sentence_id": index, "text": "x" * 500, "annotation": "a" * 500, "translation_jp": "y" * 500}
		for index in range(10)
	]})
	assert len(result["examples"]) == 6
	assert len(result["examples"][0]["text"]) <= 181
	assert len(result["examples"][0]["annotation"]) <= 281


class FakeOutput:
	def __init__(self, **values):
		self.__dict__.update(values)

	def model_dump(self, exclude_none=True):
		return dict(self.__dict__)


class FakeResponses:
	def __init__(self):
		self.calls = []

	def create(self, **kwargs):
		self.calls.append(kwargs)
		if len(self.calls) == 1:
			return SimpleNamespace(
				id="resp_first",
				output=[FakeOutput(type="function_call", name="lookup_id", call_id="call_1", arguments=json.dumps({"label": "頭an"}))],
				output_text="",
			)
		return SimpleNamespace(
			id="resp_final",
			output=[],
			output_text='{"segmented":"anna","annotation":"頭an","trsl_ai":"頭だ。","decision":"uncertain","confidence":0.6,"evidence":{}}',
		)


class FakeClient:
	def __init__(self):
		self.responses = FakeResponses()


class FakeNrdb:
	def lookup_id(self, label, schema_id):
		return {"success": True, "label": label, "schema_id": schema_id, "lexical_entries": []}

	def examples(self, label, schema_id, sentence_id, limit):
		raise AssertionError("not expected")

	def validate_analysis(self, text, segmented, annotation):
		raise AssertionError("not expected")


def test_annotation_agent_continues_tools_without_previous_response_id():
	client = FakeClient()
	agent = AnnotationAgent(FakeNrdb(), "test-model", client=client)
	result = agent.annotate(
		{"sentence_id": 1, "dialect_id": 17, "dialect_region": "Miyako", "text": "anna", "translation_jp": "頭"},
		{"annotation_schema_id": 2, "produce_translation": True},
		{"segmented": "anna", "annotation": "頭an", "huge": "x" * 20000},
	)
	assert result["model_response_id"] == "resp_final"
	assert result["trsl_ai"] == "頭だ。"
	assert len(client.responses.calls) == 2
	first = client.responses.calls[0]
	second = client.responses.calls[1]
	assert first["max_output_tokens"] == 800
	assert '"produce_translation": true' in first["input"][0]["content"]
	assert "huge" not in first["input"][0]["content"]
	assert "previous_response_id" not in second
	assert any(item.get("type") == "function_call" for item in second["input"])
	assert any(item.get("type") == "function_call_output" for item in second["input"])
