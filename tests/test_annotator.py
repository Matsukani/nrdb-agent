import json
from types import SimpleNamespace

import pytest

from nrdb_agent.annotator import AnnotationAgent, parse_final_json


def test_parse_final_json_accepts_proposal():
	result = parse_final_json('{"segmented":"an-na","annotation":"dm-top","decision":"proposed","confidence":0.91,"evidence":{}}')
	assert result["decision"] == "proposed"
	assert result["confidence"] == 0.91


def test_parse_final_json_accepts_fenced_json():
	result = parse_final_json('```json\n{"segmented":"","annotation":"","decision":"failed","confidence":0,"evidence":{}}\n```')
	assert result["decision"] == "failed"


def test_parse_final_json_rejects_bad_confidence():
	with pytest.raises(ValueError, match="confidence"):
		parse_final_json('{"segmented":"x","annotation":"x","decision":"proposed","confidence":2,"evidence":{}}')


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
			output_text='{"segmented":"anna","annotation":"頭an","decision":"uncertain","confidence":0.6,"evidence":{}}',
		)


class FakeClient:
	def __init__(self):
		self.responses = FakeResponses()


class FakeNrdb:
	def lookup_id(self, label, schema_id):
		return {"success": True, "label": label, "schema_id": schema_id}

	def examples(self, label, schema_id, sentence_id, limit):
		raise AssertionError("not expected")

	def validate_analysis(self, text, segmented, annotation):
		raise AssertionError("not expected")


def test_annotation_agent_continues_tools_without_previous_response_id():
	client = FakeClient()
	agent = AnnotationAgent(FakeNrdb(), "test-model", client=client)
	result = agent.annotate(
		{"sentence_id": 1, "dialect_id": 17, "dialect_region": "Miyako", "text": "anna", "translation_jp": "頭"},
		{"annotation_schema_id": 2},
		{"segmented": "anna", "annotation": "頭an"},
	)
	assert result["model_response_id"] == "resp_final"
	assert len(client.responses.calls) == 2
	second = client.responses.calls[1]
	assert "previous_response_id" not in second
	assert any(item.get("type") == "function_call" for item in second["input"])
	assert any(item.get("type") == "function_call_output" for item in second["input"])
