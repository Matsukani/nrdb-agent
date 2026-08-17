import pytest

from nrdb_agent.annotator import parse_final_json


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
