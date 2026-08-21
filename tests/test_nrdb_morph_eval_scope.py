import pytest

from nrdb_agent.nrdb import NrdbClient


class FakeHttp:
	def __init__(self):
		self.calls = []

	def get(self, url, params):
		self.calls.append((url, params))
		return {"success": True, "rows": [], "next_after_id": None}


def _client():
	client = NrdbClient(agent_url="http://127.0.0.1/php/agent.php")
	client.http = FakeHttp()
	return client


def test_morph_eval_without_datasets_sends_schema_and_region_scope():
	client = _client()
	assert client.morph_eval_rows(None, annotation_schema_id=2, region="宮古") == []
	params = client.http.calls[0][1]
	assert params["dataset_ids"] == ""
	assert params["annotation_schema_id"] == 2
	assert params["region"] == "宮古"


def test_morph_eval_without_datasets_rejects_unbounded_request():
	with pytest.raises(ValueError, match="requires annotation_schema_id and region"):
		_client().morph_eval_rows(None)


def test_morph_eval_compounds_and_deduplicates_dataset_ids():
	client = _client()
	client.morph_eval_rows([30, 21, 30], annotation_schema_id=2, region="宮古")
	assert client.http.calls[0][1]["dataset_ids"] == "21,30"
