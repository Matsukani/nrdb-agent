from nrdb_agent.nrdb import NrdbClient


class FailIfCalledHttp:
	def get(self, *args, **kwargs):
		raise AssertionError("HTTP should not be called for an oversized corpus expression")


def test_oversized_corpus_expression_is_rejected_locally():
	client = NrdbClient()
	client.http = FailIfCalledHttp()
	result = client.examples("A-B-C-D-E-F-G-H-I", 2, 123, 5)
	assert result["success"] is True
	assert result["query_rejected"] is True
	assert result["examples"] == []
	assert "at most 8" in result["error"]
	assert "QUERY_REJECTED" in result["label"]
