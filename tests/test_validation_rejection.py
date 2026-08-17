from nrdb_agent.nrdb import NrdbClient


class DummyHttp:
	def post(self, url, payload):
		return {"valid": False, "error": "surface mismatch", "http_status": 400}


def test_validate_analysis_returns_rejection_as_tool_evidence():
	client = NrdbClient(agent_url="http://127.0.0.1/php/agent.php", morph_url="http://127.0.0.1:8765")
	client.http = DummyHttp()
	result = client.validate_analysis("abc", "a-b-c", "x-y-z")
	assert result["valid"] is False
	assert result["error"] == "surface mismatch"
	assert result["http_status"] == 400
