import io
import json
from urllib.error import HTTPError
from unittest.mock import patch

from nrdb_agent.http import JsonHttpClient


class FakeError(HTTPError):
	def __init__(self, body, code=400):
		super().__init__("http://example.test", code, "Bad Request", {}, io.BytesIO(body.encode("utf-8")))


def test_post_returns_json_body_for_http_error():
	client = JsonHttpClient()
	with patch("nrdb_agent.http.urlopen", side_effect=FakeError(json.dumps({"valid": False, "error": "surface mismatch"}))):
		payload = client.post("http://example.test", {"x": 1})
	assert payload["valid"] is False
	assert payload["error"] == "surface mismatch"
	assert payload["http_status"] == 400
