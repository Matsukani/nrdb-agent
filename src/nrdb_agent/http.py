import json
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class JsonHttpClient:
	def __init__(self, timeout=60):
		self.timeout = float(timeout)

	def _decode(self, response):
		body = response.read().decode("utf-8")
		return json.loads(body)

	def get(self, url, params=None):
		if params:
			url = url + ("&" if "?" in url else "?") + urlencode(params)
		try:
			with urlopen(url, timeout=self.timeout) as response:
				return self._decode(response)
		except HTTPError as error:
			try:
				payload = self._decode(error)
			except (UnicodeDecodeError, json.JSONDecodeError):
				raise
			payload.setdefault("http_status", error.code)
			return payload

	def post(self, url, payload):
		request = Request(
			url,
			data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
			headers={"Content-Type": "application/json; charset=utf-8"},
			method="POST",
		)
		try:
			with urlopen(request, timeout=self.timeout) as response:
				return self._decode(response)
		except HTTPError as error:
			try:
				result = self._decode(error)
			except (UnicodeDecodeError, json.JSONDecodeError):
				raise
			result.setdefault("http_status", error.code)
			return result
