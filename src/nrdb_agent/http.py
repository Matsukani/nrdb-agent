import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class JsonHttpClient:
	def __init__(self, timeout=60):
		self.timeout = float(timeout)

	def get(self, url, params=None):
		if params:
			url = url + ("&" if "?" in url else "?") + urlencode(params)
		with urlopen(url, timeout=self.timeout) as response:
			return json.loads(response.read().decode("utf-8"))

	def post(self, url, payload):
		request = Request(
			url,
			data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
			headers={"Content-Type": "application/json; charset=utf-8"},
			method="POST",
		)
		with urlopen(request, timeout=self.timeout) as response:
			return json.loads(response.read().decode("utf-8"))
