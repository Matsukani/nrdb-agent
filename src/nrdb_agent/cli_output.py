import itertools
import sys
import threading
import time


QUIET_PREFIXES = (
	"translate:",
	"morph: model=",
	"forward-v9: uncertainty triage",
	"final: decision=",
	"translation-v7: final confidence=",
	"translation: final confidence=",
	"review-v9: action=",
	"API usage:",
)


class Spinner:
	def __init__(self, stream=None, interval=0.10):
		self.stream = stream or sys.stderr
		self.interval = float(interval)
		self._stop = threading.Event()
		self._thread = None

	def start(self):
		if self._thread is not None:
			return
		self._thread = threading.Thread(target=self._run, daemon=True)
		self._thread.start()

	def _run(self):
		for glyph in itertools.cycle("|/-\\"):
			if self._stop.is_set():
				break
			self.stream.write("\r{}".format(glyph))
			self.stream.flush()
			time.sleep(self.interval)

	def stop(self):
		if self._thread is None:
			return
		self._stop.set()
		self._thread.join(timeout=max(0.2, self.interval * 3))
		self.stream.write("\r \r")
		self.stream.flush()
		self._thread = None


class TranslationProgress:
	def __init__(self, mode="default", stream=None):
		if mode not in {"default", "quiet", "silent"}:
			raise ValueError("invalid translation output mode: {}".format(mode))
		self.mode = mode
		self.stream = stream or sys.stdout
		self.spinner = Spinner() if mode == "silent" else None

	def start(self):
		if self.spinner is not None:
			self.spinner.start()

	def stop(self):
		if self.spinner is not None:
			self.spinner.stop()

	def __call__(self, message):
		if self.mode == "silent":
			return
		text = str(message or "")
		if self.mode == "quiet":
			stripped = text.strip()
			if not any(stripped.startswith(prefix) for prefix in QUIET_PREFIXES):
				return
		print(text, file=self.stream)


def estimated_cost_text(value):
	usage = value.get("api_usage") if isinstance(value, dict) else None
	if not isinstance(usage, dict) or not usage.get("pricing_complete"):
		return "cost unknown"
	totals = usage.get("totals") or {}
	cost = totals.get("estimated_cost_usd")
	if cost is None:
		return "cost unknown"
	return "${:.4f}".format(float(cost))


def silent_translation_line(value):
	translation = str(value.get("translation") or "").strip()
	return "{} ({})".format(translation, estimated_cost_text(value))
