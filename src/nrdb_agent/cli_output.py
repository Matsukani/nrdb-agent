import itertools
import sys
import threading
import time


QUIET_PREFIXES = (
	"translate:",
	"morph: model=",
	"forward-v9: uncertainty triage",
	"reverse-v1: Japanese -> Miyako IDs",
	"id-model:",
	"reverse-v2: realize surface",
	"surface-model:",
	"final: decision=",
	"translation-v7: final confidence=",
	"translation: final confidence=",
	"review-v9: action=",
	"API usage:",
)

MILESTONE_RULES = (
	("translate:", "starting translation"),
	("morph: analyze", "morphological analysis"),
	("morph: model=", "morphological analysis"),
	("forward-v9: uncertainty triage", "annotation review"),
	("reverse-v1: Japanese -> Miyako IDs", "lexical planning"),
	("id-model:", "grammatical critic"),
	("final: decision=", "annotation finalized"),
	("translation-v7: initial response", "dictionary grounding"),
	("translation-v7: final confidence=", "Japanese realization"),
	("translation: final confidence=", "Japanese realization"),
	("reverse-v2: realize surface", "Miyako realization"),
	("surface-model:", "surface critic"),
	("review-v9: semantic consistency review", "semantic review"),
	("review-v9: action=", "semantic review"),
	("API usage:", "cost accounting"),
)


class MilestoneBar:
	def __init__(self, stream=None, interval=0.10, width=10, show_label=False):
		self.stream = stream or sys.stderr
		self.interval = float(interval)
		self.width = int(width)
		self.show_label = bool(show_label)
		self._stop = threading.Event()
		self._thread = None
		self._lock = threading.Lock()
		self._completed = 0
		self._label = "starting"
		self._seen = set()
		self._glyphs = itertools.cycle("|/-\\")

	def start(self):
		if self._thread is not None:
			return
		self._thread = threading.Thread(target=self._run, daemon=True)
		self._thread.start()

	def _clear_line(self):
		# Return to column 0 and erase the complete terminal line before every
		# redraw. Padding based on Python string length is unreliable for wide
		# characters such as Japanese labels and can leave ghost fragments.
		self.stream.write("\r\x1b[2K")

	def _render(self, glyph=None, complete=False):
		with self._lock:
			completed = min(self._completed, self.width)
			label = self._label
		if complete:
			bar = "=" * self.width
		else:
			remaining = max(0, self.width - completed - 1)
			bar = "=" * completed + (glyph or "|") + "・" * remaining
		text = bar + ("  " + label if self.show_label and label else "")
		self._clear_line()
		self.stream.write(text)
		self.stream.flush()

	def _run(self):
		while not self._stop.is_set():
			self._render(next(self._glyphs))
			time.sleep(self.interval)

	def milestone(self, key, label):
		with self._lock:
			self._label = label
			if key in self._seen:
				return
			self._seen.add(key)
			if self._completed < self.width - 1:
				self._completed += 1

	def stop(self, complete=True):
		if self._thread is None:
			return
		self._stop.set()
		self._thread.join(timeout=max(0.2, self.interval * 3))
		if complete:
			with self._lock:
				self._completed = self.width
				self._label = "complete"
			self._render(complete=True)
		self.stream.write("\n")
		self.stream.flush()
		self._thread = None


class TranslationProgress:
	def __init__(self, mode="quiet", stream=None, progress_stream=None):
		if mode not in {"quiet", "verbose", "silent", "compact"}:
			raise ValueError("invalid translation output mode: {}".format(mode))
		self.mode = mode
		self.stream = stream or sys.stdout
		self.progress_stream = progress_stream or sys.stderr
		self.bar = MilestoneBar(self.progress_stream, show_label=(mode == "compact")) if mode in {"silent", "compact"} else None

	def start(self):
		if self.bar is not None:
			self.bar.start()

	def stop(self):
		if self.bar is not None:
			self.bar.stop()

	def _milestone(self, text):
		stripped = text.strip()
		for prefix, label in MILESTONE_RULES:
			if stripped.startswith(prefix):
				return prefix, label
		return None

	def __call__(self, message):
		text = str(message or "")
		if self.bar is not None:
			milestone = self._milestone(text)
			if milestone is not None:
				self.bar.milestone(*milestone)
			return
		if self.mode == "verbose":
			print(text, file=self.stream)
			return
		stripped = text.strip()
		if any(stripped.startswith(prefix) for prefix in QUIET_PREFIXES):
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
