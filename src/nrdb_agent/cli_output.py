import itertools
import sys
import threading
import time


OUTPUT_MODES = {"quiet", "verbose", "silent", "compact"}

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
	("review-v9: human translation consistency review", "semantic review"),
	("review-v9: action=", "semantic review"),
	("API usage:", "cost accounting"),
)


def add_output_mode_args(parser):
	group = parser.add_mutually_exclusive_group()
	group.add_argument("--quiet", action="store_true", help="Show major milestones and completed results; this is the default")
	group.add_argument("--verbose", action="store_true", help="Show the full diagnostic/tool trace plus completed results")
	group.add_argument("--silent", action="store_true", help="Show an unlabeled milestone bar plus completed results")
	group.add_argument("--compact", action="store_true", help="Show a labeled milestone bar plus completed results")
	return group


def output_mode_from_args(args):
	if getattr(args, "verbose", False):
		return "verbose"
	if getattr(args, "silent", False):
		return "silent"
	if getattr(args, "compact", False):
		return "compact"
	return "quiet"


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
		self._stop = threading.Event()
		self._thread = threading.Thread(target=self._run, daemon=True)
		self._thread.start()

	def _clear_line(self):
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
		else:
			self._clear_line()
		self.stream.write("\n" if complete else "")
		self.stream.flush()
		self._thread = None


class TranslationProgress:
	def __init__(self, mode="quiet", stream=None, progress_stream=None):
		if mode not in OUTPUT_MODES:
			raise ValueError("invalid output mode: {}".format(mode))
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


class WorkflowProgress(TranslationProgress):
	"""Shared presentation policy for translate, run, and process commands."""
	def __init__(self, mode="quiet", stream=None, progress_stream=None):
		super().__init__(mode=mode, stream=stream, progress_stream=progress_stream)
		self.current_index = None
		self.current_total = None
		self.current_label = None

	def item_start(self, index, total, label=None):
		self.current_index = int(index)
		self.current_total = int(total)
		self.current_label = str(label or "")
		if self.bar is not None:
			self.bar = MilestoneBar(self.progress_stream, show_label=(self.mode == "compact"))
			self.bar.start()
		elif self.mode == "verbose":
			print("[{}/{}] {}".format(index, total, self.current_label), file=self.stream)

	def item_result(self, index, total, task, result, label=None):
		if self.bar is not None:
			self.bar.stop()
			self.bar = None
		prefix = "[{}/{}]".format(index, total)
		value = _result_text(task, result)
		cost = estimated_cost_text(result)
		print("{} {} ({})".format(prefix, value, cost), file=self.stream)
		self.stream.flush()

	def item_error(self, index, total, error, label=None):
		if self.bar is not None:
			self.bar.stop(complete=False)
			self.bar = None
		print("[{}/{}] FAILED: {}".format(index, total, error), file=self.stream)
		self.stream.flush()

	def job_summary(self, completed, total, estimated_cost_usd, failed=0, pricing_complete=True):
		cost = "${:.4f}".format(float(estimated_cost_usd)) if pricing_complete else "cost unknown"
		print("{}/{} completed | failed={} | estimated total {}".format(completed, total, failed, cost), file=self.stream)
		self.stream.flush()

	def stop(self):
		if self.bar is not None:
			self.bar.stop(complete=False)
			self.bar = None


def _result_text(task, value):
	task = str(task or "")
	translation = str(value.get("translation") or "").strip()
	if task in {"translate", "morph-translate", "reverse"} and translation:
		return translation
	segmented = str(value.get("segmented") or "").strip()
	annotation = str(value.get("annotation") or "").strip()
	if segmented and annotation:
		return "{} | {}".format(segmented, annotation)
	return annotation or segmented or "(no output)"


def estimated_cost_text(value):
	usage = value.get("api_usage") if isinstance(value, dict) else None
	if isinstance(usage, dict):
		if not usage.get("pricing_complete"):
			return "cost unknown"
		totals = usage.get("totals") or {}
		cost = totals.get("estimated_cost_usd")
		if cost is not None:
			return "${:.4f}".format(float(cost))
	if isinstance(value, dict) and value.get("estimated_cost_usd") is not None:
		return "${:.4f}".format(float(value.get("estimated_cost_usd") or 0.0))
	return "cost unknown"


def silent_translation_line(value):
	translation = str(value.get("translation") or "").strip()
	return "{} ({})".format(translation, estimated_cost_text(value))
