from io import StringIO

from nrdb_agent import cli_output
from nrdb_agent.cli_output import TranslationProgress, WorkflowProgress, estimated_cost_text, format_elapsed, silent_translation_line


def _value():
	return {
		"translation": "東はどこにあるのか。",
		"api_usage": {
			"pricing_complete": True,
			"totals": {"estimated_cost_usd": 0.09012},
		},
	}


def test_silent_translation_line_includes_estimated_cost():
	assert silent_translation_line(_value()) == "東はどこにあるのか。 ($0.0901)"
	assert estimated_cost_text(_value()) == "$0.0901"


def test_elapsed_formatting():
	assert format_elapsed(12.34) == "12.3s"
	assert format_elapsed(62) == "1m02s"
	assert format_elapsed(3661) == "1h01m01s"


def test_one_off_translation_reports_observed_cost_and_elapsed_time(monkeypatch):
	clock = iter([100.0, 112.3])
	monkeypatch.setattr(cli_output.time, "monotonic", lambda: next(clock))
	stream = StringIO()
	progress = TranslationProgress("quiet", stream=stream, progress_stream=StringIO())
	progress.start()
	progress("  API usage: requests=6 input=100 cached=20 output=30 reasoning=10 estimated_cost=$0.0529")
	progress.stop()
	value = stream.getvalue()
	assert "API usage:" in value
	assert "complete | $0.0529 | 12.3s" in value


def test_default_progress_is_quiet_and_keeps_only_major_milestones():
	stream = StringIO()
	progress = TranslationProgress(stream=stream)
	progress("  morph: analyze")
	progress("  morph: model=miyako-65k")
	progress("    -> corpus_examples(label=x)")
	progress("  forward-v9: uncertainty triage id_hotspots=1 uncertain_surfaces=2")
	progress("  final: decision=proposed confidence=0.9")
	value = stream.getvalue()
	assert "morph: analyze" not in value
	assert "corpus_examples" not in value
	assert "morph: model=miyako-65k" in value
	assert "forward-v9: uncertainty triage" in value
	assert "final: decision=proposed" in value


def test_verbose_progress_keeps_full_trace():
	stream = StringIO()
	progress = TranslationProgress("verbose", stream=stream)
	progress("    -> corpus_examples(label=x)")
	assert "corpus_examples(label=x)" in stream.getvalue()


def test_silent_progress_tracks_milestones_without_trace_output():
	trace = StringIO()
	bar_stream = StringIO()
	progress = TranslationProgress("silent", stream=trace, progress_stream=bar_stream)
	progress("translate: Miyako -> Japanese")
	progress("  morph: analyze")
	progress("  morph: model=miyako-65k")
	progress("  final: decision=proposed")
	assert trace.getvalue() == ""
	assert progress.bar._completed == 3
	assert progress.bar._label == "annotation finalized"


def test_compact_progress_tracks_current_milestone_label():
	trace = StringIO()
	bar_stream = StringIO()
	progress = TranslationProgress("compact", stream=trace, progress_stream=bar_stream)
	progress("translate: Japanese -> Miyako")
	progress("  reverse-v1: Japanese -> Miyako IDs (batch lexical triage)")
	progress("  id-model: mean_logp=-3 strong_surprises=1")
	progress.bar._render("/")
	assert trace.getvalue() == ""
	assert "grammatical critic" in bar_stream.getvalue()
	assert "===" in bar_stream.getvalue()


def test_workflow_progress_streams_translation_cost_and_elapsed_time():
	stream = StringIO()
	progress = WorkflowProgress("quiet", stream=stream, progress_stream=StringIO())
	progress.item_start(2, 100, "sentence 15456")
	progress.item_result(2, 100, "translate", _value(), "sentence 15456")
	progress.job_summary(100, 100, 0.8427, failed=0, pricing_complete=True)
	value = stream.getvalue()
	assert "[2/100] 東はどこにあるのか。 ($0.0901 | " in value
	assert "100/100 completed | failed=0 | estimated total $0.8427 | " in value


def test_compact_workflow_result_shows_source_then_translation_cost_and_time():
	stream = StringIO()
	value = _value()
	value["source"] = "agarɿ wa nzaːn aɿga"
	progress = WorkflowProgress("compact", stream=stream, progress_stream=StringIO())
	progress.item_result(7, 20, "translate", value, "sentence 77")
	lines = stream.getvalue().splitlines()
	assert lines[0] == "[7/20] agarɿ wa nzaːn aɿga"
	assert lines[1].strip().startswith("→ 東はどこにあるのか。 ($0.0901 | ")
