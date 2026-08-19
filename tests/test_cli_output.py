from io import StringIO

from nrdb_agent.cli_output import TranslationProgress, estimated_cost_text, silent_translation_line


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


def test_quiet_progress_keeps_only_major_milestones():
	stream = StringIO()
	progress = TranslationProgress("quiet", stream=stream)
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


def test_silent_progress_emits_no_trace_messages():
	stream = StringIO()
	progress = TranslationProgress("silent", stream=stream)
	progress("translate: Miyako -> Japanese")
	progress("  final: decision=proposed")
	assert stream.getvalue() == ""
