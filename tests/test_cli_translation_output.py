from nrdb_agent.cli import _print_translation


def test_quiet_translation_output_shows_raw_morph_before_agent_result(capsys):
	_print_translation({
		"source": "mmagimunura",
		"morph_baseline": {
			"source": "nrdb-morph", "segmented": "mmagi-munu-ra", "annotation": "?-物mn-fp:ra",
		},
		"segmented": "mma-gi-munu-ra", "annotation": "旨ua-sem1-物mn-fp:ra",
		"translation": "おいしそうなものだね。", "confidence": 0.94, "decision": "proposed",
	})
	lines = capsys.readouterr().out.splitlines()
	assert lines[1] == "morph seg.:  mmagi-munu-ra"
	assert lines[2] == "morph ann.:  ?-物mn-fp:ra"
	assert lines[3] == "agent seg.:  mma-gi-munu-ra"
	assert lines[4] == "agent ann.:  旨ua-sem1-物mn-fp:ra"
