from nrdb_agent.annotator import instructions_for_version


def test_annotation_v1_preserves_original_prompt():
	instructions = instructions_for_version("annotation-v1")
	assert "should normally carry ;cvb before ipf" not in instructions
	assert "second occurrence is almost certainly the reduplication marker red" not in instructions


def test_annotation_v2_adds_corrected_miyako_rules():
	instructions = instructions_for_version("annotation-v2")
	assert "should normally carry ;cvb before ipf" in instructions
	assert "second occurrence is almost certainly the reduplication marker red" in instructions
