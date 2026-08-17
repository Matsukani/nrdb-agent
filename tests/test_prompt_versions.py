from nrdb_agent.annotator import instructions_for_version


def test_annotation_v1_preserves_original_prompt():
	instructions = instructions_for_version("annotation-v1")
	assert "Do not add ;cvb to a verb when the following verbal morphology is ipf" not in instructions
	assert "Treat red as weak evidence" not in instructions


def test_annotation_v2_adds_miyako_rules():
	instructions = instructions_for_version("annotation-v2")
	assert "Do not add ;cvb to a verb when the following verbal morphology is ipf" in instructions
	assert "Treat red as weak evidence" in instructions
