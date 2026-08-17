from nrdb_agent.annotator import TRANSLATION_INSTRUCTIONS, parse_translation_json


def test_translation_instructions_freeze_annotation_and_prioritize_grounding():
	assert "FROZEN" in TRANSLATION_INSTRUCTIONS
	assert "Dictionary-attested meanings outrank guesses" in TRANSLATION_INSTRUCTIONS
	assert "corpus_examples primarily for constructional or grammatical interpretation" in TRANSLATION_INSTRUCTIONS
	assert "Do not look up every ID" in TRANSLATION_INSTRUCTIONS


def test_parse_translation_json_preserves_audit_evidence():
	result = parse_translation_json('{"trsl_ai":"畑へ行った。","confidence":0.91,"translation_evidence":{"dictionary_ids":["畑pn2","行iv"],"example_sentence_ids":[42],"ungrounded_ids":[],"note":"grounded"}}')
	assert result["trsl_ai"] == "畑へ行った。"
	assert result["confidence"] == 0.91
	assert result["translation_evidence"]["dictionary_ids"] == ["畑pn2", "行iv"]
	assert result["translation_evidence"]["example_sentence_ids"] == [42]
