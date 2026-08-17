from nrdb_agent.annotator import instructions_for_version


def test_annotation_v1_preserves_original_prompt():
	instructions = instructions_for_version("annotation-v1")
	assert "should normally carry ;cvb before ipf" not in instructions
	assert "second occurrence is almost certainly the reduplication marker red" not in instructions
	assert "Annotation and translation are separate phases" not in instructions
	assert "afn may only follow an adjectival expression" not in instructions
	assert "surface segment dui is never the focus marker foc" not in instructions


def test_annotation_v2_adds_corrected_miyako_rules():
	instructions = instructions_for_version("annotation-v2")
	assert "should normally carry ;cvb before ipf" in instructions
	assert "second occurrence is almost certainly the reduplication marker red" in instructions
	assert "Annotation and translation are separate phases" not in instructions
	assert "afn may only follow an adjectival expression" not in instructions
	assert "surface segment dui is never the focus marker foc" not in instructions


def test_annotation_v3_freezes_annotation_before_translation():
	instructions = instructions_for_version("annotation-v3")
	assert "should normally carry ;cvb before ipf" in instructions
	assert "second occurrence is almost certainly the reduplication marker red" in instructions
	assert "Annotation and translation are separate phases" in instructions
	assert "Return trsl_ai as an empty string" in instructions
	assert "afn may only follow an adjectival expression" not in instructions
	assert "surface segment dui is never the focus marker foc" not in instructions


def test_annotation_v4_adds_afn_constraint_and_keeps_v3_behavior():
	instructions = instructions_for_version("annotation-v4")
	assert "should normally carry ;cvb before ipf" in instructions
	assert "second occurrence is almost certainly the reduplication marker red" in instructions
	assert "Annotation and translation are separate phases" in instructions
	assert "afn may only follow an adjectival expression" in instructions
	assert "Do not analyze a form as afn after a non-adjectival expression" in instructions
	assert "surface segment dui is never the focus marker foc" not in instructions


def test_annotation_v5_adds_dui_and_phrase_local_red_constraints():
	instructions = instructions_for_version("annotation-v5")
	assert "surface segment dui is never the focus marker foc" in instructions
	assert "strictly phrase-local" in instructions
	assert "Never use a repeated element in another phrase elsewhere in the utterance as evidence for red" in instructions
	assert "afn may only follow an adjectival expression" in instructions
	assert "Annotation and translation are separate phases" in instructions


def test_annotation_v6_replaces_hard_dui_rule_with_regional_form_support():
	instructions = instructions_for_version("annotation-v6")
	assert "surface segment dui is never the focus marker foc" not in instructions
	assert "form_id_support(X, Y)" in instructions
	assert "current utterance's dialect region" in instructions
	assert "Zero support is evidence against a candidate, not an absolute prohibition" in instructions
	assert "surface_total" in instructions
	assert "candidate_count" in instructions
	assert "strictly phrase-local" in instructions
	assert "afn may only follow an adjectival expression" in instructions
	assert "Annotation and translation are separate phases" in instructions
