from nrdb_agent.licensed_agent import LicensedTaskAwareAnnotationAgent, morph_surfaces


class FakeNrdb:
	def __init__(self, valid=True):
		self.valid = valid
		self.validations = []

	def validate_analysis(self, text, segmented, annotation):
		self.validations.append((text, segmented, annotation))
		return {"valid": self.valid}


def make_agent(valid=True):
	agent = LicensedTaskAwareAnnotationAgent.__new__(LicensedTaskAwareAnnotationAgent)
	agent.nrdb = FakeNrdb(valid=valid)
	return agent


def licensed_match(identifier=17409, annotation="旨ua-sem:1", segmented="mma-gi", path="exact_surface", scope="same_dialect"):
	return {
		"id": identifier, "matched_surface": "mmagi", "form_romaji_seg": segmented,
		"annotation": annotation, "retrieval_path": path, "scope": scope,
		"occurrences": [{"start": 18, "end": 23}],
	}


def baseline(*matches):
	return {
		"segmented": "fai-ja mjuː-n-suga mmagi-munu-ra",
		"annotation": "食kv;cvb-top:1 見mv-neg-adv ?-物mn-fp:ra",
		"phrases": [{"raw": "mmagimunura", "segments": [{"surface": "mmagi", "label": "?"}]}],
		"licensed_realizations": {"matches": list(matches)},
	}


def test_morph_surfaces_supplies_decoder_segments_for_exact_lookup():
	assert morph_surfaces(baseline()) == ["fai", "ja", "mjuː", "n", "suga", "mmagi", "munu", "ra"]


def test_unique_same_dialect_exact_match_repairs_unknown_and_validates():
	agent = make_agent()
	result, audit = agent._repair_unknowns("faiːja mjuːnsuga mmagimunura", baseline(licensed_match()))
	assert result["segmented"] == "fai-ja mjuː-n-suga mma-gi-munu-ra"
	assert result["annotation"] == "食kv;cvb-top:1 見mv-neg-adv 旨ua-sem:1-物mn-fp:ra"
	assert audit["applied_ids"] == [17409]
	assert agent.nrdb.validations == [(
		"faiːja mjuːnsuga mmagimunura",
		"fai-ja mjuː-n-suga mma-gi-munu-ra",
		"食kv;cvb-top:1 見mv-neg-adv 旨ua-sem:1-物mn-fp:ra",
	)]
	assert result["phrases"] == []


def test_ambiguous_exact_licensed_analyses_are_not_applied():
	agent = make_agent()
	result, audit = agent._repair_unknowns(
		"faiːja mjuːnsuga mmagimunura",
		baseline(licensed_match(), licensed_match(17410, annotation="別ua-sem:2")),
	)
	assert result["annotation"] == baseline()["annotation"]
	assert audit["applied_ids"] == []
	assert audit["candidates"][0]["status"] == "ambiguous"
	assert agent.nrdb.validations == []


def test_containment_match_is_evidence_but_not_an_automatic_repair():
	agent = make_agent()
	result, audit = agent._repair_unknowns(
		"faiːja mjuːnsuga mmagimunura",
		baseline(licensed_match(path="text_containment")),
	)
	assert result["annotation"] == baseline()["annotation"]
	assert audit["applied_ids"] == []
	assert agent.nrdb.validations == []
