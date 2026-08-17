from nrdb_agent.annotator_v8 import AnnotationAgentV8, REVIEW_INSTRUCTIONS


class FakeNrdb:
	def __init__(self, valid):
		self.valid = valid
		self.validated = []

	def validate_analysis(self, text, segmented, annotation):
		self.validated.append((text, segmented, annotation))
		return {"valid": self.valid}


def test_v8_review_instructions_allow_targeted_segmentation_revision():
	assert "segmentation/annotation" in REVIEW_INSTRUCTIONS or "segmentation and annotation" in REVIEW_INSTRUCTIONS
	assert "Preserve all unaffected segmentation and IDs exactly" in REVIEW_INSTRUCTIONS


def test_v8_revised_segmentation_is_validator_gated():
	nrdb = FakeNrdb(valid=False)
	agent = AnnotationAgentV8(nrdb, "test-model", client=object())
	item = {"text": "abc"}
	result = {
		"segmented": "a-bc",
		"annotation": "A-B",
		"confidence": 0.8,
		"evidence": {"semantic_review": {}},
	}
	review = {"segmented": "ab-c", "annotation": "AB-C", "confidence": 0.9}
	validation = nrdb.validate_analysis(item["text"], review["segmented"], review["annotation"])
	assert validation["valid"] is False
	assert nrdb.validated == [("abc", "ab-c", "AB-C")]
