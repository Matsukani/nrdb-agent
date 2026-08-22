from nrdb_agent.execution import ExecutionRequest
from nrdb_agent.policy import forward_morph_policy


def test_execution_request_manifest_round_trip():
	policy = forward_morph_policy(
		review="agent", resegmentation=True,
		max_segmentation_candidates=4, morphology_source="predict", task="morph-translate",
	)
	request = ExecutionRequest(
		item={"sentence_id": 7, "dialect_id": 19, "text": "aga"},
		task="morph-translate", annotation_schema_id=2, region="宮古",
		model_name="gpt-5.6-terra", semantic_feedback="generated",
		use_constructions=True, use_licensed_forms=True, nrdb_evidence="enabled", morphology_source="predict",
		morph_policy=policy,
	)
	restored = ExecutionRequest.from_manifest(request.manifest())
	assert restored == request


def test_execution_request_rejects_unknown_format():
	try:
		ExecutionRequest.from_manifest({"format": "unknown"})
	except ValueError as error:
		assert "invalid execution request" in str(error)
	else:
		raise AssertionError("unknown execution request format was accepted")


def test_execution_request_rejects_nrdb_dependent_features_when_evidence_is_disabled():
	try:
		ExecutionRequest(
			item={"sentence_id": 7, "dialect_id": 19, "text": "aga"},
			task="translate", annotation_schema_id=2, region="宮古",
			nrdb_evidence="none", use_constructions=True,
		)
	except ValueError as error:
		assert "require --nrdb-evidence enabled" in str(error)
	else:
		raise AssertionError("NRDB constructions were accepted with NRDB evidence disabled")


def test_morphology_none_requires_translation_only_and_no_review():
	policy = forward_morph_policy(review="none", morphology_source="none", task="translate")
	request = ExecutionRequest(
		item={"sentence_id": 7, "dialect_id": 19, "text": "aga"},
		task="translate", annotation_schema_id=2, region="宮古",
		semantic_feedback="none", morphology_source="none", morph_policy=policy,
	)
	assert request.morphology_source == "none"
	try:
		ExecutionRequest(
			item=request.item, task="morph", annotation_schema_id=2, region="宮古",
			semantic_feedback="none", morphology_source="none", morph_policy=policy,
		)
	except ValueError as error:
		assert "translation-only" in str(error)
	else:
		raise AssertionError("morphology_source=none was accepted for a morphology task")
