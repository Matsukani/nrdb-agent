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
		use_constructions=True, use_licensed_forms=True, morphology_source="predict",
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
