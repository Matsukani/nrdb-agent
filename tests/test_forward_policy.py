import pytest

from nrdb_agent.annotator_v9 import AnnotationAgentV9
from nrdb_agent.policy import forward_morph_policy, policy_from_manifest


def test_policy_hashes_and_verifies_critic_artifacts(tmp_path):
	path = tmp_path / "critic.json"
	path.write_text('{"model":"v1"}', encoding="utf-8")
	manifest = forward_morph_policy(id_model=path).manifest()
	assert manifest["id_model"]["sha256"]
	assert policy_from_manifest(manifest).id_model_path == str(path.resolve())
	path.write_text('{"model":"v2"}', encoding="utf-8")
	with pytest.raises(ValueError, match="artifact hash mismatch"):
		policy_from_manifest(manifest)


def test_policy_rejects_nonorthogonal_combinations():
	with pytest.raises(ValueError, match="require --morph-review agent"):
		forward_morph_policy(review="none", resegmentation=True)
	with pytest.raises(ValueError, match="gold/existing"):
		forward_morph_policy(resegmentation=True, morphology_source="gold")
	with pytest.raises(ValueError, match="critic models"):
		forward_morph_policy(surface_model="surface.json", morphology_source="existing")


def test_host_rejects_an_untested_boundary_change():
	agent = AnnotationAgentV9.__new__(AnnotationAgentV9)
	agent.morph_policy = forward_morph_policy(resegmentation=True)
	agent._tested_segmentations = {"fa-i-ra"}
	agent.progress = lambda _message: None
	result = agent._enforce_forward_morph_policy(
		{"segmented": "fai-ra", "annotation": "食kv-fp:ra"},
		{"segmented": "fa-ira", "annotation": "X-Y", "evidence": {}, "confidence": 0.9},
		"test",
	)
	assert result["segmented"] == "fai-ra"
	assert result["annotation"] == "食kv-fp:ra"
	assert result["evidence"]["forward_morph_policy"][-1]["accepted"] is False
