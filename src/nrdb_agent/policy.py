import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path


MORPH_REVIEW_MODES = {"none", "agent"}


def _artifact(path):
	if not path:
		return None
	value = Path(path).expanduser()
	resolved = str(value.resolve())
	digest = None
	if value.is_file():
		hasher = hashlib.sha256()
		with value.open("rb") as handle:
			for block in iter(lambda: handle.read(1024 * 1024), b""):
				hasher.update(block)
		digest = hasher.hexdigest()
	return {"path": resolved, "sha256": digest}


@dataclass(frozen=True)
class ForwardMorphPolicy:
	review: str = "agent"
	resegmentation: bool = False
	id_model_path: str | None = None
	surface_model_path: str | None = None
	max_segmentation_candidates: int = 4

	def __post_init__(self):
		if self.review not in MORPH_REVIEW_MODES:
			raise ValueError("invalid morph review mode: {}".format(self.review))
		if int(self.max_segmentation_candidates) < 1 or int(self.max_segmentation_candidates) > 4:
			raise ValueError("max segmentation candidates must be between 1 and 4")
		object.__setattr__(self, "_manifest_cache", {
			"format": "nrdb-agent.forward-morph-policy.v1",
			"review": self.review,
			"resegmentation": bool(self.resegmentation),
			"max_segmentation_candidates": int(self.max_segmentation_candidates),
			"id_model": _artifact(self.id_model_path),
			"surface_model": _artifact(self.surface_model_path),
		})

	@property
	def agent_review(self):
		return self.review == "agent"

	def validate(self, morphology_source="predict", task="morph"):
		source = str(morphology_source or "predict")
		frozen = source in {"existing", "gold"}
		if source == "none" and self.agent_review:
			raise ValueError("morphology_source=none requires --morph-review none")
		if frozen and self.agent_review:
			raise ValueError("gold/existing morphology is frozen and requires --morph-review none")
		if frozen and self.resegmentation:
			raise ValueError("--resegmentation cannot be used with gold/existing morphology")
		if frozen and (self.id_model_path or self.surface_model_path):
			raise ValueError("critic models cannot be used with gold/existing morphology")
		if not self.agent_review and (self.resegmentation or self.id_model_path or self.surface_model_path):
			raise ValueError("resegmentation and critic models require --morph-review agent")
		if str(task or "morph") == "reverse" and self.resegmentation:
			raise ValueError("--resegmentation applies only to forward morphology")
		return self

	def manifest(self):
		return deepcopy(self._manifest_cache)

	def json(self):
		return json.dumps(self.manifest(), ensure_ascii=False, separators=(",", ":"))


def forward_morph_policy(review=None, resegmentation=False, id_model=None, surface_model=None,
	max_segmentation_candidates=4, morphology_source="predict", task="morph"):
	review = review or ("none" if str(morphology_source or "predict") in {"none", "existing", "gold"} else "agent")
	return ForwardMorphPolicy(
		review=str(review), resegmentation=bool(resegmentation),
		id_model_path=str(id_model) if id_model else None,
		surface_model_path=str(surface_model) if surface_model else None,
		max_segmentation_candidates=int(max_segmentation_candidates),
	).validate(morphology_source=morphology_source, task=task)


def policy_from_manifest(value, fallback_review="agent"):
	if not isinstance(value, dict):
		return forward_morph_policy(review=fallback_review)
	def artifact_path(name):
		artifact = value.get(name)
		if not isinstance(artifact, dict):
			return artifact
		path = artifact.get("path")
		expected = artifact.get("sha256")
		if path and expected:
			actual = _artifact(path)
			if actual.get("sha256") != expected:
				raise ValueError("{} artifact hash mismatch: {}".format(name, path))
		return path
	id_model = artifact_path("id_model")
	surface_model = artifact_path("surface_model")
	return forward_morph_policy(
		review=value.get("review") or fallback_review,
		resegmentation=bool(value.get("resegmentation")),
		id_model=id_model, surface_model=surface_model,
		max_segmentation_candidates=value.get("max_segmentation_candidates") or 4,
	)
