from copy import deepcopy

from .annotator_v9 import AnnotationAgentV9


class TaskAwareAnnotationAgent(AnnotationAgentV9):
	"""Expose orthogonal morphology, human-translation review, and frozen translation phases."""

	def _apply_review(self, item, result, review, evidence_key):
		result.setdefault("evidence", {})[evidence_key] = review
		self.progress("  review-v9: action={} confidence={:.3f}".format(review["action"], review["confidence"]))
		if review["action"] != "revise":
			return result
		if review["segmented"] == result["segmented"] and review["annotation"] == result["annotation"]:
			result["evidence"][evidence_key]["action"] = "keep"
			result["evidence"][evidence_key]["note"] = "Revision requested but analysis was unchanged; kept original."
			return result
		validation = self.nrdb.validate_analysis(item["text"], review["segmented"], review["annotation"])
		result["evidence"][evidence_key]["validation"] = validation
		if not validation.get("valid"):
			self.progress("  review-v9: revision rejected by validator; keeping original")
			result["evidence"][evidence_key]["action"] = "keep"
			result["evidence"][evidence_key]["note"] = "Proposed revision failed structural validation; original kept."
			return result
		self.progress("  review-v9: revised annotation accepted")
		result["segmented"] = review["segmented"]
		result["annotation"] = review["annotation"]
		result["confidence"] = min(1.0, max(float(result.get("confidence", 0.0)), review["confidence"]))
		return result

	def _review_against_human_translation(self, item, job, result):
		human_translation = str(item.get("translation_jp") or "").strip()
		if not human_translation:
			return result
		review_input = deepcopy(result)
		review_input["trsl_ai"] = human_translation
		review_input.setdefault("evidence", {})["translation"] = {
			"source": "human",
			"note": "Existing human Japanese translation supplied as semantic evidence for morphology review.",
		}
		self.progress("  review-v9: human translation consistency review")
		review = self._semantic_review(item, job, review_input)
		result = self._apply_review(item, result, review, "human_translation_review")
		result.setdefault("evidence", {})["shared_evidence"] = self._shared_evidence_compact()
		return result

	def annotate(self, item, job, morph_result):
		policy = str(job.get("translation_evidence") or "ignore")
		if policy not in {"ignore", "use", "required"}:
			raise ValueError("invalid translation_evidence policy: {}".format(policy))
		human_translation = str(item.get("translation_jp") or "").strip()
		if policy == "required" and not human_translation:
			raise ValueError("human translation is required for this morphology task")

		result = self._annotation_phase_v9(item, job, morph_result)
		human_reviewed = False
		if policy in {"use", "required"} and human_translation and result.get("annotation") and result.get("decision") != "failed":
			result = self._review_against_human_translation(item, job, result)
			human_reviewed = True

		if job.get("produce_translation") and result.get("annotation") and result.get("decision") != "failed":
			translation_item = dict(item)
			# Existing Japanese is evidence for morphology only. Never expose the target
			# translation to the generation phase itself.
			translation_item["translation_jp"] = None
			translation = self._translate_frozen_v7(translation_item, job, result)
			result["trsl_ai"] = translation["trsl_ai"]
			result.setdefault("evidence", {})["translation"] = translation["translation_evidence"]
			result["evidence"]["translation"]["confidence"] = translation["confidence"]
		else:
			result["trsl_ai"] = ""

		# If a human translation already constrained morphology, do not immediately
		# re-open the same morphology using the model-generated Japanese translation.
		if not human_reviewed and result.get("trsl_ai") and result.get("decision") != "failed":
			review = self._semantic_review(item, job, result)
			result = self._apply_review(item, result, review, "semantic_review")
			result.setdefault("evidence", {})["shared_evidence"] = self._shared_evidence_compact()
		return result

	def translate_frozen(self, item, job, segmented, annotation, confidence=1.0):
		segmented = str(segmented or "").strip()
		annotation = str(annotation or "").strip()
		if not segmented or not annotation:
			raise ValueError("frozen translation requires existing segmentation and annotation")
		validation = self.nrdb.validate_analysis(item["text"], segmented, annotation)
		if not validation.get("valid"):
			raise ValueError("existing morphology is not structurally valid: {}".format(validation.get("error") or "validation failed"))
		base = {
			"segmented": segmented,
			"annotation": annotation,
			"trsl_ai": "",
			"decision": "proposed",
			"confidence": float(confidence),
			"evidence": {"existing_morphology": {"frozen": True, "validation": validation}},
		}
		translation_item = dict(item)
		translation_item["translation_jp"] = None
		translation_job = dict(job)
		translation_job["produce_translation"] = True
		translation = self._translate_frozen_v7(translation_item, translation_job, base)
		base["trsl_ai"] = translation["trsl_ai"]
		base["confidence"] = translation["confidence"]
		base["evidence"]["translation"] = translation["translation_evidence"]
		base["evidence"]["translation"]["confidence"] = translation["confidence"]
		return base
