import json

from .annotator_v9 import AnnotationAgentV9
from .reverse_id_critic import _combined_mean_log_probability, _surprise_count
from .surface_critic import SurfaceModelCritic


FORWARD_ID_REVIEW_FORMAT = {
	"type": "json_schema",
	"name": "nrdb_forward_id_review",
	"strict": True,
	"schema": {
		"type": "object",
		"properties": {
			"annotation": {"type": "string"},
			"confidence": {"type": "number"},
			"note": {"type": "string"},
		},
		"required": ["annotation", "confidence", "note"],
		"additionalProperties": False,
	},
}


FORWARD_ID_REVIEW_INSTRUCTIONS = """You are the active grammatical critic for NRDB Miyako morphemic analysis.

You receive the OBSERVED Miyako transcription, its FIXED segmentation, a proposed NRDB annotation, and a soft ID-sequence review trained on human Miyako annotation.

Goal: make at most ONE NARROW grammatical-ID repair when the statistical critic exposes a genuinely implausible sequence.

Rules:
- The observed source transcription and segmentation are fixed evidence. NEVER rewrite, respell, resegment, or invent Miyako surface material.
- Preserve lexical/content IDs unless a grammatical conflation on that same segment must be adjusted.
- Concentrate only on grammar: case, topic/focus, TAM, converbs/dependent forms, clause linking, interrogative/final particles, and closely related control atoms.
- Use the critic's observed alternatives only as hints. Rare but valid constructions must survive.
- Preserve the exact phrase/segment structure: spaces and hyphens in the annotation must continue to align with the supplied fixed segmentation.
- If the proposed annotation is defensible, return it unchanged.
- Do not use tools and do not produce chain-of-thought.

Return exactly one JSON object:
{"annotation":"...","confidence":0.0,"note":"brief"}
"""


class ForwardCriticAnnotationAgent(AnnotationAgentV9):
	"""v9 forward analysis plus active ID repair and observed-surface compatibility criticism."""

	def __init__(self, *args, surface_model_path=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.surface_model_path = str(surface_model_path) if surface_model_path else None
		self.surface_critic = SurfaceModelCritic(surface_model_path) if surface_model_path else None

	def _surface_review(self, segmented, annotation, item, job):
		if self.surface_critic is None:
			return None
		dialects = [int(item["dialect_id"])]
		return self.surface_critic.review(segmented, annotation, dialects, int(job["annotation_schema_id"]))

	@staticmethod
	def _surface_better_or_equal(candidate, original):
		if candidate is None or original is None:
			return True
		if not candidate.get("valid_alignment"):
			return False
		orig_dis = int(original.get("strong_disagreements", 0))
		cand_dis = int(candidate.get("strong_disagreements", 0))
		if cand_dis < orig_dis:
			return True
		if cand_dis > orig_dis:
			return False
		orig_score = original.get("phonotactic_mean_log_probability")
		cand_score = candidate.get("phonotactic_mean_log_probability")
		if orig_score is None or cand_score is None:
			return True
		# Equal disagreement count: tolerate small score noise, reject material worsening.
		return float(cand_score) >= float(orig_score) - 0.25

	def _active_forward_critics(self, item, job, result):
		annotation = str(result.get("annotation") or "").strip()
		segmented = str(result.get("segmented") or "").strip()
		if not annotation or not segmented:
			return result

		evidence = result.setdefault("evidence", {})
		original_surface_review = self._surface_review(segmented, annotation, item, job)
		if original_surface_review is not None:
			evidence["forward_surface_review"] = original_surface_review
			evidence["forward_surface_model_path"] = self.surface_model_path
			self.progress("  forward-surface: phonotactic={:.3f} strong_disagreements={}".format(
				float(original_surface_review.get("phonotactic_mean_log_probability") or 0.0),
				int(original_surface_review.get("strong_disagreements", 0)),
			))

		if self.id_critic is None:
			return result
		initial_review = self.id_critic.review(annotation, int(job["annotation_schema_id"]))
		initial_compact = self.id_critic.compact(initial_review)
		initial_surprises = _surprise_count(initial_review)
		initial_mean = _combined_mean_log_probability(initial_review)
		evidence["forward_active_id_review"] = initial_compact
		self.progress("  forward-id: mean_logp={:.3f} strong_surprises={}".format(initial_mean, initial_surprises))
		if initial_surprises == 0:
			self.progress("  forward-id: no active repair needed")
			return result

		payload = {
			"source_miyako": item["text"],
			"fixed_segmented": segmented,
			"proposed_annotation": annotation,
			"id_sequence_review": initial_compact,
			"annotation_schema_id": int(job["annotation_schema_id"]),
		}
		self.progress("  forward-id: one narrow grammatical repair pass")
		response = self._create_response(
			[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
			FORWARD_ID_REVIEW_INSTRUCTIONS, tools=[], max_output_tokens=800, text_format=FORWARD_ID_REVIEW_FORMAT,
		)
		candidate = json.loads((response.output_text or "").strip())
		candidate_annotation = str(candidate.get("annotation") or "").strip()
		if not candidate_annotation or candidate_annotation == annotation:
			evidence["forward_active_id_review"]["revision_attempted"] = True
			evidence["forward_active_id_review"]["revision_accepted"] = False
			return result

		validation = self.nrdb.validate_analysis(item["text"], segmented, candidate_annotation)
		if not validation.get("valid"):
			self.progress("  forward-id: repair rejected by annotation validator")
			evidence["forward_active_id_review"]["revision_attempted"] = True
			evidence["forward_active_id_review"]["revision_accepted"] = False
			evidence["forward_active_id_review"]["candidate_validation"] = validation
			return result

		candidate_review = self.id_critic.review(candidate_annotation, int(job["annotation_schema_id"]))
		candidate_surprises = _surprise_count(candidate_review)
		candidate_mean = _combined_mean_log_probability(candidate_review)
		id_better = bool(
			candidate_surprises < initial_surprises
			or (candidate_surprises == initial_surprises and candidate_mean > initial_mean)
		)
		candidate_surface_review = self._surface_review(segmented, candidate_annotation, item, job)
		surface_ok = self._surface_better_or_equal(candidate_surface_review, original_surface_review)
		accept = bool(id_better and surface_ok)
		evidence["forward_active_id_review"]["revision_attempted"] = True
		evidence["forward_active_id_review"]["candidate"] = self.id_critic.compact(candidate_review)
		evidence["forward_active_id_review"]["candidate_validation"] = validation
		evidence["forward_active_id_review"]["revision_accepted"] = accept
		if candidate_surface_review is not None:
			evidence["forward_active_id_review"]["candidate_surface_review"] = candidate_surface_review

		if accept:
			self.progress("  forward-id: repair accepted surprises {}->{} mean_logp {:.3f}->{:.3f} surface_ok={}".format(
				initial_surprises, candidate_surprises, initial_mean, candidate_mean, surface_ok,
			))
			result["annotation"] = candidate_annotation
			result["confidence"] = min(
				float(result.get("confidence", 0.0)),
				max(0.0, min(1.0, float(candidate.get("confidence", 0.0)))),
			)
		else:
			self.progress("  forward-id: repair rejected id_better={} surface_ok={}".format(id_better, surface_ok))
		return result

	def annotate(self, item, job, morph_result):
		# Run only v9's annotation phase first so active critics operate BEFORE translation.
		result = self._annotation_phase_v9(item, job, morph_result)
		if result.get("decision") == "failed" or not result.get("annotation"):
			return result
		result = self._active_forward_critics(item, job, result)

		if job.get("produce_translation"):
			translation = self._translate_frozen_v7(item, job, result)
			result["trsl_ai"] = translation["trsl_ai"]
			result.setdefault("evidence", {})["translation"] = translation["translation_evidence"]
			result["evidence"]["translation"]["confidence"] = translation["confidence"]
		if not result.get("trsl_ai") or result.get("decision") == "failed":
			return result

		review = self._semantic_review(item, job, result)
		result.setdefault("evidence", {})["semantic_review"] = review
		result["evidence"]["shared_evidence"] = self._shared_evidence_compact()
		self.progress("  review-v9: action={} confidence={:.3f}".format(review["action"], review["confidence"]))
		if review["action"] != "revise":
			return result
		if review["segmented"] == result["segmented"] and review["annotation"] == result["annotation"]:
			result["evidence"]["semantic_review"]["action"] = "keep"
			return result
		validation = self.nrdb.validate_analysis(item["text"], review["segmented"], review["annotation"])
		result["evidence"]["semantic_review"]["validation"] = validation
		if not validation.get("valid"):
			self.progress("  review-v9: revision rejected by validator; keeping original")
			result["evidence"]["semantic_review"]["action"] = "keep"
			return result
		# Source transcription is immutable; semantic review may alter segmentation only
		# if it still reconstructs the same source and passes the existing validator.
		result["segmented"] = review["segmented"]
		result["annotation"] = review["annotation"]
		result["confidence"] = min(1.0, max(float(result.get("confidence", 0.0)), review["confidence"]))
		self.progress("  review-v9: revised annotation accepted")
		return result
