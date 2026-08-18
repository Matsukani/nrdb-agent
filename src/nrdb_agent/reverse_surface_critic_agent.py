import json

from .reverse_id_critic import IdCriticSyntaxAwareReverseSurfaceAgent
from .reverse_surface_agent import SURFACE_FORMAT
from .reverse_surface_syntax_agent import surface_alignment_error
from .surface_critic import SurfaceModelCritic


SURFACE_REVIEW_INSTRUCTIONS = """You are the final surface-only reviewer for NRDB Japanese-to-Miyako reverse translation.

The Miyako NRDB ID annotation is FROZEN and must not be revised. You receive an initial segmented trsc2 realization plus soft statistical evidence from an nrdb-morph surface model trained on held-out-safe human annotations.

NRDB realization syntax is binding:
- SPACE separates phrases.
- HYPHEN (-) separates independently realized annotation segments.
- SEMICOLON (;) conflates multiple annotation atoms inside ONE annotation segment and therefore ONE surface segment.
- Never split a semicolon-conflated annotation segment into multiple hyphenated surface morphemes.
- `消kv;cvb` is one realization slot, not `消kv` plus a separate `cvb` slot.
- `眠nv;cvb-foc-ipf` has three realization slots: `[眠nv;cvb] [foc] [ipf]`.

The statistical model provides context-conditioned surface suggestions and a dialect phonotactic score. Treat it as strong but non-absolute evidence:
- Correct clear morphophonological/allomorphic errors when a requested-dialect form is strongly preferred in the exact left/right context.
- Prefer directly attested requested-dialect realizations over an improvised form when the model strongly disagrees.
- For a conflated label such as `消kv;cvb`, prefer the model's attested COMPLETE-SEGMENT realization; do not independently realize its atoms.
- Do not blindly replace an acceptable idiomatic form merely because another attested form has a slightly higher score.
- Do not change lexical or grammatical IDs, phrase structure, semantic content, or switch an ordinary local ID into n: or vice versa.
- Keep n: Japanese lexical material in the Japanese layer; the critic intentionally treats n: interiors as opaque to Miyako phonotactics.
- Return exactly the same number of phrases and hyphen-delimited surface segments as the frozen annotation.
- Return trsc2-style romanization only.
- Do not produce chain-of-thought.

Return exactly one JSON object:
{"segmented":"...","confidence":0.0,"evidence":{"note":"brief","ids_realized":[],"fallback_ids":[],"example_sentence_ids":[]}}
"""


class SurfaceCriticReverseAgent(IdCriticSyntaxAwareReverseSurfaceAgent):
	def __init__(self, *args, surface_model_path=None, id_model_path=None, **kwargs):
		super().__init__(*args, id_model_path=id_model_path, **kwargs)
		if not surface_model_path:
			raise ValueError("surface critic requires surface_model_path")
		self.surface_critic = SurfaceModelCritic(surface_model_path)
		self.surface_model_path = str(surface_model_path)

	def _review_surface(self, item, job, result):
		dialect_ids = job.get("target_dialect_ids") or [int(item["dialect_id"])]
		annotation = result.get("annotation") or ""
		initial = result.get("segmented") or ""
		review = self.surface_critic.review(initial, annotation, dialect_ids, int(job["annotation_schema_id"]))
		self.progress("  surface-model: phonotactic={:.3f} strong_disagreements={}".format(
			float(review.get("phonotactic_mean_log_probability") or 0.0), review.get("strong_disagreements", 0),
		))
		for diagnostic in review.get("diagnostics", []):
			if not diagnostic.get("strong_disagreement"):
				continue
			suggestions = ", ".join(
				"{}({:.2f})".format(value.get("form"), float(value.get("score") or 0.0))
				for value in diagnostic.get("suggestions", [])[:3]
			)
			self.progress("    surface-model: {} after {!r} before {!r} -> [{}] current={!r} gap={}".format(
				diagnostic.get("label"), diagnostic.get("previous_surface"), diagnostic.get("next_label"), suggestions,
				diagnostic.get("generated_form"),
				"n/a" if diagnostic.get("score_gap") is None else "{:.2f}".format(float(diagnostic["score_gap"])),
			))
		if not review.get("valid_alignment") or not review.get("strong_disagreements"):
			return result, review

		payload = {
			"japanese": str(item.get("translation_jp") or "").strip(),
			"frozen_annotation": annotation,
			"initial_segmented": initial,
			"target_dialect_ids": dialect_ids,
			"surface_model_review": review,
		}
		self.progress("  surface-model: one soft revision pass")
		response = self._create_response(
			[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
			SURFACE_REVIEW_INSTRUCTIONS, tools=[], max_output_tokens=1000, text_format=SURFACE_FORMAT,
		)
		candidate = self._parse_surface(response.output_text)
		candidate_syntax_error = surface_alignment_error(candidate.get("segmented"), annotation)
		if candidate_syntax_error:
			self.progress("  surface-model: revision rejected; annotation syntax violation: {}".format(candidate_syntax_error))
			review["revision_attempted"] = True
			review["revision_accepted"] = False
			review["candidate_syntax_error"] = candidate_syntax_error
			return result, review
		candidate_review = self.surface_critic.review(
			candidate["segmented"], annotation, dialect_ids, int(job["annotation_schema_id"]),
		)
		initial_disagreements = int(review.get("strong_disagreements", 0))
		candidate_disagreements = int(candidate_review.get("strong_disagreements", initial_disagreements))
		initial_score = float(review.get("phonotactic_mean_log_probability") or -1e9)
		candidate_score = float(candidate_review.get("phonotactic_mean_log_probability") or -1e9)
		accept = bool(
			candidate_review.get("valid_alignment") and (
				candidate_disagreements < initial_disagreements or candidate_score > initial_score
			)
		)
		if accept:
			self.progress("  surface-model: revision accepted disagreements {}->{} phonotactic {:.3f}->{:.3f}".format(
				initial_disagreements, candidate_disagreements, initial_score, candidate_score,
			))
			result["segmented"] = candidate["segmented"]
			result["confidence"] = min(float(result.get("confidence", 0.0)), float(candidate.get("confidence", 0.0)))
		else:
			self.progress("  surface-model: revision rejected; keeping initial surface")
		review["revision_attempted"] = True
		review["revision_accepted"] = accept
		review["candidate_review"] = candidate_review
		return result, review

	def annotate(self, item, job, morph_result=None):
		result = super().annotate(item, job, morph_result)
		if result.get("decision") == "failed" or not result.get("segmented") or not result.get("annotation"):
			return result
		result, review = self._review_surface(item, job, result)
		result.setdefault("evidence", {})["surface_model_review"] = review
		result["evidence"]["surface_model_path"] = self.surface_model_path
		return result
