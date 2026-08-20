from copy import deepcopy

from .annotator_v9 import AnnotationAgentV9


SEMANTIC_FEEDBACK_MODES = {"none", "generated", "existing", "auto"}


class TaskAwareAnnotationAgent(AnnotationAgentV9):
	"""Expose orthogonal morphology, semantic feedback, and translation-output phases."""
	# AnnotationAgentV9's optimized semantic-review loop is bounded separately
	# from the forward annotation loop. Keep this explicit here because v8 exposes
	# max_review_evidence_calls, not a max_review_rounds attribute.
	max_review_rounds = 4

	def _shared_evidence_compact(self):
		"""Return the cached forward evidence in a compact JSON-safe review payload."""
		lookup = []
		for label, result in self._shared_evidence.get("lookup", {}).items():
			lookup.append({"label": label, "result": self._compact_v9("lookup_id", result)})
		corpus = []
		for label, result in self._shared_evidence.get("corpus", {}).items():
			corpus.append({"label": label, "result": self._compact_v9("corpus_examples", result)})
		forms = []
		for key, result in self._shared_evidence.get("form", {}).items():
			surface, candidate = (key.split("\t", 1) + [""])[:2]
			forms.append({
				"surface": surface,
				"candidate_id": candidate,
				"result": self._compact_v9("form_id_support", result),
			})
		return {"lookup": lookup, "corpus": corpus, "form": forms}

	def _review_query_already_known(self, name, arguments):
		if name == "ground_lexical_ids":
			labels = [str(value or "").strip() for value in arguments.get("labels", [])]
			labels = [value for value in labels if value]
			return bool(labels) and all(value in self._shared_evidence.get("lookup", {}) for value in labels)
		if name == "corpus_examples":
			label = str(arguments.get("label") or "").strip()
			return bool(label) and label in self._shared_evidence.get("corpus", {})
		if name == "form_id_support":
			surface = str(arguments.get("surface") or "").strip()
			candidate = str(arguments.get("candidate_id") or "").strip()
			return bool(surface and candidate) and "{}\t{}".format(surface, candidate) in self._shared_evidence.get("form", {})
		return False

	def _v8_review_tool_result(self, name, arguments, item, schema_id):
		"""Bridge v9's shared-evidence review onto the actual v8 review tool API."""
		result = self._review_tool_result(name, arguments, item, schema_id)
		if name == "corpus_examples":
			label = str(arguments.get("label") or "").strip()
			if label:
				self._cache_corpus(label, result)
		elif name == "form_id_support":
			surface = str(arguments.get("surface") or "").strip()
			candidate = str(arguments.get("candidate_id") or "").strip()
			if surface and candidate:
				self._cache_form(surface, candidate, result)
		# ground_lexical_ids already populates the lookup cache through v9.
		return result

	def _compact_review_result(self, name, result):
		return self._review_compact(name, result)

	def _parse_review(self, text, _result=None):
		# v9 historically passed the current analysis as a second argument; v8's
		# parser correctly needs only the structured review text.
		return super()._parse_review(text)

	def _force_review_finalization(self, base_input, _result, reason):
		self.progress("  review-v9: {}; forcing conservative finalization".format(reason))
		# Reuse the proven v8 no-tools finalizer. It has the same review schema and
		# conservative KEEP-unless-supported policy, and avoids another open-ended
		# tool loop when v9 review evidence is exhausted.
		return self._finalize_review(base_input, [])

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
		self.progress("  review-v9: existing translation semantic consistency review")
		review = self._semantic_review(item, job, review_input)
		result = self._apply_review(item, result, review, "existing_translation_review")
		result.setdefault("evidence", {})["semantic_feedback"] = {
			"mode": "existing",
			"source": "human",
			"translation_jp": human_translation,
		}
		result["evidence"]["shared_evidence"] = self._shared_evidence_compact()
		return result

	def _generate_translation(self, item, job, result):
		translation_item = dict(item)
		# Existing Japanese may constrain morphology but must never leak into the
		# generated-translation phase itself.
		translation_item["translation_jp"] = None
		translation_job = dict(job)
		translation_job["produce_translation"] = True
		return self._translate_frozen_v7(translation_item, translation_job, result)

	def _review_against_generated_translation(self, item, job, result, translation):
		review_input = deepcopy(result)
		review_input["trsl_ai"] = translation["trsl_ai"]
		review_input.setdefault("evidence", {})["translation"] = translation["translation_evidence"]
		review_input["evidence"]["translation"]["confidence"] = translation["confidence"]
		review_input["evidence"]["translation"]["source"] = "generated"
		self.progress("  review-v9: generated translation semantic consistency review")
		review = self._semantic_review(item, job, review_input)
		result = self._apply_review(item, result, review, "generated_translation_review")
		result.setdefault("evidence", {})["semantic_feedback"] = {
			"mode": "generated",
			"source": "generated",
			"translation_jp": translation["trsl_ai"],
			"confidence": translation["confidence"],
			"translation_evidence": translation["translation_evidence"],
		}
		result["evidence"]["shared_evidence"] = self._shared_evidence_compact()
		return result

	def _baseline_fallback(self, morph_result, reason, progress_message):
		"""Keep the morph baseline when bounded v9 review cannot produce a safe final analysis."""
		segmented = str((morph_result or {}).get("segmented") or "").strip()
		annotation = str((morph_result or {}).get("annotation") or "").strip()
		if not segmented or not annotation:
			raise RuntimeError("{} and no baseline analysis is available".format(reason))
		self.progress(progress_message)
		return {
			"segmented": segmented,
			"annotation": annotation,
			"trsl_ai": "",
			"decision": "uncertain",
			"confidence": 0.5,
			"evidence": {
				"v9_fallback": {
					"kept_baseline": True,
					"reason": reason,
				}
			},
		}

	def _round_exhaustion_fallback(self, morph_result):
		return self._baseline_fallback(
			morph_result,
			"annotation-v9 exceeded maximum tool rounds",
			"  forward-v9: maximum tool rounds reached; keeping nrdb-morph baseline as uncertain",
		)

	def _malformed_final_fallback(self, morph_result, error):
		return self._baseline_fallback(
			morph_result,
			"annotation-v9 returned malformed/empty final JSON: {}".format(error),
			"  forward-v9: malformed/empty final JSON; keeping nrdb-morph baseline as uncertain",
		)

	@staticmethod
	def _semantic_policy(job, human_translation):
		mode = job.get("semantic_feedback")
		require = bool(job.get("require_semantic_feedback"))
		if not mode:
			# Backward compatibility for jobs created before semantic_feedback became
			# an orthogonal workflow axis.
			legacy = str(job.get("translation_evidence") or "ignore")
			if legacy == "required":
				mode = "existing"
				require = True
			elif legacy == "use":
				mode = "auto"
			else:
				mode = "none"
		mode = str(mode)
		if mode not in SEMANTIC_FEEDBACK_MODES:
			raise ValueError("invalid semantic_feedback mode: {}".format(mode))
		active = mode
		if mode == "auto":
			active = "existing" if human_translation else "generated"
		return mode, active, require

	def annotate(self, item, job, morph_result):
		human_translation = str(item.get("translation_jp") or "").strip()
		semantic_mode, active_feedback, require_feedback = self._semantic_policy(job, human_translation)
		if active_feedback == "existing" and require_feedback and not human_translation:
			raise ValueError("semantic_feedback=existing is required but this item has no existing translation")

		try:
			result = self._annotation_phase_v9(item, job, morph_result)
		except RuntimeError as error:
			if str(error) != "annotation-v9 exceeded maximum tool rounds":
				raise
			result = self._round_exhaustion_fallback(morph_result)
		except ValueError as error:
			# Final structured output can occasionally be empty, truncated, or invalid
			# after a valid sequence of tool calls. Never discard a paid row for a
			# formatting failure: preserve the specialized morph baseline instead.
			result = self._malformed_final_fallback(morph_result, error)

		generated_feedback = None
		generated_feedback_matches_final = False
		if result.get("annotation") and result.get("decision") != "failed":
			if active_feedback == "existing" and human_translation:
				result = self._review_against_human_translation(item, job, result)
			elif active_feedback == "generated":
				before = (result.get("segmented"), result.get("annotation"))
				generated_feedback = self._generate_translation(item, job, result)
				result = self._review_against_generated_translation(item, job, result, generated_feedback)
				after = (result.get("segmented"), result.get("annotation"))
				generated_feedback_matches_final = before == after
			elif semantic_mode != "none" and require_feedback:
				raise ValueError("required semantic feedback could not be produced")

		if job.get("produce_translation") and result.get("annotation") and result.get("decision") != "failed":
			# Reuse the internally generated translation only if semantic review kept
			# the morphology unchanged. If morphology changed, regenerate from the
			# final analysis so output and annotation cannot diverge.
			if generated_feedback is not None and generated_feedback_matches_final:
				translation = generated_feedback
			else:
				translation = self._generate_translation(item, job, result)
			result["trsl_ai"] = translation["trsl_ai"]
			result.setdefault("evidence", {})["translation"] = translation["translation_evidence"]
			result["evidence"]["translation"]["confidence"] = translation["confidence"]
		else:
			# Generated feedback is internal evidence when the requested task is morph.
			result["trsl_ai"] = ""
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
		translation = self._generate_translation(item, job, base)
		base["trsl_ai"] = translation["trsl_ai"]
		base["confidence"] = translation["confidence"]
		base["evidence"]["translation"] = translation["translation_evidence"]
		base["evidence"]["translation"]["confidence"] = translation["confidence"]
		return base
