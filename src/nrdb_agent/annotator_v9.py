import json

from .annotator import (
	BASE_INSTRUCTIONS,
	V2_RULES,
	V3_ANNOTATION_RULES,
	V4_RULES,
	V6_RULES,
	TOOLS,
	_compact_morph,
	_compact_tool_result,
	_response_output_as_input,
	_trace_arguments,
	_trace_result,
	parse_final_json,
)
from .annotator_v8 import AnnotationAgentV8
from .reverse_id_critic import IdSequenceCritic


V9_EFFICIENCY_RULES = """

annotation-v9 evidence-efficiency rules:
- The user payload contains a cheap uncertainty map derived before the LLM call. Treat it as triage evidence, not truth.
- Do not spend corpus queries on routine grammatical sequences when the ID-sequence critic reports no strong surprise there.
- Use corpus_examples primarily for short constructions containing an explicitly flagged ID-sequence hotspot.
- Use form_id_support primarily for low-confidence/ambiguous morph segments, or when you are actively considering changing the model's proposed ID for that surface.
- High-confidence model segments with no competing analysis and no ID-sequence surprise should normally be left alone without evidence calls.
- lookup_id remains available for a genuinely unclear lexical identity, but do not look up obvious IDs merely to confirm what the model already established.
- validate_analysis is never optional for a proposed revision and does not consume the linguistic evidence budget.
- Skipped low-information requests do not consume the evidence budget; use the returned triage message and move on.
"""

V9_INSTRUCTIONS = BASE_INSTRUCTIONS + V2_RULES + V3_ANNOTATION_RULES + V4_RULES + V6_RULES + V9_EFFICIENCY_RULES


class AnnotationAgentV9(AnnotationAgentV8):
	"""Forward annotation/translation with uncertainty-gated evidence use.

	The successful v8 translation and semantic-review machinery is preserved. Only
	the first annotation phase is changed: an optional ID-sequence model and morph
	confidence produce a cheap hotspot map, and expensive corpus/form evidence is
	reserved for those hotspots.
	"""
	def __init__(self, *args, id_model_path=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.id_model_path = str(id_model_path) if id_model_path else None
		self.id_critic = IdSequenceCritic(id_model_path) if id_model_path else None
		self._forward_hotspots = None

	def _prepare_hotspots(self, morph_result, schema_id):
		annotation = str(morph_result.get("annotation") or "").strip()
		id_review = None
		hotspot_ids = set()
		if self.id_critic is not None and annotation:
			raw = self.id_critic.review(annotation, int(schema_id))
			id_review = self.id_critic.compact(raw)
			for representation in id_review.get("representations", {}).values():
				for position in representation.get("surprising_positions", []):
					token = str(position.get("token") or "").strip()
					if token and token not in {"<PB>", "<EOS>"}:
						hotspot_ids.add(token)
					for value in position.get("context", []) or []:
						value = str(value or "").strip()
						if value and value not in {"<PB>", "<BOS>"}:
							hotspot_ids.add(value)

		uncertain_surfaces = set()
		surface_labels = {}
		for phrase in morph_result.get("phrases", []) or []:
			for segment in phrase.get("segments", []) or []:
				surface = str(segment.get("surface") or "").strip()
				label = str(segment.get("label") or "").strip()
				if surface and label:
					surface_labels.setdefault(surface, set()).add(label)
				confidence = segment.get("confidence", segment.get("raw_confidence"))
				alternatives = segment.get("alternatives", []) or []
				try:
					confidence = float(confidence)
				except (TypeError, ValueError):
					confidence = 0.0
				if surface and (confidence < 0.90 or len(alternatives) > 1):
					uncertain_surfaces.add(surface)

		return {
			"id_sequence_review": id_review,
			"hotspot_ids": sorted(hotspot_ids),
			"uncertain_surfaces": sorted(uncertain_surfaces),
			"surface_labels": {key: sorted(value) for key, value in surface_labels.items()},
			"policy": "query_uncertainty_not_length_v1",
		}

	def _query_is_hotspot(self, name, arguments):
		context = self._forward_hotspots or {}
		if self.id_critic is None:
			return True, None
		if name in {"lookup_id", "validate_analysis"}:
			return True, None
		if name == "form_id_support":
			surface = str(arguments.get("surface") or "").strip()
			candidate = str(arguments.get("candidate_id") or "").strip()
			if surface in set(context.get("uncertain_surfaces", [])):
				return True, None
			original = set(context.get("surface_labels", {}).get(surface, []))
			if candidate and original and candidate not in original:
				return True, None
			return False, "High-confidence form-ID pairing is not an uncertainty hotspot; keep it unless another independent signal motivates revision."
		if name == "corpus_examples":
			label = str(arguments.get("label") or "")
			hotspots = [value for value in context.get("hotspot_ids", []) if value]
			if any(value in label for value in hotspots):
				return True, None
			return False, "No ID-sequence hotspot occurs in this construction; routine grammar should not consume corpus evidence."
		return True, None

	def _annotation_phase_v9(self, item, job, morph_result):
		self._forward_hotspots = self._prepare_hotspots(morph_result, int(job["annotation_schema_id"]))
		compact_hotspots = {
			"policy": self._forward_hotspots["policy"],
			"hotspot_ids": self._forward_hotspots["hotspot_ids"],
			"uncertain_surfaces": self._forward_hotspots["uncertain_surfaces"],
			"id_sequence_review": self._forward_hotspots["id_sequence_review"],
		}
		self.progress("  forward-v9: uncertainty triage id_hotspots={} uncertain_surfaces={}".format(
			len(compact_hotspots["hotspot_ids"]), len(compact_hotspots["uncertain_surfaces"]),
		))
		input_payload = {
			"sentence_id": int(item["sentence_id"]),
			"dialect_id": int(item["dialect_id"]),
			"dialect_region": item.get("dialect_region"),
			"text": item["text"],
			"translation_jp": None,
			"produce_translation": False,
			"annotation_schema_id": int(job["annotation_schema_id"]),
			"nrdb_morph": _compact_morph(morph_result),
			"uncertainty_triage": compact_hotspots,
		}
		base_input = [{"role": "user", "content": json.dumps(input_payload, ensure_ascii=False)}]
		evidence_summary = []
		evidence_calls = 0
		self.progress("  llm: initial response ({}; annotation-v9)".format(self.model_name))
		response = self._create_response(base_input, V9_INSTRUCTIONS)
		for round_index in range(1, self.max_rounds + 1):
			calls = [output for output in response.output if getattr(output, "type", None) == "function_call"]
			if not calls:
				result = parse_final_json(response.output_text)
				result["model_response_id"] = response.id
				result["trsl_ai"] = ""
				result.setdefault("evidence", {})["forward_query_optimization"] = compact_hotspots
				result["evidence"]["forward_query_optimization"]["evidence_calls_used"] = evidence_calls
				self.progress("  final: decision={} confidence={:.3f}".format(result["decision"], result["confidence"]))
				return result

			self.progress("  tool round {}: {} call(s)".format(round_index, len(calls)))
			continuation = list(base_input)
			if evidence_summary:
				continuation.append({"role": "user", "content": "Previously retrieved compact evidence:\n" + json.dumps(evidence_summary[-6:], ensure_ascii=False)})
			continuation.extend(_response_output_as_input(response))
			for call in calls:
				arguments = json.loads(call.arguments)
				self.progress("    -> {}({})".format(call.name, _trace_arguments(call.name, arguments)))
				allowed, reason = self._query_is_hotspot(call.name, arguments)
				if not allowed:
					compact = {"query_skipped": True, "message": reason}
					self.progress("    <- {}: skipped (not an uncertainty hotspot)".format(call.name))
				elif call.name != "validate_analysis" and evidence_calls >= self.max_evidence_calls:
					compact = {"budget_exhausted": True, "message": "Evidence-call budget exhausted; decide from existing evidence or return uncertain."}
					self.progress("    <- {}: skipped (evidence budget exhausted)".format(call.name))
				else:
					tool_result = self._tool_result(call.name, arguments, item, int(job["annotation_schema_id"]))
					compact = _compact_tool_result(call.name, tool_result)
					self.progress("    <- {}: {}".format(call.name, _trace_result(call.name, tool_result)))
					if call.name != "validate_analysis":
						evidence_calls += 1
						evidence_summary.append({"tool": call.name, "arguments": arguments, "result": compact})
				continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})
			self.progress("  llm: continue after tool round {} (evidence {}/{})".format(round_index, evidence_calls, self.max_evidence_calls))
			response = self._create_response(continuation, V9_INSTRUCTIONS)
		raise RuntimeError("annotation-v9 exceeded maximum tool rounds")

	def annotate(self, item, job, morph_result):
		result = self._annotation_phase_v9(item, job, morph_result)
		if job.get("produce_translation") and result.get("annotation") and result.get("decision") != "failed":
			# Keep v7's already efficient batch lexical grounding. It can ground up to
			# twelve content IDs in one evidence call and only uses corpus examples
			# after dictionary grounding.
			translation = self._translate_frozen_v7(item, job, result)
			result["trsl_ai"] = translation["trsl_ai"]
			result.setdefault("evidence", {})["translation"] = translation["translation_evidence"]
			result["evidence"]["translation"]["confidence"] = translation["confidence"]
		if not result.get("trsl_ai") or result.get("decision") == "failed":
			return result

		# Preserve v8 semantic review as the final safety layer. Its payload already
		# contains the dictionary-grounding evidence, so the reviewer can target
		# only residual semantic mismatches rather than rebuilding the translation.
		review = self._semantic_review(item, job, result)
		result.setdefault("evidence", {})["semantic_review"] = review
		self.progress("  review-v9: action={} confidence={:.3f}".format(review["action"], review["confidence"]))
		if review["action"] != "revise":
			return result
		if review["segmented"] == result["segmented"] and review["annotation"] == result["annotation"]:
			result["evidence"]["semantic_review"]["action"] = "keep"
			result["evidence"]["semantic_review"]["note"] = "Revision requested but analysis was unchanged; kept original."
			return result
		validation = self.nrdb.validate_analysis(item["text"], review["segmented"], review["annotation"])
		result["evidence"]["semantic_review"]["validation"] = validation
		if not validation.get("valid"):
			self.progress("  review-v9: revision rejected by validator; keeping original")
			result["evidence"]["semantic_review"]["action"] = "keep"
			result["evidence"]["semantic_review"]["note"] = "Proposed revision failed structural validation; original kept."
			return result
		self.progress("  review-v9: revised annotation accepted")
		result["segmented"] = review["segmented"]
		result["annotation"] = review["annotation"]
		result["confidence"] = min(1.0, max(float(result.get("confidence", 0.0)), review["confidence"]))
		return result
