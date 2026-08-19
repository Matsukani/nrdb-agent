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
	_response_incomplete_reason,
	_response_output_as_input,
	_trace_arguments,
	_trace_result,
	parse_final_json,
)
from .annotator_v7 import GROUND_LEXICAL_IDS_TOOL
from .annotator_v8 import AnnotationAgentV8, REVIEW_FORMAT
from .reverse_id_critic import IdSequenceCritic


V9_EFFICIENCY_RULES = """

annotation-v9 evidence-efficiency rules:
- The user payload contains a cheap uncertainty map derived before the LLM call. Treat it as triage evidence, not truth.
- Query uncertainty, not sentence length. Do not verify routine material merely because it is present.
- Do not spend corpus queries on routine grammatical sequences when the ID-sequence critic reports no strong surprise there.
- Use corpus_examples primarily for short constructions containing an explicitly flagged ID-sequence hotspot.
- For surface-to-ID evidence, use form_id_support_batch. Put competing IDs for the same surface in ONE item and put several uncertain surfaces in ONE batch call.
- High-confidence model segments with no close alternative and no ID-sequence surprise should normally be left alone without evidence calls.
- lookup_id remains available for a genuinely unclear lexical identity, but do not look up obvious IDs merely to confirm what the model already established.
- validate_analysis is never optional for a proposed revision and does not consume the linguistic evidence budget.
- Skipped low-information requests do not consume the evidence budget; use the returned triage message and move on.
"""

V9_INSTRUCTIONS = BASE_INSTRUCTIONS + V2_RULES + V3_ANNOTATION_RULES + V4_RULES + V6_RULES + V9_EFFICIENCY_RULES


FORM_ID_SUPPORT_BATCH_TOOL = {
	"type": "function",
	"name": "form_id_support_batch",
	"description": "Compare region-scoped corpus/lexicon support for several uncertain surface-to-ID choices in ONE evidence call. Group competing candidate IDs under the same surface and batch several uncertain surfaces together. Routine high-confidence pairs are filtered out by the host.",
	"parameters": {
		"type": "object",
		"properties": {
			"items": {
				"type": "array",
				"minItems": 1,
				"maxItems": 8,
				"items": {
					"type": "object",
					"properties": {
						"surface": {"type": "string", "maxLength": 128},
						"candidate_ids": {
							"type": "array", "minItems": 1, "maxItems": 5,
							"items": {"type": "string", "maxLength": 128},
						},
					},
					"required": ["surface", "candidate_ids"],
					"additionalProperties": False,
				},
			},
		},
		"required": ["items"],
		"additionalProperties": False,
	},
	"strict": True,
}


def _tool(name):
	for value in TOOLS:
		if value.get("name") == name:
			return value
	raise KeyError(name)


# Single-pair form_id_support is intentionally absent in v9. Comparative form
# evidence should be gathered in one batch call.
V9_TOOLS = [_tool("lookup_id"), _tool("corpus_examples"), FORM_ID_SUPPORT_BATCH_TOOL, _tool("validate_analysis")]
V9_REVIEW_TOOLS = [GROUND_LEXICAL_IDS_TOOL, _tool("form_id_support"), _tool("corpus_examples")]

V9_REVIEW_INSTRUCTIONS = """You are the final semantic consistency reviewer for optimized NRDB annotation-v9.
You receive a source transcription, a validated proposed segmentation/annotation, a dictionary-grounded Japanese translation, and SHARED EVIDENCE already retrieved by earlier phases.

Goal: KEEP the analysis unless a concrete residual semantic mismatch remains. Revise narrowly only when independent evidence supports it.

Efficiency rules:
- Reuse shared evidence. Do not re-ground an ID already listed in shared lexical grounding and do not re-check a surface/ID pair already listed in shared form support.
- Do not re-query a corpus construction already present in shared corpus evidence.
- Call a tool only for a genuinely NEW unresolved mismatch that could change the final analysis.
- If the translation and annotation are semantically coherent and existing evidence is adequate, KEEP immediately with no tools.

Safety rules:
- The Japanese translation is evidence, not truth.
- Preserve unaffected segmentation and IDs exactly.
- IDs are identifiers, not glosses; dictionary evidence outranks apparent spelling.
- If evidence is ambiguous, KEEP.
- Do not use or request gold annotation and do not produce chain-of-thought.

Return exactly one JSON object:
{"action":"keep|revise","segmented":"...","annotation":"...","confidence":0.0,"changed_ids":[],"note":"brief"}
"""


class AnnotationAgentV9(AnnotationAgentV8):
	"""Forward annotation/translation with uncertainty-gated, shared evidence use."""
	def __init__(self, *args, id_model_path=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.id_model_path = str(id_model_path) if id_model_path else None
		self.id_critic = IdSequenceCritic(id_model_path) if id_model_path else None
		self._forward_hotspots = None
		self._shared_evidence = {"lookup": {}, "corpus": {}, "form": {}}

	@staticmethod
	def _numeric(value):
		try:
			return float(value)
		except (TypeError, ValueError):
			return None

	def _alternative_margin(self, segment):
		values = []
		for alt in segment.get("alternatives", []) or []:
			for key in ("probability", "confidence", "score"):
				value = self._numeric(alt.get(key)) if isinstance(alt, dict) else None
				if value is not None:
					values.append(value)
					break
		if len(values) < 2:
			return None
		values.sort(reverse=True)
		return values[0] - values[1]

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
		uncertainty_reasons = {}
		for phrase in morph_result.get("phrases", []) or []:
			for segment in phrase.get("segments", []) or []:
				surface = str(segment.get("surface") or "").strip()
				label = str(segment.get("label") or "").strip()
				if surface and label:
					surface_labels.setdefault(surface, set()).add(label)
				confidence_raw = segment.get("confidence", segment.get("raw_confidence"))
				confidence = self._numeric(confidence_raw)
				margin = self._alternative_margin(segment)
				reasons = []
				if confidence is not None and confidence < 0.80:
					reasons.append("low_confidence")
				if margin is not None and margin < 0.15:
					reasons.append("small_alternative_margin")
				label_atoms = set(part.strip() for part in label.split(";") if part.strip())
				if label in hotspot_ids or label_atoms.intersection(hotspot_ids):
					reasons.append("id_sequence_hotspot")
				if surface and reasons:
					uncertain_surfaces.add(surface)
					uncertainty_reasons[surface] = reasons

		return {
			"id_sequence_review": id_review,
			"hotspot_ids": sorted(hotspot_ids),
			"uncertain_surfaces": sorted(uncertain_surfaces),
			"uncertainty_reasons": uncertainty_reasons,
			"surface_labels": {key: sorted(value) for key, value in surface_labels.items()},
			"policy": "query_uncertainty_not_length_v2",
		}

	def _pair_is_hotspot(self, surface, candidate):
		context = self._forward_hotspots or {}
		if self.id_critic is None:
			return True
		if surface in set(context.get("uncertain_surfaces", [])):
			return True
		original = set(context.get("surface_labels", {}).get(surface, []))
		if candidate and original and candidate not in original:
			return True
		return False

	def _query_is_hotspot(self, name, arguments):
		context = self._forward_hotspots or {}
		if self.id_critic is None:
			return True, None
		if name in {"lookup_id", "validate_analysis"}:
			return True, None
		if name == "form_id_support_batch":
			for item in arguments.get("items", []) or []:
				surface = str(item.get("surface") or "").strip()
				for candidate in item.get("candidate_ids", []) or []:
					if self._pair_is_hotspot(surface, str(candidate or "").strip()):
						return True, None
			return False, "All requested form-ID pairs are routine high-confidence model assignments; no comparative form query is needed."
		if name == "corpus_examples":
			label = str(arguments.get("label") or "")
			hotspots = [value for value in context.get("hotspot_ids", []) if value]
			if any(value in label for value in hotspots):
				return True, None
			return False, "No ID-sequence hotspot occurs in this construction; routine grammar should not consume corpus evidence."
		return True, None

	def _cache_lookup(self, label, result):
		self._shared_evidence["lookup"][str(label)] = result

	def _cache_corpus(self, label, result):
		self._shared_evidence["corpus"][str(label)] = result

	def _cache_form(self, surface, candidate, result):
		self._shared_evidence["form"]["{}\t{}".format(surface, candidate)] = result

	def _tool_result_v9(self, name, arguments, item, schema_id):
		if name == "form_id_support_batch":
			region = str(item.get("dialect_region") or "").strip()
			rows = []
			for request in (arguments.get("items", []) or [])[:8]:
				surface = str(request.get("surface") or "").strip()
				candidates = []
				for raw_candidate in request.get("candidate_ids", []) or []:
					candidate = str(raw_candidate or "").strip()
					if candidate and candidate not in candidates:
						candidates.append(candidate)
				candidate_rows = []
				for candidate in candidates[:5]:
					if not self._pair_is_hotspot(surface, candidate):
						candidate_rows.append({"candidate_id": candidate, "query_skipped": True, "message": "routine high-confidence model pairing"})
						continue
					key = "{}\t{}".format(surface, candidate)
					if key in self._shared_evidence["form"]:
						result = self._shared_evidence["form"][key]
						cache_hit = True
					elif not region:
						result = {"success": True, "surface": surface, "candidate_id": candidate, "region": None, "combined": {"surface_total": 0, "candidate_count": 0, "candidate_rate": None, "penalty": "none"}, "corpus": {}, "lexicon": {}}
						cache_hit = False
					else:
						result = self.nrdb.form_id_support(surface, candidate, region, schema_id)
						self._cache_form(surface, candidate, result)
						cache_hit = False
					compact = _compact_tool_result("form_id_support", result)
					compact["cache_hit"] = cache_hit
					candidate_rows.append(compact)
				rows.append({"surface": surface, "candidates": candidate_rows})
			return {"region": region or None, "items": rows}
		if name == "lookup_id":
			label = str(arguments.get("label") or "").strip()
			if label in self._shared_evidence["lookup"]:
				return self._shared_evidence["lookup"][label]
			result = self._tool_result(name, arguments, item, schema_id)
			self._cache_lookup(label, result)
			return result
		if name == "corpus_examples":
			label = str(arguments.get("label") or "").strip()
			if label in self._shared_evidence["corpus"]:
				return self._shared_evidence["corpus"][label]
			result = self._tool_result(name, arguments, item, schema_id)
			self._cache_corpus(label, result)
			return result
		return self._tool_result(name, arguments, item, schema_id)

	def _compact_v9(self, name, result):
		if name == "form_id_support_batch":
			return result
		return _compact_tool_result(name, result)

	def _trace_args_v9(self, name, arguments):
		if name == "form_id_support_batch":
			return "items={}".format([
				{"surface": value.get("surface"), "candidate_ids": value.get("candidate_ids", [])}
				for value in arguments.get("items", [])
			])
		return _trace_arguments(name, arguments)

	def _trace_result_v9(self, name, result):
		if name == "form_id_support_batch":
			checked = 0
			skipped = 0
			for item in result.get("items", []):
				for candidate in item.get("candidates", []):
					if candidate.get("query_skipped"):
						skipped += 1
					else:
						checked += 1
			return "surfaces={} checked_pairs={} skipped_pairs={}".format(len(result.get("items", [])), checked, skipped)
		return _trace_result(name, result)

	def _annotation_phase_v9(self, item, job, morph_result):
		self._forward_hotspots = self._prepare_hotspots(morph_result, int(job["annotation_schema_id"]))
		compact_hotspots = {
			"policy": self._forward_hotspots["policy"],
			"hotspot_ids": self._forward_hotspots["hotspot_ids"],
			"uncertain_surfaces": self._forward_hotspots["uncertain_surfaces"],
			"uncertainty_reasons": self._forward_hotspots["uncertainty_reasons"],
			"id_sequence_review": self._forward_hotspots["id_sequence_review"],
		}
		self.progress("  forward-v9: uncertainty triage id_hotspots={} uncertain_surfaces={}".format(
			len(compact_hotspots["hotspot_ids"]), len(compact_hotspots["uncertain_surfaces"]),
		))
		input_payload = {
			"sentence_id": int(item["sentence_id"]), "dialect_id": int(item["dialect_id"]),
			"dialect_region": item.get("dialect_region"), "text": item["text"], "translation_jp": None,
			"produce_translation": False, "annotation_schema_id": int(job["annotation_schema_id"]),
			"nrdb_morph": _compact_morph(morph_result), "uncertainty_triage": compact_hotspots,
		}
		base_input = [{"role": "user", "content": json.dumps(input_payload, ensure_ascii=False)}]
		evidence_summary = []
		evidence_calls = 0
		self.progress("  llm: initial response ({}; annotation-v9)".format(self.model_name))
		response = self._create_response(base_input, V9_INSTRUCTIONS, tools=V9_TOOLS)
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
				try:
					arguments = json.loads(call.arguments)
				except (json.JSONDecodeError, TypeError, ValueError) as error:
					compact = {
						"invalid_tool_arguments": True,
						"message": "Tool arguments were malformed or truncated JSON. Retry this tool call with one complete valid JSON object, or finalize conservatively from existing evidence.",
						"error": str(error),
					}
					self.progress("    -> {}(<malformed arguments>)".format(call.name))
					self.progress("    <- {}: malformed/truncated tool arguments; returned recoverable error".format(call.name))
					continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})
					continue
				self.progress("    -> {}({})".format(call.name, self._trace_args_v9(call.name, arguments)))
				allowed, reason = self._query_is_hotspot(call.name, arguments)
				if not allowed:
					compact = {"query_skipped": True, "message": reason}
					self.progress("    <- {}: skipped (not an uncertainty hotspot)".format(call.name))
				elif call.name != "validate_analysis" and evidence_calls >= self.max_evidence_calls:
					compact = {"budget_exhausted": True, "message": "Evidence-call budget exhausted; decide from existing evidence or return uncertain."}
					self.progress("    <- {}: skipped (evidence budget exhausted)".format(call.name))
				else:
					tool_result = self._tool_result_v9(call.name, arguments, item, int(job["annotation_schema_id"]))
					compact = self._compact_v9(call.name, tool_result)
					self.progress("    <- {}: {}".format(call.name, self._trace_result_v9(call.name, tool_result)))
					if call.name != "validate_analysis":
						evidence_calls += 1
						evidence_summary.append({"tool": call.name, "arguments": arguments, "result": compact})
				continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})
			self.progress("  llm: continue after tool round {} (evidence {}/{})".format(round_index, evidence_calls, self.max_evidence_calls))
			response = self._create_response(continuation, V9_INSTRUCTIONS, tools=V9_TOOLS)
		raise RuntimeError("annotation-v9 exceeded maximum tool rounds")

	def _ground_lexical_ids(self, labels, schema_id):
		grounded = []
		for raw_label in labels[:12]:
			label = str(raw_label or "").strip()
			if not label or len(label) > 128 or any(char.isspace() for char in label):
				grounded.append({"label": label, "lexical_entries": [], "local": None, "global": None, "grounded": False, "rejected": True, "error": "Invalid lexical-grounding label; use one exact atomic NRDB annotation ID only."})
				continue
			if label in self._shared_evidence["lookup"]:
				result = self._shared_evidence["lookup"][label]
			else:
				try:
					result = self.nrdb.lookup_id(label, schema_id)
				except RuntimeError as error:
					grounded.append({"label": label, "lexical_entries": [], "local": None, "global": None, "grounded": False, "rejected": True, "error": str(error)})
					continue
				self._cache_lookup(label, result)
			compact = _compact_tool_result("lookup_id", result)
			grounded.append({"label": label, "lexical_entries": compact.get("lexical_entries", []), "local": compact.get("local"), "global": compact.get("global"), "grounded": bool(compact.get("lexical_entries")), "rejected": False})
		return {"labels": grounded}

	def _v7_tool_result(self, name, arguments, item, schema_id):
		if name == "ground_lexical_ids":
			return self._ground_lexical_ids(arguments["labels"], schema_id)
		if name == "corpus_examples":
			label = str(arguments.get("label") or "").strip()
			if label in self._shared_evidence["corpus"]:
				return self._shared_evidence["corpus"][label]
			result = self._tool_result(name, arguments, item, schema_id)
			self._cache_corpus(label, result)
			return result
		return super()._v7_tool_result(name, arguments, item, schema_id)

	def _semantic_review(self, item, job, result):
		self.progress("  review-v9: semantic consistency review (shared evidence reuse)")
		payload = {
			"sentence_id": int(item["sentence_id"]), "source_text": item["text"],
			"segmented": result["segmented"], "annotation": result["annotation"],
			"translation_jp": result["trsl_ai"], "annotation_schema_id": int(job["annotation_schema_id"]),
			"shared_evidence": self._shared_evidence_compact(),
		}
		base_input = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
		response = self._create_response(base_input, V9_REVIEW_INSTRUCTIONS, tools=V9_REVIEW_TOOLS, max_output_tokens=1200, text_format=REVIEW_FORMAT)
		for round_index in range(1, self.max_review_rounds + 1):
			calls = [output for output in response.output if getattr(output, "type", None) == "function_call"]
			if not calls:
				return self._parse_review(response.output_text, result)
			continuation = list(base_input)
			continuation.extend(_response_output_as_input(response))
			for call in calls:
				arguments = json.loads(call.arguments)
				if self._review_query_already_known(call.name, arguments):
					compact = {"shared_evidence_reused": True, "message": "This evidence is already present in the shared cache; decide from it instead of querying again."}
				else:
					tool_result = self._v8_review_tool_result(call.name, arguments, item, int(job["annotation_schema_id"]))
					compact = self._compact_review_result(call.name, tool_result)
				continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})
			response = self._create_response(continuation, V9_REVIEW_INSTRUCTIONS, tools=V9_REVIEW_TOOLS, max_output_tokens=1200, text_format=REVIEW_FORMAT)
		return self._force_review_finalization(base_input, result, "v9 review tool budget exhausted")
