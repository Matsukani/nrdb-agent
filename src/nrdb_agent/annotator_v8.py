import json

from .annotator import TOOLS, _compact_tool_result, _response_incomplete_reason, _response_output_as_input, _trace_arguments, _trace_result
from .annotator_v7 import AnnotationAgentV7, GROUND_LEXICAL_IDS_TOOL


REVIEW_INSTRUCTIONS = """You are the final semantic consistency reviewer for NRDB annotation-v8.
You receive a source transcription, an already validated proposed segmentation/annotation, and a dictionary-grounded Japanese translation produced from that proposal.

Goal: decide whether the proposed annotation should be KEPT or narrowly REVISED.

Rules:
- The Japanese translation is evidence, not truth. Never change the annotation merely to make it agree with the translation.
- Reconsider only concrete points where the translation reveals a possible semantic mismatch in one or a few IDs.
- Use ground_lexical_ids for lexical semantics, form_id_support for region-scoped surface-to-ID evidence, and corpus_examples for short constructional comparisons.
- Prefer a revision only when independent dictionary/corpus/regional evidence supports it better than the original proposal.
- Do not rewrite an otherwise good sentence wholesale. Preserve all unaffected segmentation and IDs exactly.
- Do not use or request gold annotation.
- NRDB ID labels are identifiers, not glosses; do not infer semantics from their kanji/spelling.
- Phrase-local reduplication and all annotation-v6 constraints remain in force.
- If evidence is ambiguous, KEEP the original analysis.
- Do not produce chain-of-thought. Give only a short auditable note.

Return exactly one JSON object:
{"action":"keep|revise","segmented":"...","annotation":"...","confidence":0.0,"changed_ids":[],"note":"brief"}
"""

REVIEW_FORMAT = {
	"type": "json_schema",
	"name": "nrdb_annotation_review",
	"strict": True,
	"schema": {
		"type": "object",
		"properties": {
			"action": {"type": "string", "enum": ["keep", "revise"]},
			"segmented": {"type": "string"},
			"annotation": {"type": "string"},
			"confidence": {"type": "number"},
			"changed_ids": {"type": "array", "items": {"type": "string"}},
			"note": {"type": "string"},
		},
		"required": ["action", "segmented", "annotation", "confidence", "changed_ids", "note"],
		"additionalProperties": False,
	},
}


def _review_tool(name):
	for tool in TOOLS:
		if tool.get("name") == name:
			return tool
	raise KeyError(name)


REVIEW_TOOLS = [
	GROUND_LEXICAL_IDS_TOOL,
	_review_tool("form_id_support"),
	_review_tool("corpus_examples"),
]


class AnnotationAgentV8(AnnotationAgentV7):
	def __init__(self, *args, max_review_evidence_calls=4, **kwargs):
		super().__init__(*args, **kwargs)
		self.max_review_evidence_calls = int(max_review_evidence_calls)

	def _review_tool_result(self, name, arguments, item, schema_id):
		if name == "ground_lexical_ids":
			return self._ground_lexical_ids(arguments["labels"], schema_id)
		return self._tool_result(name, arguments, item, schema_id)

	def _review_compact(self, name, result):
		if name == "ground_lexical_ids":
			return result
		return _compact_tool_result(name, result)

	def _review_trace_arguments(self, name, arguments):
		if name == "ground_lexical_ids":
			return "labels={}".format(arguments.get("labels", []))
		return _trace_arguments(name, arguments)

	def _review_trace_result(self, name, result):
		if name == "ground_lexical_ids":
			labels = result.get("labels", [])
			return "grounded={}/{}".format(sum(1 for value in labels if value.get("grounded")), len(labels))
		return _trace_result(name, result)

	def _parse_review(self, text):
		payload = json.loads((text or "").strip())
		if payload.get("action") not in {"keep", "revise"}:
			raise ValueError("invalid review action")
		payload["segmented"] = str(payload.get("segmented") or "").strip()
		payload["annotation"] = str(payload.get("annotation") or "").strip()
		payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
		payload["changed_ids"] = payload.get("changed_ids") if isinstance(payload.get("changed_ids"), list) else []
		payload["note"] = str(payload.get("note") or "").strip()
		return payload

	def _finalize_review(self, base_input, evidence_summary):
		final_input = list(base_input)
		if evidence_summary:
			final_input.append({"role": "user", "content": "Retrieved compact review evidence:\n" + json.dumps(evidence_summary[-4:], ensure_ascii=False)})
		final_input.append({"role": "user", "content": "Evidence gathering is finished. Do not call more tools. Return KEEP unless independent evidence clearly supports a targeted revision."})
		for attempt, budget in enumerate((1000, 1500), start=1):
			self.progress("  review-v8: forced finalization attempt {}".format(attempt))
			response = self._create_response(final_input, REVIEW_INSTRUCTIONS, tools=[], max_output_tokens=budget, text_format=REVIEW_FORMAT)
			if _response_incomplete_reason(response):
				continue
			try:
				return self._parse_review(response.output_text)
			except (json.JSONDecodeError, ValueError):
				continue
		raise RuntimeError("annotation-v8 review finalization failed")

	def _semantic_review(self, item, job, result):
		payload = {
			"sentence_id": int(item["sentence_id"]),
			"source_text": item["text"],
			"dialect_region": item.get("dialect_region"),
			"proposed_segmented": result["segmented"],
			"proposed_annotation": result["annotation"],
			"proposed_translation_jp": result.get("trsl_ai", ""),
			"translation_evidence": result.get("evidence", {}).get("translation", {}),
			"annotation_confidence": result.get("confidence"),
			"annotation_schema_id": int(job["annotation_schema_id"]),
		}
		base_input = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
		evidence_summary = []
		evidence_calls = 0
		self.progress("  review-v8: semantic consistency review")
		response = self._create_response(base_input, REVIEW_INSTRUCTIONS, tools=REVIEW_TOOLS, max_output_tokens=900, text_format=REVIEW_FORMAT)
		for round_index in range(1, self.max_rounds + 1):
			calls = [output for output in response.output if getattr(output, "type", None) == "function_call"]
			if not calls:
				if _response_incomplete_reason(response):
					return self._finalize_review(base_input, evidence_summary)
				try:
					return self._parse_review(response.output_text)
				except (json.JSONDecodeError, ValueError):
					return self._finalize_review(base_input, evidence_summary)

			self.progress("  review-v8 tool round {}: {} call(s)".format(round_index, len(calls)))
			continuation = list(base_input)
			if evidence_summary:
				continuation.append({"role": "user", "content": "Previously retrieved compact review evidence:\n" + json.dumps(evidence_summary[-4:], ensure_ascii=False)})
			continuation.extend(_response_output_as_input(response))
			for call in calls:
				arguments = json.loads(call.arguments)
				self.progress("    -> {}({})".format(call.name, self._review_trace_arguments(call.name, arguments)))
				if evidence_calls >= self.max_review_evidence_calls:
					compact = {"budget_exhausted": True, "message": "Review evidence budget exhausted; keep unless existing evidence clearly supports revision."}
					self.progress("    <- {}: skipped (review evidence budget exhausted)".format(call.name))
				else:
					tool_result = self._review_tool_result(call.name, arguments, item, int(job["annotation_schema_id"]))
					compact = self._review_compact(call.name, tool_result)
					self.progress("    <- {}: {}".format(call.name, self._review_trace_result(call.name, tool_result)))
					evidence_calls += 1
					evidence_summary.append({"tool": call.name, "arguments": arguments, "result": compact})
				continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})
			if evidence_calls >= self.max_review_evidence_calls:
				return self._finalize_review(base_input, evidence_summary)
			response = self._create_response(continuation, REVIEW_INSTRUCTIONS, tools=REVIEW_TOOLS, max_output_tokens=900, text_format=REVIEW_FORMAT)
		return self._finalize_review(base_input, evidence_summary)

	def annotate(self, item, job, morph_result):
		# v8 first runs the proven v7 annotation + grounded translation pipeline.
		v7_job = dict(job)
		v7_job["prompt_version"] = "annotation-v7"
		result = super().annotate(item, v7_job, morph_result)
		if not result.get("trsl_ai") or result.get("decision") == "failed":
			return result

		review = self._semantic_review(item, job, result)
		result.setdefault("evidence", {})["semantic_review"] = review
		self.progress("  review-v8: action={} confidence={:.3f}".format(review["action"], review["confidence"]))
		if review["action"] != "revise":
			return result

		if review["segmented"] == result["segmented"] and review["annotation"] == result["annotation"]:
			result["evidence"]["semantic_review"]["action"] = "keep"
			result["evidence"]["semantic_review"]["note"] = "Revision requested but analysis was unchanged; kept original."
			return result

		validation = self.nrdb.validate_analysis(item["text"], review["segmented"], review["annotation"])
		result["evidence"]["semantic_review"]["validation"] = validation
		if not validation.get("valid"):
			self.progress("  review-v8: revision rejected by validator; keeping original")
			result["evidence"]["semantic_review"]["action"] = "keep"
			result["evidence"]["semantic_review"]["note"] = "Proposed revision failed structural validation; original kept."
			return result

		self.progress("  review-v8: revised annotation accepted")
		result["segmented"] = review["segmented"]
		result["annotation"] = review["annotation"]
		result["confidence"] = min(1.0, max(float(result.get("confidence", 0.0)), review["confidence"]))
		return result
