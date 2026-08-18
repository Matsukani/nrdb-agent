import json

from .annotator import _compact_tool_result, _response_incomplete_reason, _response_output_as_input
from .reverse_agent import ReverseIdAgent


SURFACE_INSTRUCTIONS = """You are the constrained NRDB Miyako surface-realization phase for reverse-v2.

You receive a Japanese sentence and a FROZEN predicted Miyako NRDB ID annotation. Do not revise the ID annotation in this phase.

Goal: realize that ID sequence as a plausible segmented Miyako surface string for the requested dialect priorities.

Rules:
- Retrieval first. Use surface_forms_for_id to obtain attested forms for lexical/content IDs and any grammatical IDs whose realization is unclear.
- Dialect priorities are ordered. Prefer dialect 1; use dialect 2 only when an appropriate form is unavailable in dialect 1; continue in order. Same-region forms are fallback evidence after the explicit list.
- Use corpus_examples on short ID constructions to recover target-language grammatical packaging, ordering, and morphophonological realization.
- Never copy a held-out test sentence; the evidence service excludes the evaluation cohort.
- Do not invent a lexical root when an attested requested-dialect or fallback form is available.
- Limited productive inflection/morphophonology may be composed from attested forms and corpus patterns, but mark uncertainty when evidence is weak.
- Keep the frozen annotation unchanged even if you suspect it is imperfect; this experiment separates ID transfer from surface realization.
- Return a segmented Miyako candidate using spaces for phrases and hyphens for morpheme boundaries where recoverable.
- Do not produce chain-of-thought.

Return exactly one JSON object:
{"segmented":"...","confidence":0.0,"evidence":{"note":"brief","ids_realized":[],"fallback_ids":[],"example_sentence_ids":[]}}
"""

SURFACE_FORMAT = {
	"type": "json_schema",
	"name": "nrdb_reverse_surface",
	"strict": True,
	"schema": {
		"type": "object",
		"properties": {
			"segmented": {"type": "string"},
			"confidence": {"type": "number"},
			"evidence": {
				"type": "object",
				"properties": {
					"note": {"type": "string"},
					"ids_realized": {"type": "array", "items": {"type": "string"}},
					"fallback_ids": {"type": "array", "items": {"type": "string"}},
					"example_sentence_ids": {"type": "array", "items": {"type": "integer"}},
				},
				"required": ["note", "ids_realized", "fallback_ids", "example_sentence_ids"],
				"additionalProperties": False,
			},
		},
		"required": ["segmented", "confidence", "evidence"],
		"additionalProperties": False,
	},
}

SURFACE_FORM_TOOL = {
	"type": "function",
	"name": "surface_forms_for_id",
	"description": "Retrieve attested lexical surface forms for one exact NRDB ID, ordered by the requested dialect priority and then same-region fallback.",
	"parameters": {
		"type": "object",
		"properties": {"label": {"type": "string", "maxLength": 128}},
		"required": ["label"],
		"additionalProperties": False,
	},
	"strict": True,
}


def _corpus_tool():
	from .annotator import TOOLS
	for tool in TOOLS:
		if tool.get("name") == "corpus_examples":
			return tool
	raise KeyError("corpus_examples")


SURFACE_TOOLS = [SURFACE_FORM_TOOL, _corpus_tool()]


class ReverseSurfaceAgent(ReverseIdAgent):
	def __init__(self, *args, max_surface_evidence_calls=8, **kwargs):
		super().__init__(*args, **kwargs)
		self.max_surface_evidence_calls = int(max_surface_evidence_calls)

	def _parse_surface(self, text):
		payload = json.loads((text or "").strip())
		payload["segmented"] = str(payload.get("segmented") or "").strip()
		payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
		payload["evidence"] = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
		if not payload["segmented"]:
			raise ValueError("reverse-v2 returned empty surface")
		return payload

	def _surface_result(self, name, arguments, item, job):
		if name == "surface_forms_for_id":
			return self.nrdb.surface_forms_for_id(
				arguments["label"], int(job["annotation_schema_id"]),
				job.get("target_dialect_ids") or [int(item["dialect_id"])],
				region=item.get("dialect_region"),
			)
		return self.nrdb.examples(arguments["label"], int(job["annotation_schema_id"]), int(item["sentence_id"]), min(8, arguments["limit"]))

	def _compact_surface(self, name, result):
		if name == "surface_forms_for_id":
			forms = []
			for value in result.get("forms", [])[:10]:
				forms.append({
					"dialect_id": value.get("dialect_id"), "dialect_name": value.get("dialect_name"),
					"dialect_region": value.get("dialect_region"), "form1": value.get("form1"),
					"form2": value.get("form2"), "form1_seg": value.get("form1_seg"),
					"form2_seg": value.get("form2_seg"), "meaning_jp": value.get("meaning_jp"),
				})
			return {"label": result.get("label"), "dialect_ids": result.get("dialect_ids"), "region": result.get("region"), "forms": forms}
		return _compact_tool_result(name, result)

	def _realize_surface(self, item, job, id_result):
		dialect_ids = job.get("target_dialect_ids") or [int(item["dialect_id"])]
		payload = {
			"japanese": str(item.get("translation_jp") or "").strip(),
			"frozen_annotation": id_result["annotation"],
			"target_dialect_ids": dialect_ids,
			"target_region": item.get("dialect_region"),
		}
		base_input = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
		evidence = []
		calls_used = 0
		self.progress("  reverse-v2: realize surface; dialect priority={}".format(dialect_ids))
		response = self._create_response(base_input, SURFACE_INSTRUCTIONS, tools=SURFACE_TOOLS, max_output_tokens=900, text_format=SURFACE_FORMAT)
		for round_index in range(1, self.max_rounds + 1):
			calls = [value for value in response.output if getattr(value, "type", None) == "function_call"]
			if not calls:
				if _response_incomplete_reason(response):
					break
				try:
					return self._parse_surface(response.output_text)
				except (json.JSONDecodeError, ValueError):
					break
			continuation = list(base_input)
			if evidence:
				continuation.append({"role": "user", "content": "Previously retrieved surface evidence:\n" + json.dumps(evidence[-6:], ensure_ascii=False)})
			continuation.extend(_response_output_as_input(response))
			self.progress("  reverse-v2 surface tool round {}: {} call(s)".format(round_index, len(calls)))
			for call in calls:
				arguments = json.loads(call.arguments)
				if calls_used >= self.max_surface_evidence_calls:
					compact = {"budget_exhausted": True, "message": "Surface evidence budget exhausted; finalize conservatively."}
				else:
					result = self._surface_result(call.name, arguments, item, job)
					compact = self._compact_surface(call.name, result)
					calls_used += 1
					evidence.append({"tool": call.name, "arguments": arguments, "result": compact})
					if call.name == "surface_forms_for_id":
						self.progress("    <- surface_forms_for_id({}): {} form(s)".format(arguments.get("label"), len(result.get("forms", []))))
					else:
						self.progress("    <- corpus_examples: {} example(s)".format(len(result.get("examples", []))))
				continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})
			if calls_used >= self.max_surface_evidence_calls:
				break
			response = self._create_response(continuation, SURFACE_INSTRUCTIONS, tools=SURFACE_TOOLS, max_output_tokens=900, text_format=SURFACE_FORMAT)

		final_input = list(base_input)
		if evidence:
			final_input.append({"role": "user", "content": "Retrieved surface evidence:\n" + json.dumps(evidence[-8:], ensure_ascii=False)})
		final_input.append({"role": "user", "content": "Evidence gathering is finished. Do not call tools. Return the most conservative attested-form-based segmented Miyako realization now."})
		response = self._create_response(final_input, SURFACE_INSTRUCTIONS, tools=[], max_output_tokens=1200, text_format=SURFACE_FORMAT)
		return self._parse_surface(response.output_text)

	def annotate(self, item, job, morph_result=None):
		id_job = dict(job)
		id_job["prompt_version"] = "reverse-v1"
		id_result = super().annotate(item, id_job, morph_result)
		if id_result.get("decision") == "failed" or not id_result.get("annotation"):
			return id_result
		surface = self._realize_surface(item, job, id_result)
		id_result["segmented"] = surface["segmented"]
		id_result.setdefault("evidence", {})["surface_realization"] = surface["evidence"]
		id_result["confidence"] = min(float(id_result.get("confidence", 0.0)), float(surface["confidence"]))
		return id_result
