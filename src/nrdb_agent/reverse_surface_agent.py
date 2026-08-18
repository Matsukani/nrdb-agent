import json

from .annotator import _compact_tool_result, _response_incomplete_reason, _response_output_as_input
from .reverse_agent import ReverseIdAgent


SURFACE_INSTRUCTIONS = """You are the constrained NRDB Miyako surface-realization phase for reverse-v2.

You receive a Japanese sentence and a FROZEN predicted Miyako NRDB ID annotation. Do not revise the ID annotation in this phase.

Goal: realize that ID sequence as a plausible segmented Miyako surface string for the requested dialect priorities.

Core lexical-layer rule:
- Ordinary non-n: lexical IDs belong to the local Ryukyuan lexical system. Realize them with attested local forms; do not replace them with Japanese merely because the ID label contains Japanese-readable characters.
- IDs beginning with n: belong to the synchronically Japanese lexical reservoir. Realize them as Japanese-layer lexical forms embedded inside the Ryukyuan morphosyntactic frame.
- For n: IDs, first use attested n: trsc2 corpus realizations when available. If the exact n: item is productively recruited and has no attested surface in NRDB, it is legitimate to use the ordinary Japanese pronunciation of the lexical material after n:, rendered conservatively in the project's trsc2-style romanization.
- Do not phonologically localize an n: item into an ordinary Ryukyuan lexical form unless corpus evidence shows that exact n: realization. n: marks the Japanese lexical layer, not historical Japanese origin.
- Keep surrounding grammatical morphology Ryukyuan whenever the frozen annotation is Ryukyuan.

Two-pass dialect policy:
- ID/grammar inference has already pooled evidence across the annotation schema's target region, with the requested dialects only mildly preferred. Do not reinterpret that frozen ID sequence here.
- Surface realization is much stricter. If surface evidence exists in any requested dialect, surface_forms_for_id returns ONLY requested-dialect evidence for that ID. Use the requested dialects in their supplied order.
- Same-region forms are returned only when no requested-dialect surface evidence exists for that ID. Treat such evidence as an explicit fallback and record it in fallback_ids.
- Never use an out-of-region surface form.

Rules:
- Retrieval first. Use surface_forms_for_id to obtain attested surface realizations for lexical/content IDs and grammatical/local IDs.
- Corpus surface evidence is drawn only from the romanized trsc2 segmented layer and includes sentence (sen), text (txt), and lexical-example (lxs) corpus sources. Prefer frequent attested trsc2 corpus realizations, especially for grammatical/local IDs.
- Lexicon forms are secondary complementary evidence, especially useful for lexical roots.
- Use corpus_examples on short ID constructions to recover target-language grammatical packaging, ordering, and morphophonological realization.
- Never copy a held-out test sentence; both corpus-example and surface-form evidence exclude the evaluation cohort.
- Do not invent an ordinary local lexical root when an attested requested-dialect or fallback form is available. Productive Japanese realization is allowed only for frozen n: IDs.
- Limited productive inflection/morphophonology may be composed from attested forms and corpus patterns, but mark uncertainty when evidence is weak.
- Keep the frozen annotation unchanged even if you suspect it is imperfect; this experiment separates ID transfer from surface realization.
- Return a segmented Miyako candidate in trsc2-style romanization, using spaces for phrases and hyphens for morpheme boundaries where recoverable.
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
	"description": "Retrieve annotation-schema-scoped attested trsc2 corpus surfaces (sen/txt/lxs) plus lexicon forms for one exact NRDB ID. Requested dialects are an aggressive surface scope: if any requested-dialect evidence exists, regional alternatives are suppressed; same-region evidence is returned only as fallback. A zero result for an n: ID does not prohibit productive Japanese lexical realization.",
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
		self._id_pass_dialect_ids = []

	def _tool_result_reverse(self, name, arguments, item, schema_id):
		if name == "search_japanese_evidence":
			return self.nrdb.search_japanese_evidence(
				arguments["query"], schema_id, item["sentence_id"],
				region=item.get("dialect_region"),
				dialect_ids=self._id_pass_dialect_ids,
				limit=min(8, arguments["limit"]),
			)
		return super()._tool_result_reverse(name, arguments, item, schema_id)

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
			corpus_forms = []
			for value in result.get("corpus_forms", [])[:10]:
				corpus_forms.append({
					"surface": value.get("surface"), "count": value.get("count"),
					"dialect_id": value.get("dialect_id"), "dialect_name": value.get("dialect_name"),
					"dialect_region": value.get("dialect_region"), "source_kinds": value.get("source_kinds"),
					"example_sentence_id": value.get("example_sentence_id"),
				})
			lexicon_forms = []
			for value in result.get("lexicon_forms", [])[:6]:
				lexicon_forms.append({
					"dialect_id": value.get("dialect_id"), "dialect_name": value.get("dialect_name"),
					"dialect_region": value.get("dialect_region"), "form1": value.get("form1"),
					"form2": value.get("form2"), "form1_seg": value.get("form1_seg"),
					"form2_seg": value.get("form2_seg"), "meaning_jp": value.get("meaning_jp"),
				})
			return {
				"label": result.get("label"), "annotation_schema_id": result.get("annotation_schema_id"),
				"dialect_ids": result.get("dialect_ids"), "region": result.get("region"),
				"surface_scope": result.get("surface_scope"),
				"transcription_layer": result.get("transcription_layer"),
				"corpus_source_kinds": result.get("corpus_source_kinds"),
				"corpus_forms": corpus_forms, "lexicon_forms": lexicon_forms,
			}
		return _compact_tool_result(name, result)

	def _realize_surface(self, item, job, id_result):
		dialect_ids = job.get("target_dialect_ids") or [int(item["dialect_id"])]
		payload = {
			"japanese": str(item.get("translation_jp") or "").strip(),
			"frozen_annotation": id_result["annotation"],
			"japanese_reservoir_ids": id_result.get("evidence", {}).get("japanese_reservoir_ids", []),
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
					compact = {"budget_exhausted": True, "message": "Surface evidence budget exhausted; finalize conservatively. Frozen n: IDs may still be realized productively as Japanese lexical material in trsc2."}
				else:
					result = self._surface_result(call.name, arguments, item, job)
					compact = self._compact_surface(call.name, result)
					calls_used += 1
					evidence.append({"tool": call.name, "arguments": arguments, "result": compact})
					if call.name == "surface_forms_for_id":
						self.progress("    <- surface_forms_for_id({}): scope={} corpus={} lexicon={}".format(
							arguments.get("label"), result.get("surface_scope"),
							len(result.get("corpus_forms", [])), len(result.get("lexicon_forms", [])),
						))
					else:
						self.progress("    <- corpus_examples: {} example(s)".format(len(result.get("examples", []))))
				continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})
			if calls_used >= self.max_surface_evidence_calls:
				break
			response = self._create_response(continuation, SURFACE_INSTRUCTIONS, tools=SURFACE_TOOLS, max_output_tokens=900, text_format=SURFACE_FORMAT)

		final_input = list(base_input)
		if evidence:
			final_input.append({"role": "user", "content": "Retrieved surface evidence:\n" + json.dumps(evidence[-8:], ensure_ascii=False)})
		final_input.append({"role": "user", "content": "Evidence gathering is finished. Do not call tools. Return the most conservative trsc2 segmented realization now. Respect requested-dialect evidence whenever surface_scope=requested_dialects; use regional forms only when surface_scope=region_fallback. For any frozen n: ID lacking attested surface evidence, realize its Japanese lexical content with ordinary Japanese pronunciation in trsc2 and embed it in the surrounding Ryukyuan grammar."})
		response = self._create_response(final_input, SURFACE_INSTRUCTIONS, tools=[], max_output_tokens=1200, text_format=SURFACE_FORMAT)
		return self._parse_surface(response.output_text)

	def annotate(self, item, job, morph_result=None):
		self._id_pass_dialect_ids = job.get("target_dialect_ids") or [int(item["dialect_id"])]
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
