import json
from copy import deepcopy

from .annotator import (
	AnnotationAgent,
	TRANSLATION_FORMAT,
	_compact_tool_result,
	_response_incomplete_reason,
	_response_output_as_input,
	_trace_arguments,
	_trace_result,
	parse_translation_json,
)


V7_TRANSLATION_INSTRUCTIONS = """You are the constrained NRDB Japanese translation phase for annotation-v7.
The segmentation and morphemic annotation supplied to you have already been finalized and are FROZEN. You must not revise, reinterpret, or replace them.

Core principles:
- Most NRDB annotation labels are IDENTIFIERS, not glosses. Do not normally infer lexical meaning from kanji, spelling, or other human-readable material embedded in an ID label.
- Exception: IDs in the reserved `n:` namespace explicitly represent Japanese lexical material. For example, `n:手紙` directly licenses Japanese 手紙 and does not require dictionary grounding to establish that lexical meaning. Treat the material following `n:` as Japanese lexical content, while still respecting surrounding Miyako grammar and constructional evidence.
- Outside the reserved `n:` namespace, dictionary and corpus evidence outrank apparent semantics suggested by an ID's spelling. Only as a last resort, when an otherwise ungrounded lexical ID is transparently descriptive and no retrieved evidence contradicts it, you may use that transparent lexical content conservatively. Record such use as weak/ungrounded evidence rather than as authoritative dictionary grounding.
- Schema-local IDs beginning with `l:` are local lexical IDs, `exp:` IDs are expressives, and `intj:` IDs are interjectives. Treat those three namespaces as content-bearing and dictionary-ground them when they contribute to the translation. Provisionally treat every other schema-local ID as grammatical; this includes `dm:` demonstratives. This is a routing policy, not a claim that grammatical IDs lack meaning or that lexical IDs cannot participate in constructions.
- Global IDs have a Japanese-character stem followed by a structural code such as `kn`, `nv`, or `ia`, optionally followed by a numeral, as in `漢字kn` or `漢字kn2`. The final `n`, `v`, `a`, or `o` before any numeral is coarse structural POS metadata (nominal, verbal, adjectival, or other) and may be used to match N/V/A/X construction placeholders. The Japanese characters remain semantically opaque identifiers: recover lexical meaning from dictionary evidence, not from the characters. Global IDs may also anchor or participate in grammatical constructions.
- When construction_evidence is supplied, it contains explicit grammatical knowledge curated in NRDB. A row with entry_type `morpheme` is an applicable default interpretive policy when its exact trigger_id occurs; consult it, but do not treat realization_jp as an automatic substitution. For entry_type `construction`, a trigger hit alone does NOT prove that the construction applies: verify that its full pattern fits the relevant annotation span. A matching construction specializes and may override the general morpheme policy.

Goal: produce a concise, natural Japanese translation grounded in the frozen annotation, explicit constructional evidence when enabled, bilingual dictionary data, and selectively retrieved corpus evidence.

Rules:
- First inspect any supplied construction_evidence. Pattern notation is lightweight linguistic notation over NRDB annotation: V/N/A/X are schematic content-word placeholders; literal IDs occur as written; `;`, `-`, and spaces retain their normal NRDB annotation meanings.
- Consult every retrieved `morpheme` row on its exact trigger hit, then choose its contextually appropriate realization. Apply a `construction` row only when its full pattern fits; do not apply a construction merely because its trigger_id is present.
- When a construction applies, interpret the WHOLE construction: it outranks a conflicting default atom-by-atom interpretation. Its meaning_jp is strong grammatical evidence and realization_jp is a strong Japanese realization hint, not a blind string-substitution command. Ground captured/content lexical IDs from the dictionary and realize the construction naturally in context.
- Audit grammar use by ID. List every consulted morpheme row in consulted_morpheme_entry_ids. Classify every retrieved construction row exactly once as applied or rejected in applied_construction_entry_ids or rejected_construction_entry_ids. Use only IDs supplied in construction_evidence; when no rows are supplied, return empty arrays.
- Before translating, ground the non-`n:` lexical/content IDs that contribute referential or predicate meaning with ground_lexical_ids. Batch several IDs in one call. Dictionary meaning_jp/explanation_jp outrank any apparent meaning suggested by a non-`n:` ID label.
- `n:` IDs are already explicit Japanese lexical material and need not be sent to ground_lexical_ids merely to recover their Japanese meaning.
- ground_lexical_ids accepts exact atomic NRDB annotation IDs only. Never put commentary, questions, guessed decompositions, full phrases, or explanatory text inside the labels array.
- You do not need to ground obvious purely grammatical atoms such as tense, negation, case, topic/focus, or converbal markers unless their contribution is genuinely unclear.
- If a non-`n:` content ID has no usable dictionary grounding, record it in ungrounded_ids and translate conservatively from other evidence. As a final fallback only, transparent lexical material in that ID may be used when it is strongly obvious and is not contradicted by dictionary or corpus evidence.
- Use corpus_examples primarily for constructional or grammatical interpretation and for contextual disambiguation after lexical grounding. Corpus evidence may help choose among dictionary-attested senses, but should not replace dictionary grounding with an inference from the spelling of a non-`n:` ID.
- Prefer a short informative construction query. corpus_examples accepts at most 8 hyphen-separated segments and 256 characters.
- Preserve information explicitly encoded by the frozen analysis, including negation, tense/aspect, modality, case/argument structure, focus/topic, direction/location, quantification, and semantically relevant reduplication, except where an applicable curated construction explicitly establishes a non-compositional interpretation of the whole sequence.
- Produce natural Japanese rather than an interlinear gloss.
- Do not add semantic information that is not licensed by constructional evidence, dictionary grounding, reserved `n:` Japanese material, the frozen analysis, source context, corpus evidence, or the last-resort transparent-ID fallback described above.
- When evidence is incomplete, prefer a conservative translation over an imaginative guess.
- Do not produce chain-of-thought.

Final response must be one JSON object and no surrounding prose:
{"trsl_ai":"...","confidence":0.0,"translation_evidence":{"dictionary_ids":[],"example_sentence_ids":[],"ungrounded_ids":[],"consulted_morpheme_entry_ids":[],"applied_construction_entry_ids":[],"rejected_construction_entry_ids":[],"note":"brief"}}
"""


V7_TRANSLATION_FORMAT = deepcopy(TRANSLATION_FORMAT)
V7_EVIDENCE_SCHEMA = V7_TRANSLATION_FORMAT["schema"]["properties"]["translation_evidence"]
V7_EVIDENCE_SCHEMA["properties"].update({
	"consulted_morpheme_entry_ids": {"type": "array", "items": {"type": "integer"}},
	"applied_construction_entry_ids": {"type": "array", "items": {"type": "integer"}},
	"rejected_construction_entry_ids": {"type": "array", "items": {"type": "integer"}},
})
V7_EVIDENCE_SCHEMA["required"].extend([
	"consulted_morpheme_entry_ids", "applied_construction_entry_ids", "rejected_construction_entry_ids",
])


GROUND_LEXICAL_IDS_TOOL = {
	"type": "function",
	"name": "ground_lexical_ids",
	"description": "Batch-ground exact atomic lexical/content NRDB annotation IDs in the bilingual dictionary. Reserved n: IDs already encode Japanese lexical material and normally do not need this tool merely to recover their Japanese meaning. Each labels item must be one existing annotation ID only: no commentary, questions, phrases, or explanatory text. Invalid items are returned as rejected evidence rather than aborting the tool call.",
	"parameters": {
		"type": "object",
		"properties": {
			"labels": {
				"type": "array",
				"items": {"type": "string", "maxLength": 128},
				"minItems": 1,
				"maxItems": 12,
			}
		},
		"required": ["labels"],
		"additionalProperties": False,
	},
	"strict": True,
}


class AnnotationAgentV7(AnnotationAgent):
	@staticmethod
	def _validate_grammar_audit(translation, candidates):
		evidence = translation.get("translation_evidence")
		if not isinstance(evidence, dict):
			raise ValueError("translation returned no audit evidence")
		if any(value.get("entry_type") not in ("morpheme", "construction") for value in candidates):
			raise ValueError("construction endpoint returned an invalid entry_type")

		def audit_ids(field):
			values = evidence.get(field)
			if not isinstance(values, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
				raise ValueError("translation returned invalid {}".format(field))
			if len(values) != len(set(values)):
				raise ValueError("translation returned duplicate IDs in {}".format(field))
			return set(values)

		consulted = audit_ids("consulted_morpheme_entry_ids")
		applied = audit_ids("applied_construction_entry_ids")
		rejected = audit_ids("rejected_construction_entry_ids")
		morpheme_ids = {int(value["id"]) for value in candidates if value.get("entry_type") == "morpheme"}
		construction_ids = {int(value["id"]) for value in candidates if value.get("entry_type") == "construction"}
		if consulted != morpheme_ids:
			raise ValueError("translation did not audit every retrieved morpheme row")
		if applied.intersection(rejected) or applied.union(rejected) != construction_ids:
			raise ValueError("translation did not classify every retrieved construction row exactly once")
		return translation

	def _ground_lexical_ids(self, labels, schema_id):
		grounded = []
		for raw_label in labels[:12]:
			label = str(raw_label or "").strip()
			if not label or len(label) > 128 or any(char.isspace() for char in label):
				grounded.append({
					"label": label,
					"lexical_entries": [],
					"local": None,
					"global": None,
					"grounded": False,
					"rejected": True,
					"error": "Invalid lexical-grounding label; use one exact atomic NRDB annotation ID only.",
				})
				continue
			try:
				result = self.nrdb.lookup_id(label, schema_id)
			except RuntimeError as error:
				grounded.append({
					"label": label,
					"lexical_entries": [],
					"local": None,
					"global": None,
					"grounded": False,
					"rejected": True,
					"error": str(error),
				})
				continue
			compact = _compact_tool_result("lookup_id", result)
			grounded.append({
				"label": label,
				"lexical_entries": compact.get("lexical_entries", []),
				"local": compact.get("local"),
				"global": compact.get("global"),
				"grounded": bool(compact.get("lexical_entries")),
				"rejected": False,
			})
		return {"labels": grounded}

	def _construction_candidates(self, item, job, result):
		if not job.get("use_constructions"):
			return []
		payload = self.nrdb.construction_candidates(
			result.get("annotation", ""),
			int(job["annotation_schema_id"]),
			region=item.get("dialect_region"),
			dialect_id=item.get("dialect_id"),
		)
		candidates = list(payload.get("candidates", []))[:50]
		self.progress("  translation-v7: construction pass candidates={}".format(len(candidates)))
		return candidates

	def _v7_translation_tools(self):
		return [GROUND_LEXICAL_IDS_TOOL, self._tool_by_name("corpus_examples")]

	def _tool_by_name(self, name):
		from .annotator import TOOLS
		for tool in TOOLS:
			if tool.get("name") == name:
				return tool
		raise KeyError(name)

	def _v7_tool_result(self, name, arguments, item, schema_id):
		if name == "ground_lexical_ids":
			return self._ground_lexical_ids(arguments["labels"], schema_id)
		return self._tool_result(name, arguments, item, schema_id)

	def _v7_compact_result(self, name, result):
		if name == "ground_lexical_ids":
			return result
		return _compact_tool_result(name, result)

	def _v7_trace_arguments(self, name, arguments):
		if name == "ground_lexical_ids":
			return "labels={}".format(arguments.get("labels", []))
		return _trace_arguments(name, arguments)

	def _v7_trace_result(self, name, result):
		if name == "ground_lexical_ids":
			labels = result.get("labels", [])
			return "grounded={}/{} rejected={}".format(
				sum(1 for value in labels if value.get("grounded")), len(labels),
				sum(1 for value in labels if value.get("rejected")),
			)
		return _trace_result(name, result)

	def _has_dictionary_grounding(self, evidence_summary):
		return any(entry.get("tool") == "ground_lexical_ids" for entry in evidence_summary)

	def _finalize_translation_v7(self, base_input, evidence_summary, grammar_candidates, reason="evidence complete"):
		if not self._has_dictionary_grounding(evidence_summary):
			raise RuntimeError("translation-v7 cannot finalize without dictionary grounding")
		final_input = list(base_input)
		if evidence_summary:
			final_input.append({"role": "user", "content": "Retrieved compact translation evidence:\n" + json.dumps(evidence_summary[-4:], ensure_ascii=False)})
		final_input.append({"role": "user", "content": "Evidence gathering is finished ({}). Do not call tools. Respect any applicable curated construction evidence and dictionary grounding, and return the final conservative Japanese translation now.".format(reason)})
		last_error = None
		for attempt, budget in enumerate((1200, 1800), start=1):
			self.progress("  translation-v7: forced finalization attempt {} (max_output_tokens={})".format(attempt, budget))
			response = self._create_response(final_input, V7_TRANSLATION_INSTRUCTIONS, tools=[], max_output_tokens=budget, text_format=V7_TRANSLATION_FORMAT)
			incomplete_reason = _response_incomplete_reason(response)
			if incomplete_reason:
				last_error = RuntimeError("translation response incomplete: {}".format(incomplete_reason))
				continue
			try:
				return self._validate_grammar_audit(parse_translation_json(response.output_text), grammar_candidates)
			except (json.JSONDecodeError, ValueError) as error:
				last_error = error
		if last_error:
			raise last_error
		raise RuntimeError("translation-v7 finalization failed")

	def _translate_frozen_v7(self, item, job, result):
		construction_candidates = self._construction_candidates(item, job, result)

		def finish(translation):
			self._validate_grammar_audit(translation, construction_candidates)
			translation.setdefault("translation_evidence", {})["construction_candidates"] = construction_candidates
			return translation

		payload = {
			"sentence_id": int(item["sentence_id"]),
			"source_text": item["text"],
			"existing_translation_jp": item.get("translation_jp"),
			"frozen_segmented": result["segmented"],
			"frozen_annotation": result["annotation"],
			"annotation_decision": result["decision"],
			"annotation_confidence": result["confidence"],
			"annotation_schema_id": int(job["annotation_schema_id"]),
		}
		if job.get("use_constructions"):
			payload["construction_evidence"] = {
				"enabled": True,
				"candidate_count": len(construction_candidates),
				"candidates": construction_candidates,
				"instruction": "Consult every morpheme policy on its exact trigger hit without blind substitution. Classify every construction candidate as applied or rejected; apply it only when the full pattern fits the frozen annotation span.",
			}
		base_input = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
		evidence_summary = []
		evidence_calls = 0
		tools = self._v7_translation_tools()
		self.progress("  translation-v7: initial response (dictionary-grounded; budget {}/{})".format(evidence_calls, self.max_translation_evidence_calls))
		response = self._create_response(base_input, V7_TRANSLATION_INSTRUCTIONS, tools=tools, max_output_tokens=900)
		for round_index in range(1, self.max_rounds + 1):
			calls = [output for output in response.output if getattr(output, "type", None) == "function_call"]
			if not calls:
				if not self._has_dictionary_grounding(evidence_summary):
					self.progress("  translation-v7: dictionary grounding required before finalization")
					response = self._create_response(
						base_input + [{"role": "user", "content": "You must call ground_lexical_ids for the non-n: lexical/content IDs before translating. Reserved n: IDs already encode Japanese lexical material."}],
						V7_TRANSLATION_INSTRUCTIONS, tools=tools, max_output_tokens=900,
					)
					continue
				incomplete_reason = _response_incomplete_reason(response)
				if incomplete_reason:
					return finish(self._finalize_translation_v7(base_input, evidence_summary, construction_candidates, "previous response incomplete: {}".format(incomplete_reason)))
				try:
					translation = self._validate_grammar_audit(parse_translation_json(response.output_text), construction_candidates)
				except (json.JSONDecodeError, ValueError):
					return finish(self._finalize_translation_v7(base_input, evidence_summary, construction_candidates, "previous final JSON malformed or grammar audit incomplete"))
				self.progress("  translation-v7: final confidence={:.3f}".format(translation["confidence"]))
				return finish(translation)

			self.progress("  translation-v7 tool round {}: {} call(s)".format(round_index, len(calls)))
			continuation = list(base_input)
			if evidence_summary:
				continuation.append({"role": "user", "content": "Previously retrieved compact translation evidence:\n" + json.dumps(evidence_summary[-4:], ensure_ascii=False)})
			continuation.extend(_response_output_as_input(response))
			for call in calls:
				arguments = json.loads(call.arguments)
				self.progress("    -> {}({})".format(call.name, self._v7_trace_arguments(call.name, arguments)))
				if call.name == "corpus_examples" and not self._has_dictionary_grounding(evidence_summary):
					compact = {"grounding_required": True, "message": "Ground non-n: lexical/content IDs with ground_lexical_ids before corpus construction search. Reserved n: IDs already encode Japanese lexical material."}
					self.progress("    <- corpus_examples: skipped (dictionary grounding required first)")
				elif evidence_calls >= self.max_translation_evidence_calls:
					compact = {"budget_exhausted": True, "message": "Translation evidence-call budget exhausted; translate conservatively from curated constructions, dictionary grounding, reserved n: Japanese material, and existing evidence."}
					self.progress("    <- {}: skipped (translation evidence budget exhausted)".format(call.name))
				else:
					tool_result = self._v7_tool_result(call.name, arguments, item, int(job["annotation_schema_id"]))
					compact = self._v7_compact_result(call.name, tool_result)
					self.progress("    <- {}: {}".format(call.name, self._v7_trace_result(call.name, tool_result)))
					evidence_calls += 1
					evidence_summary.append({"tool": call.name, "arguments": arguments, "result": compact})
				continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})

			self.progress("  translation-v7: continue after tool round {} (evidence {}/{})".format(round_index, evidence_calls, self.max_translation_evidence_calls))
			if evidence_calls >= self.max_translation_evidence_calls:
				return finish(self._finalize_translation_v7(base_input, evidence_summary, construction_candidates, "translation evidence budget exhausted"))
			response = self._create_response(continuation, V7_TRANSLATION_INSTRUCTIONS, tools=tools, max_output_tokens=900)
		raise RuntimeError("translation-v7 phase exceeded maximum tool rounds")

	def annotate(self, item, job, morph_result):
		annotation_job = dict(job)
		annotation_job["prompt_version"] = "annotation-v6"
		annotation_job["produce_translation"] = False
		annotation_job["blind_translation"] = False
		result = super().annotate(item, annotation_job, morph_result)
		result["trsl_ai"] = ""
		if job.get("produce_translation") and result.get("annotation") and result["decision"] != "failed":
			translation = self._translate_frozen_v7(item, job, result)
			result["trsl_ai"] = translation["trsl_ai"]
			result["evidence"]["translation"] = translation["translation_evidence"]
			result["evidence"]["translation"]["confidence"] = translation["confidence"]
		return result
