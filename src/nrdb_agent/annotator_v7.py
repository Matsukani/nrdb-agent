import json

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

Goal: produce a concise, natural Japanese translation grounded in the frozen annotation, bilingual dictionary data, and selectively retrieved corpus evidence.

Rules:
- Before translating, ground the non-`n:` lexical/content IDs that contribute referential or predicate meaning with ground_lexical_ids. Batch several IDs in one call. Dictionary meaning_jp/explanation_jp outrank any apparent meaning suggested by a non-`n:` ID label.
- `n:` IDs are already explicit Japanese lexical material and need not be sent to ground_lexical_ids merely to recover their Japanese meaning.
- ground_lexical_ids accepts exact atomic NRDB annotation IDs only. Never put commentary, questions, guessed decompositions, full phrases, or explanatory text inside the labels array.
- You do not need to ground obvious purely grammatical atoms such as tense, negation, case, topic/focus, or converbal markers unless their contribution is genuinely unclear.
- If a non-`n:` content ID has no usable dictionary grounding, record it in ungrounded_ids and translate conservatively from other evidence. As a final fallback only, transparent lexical material in that ID may be used when it is strongly obvious and is not contradicted by dictionary or corpus evidence.
- Use corpus_examples primarily for constructional or grammatical interpretation and for contextual disambiguation after lexical grounding. Corpus evidence may help choose among dictionary-attested senses, but should not replace dictionary grounding with an inference from the spelling of a non-`n:` ID.
- Prefer a short informative construction query. corpus_examples accepts at most 8 hyphen-separated segments and 256 characters.
- Preserve information explicitly encoded by the frozen analysis, including negation, tense/aspect, modality, case/argument structure, focus/topic, direction/location, quantification, and semantically relevant reduplication.
- Produce natural Japanese rather than an interlinear gloss.
- Do not add semantic information that is not licensed by dictionary grounding, reserved `n:` Japanese material, the frozen analysis, source context, corpus evidence, or the last-resort transparent-ID fallback described above.
- When evidence is incomplete, prefer a conservative translation over an imaginative guess.
- Do not produce chain-of-thought.

Final response must be one JSON object and no surrounding prose:
{"trsl_ai":"...","confidence":0.0,"translation_evidence":{"dictionary_ids":[],"example_sentence_ids":[],"ungrounded_ids":[],"note":"brief"}}
"""


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

	def _v7_translation_tools(self):
		return [
			GROUND_LEXICAL_IDS_TOOL,
			self._tool_by_name("corpus_examples"),
		]

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
				sum(1 for value in labels if value.get("grounded")),
				len(labels),
				sum(1 for value in labels if value.get("rejected")),
			)
		return _trace_result(name, result)

	def _has_dictionary_grounding(self, evidence_summary):
		return any(entry.get("tool") == "ground_lexical_ids" for entry in evidence_summary)

	def _finalize_translation_v7(self, base_input, evidence_summary, reason="evidence complete"):
		if not self._has_dictionary_grounding(evidence_summary):
			raise RuntimeError("translation-v7 cannot finalize without dictionary grounding")
		final_input = list(base_input)
		if evidence_summary:
			final_input.append({"role": "user", "content": "Retrieved compact translation evidence:\n" + json.dumps(evidence_summary[-4:], ensure_ascii=False)})
		final_input.append({"role": "user", "content": "Evidence gathering is finished ({}). Do not call tools. Respect dictionary grounding and return the final conservative Japanese translation now.".format(reason)})
		last_error = None
		for attempt, budget in enumerate((1200, 1800), start=1):
			self.progress("  translation-v7: forced finalization attempt {} (max_output_tokens={})".format(attempt, budget))
			response = self._create_response(final_input, V7_TRANSLATION_INSTRUCTIONS, tools=[], max_output_tokens=budget, text_format=TRANSLATION_FORMAT)
			incomplete_reason = _response_incomplete_reason(response)
			if incomplete_reason:
				last_error = RuntimeError("translation response incomplete: {}".format(incomplete_reason))
				continue
			try:
				return parse_translation_json(response.output_text)
			except (json.JSONDecodeError, ValueError) as error:
				last_error = error
		if last_error:
			raise last_error
		raise RuntimeError("translation-v7 finalization failed")

	def _translate_frozen_v7(self, item, job, result):
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
					return self._finalize_translation_v7(base_input, evidence_summary, "previous response incomplete: {}".format(incomplete_reason))
				try:
					translation = parse_translation_json(response.output_text)
				except (json.JSONDecodeError, ValueError):
					return self._finalize_translation_v7(base_input, evidence_summary, "previous final JSON malformed")
				self.progress("  translation-v7: final confidence={:.3f}".format(translation["confidence"]))
				return translation

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
					compact = {"budget_exhausted": True, "message": "Translation evidence-call budget exhausted; translate conservatively from dictionary grounding, reserved n: Japanese material, and existing evidence."}
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
				return self._finalize_translation_v7(base_input, evidence_summary, "translation evidence budget exhausted")
			response = self._create_response(continuation, V7_TRANSLATION_INSTRUCTIONS, tools=tools, max_output_tokens=900)
		raise RuntimeError("translation-v7 phase exceeded maximum tool rounds")

	def annotate(self, item, job, morph_result):
		# v7 intentionally reuses the proven v6 annotation behavior unchanged.
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
