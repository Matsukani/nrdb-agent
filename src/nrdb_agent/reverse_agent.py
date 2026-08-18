import json

from .annotator import AnnotationAgent, _compact_tool_result, _response_incomplete_reason, _response_output_as_input, _trace_arguments, _trace_result


REVERSE_INSTRUCTIONS = """You are the constrained NRDB Japanese-to-Miyako ID agent.

You receive ONLY a Japanese sentence plus target dialect metadata. The original Miyako transcription and gold annotation are hidden from you.

Goal: predict the most defensible Miyako NRDB morphemic-ID annotation corresponding to the Japanese sentence. This first experiment predicts IDs only; do not generate Miyako surface forms.

Core linguistic model: asymmetric dual lexical access.
- Miyako/Ryukyuan normally supplies the morphosyntactic frame, while lexical material may come either from the local Ryukyuan lexicon or from the synchronically Japanese lexical reservoir.
- The reserved namespace n: marks synchronically Japanese-layer material embedded in Ryukyuan discourse. It is NOT an etymology marker and it is NOT a closed dictionary inventory.
- n: is productive: when Japanese lexical recruitment is justified, n:<Japanese lexical lemma or lexical expression> may be created directly from the Japanese input even if that exact n: ID has never occurred in NRDB. This is the ONE exception to the no-invented-ID rule.
- Japanese lexical recruitment is especially expected for modern, institutional, technical, professional, standardized, numerical, temporal, measurement, proper-name/title and formulaic vocabulary, but it is not restricted to these domains.
- A missing or weak local Ryukyuan lexical candidate is strong evidence for using n: rather than deleting the concept or forcing an unrelated local word.
- Japanese is not merely a lexical-gap filler: if corpus evidence clearly favors an n: realization even where a local equivalent exists, the Japanese-layer choice is legitimate.
- Prefer lexical recruitment inside Ryukyuan grammar over unnecessary full Japanese switching. In ordinary mixed realization, keep case, topic/focus, clause linkage, TAM and other grammatical packaging Ryukyuan whenever evidence supports it.
- For multiword Japanese lexical expressions or institutional compounds, preserve a natural lexical unit and use one or more n: atoms as supported by corpus patterns; do not attach n: to Japanese inflectional endings or particles merely because they appear in the source sentence.

Rules:
- Use your Japanese linguistic competence directly. There is no separate Japanese parser in this baseline.
- Before finalizing, call search_japanese_evidence for the important lexical predicates/arguments and, when useful, for a short informative Japanese construction. This tool searches bilingual lexical resources and translated annotated corpus examples.
- Use lookup_id to verify ordinary candidate IDs once discovered.
- Use corpus_examples for short ID constructions after you know relevant IDs.
- Never infer lexical meaning from the visual spelling or kanji of an ordinary NRDB ID. IDs are identifiers, not glosses. The explicit exception is n:, whose Japanese lexical content is intentionally semantically transparent because n: denotes the Japanese lexical reservoir.
- Never invent an ordinary local/global NRDB ID. Every non-n: content ID must be licensed by retrieved lexical or corpus evidence. Grammatical IDs should likewise be supported by retrieved corpus patterns whenever possible.
- Preserve Japanese predicate-argument structure, negation, tense/aspect, modality, quantification, information structure, and clause relations while expressing them through attested Miyako/Ryukyuan grammatical structure where possible.
- Do not omit a content concept merely because no local Miyako lexical entry is found; consider productive n: recruitment.
- Output the predicted NRDB annotation in normal NRDB annotation syntax using spaces for phrases, hyphens for segments, and semicolons only for conflated atoms when supported by evidence.
- This is an ID-transfer experiment. Do not output Miyako transcription or segmentation.
- If grammatical evidence is genuinely insufficient, prefer UNCERTAIN rather than fabricating local grammar. Lack of a local lexical item alone is NOT a reason for failure when n: recruitment is available.
- Do not ask for or infer the hidden Miyako source or gold annotation.
- Do not produce chain-of-thought. Evidence notes must be concise and auditable.

Return exactly one JSON object:
{"annotation":"...","decision":"proposed|uncertain|failed","confidence":0.0,"evidence":{"note":"brief","japanese_queries":[],"ids_checked":[],"japanese_reservoir_ids":[],"example_sentence_ids":[]}}
"""

REVERSE_FORMAT = {
	"type": "json_schema",
	"name": "nrdb_reverse_ids",
	"strict": True,
	"schema": {
		"type": "object",
		"properties": {
			"annotation": {"type": "string"},
			"decision": {"type": "string", "enum": ["proposed", "uncertain", "failed"]},
			"confidence": {"type": "number"},
			"evidence": {
				"type": "object",
				"properties": {
					"note": {"type": "string"},
					"japanese_queries": {"type": "array", "items": {"type": "string"}},
					"ids_checked": {"type": "array", "items": {"type": "string"}},
					"japanese_reservoir_ids": {"type": "array", "items": {"type": "string"}},
					"example_sentence_ids": {"type": "array", "items": {"type": "integer"}},
				},
				"required": ["note", "japanese_queries", "ids_checked", "japanese_reservoir_ids", "example_sentence_ids"],
				"additionalProperties": False,
			},
		},
		"required": ["annotation", "decision", "confidence", "evidence"],
		"additionalProperties": False,
	},
}

SEARCH_JAPANESE_TOOL = {
	"type": "function",
	"name": "search_japanese_evidence",
	"description": "Search NRDB bilingual lexical resources and translated annotated corpus examples from a short Japanese word or phrase. Use this to discover local Miyako IDs and attested n: Japanese-layer choices from Japanese meaning. If no suitable local or attested n: lexical entry exists, productive n: recruitment may still be used for a Japanese lexical lemma.",
	"parameters": {
		"type": "object",
		"properties": {
			"query": {"type": "string"},
			"limit": {"type": "integer", "minimum": 1, "maximum": 8},
		},
		"required": ["query", "limit"],
		"additionalProperties": False,
	},
	"strict": True,
}


def _tool_by_name(name):
	from .annotator import TOOLS
	for tool in TOOLS:
		if tool.get("name") == name:
			return tool
	raise KeyError(name)


REVERSE_TOOLS = [SEARCH_JAPANESE_TOOL, _tool_by_name("lookup_id"), _tool_by_name("corpus_examples")]


class ReverseIdAgent(AnnotationAgent):
	def __init__(self, *args, max_reverse_evidence_calls=6, **kwargs):
		super().__init__(*args, **kwargs)
		self.max_reverse_evidence_calls = int(max_reverse_evidence_calls)

	def _parse(self, text):
		payload = json.loads((text or "").strip())
		if payload.get("decision") not in {"proposed", "uncertain", "failed"}:
			raise ValueError("invalid reverse decision")
		payload["annotation"] = str(payload.get("annotation") or "").strip()
		payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
		payload["evidence"] = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
		payload["segmented"] = ""
		payload["trsl_ai"] = ""
		return payload

	def _tool_result_reverse(self, name, arguments, item, schema_id):
		if name == "search_japanese_evidence":
			return self.nrdb.search_japanese_evidence(
				arguments["query"], schema_id, item["sentence_id"],
				region=item.get("dialect_region"), limit=min(8, arguments["limit"]),
			)
		return self._tool_result(name, arguments, item, schema_id)

	def _compact_reverse(self, name, result):
		if name == "search_japanese_evidence":
			lexical = []
			for entry in result.get("lexical_entries", [])[:8]:
				lexical.append({
					"label": entry.get("label"),
					"form1": entry.get("form1"),
					"form2": entry.get("form2"),
					"meaning_jp": entry.get("meaning_jp"),
					"pos": entry.get("pos"),
					"dialect_name": entry.get("dialect_name"),
				})
			examples = []
			for example in result.get("corpus_examples", [])[:8]:
				examples.append({
					"sentence_id": example.get("sentence_id"),
					"translation_jp": example.get("translation_jp"),
					"annotation": example.get("annotation"),
				})
			return {"query": result.get("query"), "region": result.get("region"), "lexical_entries": lexical, "corpus_examples": examples}
		return _compact_tool_result(name, result)

	def _trace_args_reverse(self, name, arguments):
		if name == "search_japanese_evidence":
			return "query={!r} limit={}".format(arguments.get("query", ""), arguments.get("limit", ""))
		return _trace_arguments(name, arguments)

	def _trace_result_reverse(self, name, result):
		if name == "search_japanese_evidence":
			return "lexical={} examples={}".format(len(result.get("lexical_entries", [])), len(result.get("corpus_examples", [])))
		return _trace_result(name, result)

	def _finalize(self, base_input, evidence_summary, reason):
		final_input = list(base_input)
		if evidence_summary:
			final_input.append({"role": "user", "content": "Retrieved compact NRDB evidence:\n" + json.dumps(evidence_summary[-6:], ensure_ascii=False)})
		final_input.append({"role": "user", "content": "Evidence gathering is finished ({}). Do not call tools. Return the best conservative Miyako NRDB ID annotation now. Remember that n: is a productive Japanese lexical reservoir: if a Japanese lexical concept lacks a convincing local realization, recruit it as n:<Japanese lexical lemma> and keep the surrounding grammar Ryukyuan. Return UNCERTAIN only when the grammatical analysis itself remains insufficient.".format(reason)})
		last_error = None
		for attempt, budget in enumerate((1200, 1800), start=1):
			self.progress("  reverse-v1: forced finalization attempt {}".format(attempt))
			response = self._create_response(final_input, REVERSE_INSTRUCTIONS, tools=[], max_output_tokens=budget, text_format=REVERSE_FORMAT)
			if _response_incomplete_reason(response):
				continue
			try:
				result = self._parse(response.output_text)
				result["model_response_id"] = response.id
				return result
			except (json.JSONDecodeError, ValueError) as error:
				last_error = error
		if last_error:
			raise last_error
		raise RuntimeError("reverse-v1 finalization failed")

	def annotate(self, item, job, morph_result=None):
		japanese = str(item.get("translation_jp") or "").strip()
		if not japanese:
			raise RuntimeError("reverse-v1 requires a Japanese translation")
		payload = {
			"sentence_id": int(item["sentence_id"]),
			"japanese": japanese,
			"target_dialect_id": int(item["dialect_id"]),
			"target_region": item.get("dialect_region"),
			"annotation_schema_id": int(job["annotation_schema_id"]),
		}
		base_input = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
		evidence_summary = []
		evidence_calls = 0
		searched_japanese = False
		self.progress("  reverse-v1: Japanese -> Miyako IDs")
		response = self._create_response(base_input, REVERSE_INSTRUCTIONS, tools=REVERSE_TOOLS, max_output_tokens=900)
		for round_index in range(1, self.max_rounds + 1):
			calls = [output for output in response.output if getattr(output, "type", None) == "function_call"]
			if not calls:
				if not searched_japanese:
					response = self._create_response(
						base_input + [{"role": "user", "content": "Before finalizing you must call search_japanese_evidence for the important Japanese lexical material. If a concept has no convincing local lexical realization, remember that productive n: recruitment is available."}],
						REVERSE_INSTRUCTIONS, tools=REVERSE_TOOLS, max_output_tokens=900,
					)
					continue
				if _response_incomplete_reason(response):
					return self._finalize(base_input, evidence_summary, "previous response incomplete")
				try:
					result = self._parse(response.output_text)
					result["model_response_id"] = response.id
					return result
				except (json.JSONDecodeError, ValueError):
					return self._finalize(base_input, evidence_summary, "previous final JSON malformed")

			self.progress("  reverse-v1 tool round {}: {} call(s)".format(round_index, len(calls)))
			continuation = list(base_input)
			if evidence_summary:
				continuation.append({"role": "user", "content": "Previously retrieved compact NRDB evidence:\n" + json.dumps(evidence_summary[-6:], ensure_ascii=False)})
			continuation.extend(_response_output_as_input(response))
			for call in calls:
				arguments = json.loads(call.arguments)
				self.progress("    -> {}({})".format(call.name, self._trace_args_reverse(call.name, arguments)))
				if evidence_calls >= self.max_reverse_evidence_calls:
					compact = {"budget_exhausted": True, "message": "Reverse evidence budget exhausted; finalize conservatively. Productive n: recruitment remains available for unresolved Japanese lexical concepts."}
				else:
					tool_result = self._tool_result_reverse(call.name, arguments, item, int(job["annotation_schema_id"]))
					compact = self._compact_reverse(call.name, tool_result)
					self.progress("    <- {}: {}".format(call.name, self._trace_result_reverse(call.name, tool_result)))
					evidence_calls += 1
					if call.name == "search_japanese_evidence":
						searched_japanese = True
					evidence_summary.append({"tool": call.name, "arguments": arguments, "result": compact})
				continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})
			if evidence_calls >= self.max_reverse_evidence_calls:
				return self._finalize(base_input, evidence_summary, "reverse evidence budget exhausted")
			response = self._create_response(continuation, REVERSE_INSTRUCTIONS, tools=REVERSE_TOOLS, max_output_tokens=900)
		return self._finalize(base_input, evidence_summary, "maximum tool rounds reached")
