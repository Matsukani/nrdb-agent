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

Evidence-efficiency rules:
- Your FIRST lexical evidence action should normally be search_japanese_batch. Put several high-impact lexical predicates/arguments into one call instead of spending one tool call per word.
- Batch only concepts whose resolution can materially affect the annotation. Main/subordinate predicates and ambiguous core arguments have priority over modifiers.
- Do NOT query concepts that are transparently appropriate for productive n: recruitment unless local-vs-Japanese choice is genuinely uncertain.
- Do NOT spend corpus queries on routine grammar in this phase. A separate statistical ID-sequence critic will inspect grammatical packaging after your proposal and will request corpus evidence only for actual surprise hotspots.
- After a batch result, use search_japanese_evidence only for a remaining ambiguity that the batch did not resolve.
- Use lookup_id only when a returned candidate is sparse, competing, semantically ambiguous, or otherwise decisive. Do not automatically double-verify a strong unique batch result.
- One search_japanese_batch call counts as one evidence call regardless of how many queries it contains.

Rules:
- Use your Japanese linguistic competence directly. There is no separate Japanese parser in this baseline.
- Never infer lexical meaning from the visual spelling or kanji of an ordinary NRDB ID. IDs are identifiers, not glosses. The explicit exception is n:, whose Japanese lexical content is intentionally semantically transparent because n: denotes the Japanese lexical reservoir.
- Never invent an ordinary local/global NRDB ID. Every non-n: content ID must be licensed by retrieved lexical evidence. Routine grammatical packaging may be proposed from established annotation knowledge and will be checked by the downstream ID critic.
- Preserve Japanese predicate-argument structure, negation, tense/aspect, modality, quantification, information structure, and clause relations while expressing them through Miyako/Ryukyuan grammatical structure where possible.
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
	"description": "Targeted follow-up search for one unresolved Japanese lexical ambiguity after the batch search.",
	"parameters": {
		"type": "object",
		"properties": {
			"query": {"type": "string", "maxLength": 80},
			"limit": {"type": "integer", "minimum": 1, "maximum": 8},
		},
		"required": ["query", "limit"],
		"additionalProperties": False,
	},
	"strict": True,
}

SEARCH_JAPANESE_BATCH_TOOL = {
	"type": "function",
	"name": "search_japanese_batch",
	"description": "Batch lexical discovery for several high-impact Japanese concepts in ONE evidence call. Use this first. Each query searches bilingual lexical resources and translated corpus evidence; choose predicates/core arguments whose lexical resolution matters and omit obvious productive n: material.",
	"parameters": {
		"type": "object",
		"properties": {
			"queries": {
				"type": "array", "minItems": 1, "maxItems": 8,
				"items": {"type": "string", "maxLength": 80},
			},
			"limit": {"type": "integer", "minimum": 1, "maximum": 6},
		},
		"required": ["queries", "limit"],
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


# Grammar corpus queries are intentionally absent here. The downstream ID critic
# receives that tool only after it has identified a strong grammatical surprise.
REVERSE_TOOLS = [SEARCH_JAPANESE_BATCH_TOOL, SEARCH_JAPANESE_TOOL, _tool_by_name("lookup_id")]


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

	def _one_japanese_search(self, query, item, schema_id, limit):
		return self.nrdb.search_japanese_evidence(
			query, schema_id, item["sentence_id"],
			region=item.get("dialect_region"),
			dialect_ids=getattr(self, "_id_pass_dialect_ids", None),
			limit=min(8, int(limit)),
		)

	def _tool_result_reverse(self, name, arguments, item, schema_id):
		if name == "search_japanese_batch":
			queries = []
			for value in arguments.get("queries", []):
				query = str(value or "").strip()
				if query and query not in queries:
					queries.append(query)
			queries = queries[:8]
			if not queries:
				raise ValueError("search_japanese_batch requires at least one non-empty query")
			limit = min(6, max(1, int(arguments.get("limit", 5))))
			return {
				"success": True,
				"queries": queries,
				"results": [self._one_japanese_search(query, item, schema_id, limit) for query in queries],
			}
		if name == "search_japanese_evidence":
			return self._one_japanese_search(arguments["query"], item, schema_id, arguments["limit"])
		return self._tool_result(name, arguments, item, schema_id)

	def _compact_single_search(self, result, lexical_limit=6, example_limit=4):
		lexical = []
		for entry in result.get("lexical_entries", [])[:lexical_limit]:
			lexical.append({
				"label": entry.get("label"), "form1": entry.get("form1"), "form2": entry.get("form2"),
				"meaning_jp": entry.get("meaning_jp"), "meaning_yomi": entry.get("meaning_yomi"),
				"pos": entry.get("pos"), "dialect_name": entry.get("dialect_name"),
			})
		examples = []
		for example in result.get("corpus_examples", [])[:example_limit]:
			examples.append({
				"sentence_id": example.get("sentence_id"),
				"translation_jp": example.get("translation_jp"),
				"annotation": example.get("annotation"),
			})
		return {"query": result.get("query"), "region": result.get("region"), "lexical_entries": lexical, "corpus_examples": examples}

	def _compact_reverse(self, name, result):
		if name == "search_japanese_batch":
			return {
				"queries": result.get("queries", []),
				"results": [self._compact_single_search(value, lexical_limit=5, example_limit=3) for value in result.get("results", [])[:8]],
			}
		if name == "search_japanese_evidence":
			return self._compact_single_search(result)
		return _compact_tool_result(name, result)

	def _trace_args_reverse(self, name, arguments):
		if name == "search_japanese_batch":
			return "queries={} limit={}".format(arguments.get("queries", []), arguments.get("limit", ""))
		if name == "search_japanese_evidence":
			return "query={!r} limit={}".format(arguments.get("query", ""), arguments.get("limit", ""))
		return _trace_arguments(name, arguments)

	def _trace_result_reverse(self, name, result):
		if name == "search_japanese_batch":
			lexical = sum(len(value.get("lexical_entries", [])) for value in result.get("results", []))
			examples = sum(len(value.get("corpus_examples", [])) for value in result.get("results", []))
			return "queries={} lexical={} examples={}".format(len(result.get("results", [])), lexical, examples)
		if name == "search_japanese_evidence":
			return "lexical={} examples={}".format(len(result.get("lexical_entries", [])), len(result.get("corpus_examples", [])))
		return _trace_result(name, result)

	def _finalize(self, base_input, evidence_summary, reason):
		final_input = list(base_input)
		if evidence_summary:
			final_input.append({"role": "user", "content": "Retrieved compact NRDB lexical evidence:\n" + json.dumps(evidence_summary[-6:], ensure_ascii=False)})
		final_input.append({"role": "user", "content": "Evidence gathering is finished ({}). Do not call tools. Return the best conservative Miyako NRDB ID annotation now. Routine grammar will be checked by a downstream ID-sequence critic, so do not invent extra lexical searches merely to verify common grammatical packaging. Remember that n: is a productive Japanese lexical reservoir.".format(reason)})
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
			"sentence_id": int(item["sentence_id"]), "japanese": japanese,
			"target_dialect_id": int(item["dialect_id"]), "target_region": item.get("dialect_region"),
			"annotation_schema_id": int(job["annotation_schema_id"]),
		}
		base_input = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
		evidence_summary = []
		evidence_calls = 0
		batch_searched = False
		self.progress("  reverse-v1: Japanese -> Miyako IDs (batch lexical triage)")
		response = self._create_response(base_input, REVERSE_INSTRUCTIONS, tools=REVERSE_TOOLS, max_output_tokens=900)
		for round_index in range(1, self.max_rounds + 1):
			calls = [output for output in response.output if getattr(output, "type", None) == "function_call"]
			if not calls:
				if not batch_searched:
					response = self._create_response(
						base_input + [{"role": "user", "content": "Before finalizing, make ONE search_japanese_batch call covering the high-impact lexical predicates/arguments whose local realization matters. A one-item batch is fine for a very short sentence. Do not query routine grammar or obvious productive n: material."}],
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
				continuation.append({"role": "user", "content": "Previously retrieved compact NRDB lexical evidence:\n" + json.dumps(evidence_summary[-6:], ensure_ascii=False)})
			continuation.extend(_response_output_as_input(response))
			for call in calls:
				arguments = json.loads(call.arguments)
				self.progress("    -> {}({})".format(call.name, self._trace_args_reverse(call.name, arguments)))
				if evidence_calls >= self.max_reverse_evidence_calls:
					compact = {"budget_exhausted": True, "message": "Reverse evidence budget exhausted; finalize conservatively. Productive n: recruitment remains available."}
				else:
					tool_result = self._tool_result_reverse(call.name, arguments, item, int(job["annotation_schema_id"]))
					compact = self._compact_reverse(call.name, tool_result)
					self.progress("    <- {}: {}".format(call.name, self._trace_result_reverse(call.name, tool_result)))
					evidence_calls += 1
					if call.name == "search_japanese_batch":
						batch_searched = True
					evidence_summary.append({"tool": call.name, "arguments": arguments, "result": compact})
				continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})
			if evidence_calls >= self.max_reverse_evidence_calls:
				return self._finalize(base_input, evidence_summary, "reverse evidence budget exhausted")
			response = self._create_response(continuation, REVERSE_INSTRUCTIONS, tools=REVERSE_TOOLS, max_output_tokens=900)
		return self._finalize(base_input, evidence_summary, "maximum tool rounds reached")
