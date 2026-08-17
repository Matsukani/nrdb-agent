import json
import re
import time

from openai import OpenAI, RateLimitError


BASE_INSTRUCTIONS = """You are the constrained NRDB morphemic annotation agent.
Your goal is to propose the most defensible segmentation and morphemic-ID annotation for one low-resource-language utterance.

Rules:
- Start from the nrdb-morph analysis supplied in the user message, but it is evidence rather than truth.
- Investigate only ambiguities that could change the final analysis. Do not look up every morpheme.
- Use lookup_id for targeted lexical/semantic grounding and corpus_examples for targeted human-reviewed usage.
- Use at most six evidence calls total before validation. If evidence remains incomplete, prefer UNCERTAIN over exhaustive searching.
- Never invent an annotation ID. Prefer IDs proposed by nrdb-morph or confirmed by lookup/corpus evidence.
- You may change segmentation only when evidence strongly supports it.
- Before a PROPOSED result, call validate_analysis on the complete final segmentation and annotation.
- If produce_translation is true, also generate a concise natural Japanese translation from the final analysis and return it as trsl_ai.
- If produce_translation is false, return trsl_ai as an empty string.
- If evidence is insufficient, return decision UNCERTAIN. If no defensible valid analysis exists, return FAILED.
- Do not ask for gold annotation and do not infer that it is available.
- Do not produce chain-of-thought. Evidence should contain only concise, auditable facts.

Final response must be one JSON object and no surrounding prose:
{"segmented":"...","annotation":"...","trsl_ai":"...","decision":"proposed|uncertain|failed","confidence":0.0,"evidence":{"note":"brief","labels_checked":[],"example_sentence_ids":[]}}
"""

V2_RULES = """

Miyako annotation conventions for annotation-v2:
- When a verb is followed by ipf morphology, the verbal lexical ID should normally carry ;cvb before ipf (for example V;cvb-ipf), unless strong corpus evidence shows otherwise.
- Within the same phrase, if two elements are repeated, the second occurrence is almost certainly the reduplication marker red. Prefer red for that second repeated occurrence unless strong contrary evidence exists.
"""

V3_ANNOTATION_RULES = """

annotation-v3 execution rule:
- Annotation and translation are separate phases. During this annotation phase, do not translate. Return trsl_ai as an empty string even when produce_translation is true. The validated annotation will be frozen before a separate translation phase begins.
"""

V4_RULES = """

Additional Miyako annotation convention for annotation-v4:
- afn may only follow an adjectival expression. Do not analyze a form as afn after a non-adjectival expression; in that environment choose another supported candidate such as ng:nu when the morphology and corpus evidence support it.
"""

V5_RULES = """

Additional Miyako annotation conventions for annotation-v5:
- The surface segment dui is never the focus marker foc. If a candidate assigns foc to a segment realized as dui, reject that analysis and choose another candidate supported by morphology and corpus evidence.
- The reduplication heuristic for red is strictly phrase-local. Only repetition inside the same whitespace-delimited phrase can license red. Never use a repeated element in another phrase elsewhere in the utterance as evidence for red. Inside one phrase, when an element is immediately or clearly repeated, the second occurrence is strongly expected to be red unless strong contrary evidence exists.
"""

V6_RULES = """

Additional evidence conventions for annotation-v6:
- Do not use a hard-coded surface-form exception when a form-to-ID choice can be checked empirically. For a doubtful pairing between a surface segment X and candidate annotation ID Y, use form_id_support(X, Y). This tool is automatically scoped to the current utterance's dialect region and searches aligned corpus and lexicon evidence.
- Interpret regional form-ID evidence comparatively, not categorically. A candidate with no regional support should receive a penalty, especially when the surface itself is well attested regionally under other IDs. A supported candidate may be preferred when morphology and context are otherwise compatible. Zero support is evidence against a candidate, not an absolute prohibition.
- Pay attention to the returned surface_total, candidate_count, candidate_rate, penalty and alternative IDs. A strong penalty means the surface is regionally well attested but the proposed ID is unattested; a weak penalty means evidence is sparse.
- The reduplication heuristic for red is strictly phrase-local. Only repetition inside the same whitespace-delimited phrase can license red. Never use repetition across phrase boundaries as evidence for red.
"""

TRANSLATION_INSTRUCTIONS = """You are the constrained NRDB Japanese translation phase.
The segmentation and morphemic annotation supplied to you have already been finalized and are FROZEN. You must not revise, reinterpret, or replace them.

Goal: produce a concise, natural Japanese translation grounded in the frozen annotation and NRDB evidence.

Rules:
- Translate primarily from the validated final annotation, using the source form only as supporting context.
- Use lookup_id only when the Japanese lexical meaning of a content ID is materially unclear. Dictionary-attested meanings outrank guesses.
- Use corpus_examples primarily for constructional or grammatical interpretation, especially when translated human examples can show how a multi-ID construction is realized in Japanese.
- Prefer searching an informative construction (for example A;cvb-foc or A-dat-B;cvb) over repeatedly searching obvious single grammatical IDs.
- Do not look up every ID. Search only when evidence could materially change the translation.
- Preserve information explicitly encoded by the analysis, including negation, tense/aspect, modality, case/argument structure, focus/topic, direction/location, quantification, and semantically relevant reduplication.
- Produce natural Japanese rather than an interlinear gloss.
- Do not add semantic information that is not licensed by the frozen annotation, source context, dictionary grounding, or retrieved corpus evidence.
- When evidence is incomplete, prefer a conservative translation over an imaginative guess.
- Do not produce chain-of-thought.

Final response must be one JSON object and no surrounding prose:
{"trsl_ai":"...","confidence":0.0,"translation_evidence":{"dictionary_ids":[],"example_sentence_ids":[],"ungrounded_ids":[],"note":"brief"}}
"""

TRANSLATION_FORMAT = {
	"type": "json_schema",
	"name": "nrdb_translation",
	"strict": True,
	"schema": {
		"type": "object",
		"properties": {
			"trsl_ai": {"type": "string"},
			"confidence": {"type": "number"},
			"translation_evidence": {
				"type": "object",
				"properties": {
					"dictionary_ids": {"type": "array", "items": {"type": "string"}},
					"example_sentence_ids": {"type": "array", "items": {"type": "integer"}},
					"ungrounded_ids": {"type": "array", "items": {"type": "string"}},
					"note": {"type": "string"},
				},
				"required": ["dictionary_ids", "example_sentence_ids", "ungrounded_ids", "note"],
				"additionalProperties": False,
			},
		},
		"required": ["trsl_ai", "confidence", "translation_evidence"],
		"additionalProperties": False,
	},
}


def instructions_for_version(prompt_version):
	version = str(prompt_version or "annotation-v1")
	if version == "annotation-v1":
		return BASE_INSTRUCTIONS
	if version == "annotation-v2":
		return BASE_INSTRUCTIONS + V2_RULES
	if version == "annotation-v3":
		return BASE_INSTRUCTIONS + V2_RULES + V3_ANNOTATION_RULES
	if version == "annotation-v4":
		return BASE_INSTRUCTIONS + V2_RULES + V3_ANNOTATION_RULES + V4_RULES
	if version == "annotation-v5":
		return BASE_INSTRUCTIONS + V2_RULES + V3_ANNOTATION_RULES + V4_RULES + V5_RULES
	if version == "annotation-v6":
		return BASE_INSTRUCTIONS + V2_RULES + V3_ANNOTATION_RULES + V4_RULES + V6_RULES
	raise ValueError("unsupported prompt_version: {}".format(version))


TOOLS = [
	{"type": "function", "name": "lookup_id", "description": "Look up bilingual dictionary, local-schema and UniCog grounding for one existing NRDB annotation ID.", "parameters": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"], "additionalProperties": False}, "strict": True},
	{"type": "function", "name": "corpus_examples", "description": "Retrieve human-validated corpus examples for an annotation expression. Expressions may be atomic IDs, conflated segments such as A;cvb, or segment sequences such as A-dat.", "parameters": {"type": "object", "properties": {"label": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 8}}, "required": ["label", "limit"], "additionalProperties": False}, "strict": True},
	{"type": "function", "name": "form_id_support", "description": "Check region-scoped aligned corpus and lexicon support for assigning one candidate annotation ID to one exact segmented surface form. Use this to penalize unattested form-ID pairings and compare regional alternatives.", "parameters": {"type": "object", "properties": {"surface": {"type": "string"}, "candidate_id": {"type": "string"}}, "required": ["surface", "candidate_id"], "additionalProperties": False}, "strict": True},
	{"type": "function", "name": "validate_analysis", "description": "Validate that a complete segmentation and annotation are structurally aligned and legal under nrdb-morph syntax.", "parameters": {"type": "object", "properties": {"segmented": {"type": "string"}, "annotation": {"type": "string"}}, "required": ["segmented", "annotation"], "additionalProperties": False}, "strict": True},
]

TRANSLATION_TOOLS = TOOLS[:2]


def parse_final_json(text):
	text = (text or "").strip()
	if text.startswith("```"):
		text = re.sub(r"^```(?:json)?\s*", "", text)
		text = re.sub(r"\s*```$", "", text)
	payload = json.loads(text)
	decision = str(payload.get("decision") or "").lower()
	if decision not in {"proposed", "uncertain", "failed"}:
		raise ValueError("invalid agent decision")
	confidence = float(payload.get("confidence", 0.0))
	if confidence < 0 or confidence > 1:
		raise ValueError("confidence must be between 0 and 1")
	payload["decision"] = decision
	payload["confidence"] = confidence
	payload["segmented"] = str(payload.get("segmented") or "").strip()
	payload["annotation"] = str(payload.get("annotation") or "").strip()
	payload["trsl_ai"] = str(payload.get("trsl_ai") or "").strip()
	payload["evidence"] = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
	return payload


def parse_translation_json(text):
	text = (text or "").strip()
	if text.startswith("```"):
		text = re.sub(r"^```(?:json)?\s*", "", text)
		text = re.sub(r"\s*```$", "", text)
	payload = json.loads(text)
	translation = str(payload.get("trsl_ai") or "").strip()
	if not translation:
		raise ValueError("translation phase returned empty trsl_ai")
	confidence = float(payload.get("confidence", 0.0))
	if confidence < 0 or confidence > 1:
		raise ValueError("translation confidence must be between 0 and 1")
	evidence = payload.get("translation_evidence")
	if not isinstance(evidence, dict):
		evidence = {}
	return {"trsl_ai": translation, "confidence": confidence, "translation_evidence": evidence}


def _response_output_as_input(response):
	items = []
	for output in response.output:
		if hasattr(output, "model_dump"):
			items.append(output.model_dump(exclude_none=True))
		elif isinstance(output, dict):
			items.append(dict(output))
		else:
			raise TypeError("unsupported response output item: {}".format(type(output).__name__))
	return items


def _response_incomplete_reason(response):
	if getattr(response, "status", None) != "incomplete":
		return None
	details = getattr(response, "incomplete_details", None)
	if hasattr(details, "reason"):
		return details.reason
	if isinstance(details, dict):
		return details.get("reason")
	return "unknown"


def _compact_morph(result):
	compact = {"segmented": result.get("segmented"), "annotation": result.get("annotation"), "confidence": result.get("confidence"), "mode": result.get("mode")}
	phrases = []
	for phrase in result.get("phrases", [])[:20]:
		segments = []
		for segment in phrase.get("segments", [])[:30]:
			segments.append({"surface": segment.get("surface"), "label": segment.get("label"), "confidence": segment.get("confidence", segment.get("raw_confidence")), "alternatives": [{"label": alt.get("label"), "support": alt.get("support")} for alt in segment.get("alternatives", [])[:3]]})
		phrases.append({"raw": phrase.get("raw"), "segments": segments})
	compact["phrases"] = phrases
	return compact


def _clip(value, length=240):
	value = str(value or "")
	return value if len(value) <= length else value[:length] + "…"


def _compact_tool_result(name, result):
	if name == "lookup_id":
		entries = [{"form1": entry.get("form1"), "form2": entry.get("form2"), "meaning_jp": _clip(entry.get("meaning_jp")), "pos": entry.get("pos"), "dialect_name": entry.get("dialect_name")} for entry in result.get("lexical_entries", [])[:6]]
		return {"label": result.get("label"), "lexical_entries": entries, "local": result.get("local"), "global": result.get("global")}
	if name == "corpus_examples":
		examples = [{"sentence_id": example.get("sentence_id"), "text": _clip(example.get("text"), 180), "annotation": _clip(example.get("annotation"), 280), "translation_jp": _clip(example.get("translation_jp"), 180)} for example in result.get("examples", [])[:6]]
		return {"label": result.get("label"), "examples": examples}
	if name == "form_id_support":
		return {
			"surface": result.get("surface"), "candidate_id": result.get("candidate_id"), "region": result.get("region"),
			"combined": result.get("combined"),
			"corpus": {"surface_total": result.get("corpus", {}).get("surface_total"), "candidate_count": result.get("corpus", {}).get("candidate_count"), "alternatives": result.get("corpus", {}).get("alternatives", [])[:8]},
			"lexicon": {"surface_total": result.get("lexicon", {}).get("surface_total"), "candidate_count": result.get("lexicon", {}).get("candidate_count"), "alternatives": result.get("lexicon", {}).get("alternatives", [])[:8]},
		}
	if name == "validate_analysis":
		return {"valid": bool(result.get("valid")), "error": result.get("error")}
	return result


def _trace_arguments(name, arguments):
	if name == "lookup_id":
		return "label={}".format(arguments.get("label", ""))
	if name == "corpus_examples":
		return "label={} limit={}".format(arguments.get("label", ""), arguments.get("limit", ""))
	if name == "form_id_support":
		return "surface={!r} candidate_id={}".format(arguments.get("surface", ""), arguments.get("candidate_id", ""))
	if name == "validate_analysis":
		return "segmented={!r} annotation={!r}".format(str(arguments.get("segmented", "")), str(arguments.get("annotation", "")))
	return ""


def _trace_result(name, result):
	if name == "lookup_id":
		return "lexical_entries={} local={} global={}".format(len(result.get("lexical_entries", [])), "yes" if result.get("local") else "no", "yes" if result.get("global") else "no")
	if name == "corpus_examples":
		return "examples={}".format(len(result.get("examples", [])))
	if name == "form_id_support":
		combined = result.get("combined", {})
		return "region={} support={}/{} penalty={}".format(result.get("region", ""), combined.get("candidate_count", 0), combined.get("surface_total", 0), combined.get("penalty", "none"))
	if name == "validate_analysis":
		return "valid={}".format(bool(result.get("valid")))
	return "ok"


def _retry_delay(error, attempt):
	message = str(error)
	match = re.search(r"try again in\s+([0-9.]+)s", message, re.IGNORECASE)
	if match:
		return max(1.0, float(match.group(1)) + 1.0)
	return min(60.0, 5.0 * (2 ** attempt))


class AnnotationAgent:
	def __init__(self, nrdb, model_name, client=None, max_rounds=6, max_evidence_calls=6, max_translation_evidence_calls=4, progress=None):
		self.nrdb = nrdb
		self.model_name = model_name
		self.client = client or OpenAI()
		self.max_rounds = int(max_rounds)
		self.max_evidence_calls = int(max_evidence_calls)
		self.max_translation_evidence_calls = int(max_translation_evidence_calls)
		self.progress = progress or (lambda _message: None)

	def _create_response(self, input_items, instructions, tools=TOOLS, max_output_tokens=800, text_format=None):
		for attempt in range(4):
			try:
				kwargs = {"model": self.model_name, "instructions": instructions, "input": input_items, "tools": tools, "store": False, "max_output_tokens": max_output_tokens}
				if text_format is not None:
					kwargs["text"] = {"format": text_format}
				return self.client.responses.create(**kwargs)
			except RateLimitError as error:
				message = str(error)
				requested = re.search(r"Limit\s+(\d+),\s+Requested\s+(\d+)", message)
				if requested and int(requested.group(2)) > int(requested.group(1)):
					raise
				if attempt >= 3:
					raise
				delay = _retry_delay(error, attempt)
				self.progress("  rate limit: waiting {:.0f}s before retry".format(delay))
				time.sleep(delay)

	def _tool_result(self, name, arguments, item, schema_id):
		if name == "lookup_id":
			return self.nrdb.lookup_id(arguments["label"], schema_id)
		if name == "corpus_examples":
			return self.nrdb.examples(arguments["label"], schema_id, item["sentence_id"], min(8, arguments["limit"]))
		if name == "form_id_support":
			region = str(item.get("dialect_region") or "").strip()
			if not region:
				return {"success": True, "surface": arguments["surface"], "candidate_id": arguments["candidate_id"], "region": None, "combined": {"surface_total": 0, "candidate_count": 0, "candidate_rate": None, "penalty": "none"}, "corpus": {}, "lexicon": {}, "note": "No dialect region available; no penalty applied."}
			return self.nrdb.form_id_support(arguments["surface"], arguments["candidate_id"], region, schema_id)
		if name == "validate_analysis":
			return self.nrdb.validate_analysis(item["text"], arguments["segmented"], arguments["annotation"])
		raise ValueError("unknown tool: {}".format(name))

	def _finalize_translation(self, base_input, evidence_summary, reason="evidence complete"):
		final_input = list(base_input)
		if evidence_summary:
			final_input.append({"role": "user", "content": "Retrieved compact translation evidence:\n" + json.dumps(evidence_summary[-4:], ensure_ascii=False)})
		final_input.append({"role": "user", "content": "Evidence gathering is finished ({}). Do not call tools. Return the final conservative Japanese translation now.".format(reason)})
		last_error = None
		for attempt, budget in enumerate((1200, 1800), start=1):
			self.progress("  translation: forced finalization attempt {} (max_output_tokens={})".format(attempt, budget))
			response = self._create_response(final_input, TRANSLATION_INSTRUCTIONS, tools=[], max_output_tokens=budget, text_format=TRANSLATION_FORMAT)
			incomplete_reason = _response_incomplete_reason(response)
			if incomplete_reason:
				last_error = RuntimeError("translation response incomplete: {}".format(incomplete_reason))
				self.progress("  translation: incomplete ({})".format(incomplete_reason))
				continue
			try:
				return parse_translation_json(response.output_text)
			except (json.JSONDecodeError, ValueError) as error:
				last_error = error
				self.progress("  translation: malformed final JSON; retrying")
		if last_error:
			raise last_error
		raise RuntimeError("translation finalization failed")

	def _translate_frozen(self, item, job, result):
		payload = {"sentence_id": int(item["sentence_id"]), "source_text": item["text"], "existing_translation_jp": item.get("translation_jp"), "frozen_segmented": result["segmented"], "frozen_annotation": result["annotation"], "annotation_decision": result["decision"], "annotation_confidence": result["confidence"], "annotation_schema_id": int(job["annotation_schema_id"])}
		base_input = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
		evidence_summary = []
		evidence_calls = 0
		self.progress("  translation: initial response (frozen annotation; budget {}/{})".format(evidence_calls, self.max_translation_evidence_calls))
		response = self._create_response(base_input, TRANSLATION_INSTRUCTIONS, tools=TRANSLATION_TOOLS, max_output_tokens=900)
		for round_index in range(1, self.max_rounds + 1):
			calls = [output for output in response.output if getattr(output, "type", None) == "function_call"]
			if not calls:
				incomplete_reason = _response_incomplete_reason(response)
				if incomplete_reason:
					return self._finalize_translation(base_input, evidence_summary, "previous response incomplete: {}".format(incomplete_reason))
				try:
					translation = parse_translation_json(response.output_text)
				except (json.JSONDecodeError, ValueError):
					return self._finalize_translation(base_input, evidence_summary, "previous final JSON malformed")
				self.progress("  translation: final confidence={:.3f}".format(translation["confidence"]))
				return translation

			self.progress("  translation tool round {}: {} call(s)".format(round_index, len(calls)))
			continuation = list(base_input)
			if evidence_summary:
				continuation.append({"role": "user", "content": "Previously retrieved compact translation evidence:\n" + json.dumps(evidence_summary[-4:], ensure_ascii=False)})
			continuation.extend(_response_output_as_input(response))
			for call in calls:
				arguments = json.loads(call.arguments)
				self.progress("    -> {}({})".format(call.name, _trace_arguments(call.name, arguments)))
				if evidence_calls >= self.max_translation_evidence_calls:
					compact = {"budget_exhausted": True, "message": "Translation evidence-call budget exhausted; translate conservatively from frozen analysis and existing evidence."}
					self.progress("    <- {}: skipped (translation evidence budget exhausted)".format(call.name))
				else:
					tool_result = self._tool_result(call.name, arguments, item, int(job["annotation_schema_id"]))
					compact = _compact_tool_result(call.name, tool_result)
					self.progress("    <- {}: {}".format(call.name, _trace_result(call.name, tool_result)))
					evidence_calls += 1
					evidence_summary.append({"tool": call.name, "arguments": arguments, "result": compact})
				continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})

			self.progress("  translation: continue after tool round {} (evidence {}/{})".format(round_index, evidence_calls, self.max_translation_evidence_calls))
			if evidence_calls >= self.max_translation_evidence_calls:
				final_input = continuation + [{"role": "user", "content": "Translation evidence budget is exhausted. Do not request more tools; return the final translation."}]
				response = self._create_response(final_input, TRANSLATION_INSTRUCTIONS, tools=[], max_output_tokens=1200, text_format=TRANSLATION_FORMAT)
			else:
				response = self._create_response(continuation, TRANSLATION_INSTRUCTIONS, tools=TRANSLATION_TOOLS, max_output_tokens=900)
		raise RuntimeError("translation phase exceeded maximum tool rounds")

	def annotate(self, item, job, morph_result):
		prompt_version = str(job.get("prompt_version") or "annotation-v1")
		instructions = instructions_for_version(prompt_version)
		input_payload = {"sentence_id": int(item["sentence_id"]), "dialect_id": int(item["dialect_id"]), "dialect_region": item.get("dialect_region"), "text": item["text"], "translation_jp": item.get("translation_jp"), "produce_translation": bool(job.get("produce_translation")), "annotation_schema_id": int(job["annotation_schema_id"]), "nrdb_morph": _compact_morph(morph_result)}
		base_input = [{"role": "user", "content": json.dumps(input_payload, ensure_ascii=False)}]
		evidence_summary = []
		evidence_calls = 0
		self.progress("  llm: initial response ({}; {})".format(self.model_name, prompt_version))
		response = self._create_response(base_input, instructions)
		for round_index in range(1, self.max_rounds + 1):
			calls = [output for output in response.output if getattr(output, "type", None) == "function_call"]
			if not calls:
				result = parse_final_json(response.output_text)
				result["model_response_id"] = response.id
				if prompt_version in {"annotation-v3", "annotation-v4", "annotation-v5", "annotation-v6"}:
					result["trsl_ai"] = ""
					if job.get("produce_translation") and result.get("annotation") and result["decision"] != "failed":
						translation = self._translate_frozen(item, job, result)
						result["trsl_ai"] = translation["trsl_ai"]
						result["evidence"]["translation"] = translation["translation_evidence"]
						result["evidence"]["translation"]["confidence"] = translation["confidence"]
				elif not job.get("produce_translation"):
					result["trsl_ai"] = ""
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
				if call.name != "validate_analysis" and evidence_calls >= self.max_evidence_calls:
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
			response = self._create_response(continuation, instructions)
		raise RuntimeError("agent exceeded maximum tool rounds")
