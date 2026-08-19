import json

from .annotator import _compact_tool_result, _response_output_as_input
from .reverse_agent import ReverseIdAgent
from .reverse_surface_syntax_agent import SyntaxAwareReverseSurfaceAgent


ID_REVIEW_FORMAT = {
	"type": "json_schema",
	"name": "nrdb_reverse_id_review",
	"strict": True,
	"schema": {
		"type": "object",
		"properties": {
			"annotation": {"type": "string"},
			"confidence": {"type": "number"},
			"note": {"type": "string"},
		},
		"required": ["annotation", "confidence", "note"],
		"additionalProperties": False,
	},
}

ID_REVIEW_CORPUS_TOOL = {
	"type": "function",
	"name": "corpus_examples",
	"description": "Retrieve human-validated corpus examples for ONE SHORT ID construction directly surrounding a statistical grammar surprise. Do not query unrelated or routine grammar.",
	"parameters": {
		"type": "object",
		"properties": {
			"label": {"type": "string", "maxLength": 256},
			"limit": {"type": "integer", "minimum": 1, "maximum": 6},
		},
		"required": ["label", "limit"],
		"additionalProperties": False,
	},
	"strict": True,
}

ID_REVIEW_INSTRUCTIONS = """You are the grammatical ID-sequence reviewer for NRDB Japanese-to-Miyako translation.

You receive a Japanese sentence, an already proposed Miyako NRDB annotation, its lexical evidence, and a SOFT statistical ID-sequence review trained on human Miyako annotation.

Goal: make at most a narrow grammatical revision when the statistical evidence exposes a genuinely implausible Miyako ID sequence.

Query-efficiency rules:
- The initial reverse planner has deliberately NOT spent corpus queries on routine grammar.
- corpus_examples is available here only because the statistical critic has already found a strong surprise.
- Query only a SHORT construction immediately around a surprising token when corpus evidence could distinguish competing grammatical analyses.
- Do not query every surprising atom separately if one construction query can resolve several of them.
- Do not use corpus_examples for lexical discovery or for grammar that the critic does not flag.
- At most two hotspot corpus calls are available; often zero or one is enough.

Rules:
- The statistical critic is evidence, not truth. Rare but valid constructions must survive.
- Preserve the semantic content and lexical/content IDs chosen by the reverse planner unless the original retrieved evidence itself clearly licenses an alternative.
- Concentrate revisions on grammatical packaging: case, topic/focus, interrogative/final particles, TAM, converb/dependent forms, clause linking, and other local grammatical IDs.
- Do not replace an idiomatic lexical choice merely because a different lexical ID is more frequent in the corpus.
- Segment-level and atomic bigram/trigram surprises may identify missing, wrong, or improbable grammatical material; use the listed observed continuations as hints, not commands.
- Keep NRDB annotation syntax: spaces separate phrases, hyphens separate segments, semicolons conflate atoms inside one segment.
- Productive n: Japanese lexical items remain permitted exactly as in the original reverse planner.
- If the original annotation is linguistically defensible, return it unchanged.
- Do not produce chain-of-thought.

Return exactly one JSON object:
{"annotation":"...","confidence":0.0,"note":"brief"}
"""


def _surprise_count(review):
	if "strong_surprise_count" in review:
		return int(review.get("strong_surprise_count") or 0)
	value = review.get("strong_surprises", [])
	if isinstance(value, list):
		return len(value)
	return int(value or 0)


def _combined_mean_log_probability(review):
	if "combined_mean_log_probability" in review:
		return float(review.get("combined_mean_log_probability") or 0.0)
	return float(review.get("mean_log_probability") or 0.0)


def _alternative_rows(position):
	values = position.get("alternatives") or position.get("top_observed") or []
	rows = []
	for value in values[:5]:
		if isinstance(value, dict):
			rows.append({"token": value.get("token"), "probability": value.get("probability"), "count": value.get("count")})
		elif isinstance(value, (list, tuple)) and len(value) >= 2:
			rows.append({"token": value[0], "probability": value[1], "count": None})
	return rows


class IdSequenceCritic:
	def __init__(self, model_path=None, model=None):
		if model is None:
			try:
				from nrdb_morph.id_sequence import IdSequenceModel
			except ImportError as error:
				raise RuntimeError(
					"--id-model requires nrdb-morph in the nrdb-agent environment; run: pip install -e ../nrdb-morph"
				) from error
			model = IdSequenceModel.load(model_path)
		self.model = model

	def review(self, annotation, annotation_schema_id):
		return self.model.score(annotation, int(annotation_schema_id))

	def compact(self, review):
		strong = review.get("strong_surprises", [])
		strong = strong if isinstance(strong, list) else []
		out = {"mean_log_probability": _combined_mean_log_probability(review), "strong_surprises": _surprise_count(review), "representations": {}}
		for name in ("segment", "atom"):
			value = review.get(name, {})
			positions = []
			for position in strong:
				if position.get("representation") != name:
					continue
				positions.append({
					"index": position.get("index"), "token": position.get("token"), "context": position.get("context"),
					"order": position.get("order"), "log_probability": position.get("log_probability"),
					"threshold": position.get("threshold"), "top_observed": _alternative_rows(position),
				})
			out["representations"][name] = {
				"mean_log_probability": value.get("mean_log_probability"),
				"strong_surprises": len(positions), "surprising_positions": positions[:8],
			}
		return out


class IdCriticSyntaxAwareReverseSurfaceAgent(SyntaxAwareReverseSurfaceAgent):
	def __init__(self, *args, id_model_path=None, max_id_review_evidence_calls=2, **kwargs):
		super().__init__(*args, **kwargs)
		self.id_critic = IdSequenceCritic(id_model_path) if id_model_path else None
		self.id_model_path = str(id_model_path) if id_model_path else None
		self.max_id_review_evidence_calls = int(max_id_review_evidence_calls)

	def _review_candidate(self, item, job, payload):
		base_input = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
		evidence = []
		evidence_calls = 0
		response = self._create_response(
			base_input, ID_REVIEW_INSTRUCTIONS, tools=[ID_REVIEW_CORPUS_TOOL], max_output_tokens=900, text_format=ID_REVIEW_FORMAT,
		)
		for round_index in range(1, 4):
			calls = [value for value in getattr(response, "output", []) if getattr(value, "type", None) == "function_call"]
			if not calls:
				return json.loads((response.output_text or "").strip()), evidence
			continuation = list(base_input)
			if evidence:
				continuation.append({"role": "user", "content": "Previously retrieved grammar-hotspot evidence:\n" + json.dumps(evidence, ensure_ascii=False)})
			continuation.extend(_response_output_as_input(response))
			self.progress("  id-model hotspot round {}: {} corpus call(s)".format(round_index, len(calls)))
			for call in calls:
				arguments = json.loads(call.arguments)
				if call.name != "corpus_examples":
					raise ValueError("unsupported ID-review tool: {}".format(call.name))
				if evidence_calls >= self.max_id_review_evidence_calls:
					compact = {"budget_exhausted": True, "message": "Grammar-hotspot corpus budget exhausted; revise conservatively from existing evidence."}
				else:
					result = self.nrdb.examples(
						arguments["label"], int(job["annotation_schema_id"]), int(item.get("sentence_id") or 0), min(6, int(arguments["limit"])),
					)
					compact = _compact_tool_result("corpus_examples", result)
					evidence_calls += 1
					evidence.append({"label": arguments["label"], "result": compact})
				continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})
			response = self._create_response(
				continuation, ID_REVIEW_INSTRUCTIONS,
				tools=[] if evidence_calls >= self.max_id_review_evidence_calls else [ID_REVIEW_CORPUS_TOOL],
				max_output_tokens=900, text_format=ID_REVIEW_FORMAT,
			)
		# Hard-stop finalization without additional queries.
		response = self._create_response(
			base_input + [{"role": "user", "content": "Grammar-hotspot evidence gathering is finished. Return the conservative final ID revision now."}],
			ID_REVIEW_INSTRUCTIONS, tools=[], max_output_tokens=900, text_format=ID_REVIEW_FORMAT,
		)
		return json.loads((response.output_text or "").strip()), evidence

	def _review_ids(self, item, job, id_result):
		if self.id_critic is None or not id_result.get("annotation"):
			return id_result
		annotation = id_result["annotation"]
		review = self.id_critic.review(annotation, int(job["annotation_schema_id"]))
		compact = self.id_critic.compact(review)
		initial_surprises = _surprise_count(review)
		initial_mean = _combined_mean_log_probability(review)
		self.progress("  id-model: mean_logp={:.3f} strong_surprises={}".format(initial_mean, initial_surprises))
		for representation, values in compact.get("representations", {}).items():
			for value in values.get("surprising_positions", []):
				top = ", ".join(
					"{}({:.4f})".format(entry.get("token"), float(entry.get("probability") or 0.0))
					for entry in value.get("top_observed", [])[:3]
				)
				self.progress("    id-model: {} token={!r} after {} -> [{}]".format(representation, value.get("token"), value.get("context"), top))
		id_result.setdefault("evidence", {})["id_sequence_review"] = compact
		id_result["evidence"]["id_model_path"] = self.id_model_path
		if initial_surprises == 0:
			self.progress("  id-model: no grammar hotspot; zero corpus queries")
			return id_result

		payload = {
			"japanese": str(item.get("translation_jp") or "").strip(),
			"original_annotation": annotation,
			"original_evidence": id_result.get("evidence", {}),
			"id_sequence_review": compact,
			"annotation_schema_id": int(job["annotation_schema_id"]),
			"target_region": item.get("dialect_region"),
		}
		self.progress("  id-model: one soft grammatical revision pass; corpus only for flagged hotspots")
		candidate, hotspot_evidence = self._review_candidate(item, job, payload)
		if hotspot_evidence:
			compact["hotspot_corpus_evidence"] = hotspot_evidence
		candidate_annotation = str(candidate.get("annotation") or "").strip()
		if not candidate_annotation or candidate_annotation == annotation:
			compact["revision_attempted"] = True
			compact["revision_accepted"] = False
			return id_result
		candidate_review = self.id_critic.review(candidate_annotation, int(job["annotation_schema_id"]))
		candidate_surprises = _surprise_count(candidate_review)
		candidate_mean = _combined_mean_log_probability(candidate_review)
		accept = bool(candidate_surprises < initial_surprises or (candidate_surprises == initial_surprises and candidate_mean > initial_mean))
		compact["revision_attempted"] = True
		compact["candidate"] = self.id_critic.compact(candidate_review)
		compact["revision_accepted"] = accept
		if accept:
			self.progress("  id-model: revision accepted surprises {}->{} mean_logp {:.3f}->{:.3f}".format(initial_surprises, candidate_surprises, initial_mean, candidate_mean))
			id_result["annotation"] = candidate_annotation
			id_result["confidence"] = min(
				float(id_result.get("confidence", 0.0)),
				max(0.0, min(1.0, float(candidate.get("confidence", 0.0)))),
			)
		else:
			self.progress("  id-model: revision rejected; keeping original IDs")
		return id_result

	def annotate(self, item, job, morph_result=None):
		self._id_pass_dialect_ids = job.get("target_dialect_ids") or [int(item["dialect_id"])]
		id_job = dict(job)
		id_job["prompt_version"] = "reverse-v1"
		id_result = ReverseIdAgent.annotate(self, item, id_job, morph_result)
		if id_result.get("decision") == "failed" or not id_result.get("annotation"):
			return id_result
		id_result = self._review_ids(item, job, id_result)
		surface = self._realize_surface(item, job, id_result)
		id_result["segmented"] = surface["segmented"]
		id_result.setdefault("evidence", {})["surface_realization"] = surface["evidence"]
		id_result["confidence"] = min(float(id_result.get("confidence", 0.0)), float(surface["confidence"]))
		return id_result
