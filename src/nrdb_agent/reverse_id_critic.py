import json

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

ID_REVIEW_INSTRUCTIONS = """You are the grammatical ID-sequence reviewer for NRDB Japanese-to-Miyako translation.

You receive a Japanese sentence, an already proposed Miyako NRDB annotation, its original lexical/corpus evidence, and a SOFT statistical ID-sequence review trained on human Miyako annotation.

Goal: make at most a narrow grammatical revision when the statistical evidence exposes a genuinely implausible Miyako ID sequence.

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
		out = {
			"mean_log_probability": review.get("mean_log_probability"),
			"strong_surprises": review.get("strong_surprises", 0),
			"representations": {},
		}
		for name, value in review.get("representations", {}).items():
			positions = []
			for position in value.get("positions", []):
				if not position.get("strong_surprise"):
					continue
				positions.append({
					"index": position.get("index"), "token": position.get("token"),
					"context": position.get("context"), "order": position.get("order"),
					"log_probability": position.get("log_probability"),
					"top_observed": position.get("top_observed", [])[:5],
				})
			out["representations"][name] = {
				"mean_log_probability": value.get("mean_log_probability"),
				"strong_surprises": value.get("strong_surprises", 0),
				"surprising_positions": positions[:8],
			}
		return out


class IdCriticSyntaxAwareReverseSurfaceAgent(SyntaxAwareReverseSurfaceAgent):
	def __init__(self, *args, id_model_path=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.id_critic = IdSequenceCritic(id_model_path) if id_model_path else None
		self.id_model_path = str(id_model_path) if id_model_path else None

	def _review_ids(self, item, job, id_result):
		if self.id_critic is None or not id_result.get("annotation"):
			return id_result
		annotation = id_result["annotation"]
		review = self.id_critic.review(annotation, int(job["annotation_schema_id"]))
		compact = self.id_critic.compact(review)
		self.progress("  id-model: mean_logp={:.3f} strong_surprises={}".format(
			float(review.get("mean_log_probability") or 0.0), review.get("strong_surprises", 0),
		))
		for representation, values in compact.get("representations", {}).items():
			for value in values.get("surprising_positions", []):
				top = ", ".join("{}({})".format(entry.get("token"), entry.get("count")) for entry in value.get("top_observed", [])[:3])
				self.progress("    id-model: {} token={!r} after {} -> [{}]".format(
					representation, value.get("token"), value.get("context"), top,
				))
		id_result.setdefault("evidence", {})["id_sequence_review"] = compact
		id_result["evidence"]["id_model_path"] = self.id_model_path
		if not review.get("strong_surprises"):
			return id_result

		payload = {
			"japanese": str(item.get("translation_jp") or "").strip(),
			"original_annotation": annotation,
			"original_evidence": id_result.get("evidence", {}),
			"id_sequence_review": compact,
			"annotation_schema_id": int(job["annotation_schema_id"]),
			"target_region": item.get("dialect_region"),
		}
		self.progress("  id-model: one soft grammatical revision pass")
		response = self._create_response(
			[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
			ID_REVIEW_INSTRUCTIONS, tools=[], max_output_tokens=900, text_format=ID_REVIEW_FORMAT,
		)
		candidate = json.loads((response.output_text or "").strip())
		candidate_annotation = str(candidate.get("annotation") or "").strip()
		if not candidate_annotation or candidate_annotation == annotation:
			compact["revision_attempted"] = True
			compact["revision_accepted"] = False
			return id_result
		candidate_review = self.id_critic.review(candidate_annotation, int(job["annotation_schema_id"]))
		accept = bool(
			candidate_review.get("strong_surprises", 0) < review.get("strong_surprises", 0)
			or float(candidate_review.get("mean_log_probability") or -1e9) > float(review.get("mean_log_probability") or -1e9)
		)
		compact["revision_attempted"] = True
		compact["candidate"] = self.id_critic.compact(candidate_review)
		compact["revision_accepted"] = accept
		if accept:
			self.progress("  id-model: revision accepted surprises {}->{} mean_logp {:.3f}->{:.3f}".format(
				review.get("strong_surprises", 0), candidate_review.get("strong_surprises", 0),
				float(review.get("mean_log_probability") or 0.0), float(candidate_review.get("mean_log_probability") or 0.0),
			))
			id_result["annotation"] = candidate_annotation
			id_result["confidence"] = min(float(id_result.get("confidence", 0.0)), max(0.0, min(1.0, float(candidate.get("confidence", 0.0)))))
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
