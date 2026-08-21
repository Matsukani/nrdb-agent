import csv
import io
import json
import re
from pathlib import Path

from .annotator import AnnotationAgent, _response_incomplete_reason
from .usage import UsageTracker, tracked_client


ID_ANALYSIS_INSTRUCTIONS = """You are the NRDB corpus-based grammatical-ID analyst.

You receive evidence for ONE exact annotation ID from a fixed annotation schema. The evidence contains corpus counts, exact same-phrase patterns, phrase-bounded corpus-native bigrams/trigrams, full morphemic annotations, source forms, Japanese sentence translations, existing curated grammar entries, and optional expert research notes.

Your task is to propose a compact, auditable grammatical-knowledge record for translation, plus a small number of genuinely supported construction records when appropriate.

Method and evidence rules:
- Give the ID's central function a conventional, broadly recognizable linguistic name in English and Japanese. Do not expand the opaque ID label as if it were a gloss.
- Every candidate `name` must be a concise, descriptive English `lower_snake_case` identifier. Never use Japanese, an opaque annotation ID alone, spaces, hyphens, or punctuation in `name`. Candidate names must be distinct.
- Describe the central function succinctly in Japanese. State its host/attachment and semantic or discourse scope when the evidence supports them.
- Give an operational Japanese translation policy that another translation agent can follow. A policy may explicitly license omission, reordering, lexical compensation, or a context-dependent Japanese expression.
- Expert research notes are deliberate guidance. Follow them unless the corpus evidence clearly conflicts; record any conflict in warnings rather than silently ignoring it.
- Japanese translations are sentence-level evidence, NOT word or morpheme alignments. Infer a mapping only from recurring contrasts across examples and say when it remains uncertain.
- Same-phrase patterns preserve annotation structure. N-grams are discovery/ranking evidence only and do not preserve every segment/conflation boundary.
- Propose exactly one `morpheme` candidate for the ID's general function when the corpus supports one. Its trigger_id and pattern must both be the exact target ID.
- Propose a `construction` candidate only for a recurrent complete pattern whose meaning/translation is more specific than ordinary composition. Frequent co-occurrence alone is not a construction.
- Construction patterns must be based on exact attested phrase annotations. V, N, A, or X may replace genuinely variable lexical material, while literal grammatical IDs and NRDB separators remain explicit.
- A construction candidate must use the target ID as trigger_id. Keep at most four construction candidates and prefer precision over coverage.
- Select one to three short, informative ATTESTED examples by returning their exact example_key. Never invent an example or alter its source, annotation, or Japanese translation.
- `meaning_jp` explains grammatical meaning. `realization_jp` is a concise policy for natural Japanese realization, not a mandatory string substitution.
- `note` records attachment/scope, disambiguation, limitations, and why any construction is not merely a co-occurrence. Exact examples will be attached deterministically after the response.
- Candidate records are review proposals, never automatically enabled.
- Do not produce chain-of-thought. Return only the requested JSON object.
"""


EXAMPLE_REFERENCE_SCHEMA = {
	"type": "object",
	"properties": {
		"example_key": {"type": "string"},
		"mapping_note_jp": {"type": "string"},
	},
	"required": ["example_key", "mapping_note_jp"],
	"additionalProperties": False,
}


CANDIDATE_SCHEMA = {
	"type": "object",
	"properties": {
		"entry_type": {"type": "string", "enum": ["morpheme", "construction"]},
		"name": {"type": "string"},
		"trigger_id": {"type": "string"},
		"pattern": {"type": "string"},
		"meaning_jp": {"type": "string"},
		"realization_jp": {"type": "string"},
		"note": {"type": "string"},
		"priority": {"type": "integer", "minimum": 1, "maximum": 1000},
		"confidence": {"type": "number"},
		"example_references": {"type": "array", "items": EXAMPLE_REFERENCE_SCHEMA, "minItems": 1, "maxItems": 3},
	},
	"required": [
		"entry_type", "name", "trigger_id", "pattern", "meaning_jp", "realization_jp",
		"note", "priority", "confidence", "example_references",
	],
	"additionalProperties": False,
}


ID_ANALYSIS_FORMAT = {
	"type": "json_schema",
	"name": "nrdb_grammatical_id_analysis",
	"strict": True,
	"schema": {
		"type": "object",
		"properties": {
			"target_id": {"type": "string"},
			"linguistic_name_en": {"type": "string"},
			"linguistic_name_jp": {"type": "string"},
			"central_function_jp": {"type": "string"},
			"general_translation_policy_jp": {"type": "string"},
			"candidates": {"type": "array", "items": CANDIDATE_SCHEMA, "minItems": 1, "maxItems": 5},
			"warnings": {"type": "array", "items": {"type": "string"}},
		},
		"required": [
			"target_id", "linguistic_name_en", "linguistic_name_jp", "central_function_jp",
			"general_translation_policy_jp", "candidates", "warnings",
		],
		"additionalProperties": False,
	},
}


TSV_COLUMNS = [
	"annotation_schema_id", "region", "dialect_id", "entry_type", "name", "trigger_id", "pattern",
	"meaning_jp", "realization_jp", "note", "priority", "enabled", "linguistic_name_en",
	"linguistic_name_jp", "confidence", "evidence_tokens", "evidence_sentences", "evidence_datasets",
	"evidence_dialects", "example_keys", "examples_json", "research_note", "warnings",
]


def _bounded_confidence(value):
	try:
		return max(0.0, min(1.0, float(value)))
	except (TypeError, ValueError):
		return 0.0


def _english_key(value):
	key = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
	if key and not key[0].isalpha():
		key = "grammar_" + key
	return key


def _scoped_english_name(value, target_id):
	base = _english_key(value)
	target = _english_key(target_id)
	if not base or not target:
		raise ValueError("ID analysis returned a name that cannot form an English identifier")
	suffix = "_" + target
	base = base[:128 - len(suffix)].rstrip("_")
	if not base:
		raise ValueError("ID analysis returned an unusable English name")
	return base + suffix


def _example_text(example):
	return str(
		example.get("phrase_form2") or example.get("phrase_form1") or
		example.get("sentence_trsc2") or example.get("sentence_text") or ""
	).strip()


def _materialize_examples(candidate, evidence, warnings):
	available = {str(row.get("example_key") or ""): row for row in evidence.get("examples", [])}
	selected = []
	seen = set()
	for reference in candidate.get("example_references", [])[:3]:
		key = str(reference.get("example_key") or "").strip()
		if not key or key in seen:
			continue
		row = available.get(key)
		if row is None:
			warnings.append("Candidate {} cited unavailable example {} and it was removed.".format(candidate.get("name") or "", key))
			continue
		seen.add(key)
		selected.append({
			"example_key": key,
			"source": _example_text(row),
			"phrase_annotation": str(row.get("phrase_annotation") or ""),
			"sentence_annotation": str(row.get("sentence_annotation") or ""),
			"translation_jp": str(row.get("translation_jp") or ""),
			"mapping_note_jp": str(reference.get("mapping_note_jp") or "").strip(),
			"source_kind": row.get("source_kind"),
			"dataset_id": row.get("dataset_id"),
			"source_id": row.get("source_id"),
		})
	return selected


def _note_with_examples(note, examples):
	parts = [str(note or "").strip()]
	for example in examples:
		line = "例（文訳）: {} [{}] → {}".format(
			example.get("source") or "", example.get("phrase_annotation") or "",
			example.get("translation_jp") or "",
		)
		if example.get("mapping_note_jp"):
			line += "（{}）".format(example["mapping_note_jp"])
		parts.append(line)
	return "\n".join(value for value in parts if value)


def normalize_analysis(payload, evidence):
	if not isinstance(payload, dict):
		raise ValueError("ID analysis did not return an object")
	target_id = str(evidence.get("target_id") or "")
	if str(payload.get("target_id") or "") != target_id:
		raise ValueError("ID analysis returned the wrong target_id")
	candidates = payload.get("candidates")
	if not isinstance(candidates, list) or not candidates:
		raise ValueError("ID analysis returned no candidate records")
	if any(str(value.get("entry_type") or "") not in ("morpheme", "construction") for value in candidates):
		raise ValueError("ID analysis returned an invalid entry_type")
	morpheme_count = sum(1 for value in candidates if value.get("entry_type") == "morpheme")
	if morpheme_count != 1:
		raise ValueError("ID analysis must return exactly one morpheme candidate")
	if sum(1 for value in candidates if value.get("entry_type") == "construction") > 4:
		raise ValueError("ID analysis returned too many construction candidates")
	for field in ("linguistic_name_en", "linguistic_name_jp", "central_function_jp", "general_translation_policy_jp"):
		if not str(payload.get(field) or "").strip():
			raise ValueError("ID analysis returned an empty {}".format(field))

	warnings = [str(value).strip() for value in payload.get("warnings", []) if str(value).strip()]
	normalized = []
	seen_names = set()
	for raw in candidates:
		entry_type = str(raw.get("entry_type") or "")
		name_source = payload.get("linguistic_name_en") if entry_type == "morpheme" else raw.get("name")
		candidate_name = _scoped_english_name(name_source, target_id)
		if candidate_name in seen_names:
			raise ValueError("ID analysis returned duplicate candidate names")
		seen_names.add(candidate_name)
		candidate = {
			"entry_type": entry_type,
			"name": candidate_name,
			"trigger_id": target_id,
			"pattern": target_id if entry_type == "morpheme" else str(raw.get("pattern") or "").strip(),
			"meaning_jp": str(raw.get("meaning_jp") or "").strip(),
			"realization_jp": str(raw.get("realization_jp") or "").strip(),
			"note": str(raw.get("note") or "").strip(),
			"priority": max(1, min(1000, int(raw.get("priority") or 100))),
			"confidence": _bounded_confidence(raw.get("confidence")),
			"example_references": raw.get("example_references") if isinstance(raw.get("example_references"), list) else [],
		}
		if not candidate["name"] or not candidate["pattern"] or not candidate["meaning_jp"] or not candidate["realization_jp"]:
			raise ValueError("ID analysis returned an incomplete candidate record")
		candidate["examples"] = _materialize_examples(candidate, evidence, warnings)
		candidate["note"] = _note_with_examples(candidate["note"], candidate["examples"])
		normalized.append(candidate)

	return {
		"target_id": target_id,
		"linguistic_name_en": str(payload.get("linguistic_name_en") or "").strip(),
		"linguistic_name_jp": str(payload.get("linguistic_name_jp") or "").strip(),
		"central_function_jp": str(payload.get("central_function_jp") or "").strip(),
		"general_translation_policy_jp": str(payload.get("general_translation_policy_jp") or "").strip(),
		"candidates": normalized,
		"warnings": warnings,
	}


class IdAnalysisAgent(AnnotationAgent):
	def analyze(self, evidence):
		payload = {
			"target_id": evidence.get("target_id"),
			"annotation_schema_id": evidence.get("annotation_schema_id"),
			"scope": evidence.get("scope"),
			"research_note": evidence.get("research_note"),
			"summary": evidence.get("summary"),
			"local_id_metadata": evidence.get("local_id_metadata"),
			"existing_entries": evidence.get("existing_entries"),
			"same_phrase_patterns": evidence.get("same_phrase_patterns"),
			"anchored_phrase_ngrams": evidence.get("anchored_phrase_ngrams"),
			"examples": evidence.get("examples"),
			"method_notes": evidence.get("method_notes"),
		}
		base_input = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
		last_error = None
		for attempt, budget in enumerate((3500, 5000), start=1):
			self.progress("  ID analysis: structured response attempt {}".format(attempt))
			response = self._create_response(
				base_input, ID_ANALYSIS_INSTRUCTIONS, tools=[], max_output_tokens=budget,
				text_format=ID_ANALYSIS_FORMAT,
			)
			incomplete = _response_incomplete_reason(response)
			if incomplete:
				last_error = RuntimeError("ID analysis response incomplete: {}".format(incomplete))
				continue
			try:
				analysis = normalize_analysis(json.loads((response.output_text or "").strip()), evidence)
				analysis["model_response_id"] = getattr(response, "id", None)
				return analysis
			except (json.JSONDecodeError, TypeError, ValueError) as error:
				last_error = error
		if last_error:
			raise last_error
		raise RuntimeError("ID analysis failed")


def result_rows(result):
	rows = []
	job = result["job"]
	for analysis in result.get("analyses", []):
		evidence = analysis.get("evidence") or {}
		summary = evidence.get("summary") or {}
		warnings = " | ".join(analysis.get("warnings", []))
		for candidate in analysis.get("candidates", []):
			examples = candidate.get("examples", [])
			rows.append({
				"annotation_schema_id": job.get("annotation_schema_id"),
				"region": job.get("region") or "",
				"dialect_id": job.get("dialect_id") or "",
				"entry_type": candidate.get("entry_type"),
				"name": candidate.get("name"),
				"trigger_id": candidate.get("trigger_id"),
				"pattern": candidate.get("pattern"),
				"meaning_jp": candidate.get("meaning_jp"),
				"realization_jp": candidate.get("realization_jp"),
				"note": candidate.get("note"),
				"priority": candidate.get("priority", 100),
				"enabled": 0,
				"linguistic_name_en": analysis.get("linguistic_name_en"),
				"linguistic_name_jp": analysis.get("linguistic_name_jp"),
				"confidence": candidate.get("confidence"),
				"evidence_tokens": summary.get("tokens", 0),
				"evidence_sentences": summary.get("sentences", 0),
				"evidence_datasets": summary.get("datasets", 0),
				"evidence_dialects": summary.get("dialects", 0),
				"example_keys": ",".join(str(value.get("example_key") or "") for value in examples),
				"examples_json": json.dumps(examples, ensure_ascii=False, separators=(",", ":")),
				"research_note": evidence.get("research_note") or "",
				"warnings": warnings,
			})
	return rows


def result_tsv(result):
	handle = io.StringIO(newline="")
	writer = csv.DictWriter(handle, fieldnames=TSV_COLUMNS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
	writer.writeheader()
	writer.writerows(result_rows(result))
	return handle.getvalue()


def write_result_tsv(path, value):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(value, encoding="utf-8")
	return str(path)


def run_id_analysis_job(nrdb, job_id, openai_client=None, progress=print):
	job = nrdb.id_analysis_job(job_id)["job"]
	if job.get("status") == "completed":
		raise ValueError("ID-analysis job is already completed")
	nrdb.set_id_analysis_status(job_id, "running")
	tracker = UsageTracker()
	client = tracked_client(openai_client, tracker)
	agent = IdAnalysisAgent(nrdb, job["model_name"], client=client, progress=progress)
	analyses = []
	try:
		for index, target_id in enumerate(job.get("target_ids", []), start=1):
			progress("[{}/{}] analyze ID {!r}".format(index, len(job.get("target_ids", [])), target_id))
			evidence = nrdb.id_analysis_evidence(job_id, target_id)["evidence"]
			if int((evidence.get("summary") or {}).get("tokens") or 0) == 0:
				analyses.append({
					"target_id": target_id, "linguistic_name_en": "", "linguistic_name_jp": "",
					"central_function_jp": "", "general_translation_policy_jp": "",
					"candidates": [], "warnings": ["No exact CPS attestations were found; no record was proposed."],
					"model_response_id": None, "evidence": evidence,
				})
				continue
			analysis = agent.analyze(evidence)
			analysis["evidence"] = evidence
			analyses.append(analysis)
		result = {
			"format": "nrdb-agent.id-analysis-result.v1",
			"job": {
				"id": int(job["id"]), "annotation_schema_id": int(job["annotation_schema_id"]),
				"region": job.get("region"), "dialect_id": job.get("dialect_id"),
				"target_ids": list(job.get("target_ids", [])), "source_kinds": list(job.get("source_kinds", [])),
				"dataset_ids": list(job.get("dataset_ids", [])), "minimum_ngram_count": job.get("minimum_ngram_count"),
				"example_limit": job.get("example_limit"), "model_name": job.get("model_name"),
				"prompt_version": job.get("prompt_version"),
			},
			"analyses": analyses,
			"api_usage": tracker.summary(),
		}
		tsv = result_tsv(result)
		nrdb.save_id_analysis_result(job_id, result, tsv)
		return {"result": result, "tsv": tsv}
	except BaseException as error:
		nrdb.set_id_analysis_status(job_id, "failed", str(error))
		raise
