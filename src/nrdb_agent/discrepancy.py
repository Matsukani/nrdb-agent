import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .annotator import AnnotationAgent, _response_incomplete_reason
from .metrics import annotation_ids
from .translate import translate_text
from .usage import UsageTracker, tracked_client


DISCOVERY_FORMAT = "nrdb-agent.translation-discovery.v1"
BASELINE_FORMAT = "nrdb-agent.translation-discrepancy.v1"
CHECK_FORMAT = "nrdb-agent.translation-repair-check.v1"
DISCOVERY_STAGES = {
	DISCOVERY_FORMAT: "discovery",
	BASELINE_FORMAT: "baseline",
	CHECK_FORMAT: "repair_check",
}


DISCREPANCY_INSTRUCTIONS = """You are the NRDB Japanese translation discrepancy critic.

Compare one blind generated Japanese translation with a human gold Japanese translation. Similar but differently worded translations are acceptable. Judge semantic and grammatical equivalence, not string similarity.

Pay particular attention to the requested target morphemes, but do not blame a target merely because it occurs in the sentence. Distinguish polarity, tense/aspect/modality, argument structure, factual versus hypothetical interpretation, clause linkage, discourse force, lexical choice, free gold translation, and missing discourse context. A difference unrelated to the requested morphemes must be identified as non-target. If the isolated utterance cannot resolve a lexical choice, classify it as context-dependent rather than a translation error.

Return only the requested JSON. Do not produce chain-of-thought.
"""


REPAIR_INSTRUCTIONS = """You are the NRDB construction-assisted Japanese translation repair critic.

Compare a blind baseline translation and a construction-assisted translation against the same human gold translation. Similar but differently worded translations are acceptable. Decide whether construction evidence meaningfully repaired the target grammatical phenomenon, left it unchanged, or caused a regression. Judge the target phenomenon separately from unrelated lexical or context-dependent differences. Do not reward mere wording similarity.

Return only the requested JSON. Do not produce chain-of-thought.
"""


DISCREPANCY_FORMAT = {
	"type": "json_schema", "name": "nrdb_translation_discrepancy", "strict": True,
	"schema": {
		"type": "object",
		"properties": {
			"relation": {"type": "string", "enum": ["equivalent", "acceptable_variant", "minor_difference", "substantive_difference", "context_dependent", "uncertain"]},
			"severity": {"type": "integer", "minimum": 0, "maximum": 3},
			"target_morpheme_error": {"type": "boolean"},
			"likely_responsible_ids": {"type": "array", "items": {"type": "string"}},
			"candidate_pattern": {"type": "string"},
			"discrepancy_type": {"type": "string", "enum": ["none", "lexical", "polarity", "tense_aspect", "modality", "argument_structure", "clause_relation", "factuality", "discourse_pragmatics", "morph_analysis", "free_gold_translation", "missing_context", "other", "uncertain"]},
			"summary_jp": {"type": "string"},
		},
		"required": ["relation", "severity", "target_morpheme_error", "likely_responsible_ids", "candidate_pattern", "discrepancy_type", "summary_jp"],
		"additionalProperties": False,
	},
}


REPAIR_FORMAT = {
	"type": "json_schema", "name": "nrdb_translation_repair_check", "strict": True,
	"schema": {
		"type": "object",
		"properties": {
			"outcome": {"type": "string", "enum": ["repaired", "improved_but_incomplete", "unchanged", "different_but_equivalent", "regressed", "uncertain"]},
			"baseline_relation": {"type": "string", "enum": ["equivalent", "acceptable_variant", "minor_difference", "substantive_difference", "context_dependent", "uncertain"]},
			"construction_relation": {"type": "string", "enum": ["equivalent", "acceptable_variant", "minor_difference", "substantive_difference", "context_dependent", "uncertain"]},
			"target_phenomenon_improved": {"type": "boolean"},
			"helpful_ids_or_patterns": {"type": "array", "items": {"type": "string"}},
			"remaining_problem": {"type": "string"},
			"summary_jp": {"type": "string"},
		},
		"required": ["outcome", "baseline_relation", "construction_relation", "target_phenomenon_improved", "helpful_ids_or_patterns", "remaining_problem", "summary_jp"],
		"additionalProperties": False,
	},
}


def _read(path, expected_format=None):
	path = Path(path)
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as error:
		raise ValueError("invalid discrepancy JSON: {}".format(path)) from error
	if not isinstance(payload, dict):
		raise ValueError("discrepancy artifact must contain one JSON object")
	if expected_format and payload.get("format") != expected_format:
		raise ValueError("expected {}, got {}".format(expected_format, payload.get("format")))
	return payload


def _write(path, payload):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
	return str(path)


def list_discoveries(directory=".", recursive=False, latest=None):
	directory = Path(directory)
	if not directory.is_dir():
		raise ValueError("discrepancy directory does not exist: {}".format(directory))
	paths = directory.rglob("*.json") if recursive else directory.glob("*.json")
	artifacts = []
	for path in paths:
		try:
			payload = json.loads(path.read_text(encoding="utf-8"))
		except (OSError, json.JSONDecodeError):
			continue
		stage = DISCOVERY_STAGES.get(payload.get("format")) if isinstance(payload, dict) else None
		if stage is None:
			continue
		selection = payload.get("selection") or {}
		summary = payload.get("summary") or {}
		rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
		failed = int(summary.get("failed") or 0)
		status = "created" if stage == "discovery" else ("completed" if failed == 0 else ("failed" if failed == len(rows) else "partial"))
		modified = path.stat().st_mtime
		artifacts.append({
			"path": str(path), "filename": path.name, "stage": stage, "status": status,
			"modified_at": datetime.fromtimestamp(modified, tz=timezone.utc).isoformat(), "modified_timestamp": modified,
			"target_ids": list(selection.get("target_ids") or []),
			"annotation_schema_id": selection.get("annotation_schema_id"), "region": selection.get("region"),
			"dataset_ids": list(selection.get("dataset_ids") or []), "rows": len(rows),
			"sampled_rows_by_id": dict(selection.get("sampled_rows_by_id") or {}),
			"translation_model": (payload.get("models") or {}).get("translation"),
			"discrepancy_model": (payload.get("models") or {}).get("discrepancy"),
			"counts": dict(summary.get("counts") or {}), "failed_rows": failed,
			"estimated_cost_usd": summary.get("estimated_cost_usd"), "pricing_complete": summary.get("pricing_complete"),
		})
	artifacts.sort(key=lambda value: (-value["modified_timestamp"], value["path"]))
	for artifact in artifacts:
		artifact.pop("modified_timestamp", None)
	if latest is not None:
		artifacts = artifacts[:max(0, int(latest))]
	return artifacts


def _morpheme_count(annotation):
	return sum(len(phrase.split("-")) for phrase in str(annotation or "").strip().split())


def _row_targets(row, target_ids):
	available = set(annotation_ids(row.get("gold_annotation")))
	return [target_id for target_id in target_ids if target_id in available]


def create_discovery(nrdb, target_ids, dataset_ids, annotation_schema_id, region, limit=100,
	seed=1, min_morphemes=1, require_all=False, output=None):
	target_ids = list(dict.fromkeys(str(value).strip() for value in target_ids if str(value).strip()))
	if not target_ids:
		raise ValueError("at least one target morpheme ID is required")
	dataset_ids = sorted({int(value) for value in (dataset_ids or [])})
	rows = nrdb.morph_eval_rows(dataset_ids, annotation_schema_id=annotation_schema_id, region=region)
	pools = {target_id: [] for target_id in target_ids}
	for row in rows:
		if int(row.get("annotation_schema_id") or 0) != int(annotation_schema_id):
			continue
		if str(row.get("dialect_region") or "").strip() != str(region).strip():
			continue
		if not str(row.get("translation_jp") or "").strip():
			continue
		if _morpheme_count(row.get("gold_annotation")) < int(min_morphemes):
			continue
		matched = _row_targets(row, target_ids)
		if (require_all and len(matched) != len(target_ids)) or (not require_all and not matched):
			continue
		base = {
			"sentence_id": int(row["sentence_id"]), "dataset_id": int(row["dataset_id"]),
			"example_id": str(row.get("example_id") or ""), "dialect_id": int(row["dialect_id"]),
			"dialect_region": str(row.get("dialect_region") or ""), "source": str(row.get("text") or ""),
			"gold_segmented": str(row.get("gold_segmented") or ""), "gold_annotation": str(row.get("gold_annotation") or ""),
			"gold_translation": str(row.get("translation_jp") or ""),
		}
		for target_id in matched:
			pools[target_id].append(base)
	pool_sizes = {target_id: len(pool) for target_id, pool in pools.items()}
	selected = []
	sampled_by_id = {}
	for target_id, pool in pools.items():
		pool = list(pool)
		random.Random("{}:{}".format(int(seed), target_id)).shuffle(pool)
		pool = pool[:int(limit)]
		sampled_by_id[target_id] = len(pool)
		for base in pool:
			assignment = dict(base)
			assignment["target_id"] = target_id
			assignment["matched_target_ids"] = [target_id]
			assignment["assignment_id"] = "{}:{}:{}".format(target_id, base["dataset_id"], base["sentence_id"])
			selected.append(assignment)
	if not selected:
		raise ValueError("no translated gold rows match the requested morphemes and scope")
	payload = {
		"format": DISCOVERY_FORMAT,
		"selection": {
			"target_ids": target_ids, "dataset_ids": dataset_ids,
			"annotation_schema_id": int(annotation_schema_id), "region": str(region),
			"limit_per_id": int(limit), "limit": int(limit), "seed": int(seed), "min_morphemes": int(min_morphemes),
			"match": "all" if require_all else "independent_per_id",
			"eligible_pool_size_by_id": pool_sizes, "sampled_rows_by_id": sampled_by_id,
			"eligible_assignments": sum(pool_sizes.values()), "sampled_assignments": len(selected),
		},
		"rows": selected,
	}
	if output:
		_write(output, payload)
	return payload


class DiscrepancyJudge(AnnotationAgent):
	def _judge(self, payload, instructions, text_format):
		response = self._create_response(
			[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], instructions,
			tools=[], max_output_tokens=1600, text_format=text_format,
		)
		incomplete = _response_incomplete_reason(response)
		if incomplete:
			raise RuntimeError("discrepancy judgment incomplete: {}".format(incomplete))
		result = json.loads((response.output_text or "").strip())
		result["model_response_id"] = getattr(response, "id", None)
		return result

	def discrepancy(self, row, generated):
		return self._judge({
			"target_ids": row["matched_target_ids"], "source": row["source"],
			"gold_annotation": row["gold_annotation"], "gold_translation": row["gold_translation"],
			"generated_annotation": generated.get("annotation"), "generated_translation": generated.get("translation"),
		}, DISCREPANCY_INSTRUCTIONS, DISCREPANCY_FORMAT)

	def repair(self, row, baseline, construction):
		return self._judge({
			"target_ids": row["matched_target_ids"], "source": row["source"],
			"gold_annotation": row["gold_annotation"], "gold_translation": row["gold_translation"],
			"baseline_annotation": baseline.get("annotation"), "baseline_translation": baseline.get("translation"),
			"baseline_judgment": row.get("baseline_judgment"),
			"construction_annotation": construction.get("annotation"), "construction_translation": construction.get("translation"),
		}, REPAIR_INSTRUCTIONS, REPAIR_FORMAT)


def _models_payload(translation_model, discrepancy_model):
	return {"translation": str(translation_model), "discrepancy": str(discrepancy_model), "id_critic": "nrdb_agent_default"}


def _summary(rows, field, values):
	counts = {value: 0 for value in values}
	failed = 0
	for row in rows:
		if row.get("error"):
			failed += 1
			continue
		value = (row.get(field) or {}).get("outcome" if field == "repair_judgment" else "relation")
		if value in counts:
			counts[value] += 1
	return {"rows": len(rows), "failed": failed, "counts": counts}


def _translation_usage(rows, result_key):
	totals = {"requests": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "estimated_cost_usd": 0.0}
	pricing_complete = True
	for row in rows:
		usage = ((row.get(result_key) or {}).get("api_usage") or {})
		values = usage.get("totals") or {}
		for key in ("requests", "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
			totals[key] += int(values.get(key) or 0)
		totals["estimated_cost_usd"] += float(values.get("estimated_cost_usd") or 0.0)
		pricing_complete = pricing_complete and bool(usage.get("pricing_complete", False))
	return {"pricing_complete": pricing_complete, "totals": totals}


def _candidate_morphemes(rows):
	exposure = Counter()
	attributed = Counter()
	severity = Counter()
	types = defaultdict(Counter)
	examples = defaultdict(list)
	patterns = defaultdict(Counter)
	for row in rows:
		for target_id in row.get("matched_target_ids", []):
			exposure[target_id] += 1
		judgment = row.get("baseline_judgment") or {}
		if not judgment.get("target_morpheme_error"):
			continue
		for target_id in judgment.get("likely_responsible_ids", []):
			target_id = str(target_id or "").strip()
			if not target_id or target_id not in row.get("matched_target_ids", []):
				continue
			attributed[target_id] += 1
			severity[target_id] += int(judgment.get("severity") or 0)
			types[target_id][str(judgment.get("discrepancy_type") or "uncertain")] += 1
			pattern = str(judgment.get("candidate_pattern") or "").strip()
			if pattern:
				patterns[target_id][pattern] += 1
			if len(examples[target_id]) < 10:
				examples[target_id].append({"dataset_id": row.get("dataset_id"), "sentence_id": row.get("sentence_id"), "summary_jp": judgment.get("summary_jp")})
	rows_out = []
	for target_id, count in attributed.items():
		rows_out.append({
			"morph_id": target_id, "exposures": exposure[target_id], "attributed_errors": count,
			"attributed_error_rate": count / exposure[target_id] if exposure[target_id] else None,
			"severity_sum": severity[target_id], "discrepancy_types": dict(types[target_id]),
			"candidate_patterns": [{"pattern": value, "count": amount} for value, amount in patterns[target_id].most_common()],
			"examples": examples[target_id],
		})
	rows_out.sort(key=lambda value: (-value["severity_sum"], -value["attributed_errors"], value["morph_id"]))
	return rows_out


def _with_costs(summary, rows, result_key, discrepancy_usage):
	translation_usage = _translation_usage(rows, result_key)
	discrepancy_totals = (discrepancy_usage.get("totals") or {})
	translation_cost = float((translation_usage.get("totals") or {}).get("estimated_cost_usd") or 0.0)
	discrepancy_cost = float(discrepancy_totals.get("estimated_cost_usd") or 0.0)
	summary["translation_api_usage"] = translation_usage
	summary["discrepancy_api_usage"] = discrepancy_usage
	summary["estimated_cost_usd"] = translation_cost + discrepancy_cost
	summary["pricing_complete"] = bool(translation_usage.get("pricing_complete")) and bool(discrepancy_usage.get("pricing_complete"))
	return summary


def run_discovery(nrdb, discovery_path, output, translation_model="gpt-5.6-luna",
	discrepancy_model="gpt-5.6-terra", openai_client=None, progress=print):
	discovery = _read(discovery_path, DISCOVERY_FORMAT)
	tracker = UsageTracker()
	judge = DiscrepancyJudge(nrdb, discrepancy_model, client=tracked_client(openai_client, tracker), progress=progress)
	result = {
		"format": BASELINE_FORMAT, "selection": discovery["selection"],
		"models": _models_payload(translation_model, discrepancy_model), "rows": [],
	}
	for index, source in enumerate(discovery["rows"], start=1):
		progress("[{}/{}] baseline sentence {}".format(index, len(discovery["rows"]), source["sentence_id"]))
		row = dict(source)
		try:
			generated = translate_text(
				nrdb, source["source"], "japanese", discovery["selection"]["annotation_schema_id"], discovery["selection"]["region"],
				dialect_ids=[source["dialect_id"]], model_name=translation_model, semantic_feedback="none",
				use_constructions=False, openai_client=openai_client, progress=progress,
			)
			row["baseline"] = generated
			row["baseline_judgment"] = judge.discrepancy(row, generated)
		except Exception as error:
			row["error"] = str(error)
		result["rows"].append(row)
		discrepancy_usage = tracker.summary()
		result["summary"] = _with_costs(
			_summary(result["rows"], "baseline_judgment", ["equivalent", "acceptable_variant", "minor_difference", "substantive_difference", "context_dependent", "uncertain"]),
			result["rows"], "baseline", discrepancy_usage,
		)
		result["summary"]["morphemes_to_analyse"] = _candidate_morphemes(result["rows"])
		_write(output, result)
	return result


def check_discovery(nrdb, baseline_path, output, translation_model=None,
	discrepancy_model="gpt-5.6-terra", openai_client=None, progress=print):
	baseline = _read(baseline_path, BASELINE_FORMAT)
	baseline_model = str((baseline.get("models") or {}).get("translation") or "")
	translation_model = str(translation_model or baseline_model)
	if translation_model != baseline_model:
		raise ValueError("repair check must use the baseline translation model {!r}".format(baseline_model))
	tracker = UsageTracker()
	judge = DiscrepancyJudge(nrdb, discrepancy_model, client=tracked_client(openai_client, tracker), progress=progress)
	result = {
		"format": CHECK_FORMAT, "selection": baseline["selection"],
		"models": _models_payload(translation_model, discrepancy_model), "baseline_artifact": str(baseline_path), "rows": [],
	}
	for index, source in enumerate(baseline["rows"], start=1):
		progress("[{}/{}] construction check sentence {}".format(index, len(baseline["rows"]), source["sentence_id"]))
		row = dict(source)
		if source.get("error") or not source.get("baseline"):
			row["error"] = source.get("error") or "baseline translation is missing"
		else:
			try:
				construction = translate_text(
					nrdb, source["source"], "japanese", baseline["selection"]["annotation_schema_id"], baseline["selection"]["region"],
					dialect_ids=[source["dialect_id"]], model_name=translation_model, semantic_feedback="none",
					use_constructions=True, openai_client=openai_client, progress=progress,
				)
				row["construction"] = construction
				row["repair_judgment"] = judge.repair(row, source["baseline"], construction)
			except Exception as error:
				row["error"] = str(error)
		result["rows"].append(row)
		discrepancy_usage = tracker.summary()
		result["summary"] = _with_costs(
			_summary(result["rows"], "repair_judgment", ["repaired", "improved_but_incomplete", "unchanged", "different_but_equivalent", "regressed", "uncertain"]),
			result["rows"], "construction", discrepancy_usage,
		)
		_write(output, result)
	return result
