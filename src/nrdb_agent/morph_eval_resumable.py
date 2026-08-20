import json
import os
import random
from pathlib import Path

from .dataset_io import write_json, write_tsv
from .metrics import annotation_metrics, segmentation_metrics
from .morph_eval import _paired_summary, _result_row, _run_contract
from .task_agent import SEMANTIC_FEEDBACK_MODES, TaskAwareAnnotationAgent
from .usage import UsageTracker, tracked_client


CHECKPOINT_FORMAT = "nrdb-agent.morph-ceiling-checkpoint.v2"


def _canonical_path(value):
	if value in (None, ""):
		return None
	return str(Path(value).expanduser().resolve())


def _checkpoint_path(output=None, checkpoint=None):
	if checkpoint:
		return Path(checkpoint)
	if output:
		return Path(str(output) + ".checkpoint.jsonl")
	return Path(".nrdb-agent-morph-eval.checkpoint.jsonl")


def _append_jsonl(path, value):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
		handle.flush()
		os.fsync(handle.fileno())


def _load_checkpoint(path):
	path = Path(path)
	if not path.is_file():
		return None, []
	meta = None
	rows = []
	with path.open("r", encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				value = json.loads(line)
			except json.JSONDecodeError as error:
				# A power loss can truncate only the final append. Ignore that final
				# partial line; any prior fsync'd rows remain valid.
				if line_number > 1:
					break
				raise ValueError("invalid checkpoint metadata JSON") from error
			if value.get("record_type") == "meta":
				if meta is not None:
					raise ValueError("checkpoint contains duplicate metadata records")
				meta = value
			elif value.get("record_type") == "row":
				row = value.get("row")
				if isinstance(row, dict):
					rows.append(row)
	return meta, rows


def _build_cohort(nrdb, contract, dataset_ids, limit, seed, semantic_feedback, require_semantic_feedback):
	run_dataset_ids = set(contract["dataset_ids"])
	if dataset_ids:
		selected_dataset_ids = {int(value) for value in dataset_ids}
		unknown = sorted(selected_dataset_ids - run_dataset_ids)
		if unknown:
			raise ValueError("dataset IDs are not represented in train.jsonl: {}".format(", ".join(map(str, unknown))))
	else:
		selected_dataset_ids = run_dataset_ids
	registered = nrdb.morph_eval_rows(selected_dataset_ids)
	eligible = []
	unmatched_identity = 0
	for row in registered:
		example_id = str(row.get("example_id") or "").strip()
		if not example_id:
			unmatched_identity += 1
			continue
		identity = (int(row["dataset_id"]), example_id)
		if identity in contract["train_identities"]:
			continue
		if semantic_feedback == "existing" and require_semantic_feedback and not str(row.get("translation_jp") or "").strip():
			continue
		eligible.append(row)
	eligible_pool_size = len(eligible)
	rng = random.Random(int(seed))
	rng.shuffle(eligible)
	if limit is not None:
		eligible = eligible[:max(0, int(limit))]
	if not eligible:
		raise ValueError("no eligible registered gold rows remain after excluding the morph training split")
	return {
		"rows": eligible,
		"dataset_ids": sorted(selected_dataset_ids),
		"registered_gold_rows": len(registered),
		"eligible_pool_size": eligible_pool_size,
		"unmatched_identity": unmatched_identity,
	}


def _checkpoint_meta(contract, cohort, model_name, limit, seed, expected_morph_model, id_model,
	semantic_feedback, require_semantic_feedback):
	return {
		"record_type": "meta",
		"format": CHECKPOINT_FORMAT,
		"morph_run": _canonical_path(contract["run_dir"]),
		"train_path": _canonical_path(contract["train_path"]),
		"datasets": cohort["dataset_ids"],
		"cohort_sentence_ids": [int(row["sentence_id"]) for row in cohort["rows"]],
		"limit": None if limit is None else int(limit),
		"seed": int(seed),
		"agent_model": str(model_name),
		"expected_morph_model": str(expected_morph_model or ""),
		"id_model": _canonical_path(id_model),
		"semantic_feedback": str(semantic_feedback),
		"require_semantic_feedback": bool(require_semantic_feedback),
	}


def _verify_checkpoint(expected, actual):
	if not actual:
		raise ValueError("checkpoint has no metadata record")
	for key in (
		"format", "morph_run", "train_path", "datasets", "cohort_sentence_ids",
		"limit", "seed", "agent_model", "expected_morph_model", "id_model",
		"semantic_feedback", "require_semantic_feedback",
	):
		if actual.get(key) != expected.get(key):
			raise ValueError("checkpoint does not match this evaluation: {} differs".format(key))


def evaluate_morph_agent_resumable(nrdb, run_dir, model_name="gpt-5.6", limit=None, seed=1,
	dataset_ids=None, expected_morph_model=None, id_model=None, output=None, checkpoint=None,
	resume=False, semantic_feedback="none", require_semantic_feedback=False,
	openai_client=None, progress=print):
	semantic_feedback = str(semantic_feedback or "none")
	if semantic_feedback not in SEMANTIC_FEEDBACK_MODES:
		raise ValueError("invalid semantic_feedback: {}".format(semantic_feedback))
	contract = _run_contract(run_dir)
	cohort = _build_cohort(
		nrdb, contract, dataset_ids, limit, seed, semantic_feedback, require_semantic_feedback,
	)
	checkpoint_path = _checkpoint_path(output=output, checkpoint=checkpoint)
	expected_meta = _checkpoint_meta(
		contract, cohort, model_name, limit, seed, expected_morph_model, id_model,
		semantic_feedback, require_semantic_feedback,
	)

	existing_meta, existing_rows = _load_checkpoint(checkpoint_path)
	if existing_meta is not None:
		if not resume:
			raise ValueError(
			"checkpoint already exists: {}. Use --resume to continue it, or choose a new --output/--checkpoint.".format(checkpoint_path)
		)
		_verify_checkpoint(expected_meta, existing_meta)
	else:
		if resume:
			raise ValueError("--resume requested but checkpoint does not exist: {}".format(checkpoint_path))
		_append_jsonl(checkpoint_path, expected_meta)

	rows_by_sentence = {int(row["sentence_id"]): row for row in existing_rows}
	results = [rows_by_sentence[int(source["sentence_id"])] for source in cohort["rows"] if int(source["sentence_id"]) in rows_by_sentence]
	completed_ids = set(rows_by_sentence)
	if completed_ids:
		progress("resume: loaded {} checkpointed row(s) from {}".format(len(completed_ids), checkpoint_path))

	for index, source in enumerate(cohort["rows"], start=1):
		sentence_id = int(source["sentence_id"])
		if sentence_id in completed_ids:
			progress("[{}/{}] sentence {}: checkpointed, skipping paid inference".format(index, len(cohort["rows"]), sentence_id))
			continue

		progress("[{}/{}] sentence {}".format(index, len(cohort["rows"]), sentence_id))
		baseline = nrdb.morph_analyze(source["text"], int(source["dialect_id"]), int(source["annotation_schema_id"]))
		inference = baseline.get("inference") if isinstance(baseline, dict) else None
		inference = inference if isinstance(inference, dict) else {}
		model_id = str(inference.get("model_id") or "")
		if expected_morph_model and model_id != str(expected_morph_model):
			raise RuntimeError("deployed morph model mismatch: expected {!r}, got {!r}".format(expected_morph_model, model_id))

		baseline_metrics = annotation_metrics(baseline.get("annotation"), source["gold_annotation"])
		baseline_seg = segmentation_metrics(baseline.get("segmented"), source["gold_segmented"])
		tracker = UsageTracker()
		client = tracked_client(openai_client, tracker)
		existing_translation = str(source.get("translation_jp") or "").strip()
		item = {
			"sentence_id": sentence_id,
			"dialect_id": int(source["dialect_id"]),
			"dialect_region": source.get("dialect_region") or "",
			"text": source["text"],
			"translation_jp": existing_translation if semantic_feedback in {"existing", "auto"} else None,
		}
		job = {
			"annotation_schema_id": int(source["annotation_schema_id"]),
			"model_name": model_name,
			"prompt_version": "annotation-v9",
			"task": "morph",
			"semantic_feedback": semantic_feedback,
			"require_semantic_feedback": bool(require_semantic_feedback),
			"morphology_source": "predict",
			"produce_translation": False,
			"blind_translation": False,
		}
		agent = TaskAwareAnnotationAgent(nrdb, model_name, client=client, progress=progress, id_model_path=id_model)
		agent_result = agent.annotate(item, job, baseline)
		usage = tracker.summary()
		agent_metrics = annotation_metrics(agent_result.get("annotation"), source["gold_annotation"])
		agent_seg = segmentation_metrics(agent_result.get("segmented"), source["gold_segmented"])
		row = _result_row(source, baseline, agent_result, usage, baseline_metrics, agent_metrics, baseline_seg, agent_seg)
		row["semantic_feedback"] = semantic_feedback
		row["existing_translation_present"] = int(bool(existing_translation))

		# Durable checkpoint BEFORE moving to the next paid row.
		_append_jsonl(checkpoint_path, {"record_type": "row", "row": row})
		results.append(row)
		completed_ids.add(sentence_id)
		progress("  baseline ID {:.1f}% -> agent {:.1f}% | agree={} | cost ${:.4f} | checkpointed".format(
			100.0 * baseline_metrics["id_match_rate"],
			100.0 * agent_metrics["id_match_rate"],
			"yes" if row["baseline_agent_full_agree"] else "no",
			float(row.get("agent_cost_usd") or 0.0),
		))

	# Preserve original deterministic cohort order after resume.
	row_map = {int(row["sentence_id"]): row for row in results}
	results = [row_map[int(source["sentence_id"])] for source in cohort["rows"] if int(source["sentence_id"]) in row_map]
	summary = _paired_summary(results)
	morph_model_ids = sorted({str(row.get("baseline_model_id") or "") for row in results if row.get("baseline_model_id")})
	total_cost = sum(float(row.get("agent_cost_usd") or 0.0) for row in results)
	pricing_complete = all(bool(row.get("agent_pricing_complete")) for row in results)
	summary.update({
		"format": "nrdb-agent.morph-ceiling-eval.v4",
		"morph_run": contract["run_dir"],
		"train_rows": contract["train_rows"],
		"train_rows_missing_identity": contract["train_rows_missing_identity"],
		"datasets": cohort["dataset_ids"],
		"registered_gold_rows": cohort["registered_gold_rows"],
		"eligible_pool_after_train_exclusion": cohort["eligible_pool_size"],
		"sampled_rows": len(cohort["rows"]),
		"registered_rows_without_example_id": cohort["unmatched_identity"],
		"selection_seed": int(seed),
		"morph_model_ids": morph_model_ids,
		"expected_morph_model": expected_morph_model,
		"agent_model": model_name,
		"semantic_feedback": semantic_feedback,
		"require_semantic_feedback": bool(require_semantic_feedback),
		"estimated_cost_usd": total_cost,
		"pricing_complete": pricing_complete,
		"checkpoint": str(checkpoint_path),
		"checkpointed_rows": len(results),
	})
	payload = {"summary": summary, "rows": results}
	if output:
		if Path(output).suffix.lower() == ".json":
			write_json(output, payload)
		else:
			write_tsv(output, results)
	return payload
