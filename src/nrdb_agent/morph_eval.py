import json
import random
from pathlib import Path

from .dataset_io import write_json, write_tsv
from .metrics import annotation_metrics, job_annotation_metrics, job_segmentation_metrics, segmentation_metrics
from .task_agent import TaskAwareAnnotationAgent
from .usage import UsageTracker, tracked_client


def _read_jsonl(path):
	rows = []
	with Path(path).open("r", encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				value = json.loads(line)
			except json.JSONDecodeError as error:
				raise ValueError("{}:{} is not valid JSONL".format(path, line_number)) from error
			if not isinstance(value, dict):
				raise ValueError("{}:{} must contain a JSON object".format(path, line_number))
			rows.append(value)
	return rows


def _identity(value):
	dataset_id = value.get("dataset_id")
	text_id = value.get("text_id")
	if dataset_id in (None, "") or text_id in (None, ""):
		return None
	return int(dataset_id), str(text_id).strip()


def _run_contract(run_dir):
	run_dir = Path(run_dir)
	train_path = run_dir / "train.jsonl"
	if not train_path.is_file():
		raise ValueError("morph run has no train.jsonl: {}".format(train_path))
	train = _read_jsonl(train_path)
	train_ids = set()
	dataset_ids = set()
	missing_identity = 0
	for row in train:
		identity = _identity(row)
		if identity is None:
			missing_identity += 1
			continue
		train_ids.add(identity)
		dataset_ids.add(identity[0])
	if not dataset_ids:
		raise ValueError("train.jsonl contains no registered dataset identities")
	return {
		"run_dir": str(run_dir),
		"train_path": str(train_path),
		"train_rows": len(train),
		"train_identities": train_ids,
		"dataset_ids": sorted(dataset_ids),
		"train_rows_missing_identity": missing_identity,
	}


def _agreement_flags(baseline_segmented, baseline_annotation, agent_segmented, agent_annotation):
	id_agree = bool(annotation_metrics(agent_annotation, baseline_annotation)["linguistic_exact"])
	seg_agree = bool(segmentation_metrics(agent_segmented, baseline_segmented)["exact"])
	return {
		"baseline_agent_id_agree": int(id_agree),
		"baseline_agent_seg_agree": int(seg_agree),
		"baseline_agent_full_agree": int(id_agree and seg_agree),
	}


def _result_row(source, baseline, agent, usage, baseline_metrics, agent_metrics, baseline_seg, agent_seg):
	inference = baseline.get("inference") if isinstance(baseline, dict) else None
	inference = inference if isinstance(inference, dict) else {}
	row = {
		"sentence_id": int(source["sentence_id"]),
		"dataset_id": int(source["dataset_id"]),
		"dataset_name": source.get("dataset_name") or "",
		"dataset_type": source.get("dataset_type") or "",
		"example_id": source.get("example_id") or "",
		"dialect_id": int(source["dialect_id"]),
		"dialect_region": source.get("dialect_region") or "",
		"source_text": source.get("text") or "",
		"gold_segmented": source.get("gold_segmented") or "",
		"gold_annotation": source.get("gold_annotation") or "",
		"baseline_segmented": baseline.get("segmented") or "",
		"baseline_annotation": baseline.get("annotation") or "",
		"agent_segmented": agent.get("segmented") or "",
		"agent_annotation": agent.get("annotation") or "",
		"baseline_model_id": inference.get("model_id") or "",
		"baseline_model_label": inference.get("model_label") or "",
		"baseline_id_exact": int(bool(baseline_metrics["linguistic_exact"])),
		"agent_id_exact": int(bool(agent_metrics["linguistic_exact"])),
		"baseline_id_match_rate": baseline_metrics["id_match_rate"],
		"agent_id_match_rate": agent_metrics["id_match_rate"],
		"baseline_id_edits": baseline_metrics["edits"],
		"agent_id_edits": agent_metrics["edits"],
		"baseline_seg_exact": int(bool(baseline_seg["exact"])),
		"agent_seg_exact": int(bool(agent_seg["exact"])),
		"baseline_boundary_f1": baseline_seg["boundary_f1"],
		"agent_boundary_f1": agent_seg["boundary_f1"],
		"agent_decision": agent.get("decision") or "",
		"agent_confidence": agent.get("confidence"),
		"agent_cost_usd": float(((usage.get("totals") or {}).get("estimated_cost_usd") or 0.0)),
		"agent_pricing_complete": bool(usage.get("pricing_complete")),
	}
	row.update(_agreement_flags(
		row["baseline_segmented"], row["baseline_annotation"],
		row["agent_segmented"], row["agent_annotation"],
	))
	return row


def _quality_summary(rows, total_rows):
	baseline_rows = [
		{"ai_segmented": row["baseline_segmented"], "ai_annotation": row["baseline_annotation"],
		 "gold_segmented": row["gold_segmented"], "gold_annotation": row["gold_annotation"]}
		for row in rows
	]
	agent_rows = [
		{"ai_segmented": row["agent_segmented"], "ai_annotation": row["agent_annotation"],
		 "gold_segmented": row["gold_segmented"], "gold_annotation": row["gold_annotation"]}
		for row in rows
	]
	baseline_id = job_annotation_metrics(baseline_rows)
	agent_id = job_annotation_metrics(agent_rows)
	baseline_seg = job_segmentation_metrics(baseline_rows)
	agent_seg = job_segmentation_metrics(agent_rows)
	baseline_full_exact = sum(1 for row in rows if bool(row["baseline_id_exact"]) and bool(row["baseline_seg_exact"]))
	agent_full_exact = sum(1 for row in rows if bool(row["agent_id_exact"]) and bool(row["agent_seg_exact"]))
	count = len(rows)
	return {
		"rows": count,
		"coverage": count / total_rows if total_rows else None,
		"baseline": {
			"id_exact_accuracy": baseline_id["linguistic_exact_accuracy"],
			"id_match_rate": baseline_id["id_match_rate"],
			"segmentation_exact_accuracy": baseline_seg["exact_accuracy"],
			"segmentation_boundary_f1": baseline_seg["boundary_f1"],
			"full_analysis_exact_accuracy": baseline_full_exact / count if count else None,
		},
		"agent": {
			"id_exact_accuracy": agent_id["linguistic_exact_accuracy"],
			"id_match_rate": agent_id["id_match_rate"],
			"segmentation_exact_accuracy": agent_seg["exact_accuracy"],
			"segmentation_boundary_f1": agent_seg["boundary_f1"],
			"full_analysis_exact_accuracy": agent_full_exact / count if count else None,
		},
	}


def _agreement_calibration(rows):
	total = len(rows)
	views = {}
	for name, key in (
		("id", "baseline_agent_id_agree"),
		("segmentation", "baseline_agent_seg_agree"),
		("full_analysis", "baseline_agent_full_agree"),
	):
		agree = [row for row in rows if bool(row.get(key))]
		disagree = [row for row in rows if not bool(row.get(key))]
		views[name] = {
			"agreement": _quality_summary(agree, total),
			"disagreement": _quality_summary(disagree, total),
		}
	return views


def _paired_summary(rows):
	baseline_rows = [
		{"ai_segmented": row["baseline_segmented"], "ai_annotation": row["baseline_annotation"],
		 "gold_segmented": row["gold_segmented"], "gold_annotation": row["gold_annotation"]}
		for row in rows
	]
	agent_rows = [
		{"ai_segmented": row["agent_segmented"], "ai_annotation": row["agent_annotation"],
		 "gold_segmented": row["gold_segmented"], "gold_annotation": row["gold_annotation"]}
		for row in rows
	]
	baseline_id = job_annotation_metrics(baseline_rows)
	agent_id = job_annotation_metrics(agent_rows)
	baseline_seg = job_segmentation_metrics(baseline_rows)
	agent_seg = job_segmentation_metrics(agent_rows)

	id_corrected = id_damaged = id_improved = id_harmed = 0
	seg_corrected = seg_damaged = 0
	for row in rows:
		baseline_exact = bool(row["baseline_id_exact"])
		agent_exact = bool(row["agent_id_exact"])
		if not baseline_exact and agent_exact:
			id_corrected += 1
		if baseline_exact and not agent_exact:
			id_damaged += 1
		if row["agent_id_edits"] < row["baseline_id_edits"]:
			id_improved += 1
		elif row["agent_id_edits"] > row["baseline_id_edits"]:
			id_harmed += 1
		if not bool(row["baseline_seg_exact"]) and bool(row["agent_seg_exact"]):
			seg_corrected += 1
		if bool(row["baseline_seg_exact"]) and not bool(row["agent_seg_exact"]):
			seg_damaged += 1

	baseline_id_errors = len(rows) - baseline_id["linguistic_exact_matches"]
	baseline_id_correct = baseline_id["linguistic_exact_matches"]
	return {
		"rows_scored": len(rows),
		"baseline": {"id": baseline_id, "segmentation": baseline_seg},
		"agent": {"id": agent_id, "segmentation": agent_seg},
		"paired": {
			"baseline_id_errors_corrected": id_corrected,
			"baseline_id_error_recovery_rate": id_corrected / baseline_id_errors if baseline_id_errors else None,
			"baseline_id_correct_damaged": id_damaged,
			"baseline_id_damage_rate": id_damaged / baseline_id_correct if baseline_id_correct else None,
			"rows_with_fewer_id_edits": id_improved,
			"rows_with_more_id_edits": id_harmed,
			"baseline_seg_errors_corrected": seg_corrected,
			"baseline_seg_correct_damaged": seg_damaged,
		},
		"agreement_calibration": _agreement_calibration(rows),
	}


def evaluate_morph_agent(nrdb, run_dir, model_name="gpt-5.6", limit=None, seed=1,
	dataset_ids=None, expected_morph_model=None, id_model=None, output=None,
	openai_client=None, progress=print):
	contract = _run_contract(run_dir)
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
		eligible.append(row)

	eligible_pool_size = len(eligible)
	rng = random.Random(int(seed))
	rng.shuffle(eligible)
	if limit is not None:
		eligible = eligible[:max(0, int(limit))]
	if not eligible:
		raise ValueError("no eligible registered gold rows remain after excluding the morph training split")

	results = []
	morph_model_ids = set()
	total_cost = 0.0
	pricing_complete = True
	for index, source in enumerate(eligible, start=1):
		progress("[{}/{}] sentence {}".format(index, len(eligible), source["sentence_id"]))
		baseline = nrdb.morph_analyze(source["text"], int(source["dialect_id"]), int(source["annotation_schema_id"]))
		inference = baseline.get("inference") if isinstance(baseline, dict) else None
		model_id = str((inference or {}).get("model_id") or "") if isinstance(inference, dict) else ""
		if model_id:
			morph_model_ids.add(model_id)
		if expected_morph_model and model_id != str(expected_morph_model):
			raise RuntimeError("deployed morph model mismatch: expected {!r}, got {!r}".format(expected_morph_model, model_id))

		baseline_metrics = annotation_metrics(baseline.get("annotation"), source["gold_annotation"])
		baseline_seg = segmentation_metrics(baseline.get("segmented"), source["gold_segmented"])

		tracker = UsageTracker()
		client = tracked_client(openai_client, tracker)
		item = {
			"sentence_id": int(source["sentence_id"]), "dialect_id": int(source["dialect_id"]),
			"dialect_region": source.get("dialect_region") or "", "text": source["text"],
			"translation_jp": None,
		}
		job = {
			"annotation_schema_id": int(source["annotation_schema_id"]), "model_name": model_name,
			"prompt_version": "annotation-v9", "task": "morph", "translation_evidence": "ignore",
			"morphology_source": "predict", "produce_translation": False, "blind_translation": False,
		}
		agent = TaskAwareAnnotationAgent(nrdb, model_name, client=client, progress=progress, id_model_path=id_model)
		agent_result = agent.annotate(item, job, baseline)
		usage = tracker.summary()
		cost = float(((usage.get("totals") or {}).get("estimated_cost_usd") or 0.0))
		total_cost += cost
		pricing_complete = pricing_complete and bool(usage.get("pricing_complete"))

		agent_metrics = annotation_metrics(agent_result.get("annotation"), source["gold_annotation"])
		agent_seg = segmentation_metrics(agent_result.get("segmented"), source["gold_segmented"])
		row = _result_row(source, baseline, agent_result, usage, baseline_metrics, agent_metrics, baseline_seg, agent_seg)
		results.append(row)
		progress("  baseline ID {:.1f}% -> agent {:.1f}% | agree={} | cost ${:.4f}".format(
			100.0 * baseline_metrics["id_match_rate"], 100.0 * agent_metrics["id_match_rate"],
			"yes" if row["baseline_agent_full_agree"] else "no", cost,
		))

	summary = _paired_summary(results)
	summary.update({
		"format": "nrdb-agent.morph-ceiling-eval.v2",
		"morph_run": contract["run_dir"],
		"train_rows": contract["train_rows"],
		"train_rows_missing_identity": contract["train_rows_missing_identity"],
		"datasets": sorted(selected_dataset_ids),
		"registered_gold_rows": len(registered),
		"eligible_pool_after_train_exclusion": eligible_pool_size,
		"sampled_rows": len(eligible),
		"registered_rows_without_example_id": unmatched_identity,
		"selection_seed": int(seed),
		"morph_model_ids": sorted(morph_model_ids),
		"expected_morph_model": expected_morph_model,
		"agent_model": model_name,
		"estimated_cost_usd": total_cost,
		"pricing_complete": pricing_complete,
	})
	payload = {"summary": summary, "rows": results}
	if output:
		if Path(output).suffix.lower() == ".json":
			write_json(output, payload)
		else:
			write_tsv(output, results)
	return payload
