import json
from pathlib import Path

from .dataset_io import item_matches_needs, load_dataset, output_row, write_json, write_tsv
from .execution import ExecutionRequest, execute_request
from .policy import forward_morph_policy


def _pricing_complete(result):
	usage = result.get("api_usage") if isinstance(result, dict) else None
	return bool(isinstance(usage, dict) and usage.get("pricing_complete"))


def process_dataset(nrdb, input_path, task, model_name="gpt-5.6", component=None,
	annotation_schema_id=None, region=None, default_dialect_id=None,
	semantic_feedback="none", require_semantic_feedback=False, use_constructions=False,
	use_licensed_forms=False, morphology_source="predict", needs="any",
	target_dialect_ids=None, id_model=None, surface_model=None, morph_review="agent",
	resegmentation=False, max_segmentation_candidates=4, output=None, limit=None,
	progress=print):
	bundle = load_dataset(
		input_path, component=component, annotation_schema_id=annotation_schema_id,
		region=region, default_dialect_id=default_dialect_id,
	)
	selected = [item for item in bundle["items"] if item_matches_needs(item, needs)]
	policy = forward_morph_policy(
		review=morph_review, resegmentation=resegmentation, id_model=id_model, surface_model=surface_model,
		max_segmentation_candidates=max_segmentation_candidates, morphology_source=morphology_source, task=task,
	)
	if limit is not None:
		selected = selected[:max(0, int(limit))]
	rows = []
	completed = 0
	failed = 0
	cost = 0.0
	pricing_complete = True
	for index, source_item in enumerate(selected, start=1):
		label = str(source_item.get("example_id") or source_item.get("row_id"))
		if hasattr(progress, "item_start"):
			progress.item_start(index, len(selected), label)
		else:
			progress("[{}/{}] {}".format(index, len(selected), label))
		item = dict(source_item)
		if item.get("source_status") == "invalid":
			if morphology_source == "existing":
				error = item.get("source_validation_error") or "input morphology is invalid"
				rows.append(output_row(item, error=error))
				failed += 1
				if hasattr(progress, "item_error"):
					progress.item_error(index, len(selected), error, label)
				continue
			item["existing_segmented"] = ""
			item["existing_annotation"] = ""
		try:
			request = ExecutionRequest(
				item=item, task=task, annotation_schema_id=bundle["annotation_schema_id"], region=bundle["region"],
				model_name=model_name, semantic_feedback=semantic_feedback,
				require_semantic_feedback=require_semantic_feedback,
				use_constructions=use_constructions, use_licensed_forms=use_licensed_forms,
				morphology_source=morphology_source,
				target_dialect_ids=tuple(int(value) for value in target_dialect_ids) if target_dialect_ids else None,
				morph_policy=policy,
			)
			result = execute_request(nrdb, request, progress=progress)
			rows.append(output_row(source_item, result=result))
			cost += float(result.get("estimated_cost_usd") or 0.0)
			pricing_complete = pricing_complete and _pricing_complete(result)
			completed += 1
			if hasattr(progress, "item_result"):
				progress.item_result(index, len(selected), task, result, label)
		except Exception as error:
			rows.append(output_row(source_item, error=error))
			failed += 1
			if hasattr(progress, "item_error"):
				progress.item_error(index, len(selected), error, label)
			else:
				progress("  failed: {}".format(error))

	payload = {
		"format": "nrdb-agent.batch-result.v2",
		"source": str(input_path), "source_type": bundle["source_type"],
		"dataset": bundle.get("dataset", {}), "component": bundle.get("component"),
		"task": task, "annotation_schema_id": bundle["annotation_schema_id"], "region": bundle["region"],
		"semantic_feedback": semantic_feedback, "require_semantic_feedback": bool(require_semantic_feedback),
		"use_constructions": bool(use_constructions), "use_licensed_forms": bool(use_licensed_forms),
		"morphology_source": morphology_source, "needs": needs, "model": model_name,
		"morph_review": morph_review, "resegmentation": bool(resegmentation),
		"counts": {"input": len(bundle["items"]), "selected": len(selected), "completed": completed, "failed": failed},
		"estimated_cost_usd": cost, "pricing_complete": pricing_complete, "rows": rows,
	}
	if output:
		suffix = Path(output).suffix.lower()
		if suffix == ".json": write_json(output, payload)
		else: write_tsv(output, rows)
	if hasattr(progress, "job_summary"):
		progress.job_summary(completed, len(selected), cost, failed=failed, pricing_complete=pricing_complete)
	return payload


def export_job_results_tsv(nrdb, job_id, output):
	payload = nrdb.job_results(job_id)
	rows = []
	for value in payload.get("results", []):
		evidence = value.get("evidence") or value.get("evidence_json") or {}
		usage = evidence.get("api_usage") or {}
		cost = (usage.get("totals") or {}).get("estimated_cost_usd")
		baseline = evidence.get("morph_baseline") if isinstance(evidence.get("morph_baseline"), dict) else {}
		inference = baseline.get("inference") if isinstance(baseline.get("inference"), dict) else {}
		rows.append({
			"sentence_id": value.get("sentence_id"), "example_id": value.get("example_id"),
			"source_text": value.get("source_text"), "human_translation_jp": value.get("translation_jp"),
			"morph_segmented": baseline.get("segmented", ""), "morph_annotation": baseline.get("annotation", ""),
			"morph_source": baseline.get("source", ""), "morph_model_id": inference.get("model_id", ""),
			"morph_model_label": inference.get("model_label", ""),
			"morph_inference_json": json.dumps(inference, ensure_ascii=False, separators=(",", ":")) if inference else "",
			"ai_segmented": value.get("ai_segmented"), "ai_annotation": value.get("ai_annotation"),
			"ai_translation": value.get("trsl_ai"), "ai_cost_usd": cost if cost is not None else "",
			"decision": value.get("decision"), "confidence": value.get("confidence"),
			"gold_segmented": value.get("gold_segmented"), "gold_annotation": value.get("gold_annotation"),
			"gold_translation_jp": value.get("gold_translation_jp"), "exact_match": value.get("exact_match"),
			"evidence_json": json.dumps(evidence, ensure_ascii=False, separators=(",", ":")),
		})
	write_tsv(output, rows)
	return {"output": str(output), "rows": len(rows), "job_id": int(job_id)}
