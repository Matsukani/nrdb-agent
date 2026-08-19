import json
from pathlib import Path

from .dataset_io import item_matches_needs, load_dataset, output_row, write_json, write_tsv
from .workflow import execute_item


def process_dataset(nrdb, input_path, task, model_name="gpt-5.6", component=None,
	annotation_schema_id=None, region=None, default_dialect_id=None,
	translation_evidence="ignore", morphology_source="predict", needs="any",
	target_dialect_ids=None, id_model=None, surface_model=None, output=None, limit=None,
	progress=print):
	bundle = load_dataset(
		input_path, component=component, annotation_schema_id=annotation_schema_id,
		region=region, default_dialect_id=default_dialect_id,
	)
	selected = [item for item in bundle["items"] if item_matches_needs(item, needs)]
	if limit is not None:
		selected = selected[:max(0, int(limit))]
	rows = []
	completed = 0
	failed = 0
	cost = 0.0
	for index, source_item in enumerate(selected, start=1):
		progress("[{}/{}] {}".format(index, len(selected), source_item.get("example_id") or source_item.get("row_id")))
		item = dict(source_item)
		if item.get("source_status") == "invalid":
			if morphology_source == "existing":
				error = item.get("source_validation_error") or "input morphology is invalid"
				rows.append(output_row(item, error=error))
				failed += 1
				continue
			# Invalid prior morphology is not authoritative in predict/auto modes.
			# Treat it as absent so a clean analysis can replace it.
			item["existing_segmented"] = ""
			item["existing_annotation"] = ""
		try:
			result = execute_item(
				nrdb, item, task, bundle["annotation_schema_id"], bundle["region"],
				model_name=model_name, translation_evidence=translation_evidence,
				morphology_source=morphology_source, target_dialect_ids=target_dialect_ids,
				id_model=id_model, surface_model=surface_model, progress=progress,
			)
			rows.append(output_row(source_item, result=result))
			cost += float(result.get("estimated_cost_usd") or 0.0)
			completed += 1
		except Exception as error:
			rows.append(output_row(source_item, error=error))
			failed += 1
			progress("  failed: {}".format(error))

	payload = {
		"format": "nrdb-agent.batch-result.v1",
		"source": str(input_path), "source_type": bundle["source_type"],
		"dataset": bundle.get("dataset", {}), "component": bundle.get("component"),
		"task": task, "annotation_schema_id": bundle["annotation_schema_id"], "region": bundle["region"],
		"translation_evidence": translation_evidence, "morphology_source": morphology_source,
		"needs": needs, "model": model_name,
		"counts": {"input": len(bundle["items"]), "selected": len(selected), "completed": completed, "failed": failed},
		"estimated_cost_usd": cost, "rows": rows,
	}
	if output:
		suffix = Path(output).suffix.lower()
		if suffix == ".json":
			write_json(output, payload)
		else:
			write_tsv(output, rows)
	return payload


def export_job_results_tsv(nrdb, job_id, output):
	payload = nrdb.job_results(job_id)
	rows = []
	for value in payload.get("results", []):
		rows.append({
			"sentence_id": value.get("sentence_id"), "example_id": value.get("example_id"),
			"source_text": value.get("source_text"), "human_translation_jp": value.get("translation_jp"),
			"ai_segmented": value.get("ai_segmented"), "ai_annotation": value.get("ai_annotation"),
			"ai_translation": value.get("trsl_ai"), "decision": value.get("decision"),
			"confidence": value.get("confidence"), "gold_segmented": value.get("gold_segmented"),
			"gold_annotation": value.get("gold_annotation"), "gold_translation_jp": value.get("gold_translation_jp"),
			"exact_match": value.get("exact_match"), "evidence_json": json.dumps(value.get("evidence") or value.get("evidence_json") or {}, ensure_ascii=False, separators=(",", ":")),
		})
	write_tsv(output, rows)
	return {"output": str(output), "rows": len(rows), "job_id": int(job_id)}
