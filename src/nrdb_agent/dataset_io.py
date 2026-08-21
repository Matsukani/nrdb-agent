import csv
import json
from pathlib import Path


TEXT_FIELDS = ("text", "sentence_trsc2", "trsc2", "sentence")
DIALECT_FIELDS = ("dialect_id", "target_dialect_id")
SEGMENTED_FIELDS = ("segmented", "sentence_trsc2_segmented", "trsc2_seg", "sentence_segmented")
ANNOTATION_FIELDS = ("annotation", "sentence_annotation", "annotation_r")
TRANSLATION_FIELDS = ("translation_jp", "translation", "trsl")
ID_FIELDS = ("example_id", "id", "sentence_id", "utt_id")


def _first(row, fields):
	for field in fields:
		value = row.get(field)
		if value is not None and str(value).strip() != "":
			return str(value).strip()
	return ""


def _positive_int(value, label):
	try:
		value = int(value)
	except (TypeError, ValueError) as error:
		raise ValueError("{} must be a positive integer".format(label)) from error
	if value < 1:
		raise ValueError("{} must be a positive integer".format(label))
	return value


def load_portable_xlsx(path, component=None):
	try:
		from nrdb_morph.job import import_annotation_job
	except ImportError as error:
		raise RuntimeError(
			"Portable XLSX input requires nrdb-morph in the active environment; install the local nrdb-morph package first."
		) from error
	job = import_annotation_job(path, component=component)
	items = []
	for index, source in enumerate(job.get("items", []), start=1):
		item = {
			"row_id": index,
			"sentence_id": index,
			"example_id": source.get("example_id") or str(index),
			"text": source.get("text") or "",
			"dialect_id": source.get("target_dialect_id"),
			"dialect_region": job.get("region"),
			"translation_jp": source.get("translation") or "",
			"existing_segmented": source.get("segmented") or "",
			"existing_annotation": source.get("annotation") or "",
			"source_status": source.get("status") or "",
			"source_validation_error": source.get("validation_error"),
			"source_row": source.get("excel_row"),
			"source_filename": source.get("filename") or "",
			"_original": {
				"example_id": source.get("example_id") or str(index),
				"text": source.get("text") or "",
				"dialect_id": source.get("target_dialect_id"),
				"translation_jp": source.get("translation") or "",
				"segmented": source.get("segmented") or "",
				"annotation": source.get("annotation") or "",
				"status": source.get("status") or "",
				"source_row": source.get("excel_row"),
				"filename": source.get("filename") or "",
			},
		}
		for key in ("entry_id", "meaning_id", "meaning_nb"):
			if key in source:
				item["_original"][key] = source.get(key)
		items.append(item)
	return {
		"format": "nrdb-agent.input.v1",
		"source_type": "xlsx",
		"dataset": job.get("dataset", {}),
		"annotation_schema_id": int(job["annotation_schema_id"]),
		"region": job.get("region") or "",
		"component": (job.get("source") or {}).get("kind"),
		"items": items,
	}


def load_tsv(path, annotation_schema_id, region, default_dialect_id=None):
	annotation_schema_id = _positive_int(annotation_schema_id, "annotation_schema_id")
	region = str(region or "").strip()
	if not region:
		raise ValueError("TSV input requires --region")
	path = Path(path)
	with path.open("r", encoding="utf-8-sig", newline="") as handle:
		reader = csv.DictReader(handle, delimiter="\t")
		if not reader.fieldnames:
			raise ValueError("TSV input has no header row")
		items = []
		for index, row in enumerate(reader, start=1):
			text = _first(row, TEXT_FIELDS)
			translation = _first(row, TRANSLATION_FIELDS)
			if not text and not translation:
				continue
			dialect_value = _first(row, DIALECT_FIELDS) or default_dialect_id
			dialect_id = _positive_int(dialect_value, "dialect_id on TSV row {}".format(index + 1))
			segmented = _first(row, SEGMENTED_FIELDS)
			annotation = _first(row, ANNOTATION_FIELDS)
			items.append({
				"row_id": index, "sentence_id": index,
				"example_id": _first(row, ID_FIELDS) or str(index),
				"text": text, "dialect_id": dialect_id, "dialect_region": region,
				"translation_jp": translation,
				"existing_segmented": segmented, "existing_annotation": annotation,
				"source_status": "annotated" if segmented and annotation else "unannotated",
				"_original": dict(row),
			})
	return {
		"format": "nrdb-agent.input.v1", "source_type": "tsv",
		"dataset": {"name": path.stem, "id": None, "type": "external"},
		"annotation_schema_id": annotation_schema_id, "region": region, "component": None,
		"items": items,
	}


def load_dataset(path, component=None, annotation_schema_id=None, region=None, default_dialect_id=None):
	suffix = Path(path).suffix.lower()
	if suffix in {".xlsx", ".xlsm"}:
		return load_portable_xlsx(path, component=component)
	if suffix in {".tsv", ".txt"}:
		if annotation_schema_id is None:
			raise ValueError("TSV input requires --annotation-schema")
		return load_tsv(path, annotation_schema_id, region, default_dialect_id=default_dialect_id)
	raise ValueError("unsupported dataset input {}; use portable .xlsx or .tsv".format(suffix or path))


def item_matches_needs(item, needs):
	needs = str(needs or "any")
	has_annotation = bool(str(item.get("existing_segmented") or "").strip() and str(item.get("existing_annotation") or "").strip())
	has_translation = bool(str(item.get("translation_jp") or "").strip())
	if needs == "any":
		return True
	if needs == "annotation":
		return not has_annotation
	if needs == "translation":
		return not has_translation
	if needs == "either":
		return not has_annotation or not has_translation
	if needs == "both":
		return not has_annotation and not has_translation
	raise ValueError("invalid needs filter: {}".format(needs))


def output_row(item, result=None, error=None):
	row = dict(item.get("_original") or {})
	result = result or {}
	baseline = result.get("morph_baseline") if isinstance(result.get("morph_baseline"), dict) else {}
	inference = baseline.get("inference") if isinstance(baseline.get("inference"), dict) else {}
	row.update({
		"morph_segmented": baseline.get("segmented", ""),
		"morph_annotation": baseline.get("annotation", ""),
		"morph_source": baseline.get("source", ""),
		"morph_model_id": inference.get("model_id", ""),
		"ai_segmented": result.get("segmented", ""),
		"ai_annotation": result.get("annotation", ""),
		"ai_translation": result.get("translation", ""),
		"ai_decision": result.get("decision", "failed" if error else ""),
		"ai_confidence": result.get("confidence", ""),
		"ai_cost_usd": "{:.6f}".format(float(result.get("estimated_cost_usd") or 0.0)) if result else "",
		"ai_model": result.get("model", ""),
		"ai_error": str(error or ""),
		"ai_evidence_json": json.dumps(result.get("evidence", {}), ensure_ascii=False, separators=(",", ":")) if result else "",
		"ai_forward_morph_policy_json": json.dumps(result.get("forward_morph_policy", {}), ensure_ascii=False, separators=(",", ":")) if result else "",
	})
	return row


def write_tsv(path, rows):
	rows = list(rows)
	fieldnames = []
	for row in rows:
		for key in row:
			if key not in fieldnames:
				fieldnames.append(key)
	with Path(path).open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore", lineterminator="\n")
		writer.writeheader()
		writer.writerows(rows)


def write_json(path, payload):
	Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
