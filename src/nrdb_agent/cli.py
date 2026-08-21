import argparse
import json

from .asr_review import review_asr_predictions
from .batch import export_job_results_tsv, process_dataset
from .cli_output import TranslationProgress, WorkflowProgress, add_output_mode_args, output_mode_from_args, silent_translation_line
from .discrepancy import check_discovery, create_discovery, list_discoveries, run_discovery
from .id_analysis import run_id_analysis_job, write_result_tsv
from .metrics import annotation_metrics, job_annotation_metrics, job_segmentation_metrics, segmentation_metrics
from .nrdb import NrdbClient
from .runner import run_job
from .translate import translate_text
from .workflow import run_workflow_job


SEMANTIC_FEEDBACK_CHOICES = ["none", "generated", "existing", "auto"]


def _print_json(value):
	print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _print_translation(value):
	print("source:      {}".format(value.get("source", "")))
	if value.get("segmented"):
		print("segmented:   {}".format(value.get("segmented", "")))
	print("annotation:  {}".format(value.get("annotation", "")))
	print("translation: {}".format(value.get("translation", "")))
	print("confidence:  {} | decision: {}".format(value.get("confidence", ""), value.get("decision", "")))


def _print_discrepancy_summary(label, value, output):
	summary = value.get("summary") or {}
	cost = "${:.4f}".format(float(summary.get("estimated_cost_usd") or 0.0)) if summary.get("pricing_complete") else "cost unknown"
	print("{}: rows={} failed={} {} counts={}".format(label, summary.get("rows", 0), summary.get("failed", 0), cost, json.dumps(summary.get("counts") or {}, ensure_ascii=False)))
	candidates = summary.get("morphemes_to_analyse") or []
	if candidates:
		print("morphemes to analyse: {}".format(", ".join("{} ({})".format(row["morph_id"], row["attributed_errors"]) for row in candidates)))
	print("wrote {}".format(output))


def _print_discrepancy_list(rows):
	if not rows:
		print("no discrepancy artifacts found")
		return
	for row in rows:
		models = ""
		if row.get("translation_model") or row.get("discrepancy_model"):
			models = " | models={}/{}".format(row.get("translation_model") or "-", row.get("discrepancy_model") or "-")
		conditions = " | morph={} constructions={}".format(row.get("morphology") or "-", row.get("constructions") or "-")
		cost = ""
		if row.get("pricing_complete"):
			cost = " | ${:.4f}".format(float(row.get("estimated_cost_usd") or 0.0))
		print("{} | {} | {} | rows={} | schema={} region={} | ids={}{}{} | {}".format(
			row.get("modified_at", "")[:19], row.get("stage"), row.get("status"), row.get("rows", 0),
			row.get("annotation_schema_id") or "-", row.get("region") or "-", ",".join(row.get("target_ids") or []),
			models + conditions, cost, row.get("path"),
		))


def _print_asr_review(value):
	print("ASR REVIEW")
	print("  rows scored:                 {}".format(value.get("rows_scored", 0)))
	print("  top-1 UER:                   {}".format(_pct(value.get("top1_UER"))))
	print("  deterministic baseline UER:  {}".format(_pct(value.get("baseline_UER"))))
	print("  agent-selected UER:          {}".format(_pct(value.get("agent_UER"))))
	print("  oracle UER:                  {}".format(_pct(value.get("oracle_UER"))))
	print("  agent headroom recovered:    {}".format(_pct(value.get("agent_headroom_recovered"))))
	print("  baseline headroom recovered: {}".format(_pct(value.get("baseline_headroom_recovered"))))
	print("  rank-1 retained:             {}".format(value.get("rank1_retained", 0)))
	print("  changed:                     {}".format(value.get("agent_changed", 0)))
	print("  improved / harmful / neutral: {} / {} / {}".format(value.get("improved_rows", 0), value.get("harmful_rows", 0), value.get("neutral_rows", 0)))
	print("  oracle-rank selections:      {}".format(value.get("oracle_rank_selected_rows", 0)))


def _pct(value):
	return "n/a" if value is None else "{:.1f}%".format(100.0 * value)


def _parse_dialect_ids(value):
	if not value:
		return None
	out = []
	for part in str(value).split(","):
		part = part.strip()
		if not part:
			continue
		try:
			item = int(part)
		except ValueError:
			raise argparse.ArgumentTypeError("target dialects must be comma-separated integer IDs")
		if item <= 0:
			raise argparse.ArgumentTypeError("target dialect IDs must be positive")
		if item not in out:
			out.append(item)
	return out or None


def _parse_sentence_range(value):
	text = str(value or "").strip()
	if not text:
		return None
	parts = text.split(":", 1)
	try:
		start = int(parts[0])
		end = int(parts[1]) if len(parts) == 2 and parts[1] else start
	except ValueError as error:
		raise argparse.ArgumentTypeError("sentence scope must be ID or START:END") from error
	if start < 1 or end < start:
		raise argparse.ArgumentTypeError("sentence scope must use positive IDs with END >= START")
	return (start, end)


def _positive_count(value):
	try:
		value = int(value)
	except (TypeError, ValueError) as error:
		raise argparse.ArgumentTypeError("count must be a positive integer") from error
	if value < 1:
		raise argparse.ArgumentTypeError("count must be a positive integer")
	return value


def _id_analysis_notes(values, target_ids, instructions=None):
	target_ids = set(target_ids)
	notes = {}
	if str(instructions or "").strip():
		notes["*"] = str(instructions).strip()
	for raw in values or []:
		text = str(raw or "")
		if "=" not in text:
			raise ValueError("--note must use ID=TEXT")
		target_id, note = text.split("=", 1)
		target_id = target_id.strip()
		note = note.strip()
		if target_id not in target_ids:
			raise ValueError("--note refers to an ID not requested in this job: {}".format(target_id))
		if not note:
			raise ValueError("--note requires non-empty text")
		notes[target_id] = note
	return notes


def _semantic_settings(args):
	mode = str(getattr(args, "semantic_feedback", None) or "none")
	require = bool(getattr(args, "require_semantic_feedback", False))
	legacy = getattr(args, "translation_evidence", None)
	if legacy is not None:
		if mode != "none" or require:
			raise ValueError("--translation-evidence cannot be combined with --semantic-feedback or --require-semantic-feedback")
		if legacy == "required": return "existing", True
		if legacy == "use": return "existing", False
		return "none", False
	return mode, require


def _job_task(job):
	return str(job.get("task") or ("reverse" if job.get("prompt_version") == "reverse-v1" else "morph"))


def _select_jobs(jobs, latest=None):
	ordered = sorted(list(jobs or []), key=lambda job: int(job.get("id") or 0))
	if latest is not None:
		ordered = ordered[-int(latest):]
	return ordered


def _job_scope_text(job):
	parts = []
	if job.get("scope_text_id") not in (None, ""):
		parts.append("text={}".format(job["scope_text_id"]))
	if job.get("scope_sentence_start") not in (None, ""):
		start = job["scope_sentence_start"]
		end = job.get("scope_sentence_end") or start
		parts.append("sentences={}:{}".format(start, end))
	return " ".join(parts) or "all"


def _print_jobs(jobs, short=False):
	if not jobs:
		print("No jobs.")
		return
	for job in jobs:
		job_id = int(job.get("id") or 0)
		status = str(job.get("status") or "")
		dataset_id = job.get("dataset_id") or "?"
		dataset_name = str(job.get("dataset_name") or "")
		task = _job_task(job)
		model = str(job.get("model_name") or "")
		limit = job.get("item_limit") or "?"
		created = str(job.get("created_at") or "")
		flags = []
		if job.get("use_constructions"): flags.append("constructions")
		if job.get("use_licensed_forms"): flags.append("licensed")
		flag_text = (" " + "+".join(flags)) if flags else ""
		if short:
			print("#{:<4} {:<10} ds={} {:<18} {:<15} model={} limit={} scope={}{} {}".format(job_id, status, dataset_id, dataset_name[:18], task, model, limit, _job_scope_text(job), flag_text, created).rstrip())
			continue
		print("#{}  {}  dataset {}{}  task={}  model={}".format(job_id, status, dataset_id, " ({})".format(dataset_name) if dataset_name else "", task, model))
		print("  limit={} seed={} scope={} created={}".format(limit, job.get("selection_seed") or "?", _job_scope_text(job), created or "?"))
		if job.get("morphology_source") or job.get("semantic_feedback") or job.get("needs_filter"):
			print("  morphology={} semantic_feedback={} constructions={} licensed={} needs={}".format(job.get("morphology_source") or "-", job.get("semantic_feedback") or "-", "on" if job.get("use_constructions") else "off", "on" if job.get("use_licensed_forms") else "off", job.get("needs_filter") or "-"))


def _print_results(payload, show_sentences=0):
	job = payload["job"]
	rows = payload.get("results", [])
	reverse = job.get("prompt_version") == "reverse-v1"
	print("job {} | dataset {} ({}) | status {} | model {} | prompt {}".format(job["id"], job["dataset_id"], job.get("dataset_name", ""), job.get("status", ""), job.get("model_name", ""), job.get("prompt_version", "")))
	print("=" * 80)
	if show_sentences:
		shown_rows = rows if show_sentences < 0 else rows[:show_sentences]
		for index, row in enumerate(shown_rows, start=1):
			print("[{}] sentence {}{}".format(index, row["sentence_id"], " / " + str(row["example_id"]) if row.get("example_id") else ""))
			if reverse:
				print("Japanese:    {}".format(row.get("translation_jp") or row.get("gold_translation_jp") or ""))
				print("pred IDs:    {}".format(row.get("ai_annotation") or ""))
				if row.get("ai_segmented"): print("pred Miyako: {}".format(row.get("ai_segmented") or ""))
				print("gold IDs:    {}".format(row.get("gold_annotation") or ""))
				print("gold Miyako: {}".format(row.get("source_text") or ""))
			else:
				print("source:      {}".format(row.get("source_text") or ""))
				print("segmented:   {}".format(row.get("ai_segmented") or ""))
				if row.get("gold_segmented"): print("gold seg:    {}".format(row.get("gold_segmented") or ""))
				print("annotation:  {}".format(row.get("ai_annotation") or ""))
				if row.get("trsl_ai"): print("translation: {}".format(row["trsl_ai"]))
				if row.get("gold_translation_jp"): print("gold trsl:   {}".format(row["gold_translation_jp"]))
				elif row.get("translation_jp"): print("human trsl:  {}".format(row["translation_jp"]))
				if row.get("gold_annotation"): print("gold ann:    {}".format(row.get("gold_annotation") or ""))
			metrics = annotation_metrics(row.get("ai_annotation"), row.get("gold_annotation")) if row.get("gold_annotation") else None
			linguistic_exact = int(metrics["linguistic_exact"]) if metrics is not None else "n/a"
			raw_exact = row.get("exact_match") if row.get("exact_match") is not None else "n/a"
			exact_suffix = ""
			if not reverse and metrics is not None and raw_exact != "n/a" and int(raw_exact) != linguistic_exact:
				exact_suffix = " | raw exact: {}".format(raw_exact)
			print("decision:    {} | confidence: {} | exact: {}{}".format(row.get("decision") or "", row.get("confidence") or "", linguistic_exact, exact_suffix))
			if not reverse and row.get("gold_segmented"):
				seg_metrics = segmentation_metrics(row.get("ai_segmented"), row.get("gold_segmented"))
				print("SEG boundary: P={} R={} F1={} | exact: {} | FP:{} FN:{}".format(_pct(seg_metrics["boundary_precision"]), _pct(seg_metrics["boundary_recall"]), _pct(seg_metrics["boundary_f1"]), int(seg_metrics["exact"]), seg_metrics["false_positive_boundaries"], seg_metrics["false_negative_boundaries"]))
			if metrics is not None:
				print("ID match:    {} ({}/{}) | S:{} I:{} D:{}".format(_pct(metrics["id_match_rate"]), metrics["matches"], max(metrics["gold_ids"], metrics["predicted_ids"]), metrics["substitutions"], metrics["insertions"], metrics["deletions"]))
			print("-" * 80)
		if show_sentences > 0 and len(rows) > show_sentences:
			print("showing {} of {} sentences (use --show-sentences with no number to show all)".format(show_sentences, len(rows)))
			print("-" * 80)
	if not reverse:
		seg_metrics = job_segmentation_metrics(rows)
		if seg_metrics["sentences_scored"]:
			print("SEGMENTATION METRICS")
			print("  sentences scored: {}".format(seg_metrics["sentences_scored"]))
			print("  exact matches:    {} ({})".format(seg_metrics["exact_matches"], _pct(seg_metrics["exact_accuracy"])))
			print("  boundary precision: {} ({}/{} predicted boundaries)".format(_pct(seg_metrics["boundary_precision"]), seg_metrics["correct_boundaries"], seg_metrics["predicted_boundaries"]))
			print("  boundary recall:    {} ({}/{} gold boundaries)".format(_pct(seg_metrics["boundary_recall"]), seg_metrics["correct_boundaries"], seg_metrics["gold_boundaries"]))
			print("  boundary F1:        {}".format(_pct(seg_metrics["boundary_f1"])))
			print("  false positives:    {}".format(seg_metrics["false_positive_boundaries"]))
			print("  false negatives:    {}".format(seg_metrics["false_negative_boundaries"]))
			if seg_metrics["surface_mismatches"]: print("  surface mismatches: {}".format(seg_metrics["surface_mismatches"]))
			print()
	metrics = job_annotation_metrics(rows)
	if metrics["sentences_scored"]:
		print("REVERSE ID METRICS" if reverse else "ID METRICS")
		print("  sentences scored: {}".format(metrics["sentences_scored"]))
		print("  exact matches:    {} ({})".format(metrics["linguistic_exact_matches"], _pct(metrics["linguistic_exact_accuracy"])))
		print("  ID match rate:    {} ({}/{} aligned IDs)".format(_pct(metrics["id_match_rate"]), metrics["matches"], max(metrics["gold_ids"], metrics["predicted_ids"])))
		print("  ID error rate:    {} ({} edits / {} gold IDs)".format(_pct(metrics["id_error_rate"]), metrics["edits"], metrics["gold_ids"]))
		print("  substitutions:    {}".format(metrics["substitutions"]))
		print("  insertions:       {}".format(metrics["insertions"]))
		print("  deletions:        {}".format(metrics["deletions"]))
		if metrics["confusions"]:
			print("CONFUSION MATRIX (gold -> predicted)")
			for entry in metrics["confusions"]:
				print("  {:>4}  {} -> {}".format(entry["count"], entry["gold"], entry["predicted"]))


def _show_with_linguistic_metrics(nrdb, job_id):
	summary = nrdb.summary(job_id)
	results = nrdb.job_results(job_id)
	rows = results.get("results", [])
	reverse = results.get("job", {}).get("prompt_version") == "reverse-v1"
	metrics = job_annotation_metrics(rows)
	seg_metrics = job_segmentation_metrics(rows) if not reverse else {"sentences_scored": 0}
	if metrics["sentences_scored"]:
		summary["summary"]["raw_exact_matches"] = summary["summary"].get("exact_matches")
		summary["summary"]["raw_exact_accuracy"] = summary["summary"].get("exact_accuracy")
		summary["summary"]["exact_matches"] = metrics["linguistic_exact_matches"]
		summary["summary"]["exact_accuracy"] = metrics["linguistic_exact_accuracy"]
		summary["summary"]["id_match_rate"] = metrics["id_match_rate"]
		summary["summary"]["id_error_rate"] = metrics["id_error_rate"]
	if seg_metrics["sentences_scored"]:
		summary["summary"]["segmentation_exact_matches"] = seg_metrics["exact_matches"]
		summary["summary"]["segmentation_exact_accuracy"] = seg_metrics["exact_accuracy"]
		summary["summary"]["segmentation_boundary_precision"] = seg_metrics["boundary_precision"]
		summary["summary"]["segmentation_boundary_recall"] = seg_metrics["boundary_recall"]
		summary["summary"]["segmentation_boundary_f1"] = seg_metrics["boundary_f1"]
	return summary


def _explicit_output_mode(args):
	return any(getattr(args, name, False) for name in ("quiet", "verbose", "silent", "compact"))


def main():
	parser = argparse.ArgumentParser(description="Run constrained NRDB AI morphology and translation workflows")
	parser.add_argument("--agent-url", default=None)
	parser.add_argument("--morph-url", default=None)
	sub = parser.add_subparsers(dest="command", required=True)

	create = sub.add_parser("create", help="Create a scoped NRDB morphology/translation job")
	create.add_argument("--dataset-id", type=int, required=True)
	create.add_argument("--task", choices=["morph", "translate", "morph-translate", "reverse"], default="morph")
	create.add_argument("--semantic-feedback", choices=SEMANTIC_FEEDBACK_CHOICES, default="none", help="Morphology semantic review: none, generated Japanese, existing data translation, or auto")
	create.add_argument("--require-semantic-feedback", action="store_true", help="Fail rows when the selected semantic-feedback source cannot be supplied")
	create.add_argument("--constructions", action="store_true", help="Use curated NRDB constructional evidence during Japanese translation")
	create.add_argument("--licensed", action="store_true", help="Use grammar-licensed generated forms as forward morphology evidence")
	create.add_argument("--morphology-source", choices=["predict", "existing", "auto"], default="predict", help="Predict morphology, freeze existing morphology, or use existing when available")
	create.add_argument("--needs", choices=["any", "annotation", "translation", "either", "both"], default="any", help="Select rows missing annotation, translation, either, both, or no missingness filter")
	create.add_argument("--text-id", type=int, default=None, help="Restrict a text dataset to one internal text_id")
	create.add_argument("--sentence-id", type=_parse_sentence_range, default=None, metavar="ID|START:END", help="Restrict sentence/lxs rows by internal ex_sen_lx sentence ID")
	create.add_argument("--limit", type=int, default=100)
	create.add_argument("--seed", type=int, default=1)
	create.add_argument("--model", default="gpt-5.6")
	create.add_argument("--translation-evidence", choices=["ignore", "use", "required"], default=None, help=argparse.SUPPRESS)
	create.add_argument("--mode", choices=["blind_gold", "unannotated"], default=None, help=argparse.SUPPRESS)
	create.add_argument("--prompt-version", choices=["annotation-v1", "annotation-v2", "annotation-v3", "annotation-v4", "annotation-v5", "annotation-v6", "annotation-v7", "annotation-v8", "annotation-v9", "reverse-v1"], default=None, help=argparse.SUPPRESS)
	create.add_argument("--translate", action="store_true", help=argparse.SUPPRESS)
	create.add_argument("--blind-translation", action="store_true", help=argparse.SUPPRESS)

	list_cmd = sub.add_parser("list", help="List jobs oldest to newest")
	list_cmd.add_argument("--latest", nargs="?", const=1, default=None, type=_positive_count, metavar="N", help="Show only the latest job; optionally show the latest N jobs")
	list_cmd.add_argument("--short", action="store_true", help="Print exactly one line per job")
	list_cmd.add_argument("--json", action="store_true", help="Print selected jobs as JSON")

	run = sub.add_parser("run", help="Run one existing job")
	run.add_argument("job_id", type=int)
	run.add_argument("--max-items", type=int, default=None)
	run.add_argument("--target-dialects", type=_parse_dialect_ids, default=None, metavar="ID1,ID2,...")
	run.add_argument("--id-model", default=None, help="Optional nrdb-morph ID-sequence critic; also reads NRDB_ID_MODEL")
	run.add_argument("--surface-model", default=None, help="Optional nrdb-morph surface critic; also reads NRDB_SURFACE_MODEL")
	run.add_argument("--json", action="store_true", help="Suppress terminal progress and print the final job summary as JSON")
	add_output_mode_args(run)

	process = sub.add_parser("process", help="Process a portable _meta_/_cf_ XLSX or TSV dataset without registering it in NRDB")
	process.add_argument("input")
	process.add_argument("--task", choices=["morph", "translate", "morph-translate", "reverse"], required=True)
	process.add_argument("--component", choices=["sen", "utt", "lxs", "lexeme"], default=None, help="Portable XLSX component; required only when more than one is enabled")
	process.add_argument("--annotation-schema", dest="annotation_schema_id", type=int, default=None, help="Required for TSV; XLSX reads _meta_")
	process.add_argument("--region", default=None, help="Required for TSV; XLSX reads _meta_")
	process.add_argument("--dialect", dest="dialect_id", type=int, default=None, help="Fallback dialect_id for TSV rows without one")
	process.add_argument("--dialects", type=_parse_dialect_ids, default=None, metavar="ID1,ID2,...", help="Ordered target dialects for reverse")
	process.add_argument("--semantic-feedback", choices=SEMANTIC_FEEDBACK_CHOICES, default="none")
	process.add_argument("--require-semantic-feedback", action="store_true")
	process.add_argument("--constructions", action="store_true", help="Use curated NRDB constructional evidence during Japanese translation")
	process.add_argument("--licensed", action="store_true", help="Use grammar-licensed generated forms as forward morphology evidence")
	process.add_argument("--morphology-source", choices=["predict", "existing", "auto"], default="predict")
	process.add_argument("--needs", choices=["any", "annotation", "translation", "either", "both"], default="any")
	process.add_argument("--model", default="gpt-5.6")
	process.add_argument("--id-model", default=None)
	process.add_argument("--surface-model", default=None)
	process.add_argument("--limit", type=int, default=None)
	process.add_argument("--output", default=None, help="Write .tsv (default by extension) or .json")
	process.add_argument("--json", action="store_true", help="Suppress terminal progress and print the complete batch result JSON")
	process.add_argument("--translation-evidence", choices=["ignore", "use", "required"], default=None, help=argparse.SUPPRESS)
	add_output_mode_args(process)

	translate = sub.add_parser("translate", help="Translate arbitrary Miyako or Japanese text without creating a job")
	translate.add_argument("text")
	translate.add_argument("--target", choices=["japanese", "miyako"], required=True)
	translate.add_argument("--annotation-schema", dest="annotation_schema_id", type=int, required=True)
	translate.add_argument("--region", required=True)
	translate.add_argument("--dialects", type=_parse_dialect_ids, default=None, metavar="ID1,ID2,...")
	translate.add_argument("--semantic-feedback", choices=SEMANTIC_FEEDBACK_CHOICES, default=None, help="For Miyako→Japanese, default is generated; for Japanese→Miyako, semantic feedback is not used")
	translate.add_argument("--require-semantic-feedback", action="store_true")
	translate.add_argument("--constructions", action="store_true", help="Use curated NRDB constructional evidence before Japanese translation")
	translate.add_argument("--licensed", action="store_true", help="Use grammar-licensed generated forms as forward morphology evidence")
	translate.add_argument("--existing-translation", default=None, help="Existing Japanese translation used only when semantic feedback is existing/auto; never exposed to output generation")
	translate.add_argument("--surface-model", default=None)
	translate.add_argument("--model", default="gpt-5.6")
	translate.add_argument("--json", action="store_true")
	add_output_mode_args(translate)

	discrepancy_create = sub.add_parser("discrepancy-create", help="Freeze a translated gold cohort containing selected morpheme IDs")
	discrepancy_create.add_argument("ids", nargs="+", help="Exact atomic annotation IDs; quote shell metacharacters such as 'ppt>2'")
	discrepancy_create.add_argument("--dataset-id", dest="dataset_ids", action="extend", nargs="+", type=int, default=None, help="Optional dataset restriction; accept one or more IDs and may be repeated")
	discrepancy_create.add_argument("--annotation-schema", dest="annotation_schema_id", type=int, required=True)
	discrepancy_create.add_argument("--region", required=True)
	discrepancy_create.add_argument("--limit", type=_positive_count, default=100)
	discrepancy_create.add_argument("--seed", type=int, default=1)
	discrepancy_create.add_argument("--min-morphemes", type=_positive_count, default=1)
	discrepancy_create.add_argument("--require-all", action="store_true", help="Require every requested ID; default matches any requested ID")
	discrepancy_create.add_argument("--gold-morph", action="store_true", help="Keep only rows with gold segmentation and annotation, for frozen-morphology translation")
	discrepancy_create.add_argument("--constructions", action="store_true", help="Declare that the baseline will use the currently enabled grammatical constructions")
	discrepancy_create.add_argument("--output", required=True, help="Write the frozen discovery cohort JSON")

	discrepancy_run = sub.add_parser("discrepancy-run", help="Generate blind translations and judge them against gold")
	discrepancy_run.add_argument("discovery")
	discrepancy_run.add_argument("--translation-model", default="gpt-5.6-luna", help="Model that generates blind translations")
	discrepancy_run.add_argument("--discrepancy-model", default="gpt-5.6-terra", help="Model that judges semantic discrepancies")
	discrepancy_run.add_argument("--gold-morph", action="store_true", help="Translate from the frozen gold segmentation and annotation instead of predicted morphology")
	discrepancy_run.add_argument("--constructions", action="store_true", help="Use currently enabled grammatical constructions; must match cohort creation")
	discrepancy_run.add_argument("--output", required=True, help="Write baseline translations and judgments JSON")

	discrepancy_check = sub.add_parser("discrepancy-check", help="Rerun the frozen baseline cohort with grammatical constructions")
	discrepancy_check.add_argument("baseline")
	discrepancy_check.add_argument("--translation-model", default=None, help="Must match the baseline generation model; defaults to it")
	discrepancy_check.add_argument("--discrepancy-model", default="gpt-5.6-terra", help="Model that judges repair versus regression")
	discrepancy_check.add_argument("--output", required=True, help="Write construction-assisted translations and repair judgments JSON")

	discrepancy_list = sub.add_parser("discrepancy-list", help="List local discrepancy discovery, baseline, and repair artifacts")
	discrepancy_list.add_argument("--directory", default=".", help="Directory containing discrepancy JSON artifacts")
	discrepancy_list.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
	discrepancy_list.add_argument("--latest", nargs="?", const=1, type=_positive_count, default=None, metavar="N", help="Show only the latest artifact; optionally latest N")
	discrepancy_list.add_argument("--json", action="store_true")

	asr_review = sub.add_parser("asr-review", help="Blindly rerank ASR n-best hypotheses with NRDB linguistic evidence")
	asr_review.add_argument("predictions")
	asr_review.add_argument("--out-dir", required=True)
	asr_review.add_argument("--annotation-schema", dest="annotation_schema_id", type=int, required=True)
	asr_review.add_argument("--region", required=True)
	asr_review.add_argument("--dialect", dest="dialect_id", type=int, required=True)
	asr_review.add_argument("--model", default="gpt-5.6")
	asr_review.add_argument("--id-model", default=None)
	asr_review.add_argument("--surface-model", default=None)
	asr_review.add_argument("--phrase-boundary-model", default=None)
	asr_review.add_argument("--limit", type=int, default=None)
	asr_review.add_argument("--max-candidates", type=int, default=None)
	asr_review.add_argument("--no-llm", action="store_true")
	asr_review.add_argument("--json", action="store_true")

	show = sub.add_parser("show", help="Show audit summary for one job")
	show.add_argument("job_id", type=int)

	results = sub.add_parser("results", help="Show aggregate metrics or export job results")
	results.add_argument("job_id", type=int)
	results.add_argument("--show-sentences", nargs="?", const=-1, default=0, type=int, metavar="N")
	results.add_argument("--output", default=None, help="Export results as TSV")

	id_analysis = sub.add_parser("id-analysis", help="Create, run, and inspect corpus-based grammatical-ID analyses")
	id_sub = id_analysis.add_subparsers(dest="id_analysis_command", required=True)
	id_create = id_sub.add_parser("create", help="Create an ID-analysis job for one or more exact annotation IDs")
	id_create.add_argument("ids", nargs="+", help="Exact atomic annotation IDs; quote IDs containing shell metacharacters such as 'ppt>2'")
	id_create.add_argument("--annotation-schema", dest="annotation_schema_id", type=int, required=True)
	id_create.add_argument("--region", default=None)
	id_create.add_argument("--dialect", dest="dialect_id", type=int, default=None)
	id_create.add_argument("--source", dest="source_kinds", action="append", choices=["txt", "sen", "lxs"], default=None, help="Restrict evidence sources; repeat as needed")
	id_create.add_argument("--dataset-id", dest="dataset_ids", action="append", type=int, default=None, help="Restrict evidence to a dataset; repeat as needed")
	id_create.add_argument("--instructions", default=None, help="Expert guidance applying to every requested ID")
	id_create.add_argument("--note", action="append", default=None, metavar="ID=TEXT", help="Expert guidance for one requested ID; repeat as needed")
	id_create.add_argument("--minimum-ngram-count", type=int, default=2)
	id_create.add_argument("--example-limit", type=int, default=30)
	id_create.add_argument("--model", default="gpt-5.6")
	id_run = id_sub.add_parser("run", help="Run one queued ID-analysis job")
	id_run.add_argument("job_id", type=int)
	id_run.add_argument("--output", default=None, help="Write candidate records to a TSV file")
	id_run.add_argument("--json", action="store_true", help="Print the complete audited JSON result")
	id_show = id_sub.add_parser("show", help="Show one stored ID-analysis job/result")
	id_show.add_argument("job_id", type=int)
	id_show.add_argument("--output", default=None, help="Write the stored candidate TSV to a file")
	id_show.add_argument("--json", action="store_true")
	id_list = id_sub.add_parser("list", help="List recent ID-analysis jobs")
	id_list.add_argument("--json", action="store_true")

	args = parser.parse_args()
	nrdb = NrdbClient(args.agent_url, args.morph_url)
	if args.command == "create":
		legacy = args.mode is not None or args.prompt_version is not None or args.translate or args.blind_translation
		if legacy:
			mode = args.mode or "blind_gold"
			prompt = args.prompt_version or "annotation-v8"
			_print_json(nrdb.create_job(args.dataset_id, mode, args.limit, args.model, prompt, args.seed, args.translate, args.blind_translation))
		else:
			try:
				semantic_feedback, require_semantic_feedback = _semantic_settings(args)
			except ValueError as error:
				parser.error(str(error))
			if args.task == "reverse" and (args.constructions or args.licensed):
				parser.error("--constructions and --licensed apply only to Miyako -> Japanese/forward morphology workflows")
			start = end = None
			if args.sentence_id: start, end = args.sentence_id
			_print_json(nrdb.create_workflow_job(
				args.dataset_id, args.task, args.limit, args.model, args.seed,
				semantic_feedback=semantic_feedback, require_semantic_feedback=require_semantic_feedback,
				use_constructions=args.constructions, use_licensed_forms=args.licensed,
				morphology_source=args.morphology_source, needs_filter=args.needs,
				scope_text_id=args.text_id, scope_sentence_start=start, scope_sentence_end=end,
			))
	elif args.command == "list":
		jobs = _select_jobs(nrdb.jobs(), latest=args.latest)
		_print_json(jobs) if args.json else _print_jobs(jobs, short=args.short)
	elif args.command == "run":
		if args.json and _explicit_output_mode(args): parser.error("--json cannot be combined with --quiet, --verbose, --silent, or --compact")
		display = (lambda _message: None) if args.json else WorkflowProgress(output_mode_from_args(args))
		try:
			try:
				value = run_workflow_job(nrdb, args.job_id, max_items=args.max_items, target_dialects=args.target_dialects, id_model=args.id_model, surface_model=args.surface_model, progress=display)
			except RuntimeError as error:
				if "Legacy job" not in str(error): raise
				value = run_job(nrdb, args.job_id, max_items=args.max_items, target_dialects=args.target_dialects, surface_model=args.surface_model, id_model=args.id_model, progress=display)
		finally:
			if hasattr(display, "stop"): display.stop()
		if args.json: _print_json(value)
	elif args.command == "process":
		if args.json and _explicit_output_mode(args): parser.error("--json cannot be combined with --quiet, --verbose, --silent, or --compact")
		try:
			semantic_feedback, require_semantic_feedback = _semantic_settings(args)
		except ValueError as error:
			parser.error(str(error))
		if args.task == "reverse" and (args.constructions or args.licensed): parser.error("--constructions and --licensed apply only to Miyako -> Japanese/forward morphology workflows")
		display = (lambda _message: None) if args.json else WorkflowProgress(output_mode_from_args(args))
		try:
			value = process_dataset(
				nrdb, args.input, args.task, model_name=args.model, component=args.component,
				annotation_schema_id=args.annotation_schema_id, region=args.region, default_dialect_id=args.dialect_id,
				semantic_feedback=semantic_feedback, require_semantic_feedback=require_semantic_feedback,
				use_constructions=args.constructions, use_licensed_forms=args.licensed,
				morphology_source=args.morphology_source, needs=args.needs,
				target_dialect_ids=args.dialects, id_model=args.id_model, surface_model=args.surface_model,
				output=args.output, limit=args.limit, progress=display,
			)
		finally:
			if hasattr(display, "stop"): display.stop()
		if args.json: _print_json(value)
	elif args.command == "translate":
		if args.target == "miyako" and not args.dialects: parser.error("Japanese -> Miyako translation requires --dialects ID1,ID2,...")
		if args.json and _explicit_output_mode(args): parser.error("--json cannot be combined with --quiet, --verbose, --silent, or --compact")
		semantic_feedback = args.semantic_feedback
		if semantic_feedback is None: semantic_feedback = "generated" if args.target == "japanese" else "none"
		if args.target == "miyako" and (semantic_feedback != "none" or args.require_semantic_feedback or args.existing_translation or args.constructions or args.licensed):
			parser.error("semantic feedback, constructions and licensed forms are only applicable to direct Miyako -> Japanese translation")
		kwargs = dict(
			dialect_ids=args.dialects, model_name=args.model, surface_model=args.surface_model,
			semantic_feedback=semantic_feedback, require_semantic_feedback=args.require_semantic_feedback,
			use_constructions=args.constructions, use_licensed_forms=args.licensed,
			existing_translation=args.existing_translation,
		)
		if args.json:
			value = translate_text(nrdb, args.text, args.target, args.annotation_schema_id, args.region, progress=lambda _message: None, **kwargs)
			_print_json(value)
		else:
			mode = output_mode_from_args(args)
			display = TranslationProgress(mode)
			display.start()
			try:
				value = translate_text(nrdb, args.text, args.target, args.annotation_schema_id, args.region, progress=display, **kwargs)
			finally:
				display.stop()
			if mode in {"silent", "compact"}: print(silent_translation_line(value))
			else: _print_translation(value)
	elif args.command == "discrepancy-create":
		value = create_discovery(
			nrdb, args.ids, args.dataset_ids, args.annotation_schema_id, args.region,
			limit=args.limit, seed=args.seed, min_morphemes=args.min_morphemes,
			require_all=args.require_all, require_gold_morph=args.gold_morph,
			use_constructions=args.constructions, output=args.output,
		)
		print("created {} frozen assignment(s), up to {} per ID: {}".format(len(value["rows"]), value["selection"]["limit_per_id"], args.output))
		print("per ID: {}".format(json.dumps(value["selection"]["sampled_rows_by_id"], ensure_ascii=False)))
	elif args.command == "discrepancy-run":
		value = run_discovery(
			nrdb, args.discovery, args.output, translation_model=args.translation_model,
			discrepancy_model=args.discrepancy_model, use_gold_morph=args.gold_morph,
			use_constructions=args.constructions,
		)
		_print_discrepancy_summary("baseline complete", value, args.output)
	elif args.command == "discrepancy-check":
		value = check_discovery(
			nrdb, args.baseline, args.output, translation_model=args.translation_model,
			discrepancy_model=args.discrepancy_model,
		)
		_print_discrepancy_summary("repair check complete", value, args.output)
	elif args.command == "discrepancy-list":
		value = list_discoveries(args.directory, recursive=args.recursive, latest=args.latest)
		_print_json(value) if args.json else _print_discrepancy_list(value)
	elif args.command == "asr-review":
		value = review_asr_predictions(nrdb, args.predictions, args.out_dir, args.annotation_schema_id, args.region, args.dialect_id, model_name=args.model, id_model_path=args.id_model, surface_model_path=args.surface_model, phrase_boundary_model_path=args.phrase_boundary_model, limit=args.limit, max_candidates=args.max_candidates, use_llm=not args.no_llm)
		_print_json(value) if args.json else _print_asr_review(value)
	elif args.command == "show":
		_print_json(_show_with_linguistic_metrics(nrdb, args.job_id))
	elif args.command == "results":
		if args.output:
			_print_json(export_job_results_tsv(nrdb, args.job_id, args.output))
		else:
			if args.show_sentences < -1: parser.error("--show-sentences must be a non-negative number, or omitted to show all")
			_print_results(nrdb.job_results(args.job_id), show_sentences=args.show_sentences)
	elif args.command == "id-analysis":
		if args.id_analysis_command == "create":
			target_ids = list(dict.fromkeys(str(value).strip() for value in args.ids if str(value).strip()))
			try:
				notes = _id_analysis_notes(args.note, target_ids, args.instructions)
			except ValueError as error:
				parser.error(str(error))
			_print_json(nrdb.create_id_analysis_job(
				args.annotation_schema_id, target_ids, args.model, region=args.region,
				dialect_id=args.dialect_id, research_notes=notes, source_kinds=args.source_kinds,
				dataset_ids=args.dataset_ids, minimum_ngram_count=args.minimum_ngram_count,
				example_limit=args.example_limit,
			))
		elif args.id_analysis_command == "run":
			progress = (lambda _message: None) if args.json else print
			bundle = run_id_analysis_job(nrdb, args.job_id, progress=progress)
			if args.output:
				write_result_tsv(args.output, bundle["tsv"])
			if args.json:
				_print_json(bundle["result"])
			elif args.output:
				candidate_count = sum(len(value.get("candidates", [])) for value in bundle["result"].get("analyses", []))
				print("wrote {} candidate row(s) to {}".format(candidate_count, args.output))
			else:
				print(bundle["tsv"], end="")
		elif args.id_analysis_command == "show":
			job = nrdb.id_analysis_job(args.job_id)["job"]
			tsv = str(job.get("result_tsv") or "")
			if args.output:
				if not tsv: parser.error("ID-analysis job has no stored TSV result")
				write_result_tsv(args.output, tsv)
			if args.json:
				_print_json(job)
			elif args.output:
				print("wrote stored candidate TSV to {}".format(args.output))
			elif tsv:
				print(tsv, end="")
			else:
				_print_json(job)
		elif args.id_analysis_command == "list":
			jobs = nrdb.id_analysis_jobs()
			if args.json:
				_print_json(jobs)
			else:
				for job in jobs:
					print("#{} {} schema={} ids={} model={}".format(job.get("id"), job.get("status"), job.get("annotation_schema_id"), ",".join(job.get("target_ids", [])), job.get("model_name")))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
