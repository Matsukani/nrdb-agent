import argparse
import json

from .asr_review import review_asr_predictions
from .metrics import annotation_metrics, job_annotation_metrics, job_segmentation_metrics, segmentation_metrics
from .nrdb import NrdbClient
from .runner import run_job
from .translate import translate_text


def _print_json(value):
	print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _print_translation(value):
	print("source:      {}".format(value.get("source", "")))
	if value.get("segmented"):
		print("segmented:   {}".format(value.get("segmented", "")))
	print("annotation:  {}".format(value.get("annotation", "")))
	print("translation: {}".format(value.get("translation", "")))
	print("confidence:  {} | decision: {}".format(value.get("confidence", ""), value.get("decision", "")))


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
	print("  improved / harmful / neutral: {} / {} / {}".format(
		value.get("improved_rows", 0), value.get("harmful_rows", 0), value.get("neutral_rows", 0),
	))
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
				if row.get("ai_segmented"):
					print("pred Miyako: {}".format(row.get("ai_segmented") or ""))
				print("gold IDs:    {}".format(row.get("gold_annotation") or ""))
				print("gold Miyako: {}".format(row.get("source_text") or ""))
			else:
				print("source:      {}".format(row.get("source_text") or ""))
				print("segmented:   {}".format(row.get("ai_segmented") or ""))
				if row.get("gold_segmented"):
					print("gold seg:    {}".format(row.get("gold_segmented") or ""))
				print("annotation:  {}".format(row.get("ai_annotation") or ""))
				if row.get("trsl_ai"):
					print("translation: {}".format(row["trsl_ai"]))
				if row.get("gold_translation_jp"):
					print("gold trsl:   {}".format(row["gold_translation_jp"]))
				elif row.get("translation_jp"):
					print("human trsl:  {}".format(row["translation_jp"]))
				if row.get("gold_annotation"):
					print("gold ann:    {}".format(row.get("gold_annotation") or ""))
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
			if seg_metrics["surface_mismatches"]:
				print("  surface mismatches: {}".format(seg_metrics["surface_mismatches"]))
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


def main():
	parser = argparse.ArgumentParser(description="Run constrained NRDB AI annotation jobs")
	parser.add_argument("--agent-url", default=None)
	parser.add_argument("--morph-url", default=None)
	sub = parser.add_subparsers(dest="command", required=True)

	create = sub.add_parser("create", help="Create an explicit annotation job")
	create.add_argument("--dataset-id", type=int, required=True)
	create.add_argument("--mode", choices=["blind_gold", "unannotated"], default="blind_gold")
	create.add_argument("--limit", type=int, default=100)
	create.add_argument("--seed", type=int, default=1)
	create.add_argument("--model", default="gpt-5.6")
	create.add_argument("--prompt-version", choices=["annotation-v1", "annotation-v2", "annotation-v3", "annotation-v4", "annotation-v5", "annotation-v6", "annotation-v7", "annotation-v8", "reverse-v1"], default="annotation-v8")
	create.add_argument("--translate", action="store_true", help="Also generate a Japanese translation and store it as trsl_ai")
	create.add_argument("--blind-translation", action="store_true", help="Generate trsl_ai without exposing translation_jp to the agent; implies --translate")

	sub.add_parser("list", help="List recent jobs")

	run = sub.add_parser("run", help="Run one existing job")
	run.add_argument("job_id", type=int)
	run.add_argument("--max-items", type=int, default=None, help="Process only this many items; useful for a smoke test")
	run.add_argument("--target-dialects", type=_parse_dialect_ids, default=None, metavar="ID1,ID2,...", help="For reverse-v1, also realize Miyako surface forms using this ordered dialect priority")
	run.add_argument("--surface-model", default=None, help="Optional nrdb-morph surface_model.json for reverse surface criticism")

	translate = sub.add_parser("translate", help="Translate arbitrary Miyako or Japanese text without creating a job")
	translate.add_argument("text")
	translate.add_argument("--target", choices=["japanese", "miyako"], required=True)
	translate.add_argument("--annotation-schema", dest="annotation_schema_id", type=int, required=True)
	translate.add_argument("--region", required=True)
	translate.add_argument("--dialects", type=_parse_dialect_ids, default=None, metavar="ID1,ID2,...", help="Ordered target dialects; required for Japanese -> Miyako")
	translate.add_argument("--surface-model", default=None, help="Optional nrdb-morph surface_model.json; also reads NRDB_SURFACE_MODEL")
	translate.add_argument("--model", default="gpt-5.6", help="LLM model name; does not select the nrdb-morph model")
	translate.add_argument("--json", action="store_true", help="Print complete translation result as JSON")

	asr_review = sub.add_parser("asr-review", help="Blindly rerank ASR n-best hypotheses with NRDB linguistic evidence")
	asr_review.add_argument("predictions", help="predictions.tsv from nrdb-asr eval-manifest --nbest")
	asr_review.add_argument("--out-dir", required=True)
	asr_review.add_argument("--annotation-schema", dest="annotation_schema_id", type=int, required=True)
	asr_review.add_argument("--region", required=True)
	asr_review.add_argument("--dialect", dest="dialect_id", type=int, required=True)
	asr_review.add_argument("--model", default="gpt-5.6")
	asr_review.add_argument("--id-model", default=None, help="Optional nrdb-morph ID-sequence model; also reads NRDB_ID_MODEL")
	asr_review.add_argument("--surface-model", default=None, help="Optional nrdb-morph surface model; also reads NRDB_SURFACE_MODEL")
	asr_review.add_argument("--phrase-boundary-model", default=None, help="Optional nrdb-asr/nrdb-morph phrase-boundary model manifest")
	asr_review.add_argument("--limit", type=int, default=None)
	asr_review.add_argument("--max-candidates", type=int, default=None, help="Inspect only the first N hypotheses from each n-best list")
	asr_review.add_argument("--no-llm", action="store_true", help="Run only the deterministic linguistic baseline")
	asr_review.add_argument("--json", action="store_true", help="Print complete summary JSON")

	show = sub.add_parser("show", help="Show audit summary for one job")
	show.add_argument("job_id", type=int)

	results = sub.add_parser("results", help="Show aggregate metrics and optionally sentence-level results for one job")
	results.add_argument("job_id", type=int)
	results.add_argument("--show-sentences", nargs="?", const=-1, default=0, type=int, metavar="N", help="Show N sentence results; omit N to show all")

	args = parser.parse_args()
	nrdb = NrdbClient(args.agent_url, args.morph_url)
	if args.command == "create":
		if args.prompt_version == "reverse-v1" and args.mode != "blind_gold":
			parser.error("reverse-v1 currently requires --mode blind_gold for hidden-gold evaluation")
		if args.prompt_version == "reverse-v1" and (args.translate or args.blind_translation):
			parser.error("reverse-v1 uses Japanese as input; do not use --translate")
		_print_json(nrdb.create_job(args.dataset_id, args.mode, args.limit, args.model, args.prompt_version, args.seed, args.translate, args.blind_translation))
	elif args.command == "list":
		_print_json(nrdb.jobs())
	elif args.command == "run":
		_print_json(run_job(nrdb, args.job_id, max_items=args.max_items, target_dialects=args.target_dialects, surface_model=args.surface_model))
	elif args.command == "translate":
		if args.target == "miyako" and not args.dialects:
			parser.error("Japanese -> Miyako translation requires --dialects ID1,ID2,...")
		value = translate_text(
			nrdb, args.text, args.target, args.annotation_schema_id, args.region,
			dialect_ids=args.dialects, model_name=args.model, surface_model=args.surface_model,
		)
		_print_json(value) if args.json else _print_translation(value)
	elif args.command == "asr-review":
		value = review_asr_predictions(
			nrdb, args.predictions, args.out_dir, args.annotation_schema_id, args.region, args.dialect_id,
			model_name=args.model, id_model_path=args.id_model, surface_model_path=args.surface_model,
			phrase_boundary_model_path=args.phrase_boundary_model, limit=args.limit,
			max_candidates=args.max_candidates, use_llm=not args.no_llm,
		)
		_print_json(value) if args.json else _print_asr_review(value)
	elif args.command == "show":
		_print_json(_show_with_linguistic_metrics(nrdb, args.job_id))
	elif args.command == "results":
		if args.show_sentences < -1:
			parser.error("--show-sentences must be a non-negative number, or omitted to show all")
		_print_results(nrdb.job_results(args.job_id), show_sentences=args.show_sentences)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
