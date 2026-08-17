import argparse
import json

from .metrics import annotation_metrics, job_annotation_metrics
from .nrdb import NrdbClient
from .runner import run_job


def _print_json(value):
	print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _pct(value):
	return "n/a" if value is None else "{:.1f}%".format(100.0 * value)


def _print_results(payload):
	job = payload["job"]
	rows = payload.get("results", [])
	print("job {} | dataset {} ({}) | status {} | model {} | prompt {}".format(
		job["id"], job["dataset_id"], job.get("dataset_name", ""), job.get("status", ""),
		job.get("model_name", ""), job.get("prompt_version", "")
	))
	print("=" * 80)
	for index, row in enumerate(rows, start=1):
		print("[{}] sentence {}{}".format(index, row["sentence_id"], " / " + str(row["example_id"]) if row.get("example_id") else ""))
		print("source:      {}".format(row.get("source_text") or ""))
		print("segmented:   {}".format(row.get("ai_segmented") or ""))
		print("annotation:  {}".format(row.get("ai_annotation") or ""))
		if row.get("trsl_ai"):
			print("translation: {}".format(row["trsl_ai"]))
		if row.get("gold_translation_jp"):
			print("gold trsl:   {}".format(row["gold_translation_jp"]))
		elif row.get("translation_jp"):
			print("human trsl:  {}".format(row["translation_jp"]))
		if row.get("gold_annotation"):
			print("gold ann:    {}".format(row["gold_annotation"]))
		print("decision:    {} | confidence: {} | exact: {}".format(
			row.get("decision") or "", row.get("confidence") or "",
			row.get("exact_match") if row.get("exact_match") is not None else "n/a"
		))
		if row.get("gold_annotation"):
			metrics = annotation_metrics(row.get("ai_annotation"), row.get("gold_annotation"))
			print("ID match:    {} ({}/{}) | S:{} I:{} D:{}".format(
				_pct(metrics["id_match_rate"]), metrics["matches"], max(metrics["gold_ids"], metrics["predicted_ids"]),
				metrics["substitutions"], metrics["insertions"], metrics["deletions"],
			))
		print("-" * 80)

	metrics = job_annotation_metrics(rows)
	if metrics["sentences_scored"]:
		print("ID METRICS")
		print("  sentences scored: {}".format(metrics["sentences_scored"]))
		print("  ID match rate:    {} ({}/{} aligned IDs)".format(
			_pct(metrics["id_match_rate"]), metrics["matches"], max(metrics["gold_ids"], metrics["predicted_ids"])
		))
		print("  ID error rate:    {} ({} edits / {} gold IDs)".format(
			_pct(metrics["id_error_rate"]), metrics["edits"], metrics["gold_ids"]
		))
		print("  substitutions:    {}".format(metrics["substitutions"]))
		print("  insertions:       {}".format(metrics["insertions"]))
		print("  deletions:        {}".format(metrics["deletions"]))
		if metrics["confusions"]:
			print("CONFUSION MATRIX (gold -> predicted)")
			for entry in metrics["confusions"]:
				print("  {:>4}  {} -> {}".format(entry["count"], entry["gold"], entry["predicted"]))


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
	create.add_argument("--prompt-version", choices=["annotation-v1", "annotation-v2", "annotation-v3", "annotation-v4", "annotation-v5", "annotation-v6"], default="annotation-v6")
	create.add_argument("--translate", action="store_true", help="Also generate a Japanese translation and store it as trsl_ai")
	create.add_argument("--blind-translation", action="store_true", help="Generate trsl_ai without exposing translation_jp to the agent; implies --translate")

	sub.add_parser("list", help="List recent jobs")

	run = sub.add_parser("run", help="Run one existing job")
	run.add_argument("job_id", type=int)
	run.add_argument("--max-items", type=int, default=None, help="Process only this many items; useful for a smoke test")

	show = sub.add_parser("show", help="Show audit summary for one job")
	show.add_argument("job_id", type=int)

	results = sub.add_parser("results", help="Show stored sentence-level results for one job")
	results.add_argument("job_id", type=int)

	args = parser.parse_args()
	nrdb = NrdbClient(args.agent_url, args.morph_url)
	if args.command == "create":
		_print_json(nrdb.create_job(
			args.dataset_id, args.mode, args.limit, args.model, args.prompt_version,
			args.seed, args.translate, args.blind_translation,
		))
	elif args.command == "list":
		_print_json(nrdb.jobs())
	elif args.command == "run":
		_print_json(run_job(nrdb, args.job_id, max_items=args.max_items))
	elif args.command == "show":
		_print_json(nrdb.summary(args.job_id))
	elif args.command == "results":
		_print_results(nrdb.job_results(args.job_id))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
