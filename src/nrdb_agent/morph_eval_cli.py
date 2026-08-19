import argparse
import json
import os

from .morph_eval import evaluate_morph_agent
from .nrdb import NrdbClient


def _dataset_ids(value):
	if not value:
		return None
	out = []
	for part in str(value).split(","):
		part = part.strip()
		if not part:
			continue
		try:
			parsed = int(part)
		except ValueError as error:
			raise argparse.ArgumentTypeError("dataset IDs must be comma-separated integers") from error
		if parsed < 1:
			raise argparse.ArgumentTypeError("dataset IDs must be positive")
		if parsed not in out:
			out.append(parsed)
	return out or None


def _pct(value):
	return "n/a" if value is None else "{:.1f}%".format(100.0 * float(value))


def _print_summary(payload):
	summary = payload["summary"]
	baseline = summary["baseline"]
	agent = summary["agent"]
	paired = summary["paired"]
	print("MORPH CEILING EVALUATION")
	print("  morph run:                  {}".format(summary["morph_run"]))
	print("  train rows excluded:        {}".format(summary["train_rows"]))
	print("  datasets:                   {}".format(",".join(map(str, summary["datasets"]))))
	print("  rows scored:                {}".format(summary["rows_scored"]))
	print("  morph model(s):             {}".format(", ".join(summary["morph_model_ids"]) or "unknown"))
	print("  agent model:                {}".format(summary["agent_model"]))
	print()
	print("BASELINE -> AGENT")
	print("  ID match rate:              {} -> {}".format(_pct(baseline["id"]["id_match_rate"]), _pct(agent["id"]["id_match_rate"])))
	print("  ID exact accuracy:          {} -> {}".format(_pct(baseline["id"]["linguistic_exact_accuracy"]), _pct(agent["id"]["linguistic_exact_accuracy"])))
	print("  segmentation boundary F1:   {} -> {}".format(_pct(baseline["segmentation"]["boundary_f1"]), _pct(agent["segmentation"]["boundary_f1"])))
	print("  segmentation exact:         {} -> {}".format(_pct(baseline["segmentation"]["exact_accuracy"]), _pct(agent["segmentation"]["exact_accuracy"])))
	print()
	print("PAIRED CEILING BREAK")
	print("  baseline ID errors fixed:   {} ({})".format(paired["baseline_id_errors_corrected"], _pct(paired["baseline_id_error_recovery_rate"])))
	print("  baseline ID correct damaged:{} ({})".format(paired["baseline_id_correct_damaged"], _pct(paired["baseline_id_damage_rate"])))
	print("  fewer / more ID edits:      {} / {}".format(paired["rows_with_fewer_id_edits"], paired["rows_with_more_id_edits"]))
	print("  seg errors fixed / damaged: {} / {}".format(paired["baseline_seg_errors_corrected"], paired["baseline_seg_correct_damaged"]))
	cost = summary.get("estimated_cost_usd")
	cost_text = "unknown" if not summary.get("pricing_complete") else "${:.4f}".format(float(cost or 0.0))
	print("  estimated agent cost:       {}".format(cost_text))


def main(argv=None):
	parser = argparse.ArgumentParser(description="Paired NRDB morph-model versus agent evaluation excluding the morph training split")
	parser.add_argument("morph_run", help="nrdb-morph training-run directory containing train.jsonl")
	parser.add_argument("--model", default="gpt-5.6-terra", help="agent LLM model")
	parser.add_argument("--dataset-ids", type=_dataset_ids, default=None, metavar="ID1,ID2,...", help="optional subset of datasets represented in train.jsonl")
	parser.add_argument("--limit", type=int, default=None, help="score at most N eligible non-training rows")
	parser.add_argument("--seed", type=int, default=1, help="deterministic cohort shuffle seed")
	parser.add_argument("--expected-morph-model", default=None, help="fail if /analyze reports a different deployed morph model ID")
	parser.add_argument("--id-model", default=None, help="ID-sequence critic; defaults to NRDB_ID_MODEL")
	parser.add_argument("--output", default=None, help="write per-row .tsv or complete .json")
	parser.add_argument("--json", action="store_true", help="print complete JSON instead of human summary")
	parser.add_argument("--quiet", action="store_true", help="suppress per-row progress")
	parser.add_argument("--agent-url", default=None)
	parser.add_argument("--morph-url", default=None)
	args = parser.parse_args(argv)
	if args.limit is not None and args.limit < 1:
		parser.error("--limit must be positive")
	progress = (lambda _message: None) if args.quiet or args.json else print
	nrdb = NrdbClient(args.agent_url, args.morph_url)
	value = evaluate_morph_agent(
		nrdb, args.morph_run, model_name=args.model, limit=args.limit, seed=args.seed,
		dataset_ids=args.dataset_ids, expected_morph_model=args.expected_morph_model,
		id_model=args.id_model or os.environ.get("NRDB_ID_MODEL"), output=args.output,
		progress=progress,
	)
	if args.json:
		print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
	else:
		_print_summary(value)
		if args.output:
			print("  output:                     {}".format(args.output))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
