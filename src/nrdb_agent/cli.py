import argparse
import json

from .nrdb import NrdbClient
from .runner import run_job


def _print_json(value):
	print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


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
	create.add_argument("--prompt-version", default="annotation-v1")
	create.add_argument("--translate", action="store_true", help="Also generate a Japanese translation and store it as trsl_ai")

	sub.add_parser("list", help="List recent jobs")

	run = sub.add_parser("run", help="Run one existing job")
	run.add_argument("job_id", type=int)
	run.add_argument("--max-items", type=int, default=None, help="Process only this many items; useful for a smoke test")

	show = sub.add_parser("show", help="Show audit summary for one job")
	show.add_argument("job_id", type=int)

	args = parser.parse_args()
	nrdb = NrdbClient(args.agent_url, args.morph_url)
	if args.command == "create":
		_print_json(nrdb.create_job(args.dataset_id, args.mode, args.limit, args.model, args.prompt_version, args.seed, args.translate))
	elif args.command == "list":
		_print_json(nrdb.jobs())
	elif args.command == "run":
		_print_json(run_job(nrdb, args.job_id, max_items=args.max_items))
	elif args.command == "show":
		_print_json(nrdb.summary(args.job_id))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
