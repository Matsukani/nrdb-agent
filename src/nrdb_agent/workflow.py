from .execution import ExecutionRequest, execute_request
from .policy import policy_from_manifest


def _pricing_complete(result):
	usage = result.get("api_usage") if isinstance(result, dict) else None
	return bool(isinstance(usage, dict) and usage.get("pricing_complete"))


def _item_start(progress, index, total, label):
	if hasattr(progress, "item_start"): progress.item_start(index, total, label)
	else: progress("[{}/{}] {}".format(index, total, label))


def _item_result(progress, index, total, task, result, label):
	if hasattr(progress, "item_result"): progress.item_result(index, total, task, result, label)


def _job_summary(progress, completed, total, cost, failed, pricing_complete):
	if hasattr(progress, "job_summary"):
		progress.job_summary(completed, total, cost, failed=failed, pricing_complete=pricing_complete)


def _job_scope(progress, bundle):
	scope = bundle.get("scope") if isinstance(bundle, dict) else None
	if not isinstance(scope, dict): return
	extra = []
	if scope.get("internal_text_id") is not None: extra.append("text={}".format(scope["internal_text_id"]))
	if scope.get("sentence_start") is not None: extra.append("sentences={}:{}".format(scope["sentence_start"], scope.get("sentence_end")))
	suffix = " | " + " ".join(extra) if extra else ""
	progress("job scope: eligible={} selected={} completed={} remaining={} limit={}{}".format(
		scope.get("eligible_utterances", 0), scope.get("selected_utterances", 0),
		scope.get("completed_utterances", 0), scope.get("remaining_utterances", 0),
		scope.get("item_limit", 0), suffix,
	))


def _stored_policy(job):
	value = job.get("execution_policy") if isinstance(job.get("execution_policy"), dict) else None
	if value is None and isinstance(job.get("execution_policy_json"), str) and job.get("execution_policy_json"):
		import json
		value = json.loads(job["execution_policy_json"])
	if not value:
		raise ValueError("job has no execution policy; recreate it with the current nrdb-agent create command")
	return policy_from_manifest(value).validate(
		morphology_source=job.get("morphology_source") or "predict", task=job.get("task") or "morph",
	)


def run_workflow_job(nrdb, job_id, max_items=None, progress=print, target_dialects=None, openai_client=None):
	bundle = nrdb.workflow_job_items(job_id)
	job = bundle["job"]
	policy = _stored_policy(job)
	items = list(bundle.get("items", []))
	_job_scope(progress, bundle)
	if max_items is not None: items = items[:max(0, int(max_items))]
	nrdb.exclude_job_id = int(job_id)
	nrdb.set_job_status(job_id, "running")
	completed = 0
	cost = 0.0
	pricing_complete = True
	try:
		for index, raw in enumerate(items, start=1):
			label = "sentence {}".format(raw["sentence_id"])
			_item_start(progress, index, len(items), label)
			request = ExecutionRequest(
				item=dict(raw), task=job["task"], annotation_schema_id=int(job["annotation_schema_id"]),
				region=str(raw.get("dialect_region") or ""), model_name=job["model_name"],
				semantic_feedback=job.get("semantic_feedback") or "none",
				require_semantic_feedback=bool(job.get("require_semantic_feedback")),
				use_constructions=bool(job.get("use_constructions")), use_licensed_forms=bool(job.get("use_licensed_forms")),
				nrdb_evidence=job.get("nrdb_evidence") or "enabled",
				morphology_source=job.get("morphology_source") or "predict",
				target_dialect_ids=tuple(int(value) for value in target_dialects) if target_dialects else None,
				morph_policy=policy,
			)
			result = execute_request(nrdb, request, openai_client=openai_client, progress=progress)
			cost += float(result.get("estimated_cost_usd") or 0.0)
			pricing_complete = pricing_complete and _pricing_complete(result)
			evidence = dict(result.get("evidence") or {})
			evidence["api_usage"] = result.get("api_usage", {})
			evidence["execution_request"] = request.manifest()
			if result.get("morph_baseline"): evidence["morph_baseline"] = result["morph_baseline"]
			if result.get("forward_morph_policy"): evidence["forward_morph_policy_manifest"] = result["forward_morph_policy"]
			nrdb.save_result(job_id=job_id, sentence_id=raw["sentence_id"], segmented=result.get("segmented", ""), annotation=result.get("annotation", ""), trsl_ai=result.get("translation", "") if job.get("produce_translation") else "", decision=result.get("decision") or "proposed", confidence=result.get("confidence"), evidence=evidence, model_response_id=None)
			completed += 1
			_item_result(progress, index, len(items), job["task"], result, label)
		if max_items is None or completed >= len(bundle.get("items", [])): nrdb.set_job_status(job_id, "completed")
		_job_summary(progress, completed, len(items), cost, 0, pricing_complete)
		summary = nrdb.summary(job_id)
		if isinstance(summary, dict):
			summary["workflow"] = {"task": job["task"], "forward_morph_policy": policy.manifest(), "completed": completed, "failed": 0, "estimated_cost_usd": cost, "pricing_complete": pricing_complete, "scope": bundle.get("scope", {})}
		return summary
	except BaseException as error:
		nrdb.set_job_status(job_id, "failed", str(error))
		if hasattr(progress, "item_error"): progress.item_error(completed + 1, len(items), error)
		raise
