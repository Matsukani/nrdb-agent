import os

from .licensed_agent import LicensedTaskAwareAnnotationAgent
from .reverse_id_critic import IdCriticSyntaxAwareReverseSurfaceAgent
from .reverse_surface_critic_agent import SurfaceCriticReverseAgent
from .reverse_surface_syntax_agent import SyntaxAwareReverseSurfaceAgent
from .task_agent import SEMANTIC_FEEDBACK_MODES, TaskAwareAnnotationAgent
from .usage import UsageTracker, tracked_client


TASKS = {"morph", "translate", "morph-translate", "reverse"}
MORPHOLOGY_SOURCES = {"predict", "existing", "auto"}
LEGACY_TRANSLATION_EVIDENCE = {"ignore", "use", "required"}


def _existing_morphology(item):
	segmented = str(item.get("existing_segmented") or item.get("segmented") or "").strip()
	annotation = str(item.get("existing_annotation") or item.get("annotation") or "").strip()
	return segmented, annotation


def _semantic_feedback(semantic_feedback=None, require_semantic_feedback=False, translation_evidence=None):
	if semantic_feedback is None:
		legacy = str(translation_evidence or "ignore")
		if legacy not in LEGACY_TRANSLATION_EVIDENCE:
			raise ValueError("invalid legacy translation_evidence: {}".format(legacy))
		if legacy == "required": return "existing", True
		if legacy == "use": return "auto", bool(require_semantic_feedback)
		return "none", bool(require_semantic_feedback)
	mode = str(semantic_feedback or "none")
	if mode not in SEMANTIC_FEEDBACK_MODES:
		raise ValueError("invalid semantic_feedback: {}".format(mode))
	return mode, bool(require_semantic_feedback)


def _usage_cost(usage):
	try:
		return float((usage.get("totals") or {}).get("estimated_cost_usd") or 0.0)
	except (AttributeError, TypeError, ValueError):
		return 0.0


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


def _morph_baseline(morph, source="nrdb-morph"):
	if not isinstance(morph, dict): return None
	segmented = str(morph.get("segmented") or "").strip()
	annotation = str(morph.get("annotation") or "").strip()
	if not segmented and not annotation: return None
	inference = morph.get("inference") if isinstance(morph.get("inference"), dict) else {}
	return {"source": source, "segmented": segmented, "annotation": annotation, "inference": inference}


def execute_item(nrdb, item, task, annotation_schema_id, region, model_name="gpt-5.6",
	semantic_feedback=None, require_semantic_feedback=False, use_constructions=False,
	use_licensed_forms=False, morphology_source="predict", target_dialect_ids=None,
	id_model=None, surface_model=None, openai_client=None, progress=print,
	translation_evidence=None):
	task = str(task or "morph")
	morphology_source = str(morphology_source or "predict")
	use_constructions = bool(use_constructions)
	use_licensed_forms = bool(use_licensed_forms)
	semantic_feedback, require_semantic_feedback = _semantic_feedback(semantic_feedback, require_semantic_feedback, translation_evidence)
	if task not in TASKS: raise ValueError("invalid task: {}".format(task))
	if morphology_source not in MORPHOLOGY_SOURCES: raise ValueError("invalid morphology_source: {}".format(morphology_source))

	annotation_schema_id = int(annotation_schema_id)
	dialect_id = int(item.get("dialect_id") or item.get("target_dialect_id") or 0)
	if dialect_id <= 0: raise ValueError("item has no valid dialect_id")
	region = str(item.get("dialect_region") or region or "").strip()
	text = str(item.get("text") or "").strip()
	if task != "reverse" and not text: raise ValueError("item has no Miyako source text")

	id_model = id_model or os.environ.get("NRDB_ID_MODEL")
	surface_model = surface_model or os.environ.get("NRDB_SURFACE_MODEL")
	tracker = UsageTracker()
	client = tracked_client(openai_client, tracker)

	if task == "reverse":
		if semantic_feedback != "none" or require_semantic_feedback or use_constructions or use_licensed_forms:
			raise ValueError("semantic feedback, constructions and licensed forms are not used for reverse tasks")
		japanese = str(item.get("translation_jp") or item.get("translation") or item.get("japanese") or text or "").strip()
		if not japanese: raise ValueError("reverse task requires Japanese input")
		dialects = [int(value) for value in (target_dialect_ids or [dialect_id])]
		reverse_item = {"sentence_id": int(item.get("sentence_id") or item.get("row_id") or 0), "dialect_id": dialects[0], "dialect_region": region, "text": "", "translation_jp": japanese}
		job = {"annotation_schema_id": annotation_schema_id, "model_name": model_name, "prompt_version": "reverse-v1", "produce_translation": False, "blind_translation": False, "target_dialect_ids": dialects}
		if surface_model: agent = SurfaceCriticReverseAgent(nrdb, model_name, client=client, progress=progress, surface_model_path=surface_model, id_model_path=id_model)
		elif id_model: agent = IdCriticSyntaxAwareReverseSurfaceAgent(nrdb, model_name, client=client, progress=progress, id_model_path=id_model)
		else: agent = SyntaxAwareReverseSurfaceAgent(nrdb, model_name, client=client, progress=progress)
		result = agent.annotate(reverse_item, job, None)
		usage = tracker.summary()
		return {"source": japanese, "segmented": result.get("segmented", ""), "annotation": result.get("annotation", ""), "translation": result.get("segmented", ""), "decision": result.get("decision"), "confidence": result.get("confidence"), "evidence": result.get("evidence", {}), "api_usage": usage, "estimated_cost_usd": _usage_cost(usage), "model": model_name, "morph_baseline": None, "use_constructions": False, "use_licensed_forms": False}

	segmented, annotation = _existing_morphology(item)
	has_existing = bool(segmented and annotation)
	use_existing = morphology_source == "existing" or (morphology_source == "auto" and has_existing)
	if morphology_source == "existing" and not has_existing: raise ValueError("morphology_source=existing requires segmentation and annotation")

	human_translation = str(item.get("translation_jp") or item.get("translation") or "").strip()
	if semantic_feedback == "existing" and require_semantic_feedback and not human_translation and not use_existing:
		raise ValueError("semantic_feedback=existing is required but this item has no existing translation")

	forward_item = {"sentence_id": int(item.get("sentence_id") or item.get("row_id") or 0), "dialect_id": dialect_id, "dialect_region": region, "text": text, "translation_jp": human_translation if semantic_feedback in {"existing", "auto"} else None}
	job = {
		"annotation_schema_id": annotation_schema_id, "model_name": model_name,
		"prompt_version": "annotation-v9", "task": task,
		"semantic_feedback": semantic_feedback, "require_semantic_feedback": require_semantic_feedback,
		"use_constructions": use_constructions, "use_licensed_forms": use_licensed_forms,
		"morphology_source": morphology_source, "produce_translation": task in {"translate", "morph-translate"},
		"blind_translation": False,
	}
	agent_class = LicensedTaskAwareAnnotationAgent if use_licensed_forms else TaskAwareAnnotationAgent
	agent = agent_class(nrdb, model_name, client=client, progress=progress, id_model_path=id_model, surface_model_path=surface_model)
	morph_baseline = None

	if use_existing:
		morph_baseline = {"source": "existing", "segmented": segmented, "annotation": annotation, "inference": {}}
		if task == "morph":
			result = {"segmented": segmented, "annotation": annotation, "trsl_ai": "", "decision": "proposed", "confidence": 1.0, "evidence": {"existing_morphology": {"frozen": True}}}
		else:
			result = agent.translate_frozen(forward_item, job, segmented, annotation)
	else:
		progress("  morph: analyze")
		morph = nrdb.morph_analyze(text, dialect_id, annotation_schema_id)
		morph_baseline = _morph_baseline(morph)
		result = agent.annotate(forward_item, job, morph)

	usage = tracker.summary()
	return {
		"source": text,
		"segmented": result.get("segmented", ""), "annotation": result.get("annotation", ""),
		"translation": result.get("trsl_ai", ""), "decision": result.get("decision"),
		"confidence": result.get("confidence"), "evidence": result.get("evidence", {}),
		"api_usage": usage, "estimated_cost_usd": _usage_cost(usage), "model": model_name,
		"semantic_feedback": semantic_feedback, "use_constructions": use_constructions,
		"use_licensed_forms": use_licensed_forms, "morph_baseline": morph_baseline,
	}


def run_workflow_job(nrdb, job_id, max_items=None, progress=print, target_dialects=None,
	id_model=None, surface_model=None, openai_client=None):
	bundle = nrdb.workflow_job_items(job_id)
	job = bundle["job"]
	items = list(bundle.get("items", []))
	_job_scope(progress, bundle)
	if max_items is not None: items = items[:max(0, int(max_items))]
	nrdb.exclude_job_id = int(job_id)
	nrdb.set_job_status(job_id, "running")
	completed = 0
	failed = 0
	cost = 0.0
	pricing_complete = True
	try:
		for index, raw in enumerate(items, start=1):
			label = "sentence {}".format(raw["sentence_id"])
			_item_start(progress, index, len(items), label)
			item = dict(raw)
			result = execute_item(
				nrdb, item, job["task"], job["annotation_schema_id"], item.get("dialect_region"),
				model_name=job["model_name"], semantic_feedback=job.get("semantic_feedback"),
				require_semantic_feedback=bool(job.get("require_semantic_feedback")),
				use_constructions=bool(job.get("use_constructions")),
				use_licensed_forms=bool(job.get("use_licensed_forms")),
				translation_evidence=job.get("translation_evidence"), morphology_source=job.get("morphology_source") or "predict",
				target_dialect_ids=target_dialects, id_model=id_model, surface_model=surface_model,
				openai_client=openai_client, progress=progress,
			)
			cost += float(result.get("estimated_cost_usd") or 0.0)
			pricing_complete = pricing_complete and _pricing_complete(result)
			evidence = dict(result.get("evidence") or {})
			evidence["api_usage"] = result.get("api_usage", {})
			if result.get("morph_baseline"): evidence["morph_baseline"] = result["morph_baseline"]
			nrdb.save_result(job_id=job_id, sentence_id=item["sentence_id"], segmented=result.get("segmented", ""), annotation=result.get("annotation", ""), trsl_ai=result.get("translation", "") if job.get("produce_translation") else "", decision=result.get("decision") or "proposed", confidence=result.get("confidence"), evidence=evidence, model_response_id=None)
			completed += 1
			_item_result(progress, index, len(items), job["task"], result, label)
		if max_items is None or completed >= len(bundle.get("items", [])): nrdb.set_job_status(job_id, "completed")
		_job_summary(progress, completed, len(items), cost, failed, pricing_complete)
		summary = nrdb.summary(job_id)
		if isinstance(summary, dict):
			summary["workflow"] = {"task": job["task"], "semantic_feedback": job.get("semantic_feedback"), "require_semantic_feedback": bool(job.get("require_semantic_feedback")), "use_constructions": bool(job.get("use_constructions")), "use_licensed_forms": bool(job.get("use_licensed_forms")), "completed": completed, "failed": failed, "estimated_cost_usd": cost, "pricing_complete": pricing_complete, "scope": bundle.get("scope", {})}
		return summary
	except BaseException as error:
		nrdb.set_job_status(job_id, "failed", str(error))
		if hasattr(progress, "item_error"): progress.item_error(completed + 1, len(items), error)
		raise
