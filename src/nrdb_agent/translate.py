import json
import os

from .licensed_agent import LicensedTaskAwareAnnotationAgent
from .reverse_id_critic import IdCriticSyntaxAwareReverseSurfaceAgent
from .reverse_surface_critic_agent import SurfaceCriticReverseAgent
from .reverse_surface_syntax_agent import SyntaxAwareReverseSurfaceAgent
from .task_agent import SEMANTIC_FEEDBACK_MODES, TaskAwareAnnotationAgent
from .usage import UsageTracker, tracked_client


DIRECT_JSON_ATTEMPTS = 3


def _dialect_ids(nrdb, region, annotation_schema_id, dialect_ids=None):
	if dialect_ids:
		return [int(value) for value in dialect_ids]
	rows = nrdb.region_dialects(region, annotation_schema_id)
	if not rows:
		raise RuntimeError("no dialects available for region {!r} under annotation schema {}".format(region, annotation_schema_id))
	return [int(row["id"]) for row in rows]


def _morph_provenance(morph):
	value = morph.get("inference") if isinstance(morph, dict) else None
	return value if isinstance(value, dict) else {}


def _trace_morph_provenance(progress, morph):
	inference = _morph_provenance(morph)
	if not inference:
		progress("  morph: model provenance unavailable")
		return
	progress("  morph: model={} ({}) backend={} decoding={} top-k={} id-weight={}".format(
		inference.get("model_id", ""), inference.get("model_label", ""), inference.get("backend", ""),
		inference.get("segmentation_mode", ""), inference.get("segmentation_top_k", ""), inference.get("segmentation_id_weight", ""),
	))


def _trace_usage(progress, usage):
	totals = usage.get("totals", {})
	cost = totals.get("estimated_cost_usd")
	cost_text = "unknown" if not usage.get("pricing_complete") else "${:.4f}".format(float(cost or 0.0))
	progress("  API usage: requests={} input={} cached={} output={} reasoning={} estimated_cost={}".format(
		totals.get("requests", 0), totals.get("input_tokens", 0), totals.get("cached_input_tokens", 0),
		totals.get("output_tokens", 0), totals.get("reasoning_tokens", 0), cost_text,
	))
	for stage, values in usage.get("by_stage", {}).items():
		progress("    cost {}: requests={} tokens={}+{} cost=${:.4f}".format(
			stage, values.get("requests", 0), values.get("input_tokens", 0), values.get("output_tokens", 0),
			float(values.get("estimated_cost_usd") or 0.0),
		))


def _annotate_with_json_retry(agent, item, job, morph, progress):
	last_error = None
	for attempt in range(1, DIRECT_JSON_ATTEMPTS + 1):
		try:
			return agent.annotate(item, job, morph)
		except json.JSONDecodeError as error:
			last_error = error
			if attempt >= DIRECT_JSON_ATTEMPTS:
				raise
			progress("  llm: malformed/truncated tool or final JSON (attempt {}/{}): {}; retrying translation".format(attempt, DIRECT_JSON_ATTEMPTS, error))
	if last_error:
		raise last_error
	raise RuntimeError("direct translation JSON retry failed")


def _translate_frozen_with_json_retry(agent, item, job, segmented, annotation, progress):
	last_error = None
	for attempt in range(1, DIRECT_JSON_ATTEMPTS + 1):
		try:
			return agent.translate_frozen(item, job, segmented, annotation)
		except json.JSONDecodeError as error:
			last_error = error
			if attempt >= DIRECT_JSON_ATTEMPTS:
				raise
			progress("  llm: malformed/truncated frozen-translation JSON (attempt {}/{}): {}; retrying translation".format(attempt, DIRECT_JSON_ATTEMPTS, error))
	if last_error:
		raise last_error
	raise RuntimeError("frozen translation JSON retry failed")


def translate_text(nrdb, text, target, annotation_schema_id, region, dialect_ids=None,
	model_name="gpt-5.6", surface_model=None, id_model=None, semantic_feedback="generated",
	require_semantic_feedback=False, use_constructions=False, use_licensed_forms=False,
	existing_translation=None, fixed_segmented=None, fixed_annotation=None,
	sentence_id=0,
	openai_client=None, progress=print):
	text = str(text or "").strip()
	region = str(region or "").strip()
	use_constructions = bool(use_constructions)
	use_licensed_forms = bool(use_licensed_forms)
	if not text:
		raise ValueError("translation text cannot be empty")
	if not region:
		raise ValueError("translation region cannot be empty")
	annotation_schema_id = int(annotation_schema_id)
	if annotation_schema_id <= 0:
		raise ValueError("annotation schema ID must be positive")
	target = str(target or "").strip().lower()
	if target not in {"japanese", "miyako"}:
		raise ValueError("target must be japanese or miyako")
	semantic_feedback = str(semantic_feedback or "none")
	if semantic_feedback not in SEMANTIC_FEEDBACK_MODES:
		raise ValueError("invalid semantic_feedback: {}".format(semantic_feedback))

	dialects = _dialect_ids(nrdb, region, annotation_schema_id, dialect_ids)
	nrdb.exclude_job_id = 0
	id_model = id_model or os.environ.get("NRDB_ID_MODEL")
	usage_tracker = UsageTracker()
	client = tracked_client(openai_client, usage_tracker)

	if target == "japanese":
		dialect_id = dialects[0]
		fixed_segmented = str(fixed_segmented or "").strip()
		fixed_annotation = str(fixed_annotation or "").strip()
		if bool(fixed_segmented) != bool(fixed_annotation):
			raise ValueError("fixed morphology requires both segmentation and annotation")
		use_fixed_morphology = bool(fixed_segmented and fixed_annotation)
		existing_translation = str(existing_translation or "").strip()
		if semantic_feedback == "existing" and require_semantic_feedback and not existing_translation:
			raise ValueError("semantic_feedback=existing requires --existing-translation")
		progress("translate: Miyako -> Japanese | region={} morph_dialect={} schema={} forward=annotation-v9 model={} semantic_feedback={} constructions={} licensed={} morphology={}".format(
			region, dialect_id, annotation_schema_id, model_name, semantic_feedback,
			"on" if use_constructions else "off", "on" if use_licensed_forms else "off",
			"gold" if use_fixed_morphology else "predicted",
		))
		if use_fixed_morphology:
			progress("  morph: use frozen gold segmentation and annotation")
			morph = {"segmented": fixed_segmented, "annotation": fixed_annotation, "inference": {}}
		else:
			progress("  morph: analyze")
			morph = nrdb.morph_analyze(text, dialect_id, annotation_schema_id)
		licensed = None
		if use_licensed_forms:
			licensed = nrdb.licensed_forms_in_text(text, annotation_schema_id, region, dialect_id)
			morph["licensed_realizations"] = licensed
			progress("  licensed: grammar-derived surface matches={}".format(len(licensed.get("matches", []))))
		if not use_fixed_morphology:
			_trace_morph_provenance(progress, morph)
		progress("  morph: segmented={!r} annotation={!r}".format(morph.get("segmented", ""), morph.get("annotation", "")))
		if id_model and not use_fixed_morphology:
			progress("  forward ID critic: {}".format(id_model))
		item = {"sentence_id": int(sentence_id or 0), "dialect_id": dialect_id, "dialect_region": region, "text": text, "translation_jp": existing_translation if semantic_feedback in {"existing", "auto"} else None}
		job = {
			"annotation_schema_id": annotation_schema_id, "model_name": model_name,
			"prompt_version": "annotation-v9", "task": "morph-translate",
			"semantic_feedback": semantic_feedback, "require_semantic_feedback": bool(require_semantic_feedback),
			"use_constructions": use_constructions, "use_licensed_forms": use_licensed_forms,
			"morphology_source": "existing" if use_fixed_morphology else "predict", "produce_translation": True, "blind_translation": False,
		}
		agent_class = LicensedTaskAwareAnnotationAgent if use_licensed_forms else TaskAwareAnnotationAgent
		agent = agent_class(nrdb, model_name, client=client, progress=progress, id_model_path=None if use_fixed_morphology else id_model)
		if use_fixed_morphology:
			result = _translate_frozen_with_json_retry(agent, item, job, fixed_segmented, fixed_annotation, progress)
		else:
			result = _annotate_with_json_retry(agent, item, job, morph, progress)
		if licensed is not None:
			result.setdefault("evidence", {})["licensed_realizations"] = licensed
		usage = usage_tracker.summary()
		_trace_usage(progress, usage)
		return {
			"direction": "miyako_to_japanese", "source": text, "region": region,
			"annotation_schema_id": annotation_schema_id, "morph_dialect_id": dialect_id,
			"morph_inference": _morph_provenance(morph), "llm_model": model_name,
			"morphology_source": "gold" if use_fixed_morphology else "predicted",
			"semantic_feedback": semantic_feedback, "use_constructions": use_constructions,
			"use_licensed_forms": use_licensed_forms,
			"segmented": result.get("segmented", ""), "annotation": result.get("annotation", ""),
			"translation": result.get("trsl_ai", ""), "decision": result.get("decision"),
			"confidence": result.get("confidence"), "api_usage": usage, "evidence": result.get("evidence", {}),
		}

	if semantic_feedback != "none" or require_semantic_feedback or use_constructions or use_licensed_forms:
		raise ValueError("semantic feedback, constructions and licensed forms are not used for Japanese -> Miyako direct translation")
	if not dialect_ids:
		raise ValueError("Japanese -> Miyako translation requires an ordered --dialects list")
	surface_model = surface_model or os.environ.get("NRDB_SURFACE_MODEL")
	progress("translate: Japanese -> Miyako | region={} dialects={} schema={} model={}".format(region, dialects, annotation_schema_id, model_name))
	if id_model:
		progress("  ID critic: {}".format(id_model))
	if surface_model:
		progress("  surface critic: {}".format(surface_model))
	item = {"sentence_id": 0, "dialect_id": dialects[0], "dialect_region": region, "text": "", "translation_jp": text}
	job = {"annotation_schema_id": annotation_schema_id, "model_name": model_name, "prompt_version": "reverse-v1", "produce_translation": False, "blind_translation": False, "target_dialect_ids": dialects}
	if surface_model:
		agent = SurfaceCriticReverseAgent(nrdb, model_name, client=client, progress=progress, surface_model_path=surface_model, id_model_path=id_model)
	elif id_model:
		agent = IdCriticSyntaxAwareReverseSurfaceAgent(nrdb, model_name, client=client, progress=progress, id_model_path=id_model)
	else:
		agent = SyntaxAwareReverseSurfaceAgent(nrdb, model_name, client=client, progress=progress)
	result = _annotate_with_json_retry(agent, item, job, None, progress)
	usage = usage_tracker.summary()
	_trace_usage(progress, usage)
	return {
		"direction": "japanese_to_miyako", "source": text, "region": region,
		"annotation_schema_id": annotation_schema_id, "target_dialect_ids": dialects, "llm_model": model_name,
		"annotation": result.get("annotation", ""), "translation": result.get("segmented", ""),
		"decision": result.get("decision"), "confidence": result.get("confidence"),
		"api_usage": usage, "evidence": result.get("evidence", {}),
	}
