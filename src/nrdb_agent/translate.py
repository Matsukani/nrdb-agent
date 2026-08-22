from .execution import ExecutionRequest, execute_request
from .policy import forward_morph_policy
from .task_agent import SEMANTIC_FEEDBACK_MODES


def _dialect_ids(nrdb, region, annotation_schema_id, dialect_ids=None):
	if dialect_ids:
		return [int(value) for value in dialect_ids]
	rows = nrdb.region_dialects(region, annotation_schema_id)
	if not rows:
		raise RuntimeError("no dialects available for region {!r} under annotation schema {}".format(region, annotation_schema_id))
	return [int(row["id"]) for row in rows]


def _trace_morph_provenance(progress, baseline):
	inference = baseline.get("inference") if isinstance(baseline, dict) else None
	if not isinstance(inference, dict) or not inference:
		return
	progress("  morph: model={} ({}) backend={} decoding={} top-k={} id-weight={}".format(
		inference.get("model_id", ""), inference.get("model_label", ""), inference.get("backend", ""),
		inference.get("segmentation_mode", ""), inference.get("segmentation_top_k", ""), inference.get("segmentation_id_weight", ""),
	))


def _trace_usage(progress, usage):
	totals = usage.get("totals", {})
	cost = totals.get("estimated_cost_usd")
	cost_text = "unknown" if not usage.get("pricing_complete") else "USD {:.4f}".format(float(cost or 0.0))
	progress("  API usage: requests={} input={} cached={} output={} reasoning={} estimated_cost={}".format(
		totals.get("requests", 0), totals.get("input_tokens", 0), totals.get("cached_input_tokens", 0),
		totals.get("output_tokens", 0), totals.get("reasoning_tokens", 0), cost_text,
	))


def translate_text(nrdb, text, target, annotation_schema_id, region, dialect_ids=None,
	model_name="gpt-5.6", surface_model=None, id_model=None, semantic_feedback=None,
	require_semantic_feedback=False, use_constructions=False, use_licensed_forms=False,
	nrdb_evidence="enabled", morphology_source=None,
	existing_translation=None, fixed_segmented=None, fixed_annotation=None,
	sentence_id=0, morph_review=None, resegmentation=False, max_segmentation_candidates=4,
	morph_policy=None, openai_client=None, progress=print):
	text = str(text or "").strip()
	region = str(region or "").strip()
	if not text: raise ValueError("translation text cannot be empty")
	if not region: raise ValueError("translation region cannot be empty")
	annotation_schema_id = int(annotation_schema_id)
	if annotation_schema_id <= 0: raise ValueError("annotation schema ID must be positive")
	target = str(target or "").strip().lower()
	if target not in {"japanese", "miyako"}: raise ValueError("target must be japanese or miyako")
	semantic_feedback = str(semantic_feedback if semantic_feedback is not None else ("generated" if target == "japanese" else "none"))
	if semantic_feedback not in SEMANTIC_FEEDBACK_MODES: raise ValueError("invalid semantic_feedback: {}".format(semantic_feedback))
	dialects = _dialect_ids(nrdb, region, annotation_schema_id, dialect_ids)
	nrdb.exclude_job_id = 0

	if target == "japanese":
		fixed_segmented = str(fixed_segmented or "").strip()
		fixed_annotation = str(fixed_annotation or "").strip()
		if bool(fixed_segmented) != bool(fixed_annotation):
			raise ValueError("fixed morphology requires both segmentation and annotation")
		use_fixed = bool(fixed_segmented and fixed_annotation)
		morphology_source = str(morphology_source or ("existing" if use_fixed else "predict"))
		if use_fixed and morphology_source not in {"existing", "auto"}:
			raise ValueError("fixed morphology requires morphology_source=existing or auto")
		if morphology_source == "existing" and not use_fixed:
			raise ValueError("morphology_source=existing requires fixed segmentation and annotation")
		if morphology_source == "auto":
			morphology_source = "existing" if use_fixed else "predict"
		policy = morph_policy or forward_morph_policy(
			review=morph_review, resegmentation=resegmentation, id_model=id_model, surface_model=surface_model,
			max_segmentation_candidates=max_segmentation_candidates,
			morphology_source=morphology_source, task="translate" if morphology_source == "none" else "morph-translate",
		)
		progress("translate: Miyako -> Japanese | region={} morph_dialect={} schema={} model={} morphology={}".format(
			region, dialects[0], annotation_schema_id, model_name, morphology_source,
		))
		request = ExecutionRequest(
			item={
				"sentence_id": int(sentence_id or 0), "dialect_id": dialects[0], "dialect_region": region,
				"text": text, "translation_jp": str(existing_translation or "").strip(),
				"existing_segmented": fixed_segmented, "existing_annotation": fixed_annotation,
			},
			task="translate" if morphology_source == "none" else "morph-translate", annotation_schema_id=annotation_schema_id, region=region,
			model_name=model_name, semantic_feedback=semantic_feedback,
			require_semantic_feedback=bool(require_semantic_feedback),
			use_constructions=bool(use_constructions), use_licensed_forms=bool(use_licensed_forms),
			nrdb_evidence=nrdb_evidence, morphology_source=morphology_source, morph_policy=policy,
		)
		result = execute_request(nrdb, request, openai_client=openai_client, progress=progress)
		_trace_morph_provenance(progress, result.get("morph_baseline"))
		_trace_usage(progress, result.get("api_usage") or {})
		return {
			**result,
			"direction": "miyako_to_japanese", "region": region,
			"annotation_schema_id": annotation_schema_id, "morph_dialect_id": dialects[0],
			"morph_inference": (result.get("morph_baseline") or {}).get("inference") or {},
			"llm_model": model_name, "morphology_source": "gold" if use_fixed else morphology_source,
			"execution_request": request.manifest(),
		}

	if semantic_feedback != "none" or require_semantic_feedback or use_constructions or use_licensed_forms:
		raise ValueError("semantic feedback, constructions and licensed forms are not used for Japanese -> Miyako direct translation")
	if not dialect_ids:
		raise ValueError("Japanese -> Miyako translation requires an ordered --dialects list")
	policy = morph_policy or forward_morph_policy(
		review=morph_review, resegmentation=False, id_model=id_model, surface_model=surface_model,
		max_segmentation_candidates=max_segmentation_candidates, morphology_source="predict", task="reverse",
	)
	progress("translate: Japanese -> Miyako | region={} dialects={} schema={} model={}".format(region, dialects, annotation_schema_id, model_name))
	request = ExecutionRequest(
		item={"sentence_id": int(sentence_id or 0), "dialect_id": dialects[0], "dialect_region": region, "text": "", "translation_jp": text},
		task="reverse", annotation_schema_id=annotation_schema_id, region=region, model_name=model_name,
		semantic_feedback="none", nrdb_evidence=nrdb_evidence,
		target_dialect_ids=tuple(dialects), morph_policy=policy,
	)
	result = execute_request(nrdb, request, openai_client=openai_client, progress=progress)
	_trace_usage(progress, result.get("api_usage") or {})
	return {
		**result,
		"direction": "japanese_to_miyako", "source": text, "region": region,
		"annotation_schema_id": annotation_schema_id, "target_dialect_ids": dialects,
		"llm_model": model_name, "execution_request": request.manifest(),
	}
