import json
from dataclasses import dataclass

from .licensed_agent import LicensedTaskAwareAnnotationAgent
from .policy import ForwardMorphPolicy, forward_morph_policy, policy_from_manifest
from .reverse_id_critic import IdCriticSyntaxAwareReverseSurfaceAgent
from .reverse_surface_critic_agent import SurfaceCriticReverseAgent
from .reverse_surface_syntax_agent import SyntaxAwareReverseSurfaceAgent
from .task_agent import SEMANTIC_FEEDBACK_MODES, TaskAwareAnnotationAgent
from .usage import UsageTracker, tracked_client


TASKS = {"morph", "translate", "morph-translate", "reverse"}
MORPHOLOGY_SOURCES = {"none", "predict", "existing", "auto"}
NRDB_EVIDENCE_MODES = {"none", "enabled"}
EXECUTION_JSON_ATTEMPTS = 3


def _existing_morphology(item):
	segmented = str(item.get("existing_segmented") or item.get("segmented") or "").strip()
	annotation = str(item.get("existing_annotation") or item.get("annotation") or "").strip()
	return segmented, annotation


def _semantic_feedback(semantic_feedback=None, require_semantic_feedback=False):
	mode = str(semantic_feedback or "none")
	if mode not in SEMANTIC_FEEDBACK_MODES:
		raise ValueError("invalid semantic_feedback: {}".format(mode))
	return mode, bool(require_semantic_feedback)


def _usage_cost(usage):
	try:
		return float((usage.get("totals") or {}).get("estimated_cost_usd") or 0.0)
	except (AttributeError, TypeError, ValueError):
		return 0.0


def _call_with_json_retry(call, progress, stage):
	for attempt in range(1, EXECUTION_JSON_ATTEMPTS + 1):
		try:
			return call()
		except json.JSONDecodeError as error:
			if attempt >= EXECUTION_JSON_ATTEMPTS:
				raise
			progress("  {}: malformed/truncated JSON (attempt {}/{}): {}; retrying".format(stage, attempt, EXECUTION_JSON_ATTEMPTS, error))
	raise RuntimeError("{} JSON retry failed".format(stage))


def _morph_baseline(morph, source="nrdb-morph"):
	if not isinstance(morph, dict): return None
	segmented = str(morph.get("segmented") or "").strip()
	annotation = str(morph.get("annotation") or "").strip()
	if not segmented and not annotation: return None
	inference = morph.get("inference") if isinstance(morph.get("inference"), dict) else {}
	return {"source": source, "segmented": segmented, "annotation": annotation, "inference": inference}


@dataclass(frozen=True)
class ExecutionRequest:
	item: dict
	task: str
	annotation_schema_id: int
	region: str
	model_name: str = "gpt-5.6"
	semantic_feedback: str | None = "none"
	require_semantic_feedback: bool = False
	use_constructions: bool = False
	use_licensed_forms: bool = False
	nrdb_evidence: str = "enabled"
	morphology_source: str = "predict"
	target_dialect_ids: tuple[int, ...] | None = None
	morph_policy: ForwardMorphPolicy | None = None
	predicted_morphology: dict | None = None

	def __post_init__(self):
		validate_execution_configuration(
			task=self.task, morphology_source=self.morphology_source,
			nrdb_evidence=self.nrdb_evidence, semantic_feedback=self.semantic_feedback,
			require_semantic_feedback=self.require_semantic_feedback,
			use_constructions=self.use_constructions, use_licensed_forms=self.use_licensed_forms,
			morph_policy=self.morph_policy, predicted_morphology=self.predicted_morphology,
		)

	def manifest(self):
		return {
			"format": "nrdb-agent.execution-request.v1",
			"item": dict(self.item),
			"task": self.task,
			"annotation_schema_id": int(self.annotation_schema_id),
			"region": str(self.region or ""),
			"model_name": self.model_name,
			"semantic_feedback": self.semantic_feedback,
			"require_semantic_feedback": bool(self.require_semantic_feedback),
			"use_constructions": bool(self.use_constructions),
			"use_licensed_forms": bool(self.use_licensed_forms),
			"nrdb_evidence": self.nrdb_evidence,
			"morphology_source": self.morphology_source,
			"target_dialect_ids": list(self.target_dialect_ids or []),
			"forward_morph_policy": self.morph_policy.manifest() if self.morph_policy else None,
			"predicted_morphology": dict(self.predicted_morphology) if self.predicted_morphology else None,
		}

	@classmethod
	def from_manifest(cls, value):
		if not isinstance(value, dict) or value.get("format") != "nrdb-agent.execution-request.v1":
			raise ValueError("invalid execution request")
		item = value.get("item")
		if not isinstance(item, dict):
			raise ValueError("execution request item must be an object")
		task = str(value.get("task") or "")
		morphology_source = str(value.get("morphology_source") or "predict")
		nrdb_evidence = str(value.get("nrdb_evidence") or "enabled")
		policy_value = value.get("forward_morph_policy")
		policy = policy_from_manifest(policy_value).validate(morphology_source=morphology_source, task=task) if policy_value else None
		return cls(
			item=dict(item), task=task, annotation_schema_id=int(value["annotation_schema_id"]),
			region=str(value.get("region") or ""), model_name=str(value.get("model_name") or "gpt-5.6"),
			semantic_feedback=value.get("semantic_feedback"),
			require_semantic_feedback=bool(value.get("require_semantic_feedback")),
			use_constructions=bool(value.get("use_constructions")), use_licensed_forms=bool(value.get("use_licensed_forms")),
			nrdb_evidence=nrdb_evidence,
			morphology_source=morphology_source,
			target_dialect_ids=tuple(int(item) for item in (value.get("target_dialect_ids") or [])) or None,
			morph_policy=policy,
			predicted_morphology=dict(value["predicted_morphology"]) if isinstance(value.get("predicted_morphology"), dict) else None,
		)


def execute_request(nrdb, request, openai_client=None, progress=print):
	if not isinstance(request, ExecutionRequest):
		raise TypeError("request must be an ExecutionRequest")
	result = _execute_item(
		nrdb, request.item, request.task, request.annotation_schema_id, request.region,
		model_name=request.model_name, semantic_feedback=request.semantic_feedback,
		require_semantic_feedback=request.require_semantic_feedback,
		use_constructions=request.use_constructions, use_licensed_forms=request.use_licensed_forms,
		nrdb_evidence=request.nrdb_evidence,
		morphology_source=request.morphology_source, target_dialect_ids=request.target_dialect_ids,
		morph_policy=request.morph_policy, predicted_morphology=request.predicted_morphology,
		openai_client=openai_client, progress=progress,
	)
	return {
		"format": "nrdb-agent.execution-result.v1",
		**result,
		"execution_request": request.manifest(),
	}


def validate_execution_configuration(task, morphology_source="predict", nrdb_evidence="enabled",
	semantic_feedback="none", require_semantic_feedback=False, use_constructions=False,
	use_licensed_forms=False, morph_policy=None, predicted_morphology=None):
	task = str(task or "morph")
	source = str(morphology_source or "predict")
	evidence = str(nrdb_evidence or "enabled")
	semantic = str(semantic_feedback or "none")
	if task not in TASKS: raise ValueError("invalid task: {}".format(task))
	if source not in MORPHOLOGY_SOURCES: raise ValueError("invalid morphology_source: {}".format(source))
	if evidence not in NRDB_EVIDENCE_MODES: raise ValueError("invalid nrdb_evidence: {}".format(evidence))
	if semantic not in SEMANTIC_FEEDBACK_MODES: raise ValueError("invalid semantic_feedback: {}".format(semantic))
	if task == "morph" and use_constructions:
		raise ValueError("--constructions requires a translation task")
	if evidence == "none" and (use_constructions or use_licensed_forms):
		raise ValueError("--constructions and --licensed require --nrdb-evidence enabled")
	if task == "reverse" and evidence == "none":
		raise ValueError("nrdb_evidence=none is not supported for reverse realization")
	if task == "reverse" and source != "predict":
		raise ValueError("--morphology-source is not applicable to reverse realization")
	if source == "none":
		if task != "translate":
			raise ValueError("morphology_source=none is valid only for translation-only tasks")
		if semantic != "none" or require_semantic_feedback:
			raise ValueError("morphology_source=none cannot use morphology semantic feedback")
		if use_constructions or use_licensed_forms:
			raise ValueError("morphology_source=none cannot use constructions or licensed morphology")
		if predicted_morphology:
			raise ValueError("morphology_source=none cannot receive predicted morphology")
	if predicted_morphology and source != "predict":
		raise ValueError("predicted_morphology requires morphology_source=predict")
	if morph_policy is not None:
		morph_policy.validate(morphology_source=source, task=task)
		if not morph_policy.agent_review and semantic != "none":
			raise ValueError("semantic feedback requires --morph-review agent")
		if not morph_policy.agent_review and use_licensed_forms:
			raise ValueError("licensed morphology evidence requires --morph-review agent")
	return True


def _execute_item(nrdb, item, task, annotation_schema_id, region, model_name="gpt-5.6",
	semantic_feedback=None, require_semantic_feedback=False, use_constructions=False,
	use_licensed_forms=False, nrdb_evidence="enabled", morphology_source="predict", target_dialect_ids=None,
	morph_review=None, resegmentation=False, max_segmentation_candidates=4,
	id_model=None, surface_model=None, morph_policy=None, predicted_morphology=None, openai_client=None, progress=print):
	task = str(task or "morph")
	morphology_source = str(morphology_source or "predict")
	nrdb_evidence = str(nrdb_evidence or "enabled")
	use_constructions = bool(use_constructions)
	use_licensed_forms = bool(use_licensed_forms)
	semantic_feedback, require_semantic_feedback = _semantic_feedback(semantic_feedback, require_semantic_feedback)
	validate_execution_configuration(
		task, morphology_source, nrdb_evidence, semantic_feedback, require_semantic_feedback,
		use_constructions, use_licensed_forms, morph_policy, predicted_morphology,
	)

	annotation_schema_id = int(annotation_schema_id)
	dialect_id = int(item.get("dialect_id") or item.get("target_dialect_id") or 0)
	if dialect_id <= 0: raise ValueError("item has no valid dialect_id")
	region = str(item.get("dialect_region") or region or "").strip()
	text = str(item.get("text") or "").strip()
	if task != "reverse" and not text: raise ValueError("item has no Miyako source text")

	tracker = UsageTracker()
	client = tracked_client(openai_client, tracker)

	if task == "reverse":
		if morph_policy is not None:
			id_model = morph_policy.id_model_path
			surface_model = morph_policy.surface_model_path
		if resegmentation:
			raise ValueError("--resegmentation applies only to forward morphology")
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
		result = _call_with_json_retry(lambda: agent.annotate(reverse_item, job, None), progress, "reverse")
		usage = tracker.summary()
		return {"source": japanese, "segmented": result.get("segmented", ""), "annotation": result.get("annotation", ""), "translation": result.get("segmented", ""), "decision": result.get("decision"), "confidence": result.get("confidence"), "evidence": result.get("evidence", {}), "api_usage": usage, "estimated_cost_usd": _usage_cost(usage), "model": model_name, "morph_baseline": None, "use_constructions": False, "use_licensed_forms": False}

	segmented, annotation = _existing_morphology(item)
	has_existing = bool(segmented and annotation)
	use_existing = morphology_source == "existing" or (morphology_source == "auto" and has_existing)
	if morphology_source == "existing" and not has_existing: raise ValueError("morphology_source=existing requires segmentation and annotation")
	policy = morph_policy or forward_morph_policy(
		review=morph_review, resegmentation=resegmentation, id_model=id_model, surface_model=surface_model,
		max_segmentation_candidates=max_segmentation_candidates,
		morphology_source="existing" if use_existing else morphology_source, task=task,
	)
	if not policy.agent_review and semantic_feedback != "none":
		raise ValueError("semantic feedback requires --morph-review agent")
	if not policy.agent_review and use_licensed_forms:
		raise ValueError("licensed morphology evidence requires --morph-review agent")

	human_translation = str(item.get("translation_jp") or item.get("translation") or "").strip()
	if semantic_feedback == "existing" and require_semantic_feedback and not human_translation and not use_existing:
		raise ValueError("semantic_feedback=existing is required but this item has no existing translation")

	forward_item = {"sentence_id": int(item.get("sentence_id") or item.get("row_id") or 0), "dialect_id": dialect_id, "dialect_region": region, "text": text, "translation_jp": human_translation if semantic_feedback in {"existing", "auto"} else None}
	job = {
		"annotation_schema_id": annotation_schema_id, "model_name": model_name,
		"prompt_version": "annotation-v9", "task": task,
		"semantic_feedback": semantic_feedback, "require_semantic_feedback": require_semantic_feedback,
		"use_constructions": use_constructions, "use_licensed_forms": use_licensed_forms,
		"nrdb_evidence": nrdb_evidence,
		"morphology_source": morphology_source, "produce_translation": task in {"translate", "morph-translate"},
		"blind_translation": False,
	}
	agent_class = LicensedTaskAwareAnnotationAgent if use_licensed_forms else TaskAwareAnnotationAgent
	agent = agent_class(nrdb, model_name, client=client, progress=progress, morph_policy=policy)
	morph_baseline = None

	if morphology_source == "none":
		progress("  morph: disabled")
		result = _call_with_json_retry(
			lambda: agent.translate_frozen(forward_item, job, "", ""), progress, "raw translation",
		)
		usage = tracker.summary()
		return {
			"source": text, "segmented": "", "annotation": "", "translation": result.get("trsl_ai", ""),
			"decision": result.get("decision"), "confidence": result.get("confidence"),
			"evidence": result.get("evidence", {}), "api_usage": usage,
			"estimated_cost_usd": _usage_cost(usage), "model": model_name,
			"semantic_feedback": semantic_feedback, "nrdb_evidence": nrdb_evidence,
			"use_constructions": False, "use_licensed_forms": False, "morph_baseline": None,
			"forward_morph_policy": policy.manifest(),
		}

	if use_existing:
		morph_baseline = {"source": "existing", "segmented": segmented, "annotation": annotation, "inference": {}}
		if task == "morph":
			result = {"segmented": segmented, "annotation": annotation, "trsl_ai": "", "decision": "proposed", "confidence": 1.0, "evidence": {"existing_morphology": {"frozen": True}}}
		else:
			result = _call_with_json_retry(lambda: agent.translate_frozen(forward_item, job, segmented, annotation), progress, "translation")
	else:
		if predicted_morphology is None:
			progress("  morph: analyze")
			morph = nrdb.morph_analyze(text, dialect_id, annotation_schema_id)
		else:
			progress("  morph: use supplied predicted baseline")
			morph = dict(predicted_morphology)
		morph_baseline = _morph_baseline(morph)
		if policy.agent_review:
			result = _call_with_json_retry(lambda: agent.annotate(forward_item, job, morph), progress, "morphology")
		elif task == "morph":
			result = {"segmented": morph.get("segmented", ""), "annotation": morph.get("annotation", ""), "trsl_ai": "", "decision": "proposed", "confidence": 1.0, "evidence": {"morph_review": {"mode": "none"}}}
		else:
			result = _call_with_json_retry(lambda: agent.translate_frozen(forward_item, job, morph.get("segmented", ""), morph.get("annotation", "")), progress, "translation")

	usage = tracker.summary()
	return {
		"source": text,
		"segmented": result.get("segmented", ""), "annotation": result.get("annotation", ""),
		"translation": result.get("trsl_ai", ""), "decision": result.get("decision"),
		"confidence": result.get("confidence"), "evidence": result.get("evidence", {}),
		"api_usage": usage, "estimated_cost_usd": _usage_cost(usage), "model": model_name,
		"semantic_feedback": semantic_feedback, "use_constructions": use_constructions,
		"nrdb_evidence": nrdb_evidence,
		"use_licensed_forms": use_licensed_forms, "morph_baseline": morph_baseline,
		"forward_morph_policy": policy.manifest(),
	}
