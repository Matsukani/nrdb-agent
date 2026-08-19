import json
import os

from openai import OpenAI

from .annotator import AnnotationAgent
from .annotator_v7 import AnnotationAgentV7
from .annotator_v8 import AnnotationAgentV8
from .annotator_v9 import AnnotationAgentV9
from .reverse_agent import ReverseIdAgent
from .reverse_id_critic import IdCriticSyntaxAwareReverseSurfaceAgent
from .reverse_surface_syntax_agent import SyntaxAwareReverseSurfaceAgent
from .reverse_surface_critic_agent import SurfaceCriticReverseAgent
from .usage import UsageTracker, tracked_client


AGENT_JSON_ATTEMPTS = 3


def _trace_morph_inference(progress, morph):
	inference = morph.get("inference") if isinstance(morph, dict) else None
	if not isinstance(inference, dict):
		return
	progress(
		"  morph: model={} ({}) backend={} decoding={} top-k={} id-weight={}".format(
			inference.get("model_id", ""), inference.get("model_label", ""),
			inference.get("backend", ""), inference.get("segmentation_mode", ""),
			inference.get("segmentation_top_k", ""), inference.get("segmentation_id_weight", ""),
		)
	)


def _trace_usage(progress, usage, prefix="API usage"):
	totals = usage.get("totals", {})
	cost = totals.get("estimated_cost_usd")
	cost_text = "unknown" if not usage.get("pricing_complete") else "${:.4f}".format(float(cost or 0.0))
	progress("  {}: requests={} input={} cached={} output={} estimated_cost={}".format(
		prefix, totals.get("requests", 0), totals.get("input_tokens", 0), totals.get("cached_input_tokens", 0),
		totals.get("output_tokens", 0), cost_text,
	))


def run_job(nrdb, job_id, max_items=None, openai_client=None, progress=print, target_dialects=None, surface_model=None, id_model=None):
	surface_model = surface_model or os.environ.get("NRDB_SURFACE_MODEL")
	id_model = id_model or os.environ.get("NRDB_ID_MODEL")
	bundle = nrdb.job_items(job_id)
	job = bundle["job"]
	items = bundle["items"]
	nrdb.exclude_job_id = int(job_id)
	if target_dialects:
		job["target_dialect_ids"] = [int(value) for value in target_dialects]
	if max_items is not None:
		items = items[:max(0, int(max_items))]
	prompt_version = job.get("prompt_version")
	if prompt_version == "reverse-v1" and target_dialects and surface_model:
		agent_class = SurfaceCriticReverseAgent
	elif prompt_version == "reverse-v1" and target_dialects and id_model:
		agent_class = IdCriticSyntaxAwareReverseSurfaceAgent
	elif prompt_version == "reverse-v1" and target_dialects:
		agent_class = SyntaxAwareReverseSurfaceAgent
	elif prompt_version == "reverse-v1":
		agent_class = ReverseIdAgent
	elif prompt_version == "annotation-v9":
		agent_class = AnnotationAgentV9
	elif prompt_version == "annotation-v8":
		agent_class = AnnotationAgentV8
	elif prompt_version == "annotation-v7":
		agent_class = AnnotationAgentV7
	else:
		agent_class = AnnotationAgent

	usage_tracker = UsageTracker()
	base_client = openai_client or OpenAI()
	client = tracked_client(base_client, usage_tracker)
	agent_kwargs = {"client": client, "progress": progress}
	if agent_class is SurfaceCriticReverseAgent:
		agent_kwargs["surface_model_path"] = surface_model
		agent_kwargs["id_model_path"] = id_model
	elif agent_class is IdCriticSyntaxAwareReverseSurfaceAgent:
		agent_kwargs["id_model_path"] = id_model
	elif agent_class is AnnotationAgentV9:
		agent_kwargs["id_model_path"] = id_model
	agent = agent_class(nrdb, job["model_name"], **agent_kwargs)
	nrdb.set_job_status(job_id, "running")
	completed = 0
	try:
		for index, item in enumerate(items, start=1):
			progress("[{}/{}] sentence {}".format(index, len(items), item["sentence_id"]))
			usage_start = usage_tracker.snapshot()
			try:
				if prompt_version == "reverse-v1":
					progress("  reverse-v1: Japanese={!r}".format(item.get("translation_jp") or ""))
					if target_dialects:
						progress("  reverse surface: target dialect priority={}".format(job["target_dialect_ids"]))
						if id_model:
							progress("  reverse IDs: nrdb-morph critic={}".format(id_model))
						if surface_model:
							progress("  reverse surface: nrdb-morph critic={}".format(surface_model))
					morph = None
				else:
					progress("  morph: analyze")
					morph = nrdb.morph_analyze(item["text"], item["dialect_id"], job["annotation_schema_id"])
					_trace_morph_inference(progress, morph)
					progress("  morph: segmented={!r} annotation={!r}".format(morph.get("segmented", ""), morph.get("annotation", "")))
					if prompt_version == "annotation-v9" and id_model:
						progress("  forward IDs: nrdb-morph critic={}".format(id_model))
				for attempt in range(1, AGENT_JSON_ATTEMPTS + 1):
					try:
						result = agent.annotate(item, job, morph)
						break
					except json.JSONDecodeError as error:
						if attempt >= AGENT_JSON_ATTEMPTS:
							raise
						progress("  llm: malformed/truncated JSON (attempt {}/{}): {}; retrying sentence".format(attempt, AGENT_JSON_ATTEMPTS, error))
			except Exception as error:
				progress("  infrastructure failure: {}".format(error))
				raise

			sentence_usage = usage_tracker.summary(since=usage_start)
			result.setdefault("evidence", {})["api_usage"] = sentence_usage
			_trace_usage(progress, sentence_usage)
			progress("  save: AI result")
			nrdb.save_result(
				job_id=job_id,
				sentence_id=item["sentence_id"],
				segmented=result.get("segmented", ""),
				annotation=result["annotation"],
				trsl_ai=result.get("trsl_ai", "") if job.get("produce_translation") else "",
				decision=result["decision"],
				confidence=result["confidence"],
				evidence=result["evidence"],
				model_response_id=result.get("model_response_id"),
			)
			if prompt_version == "reverse-v1":
				progress("  reverse IDs: {!r}".format(result.get("annotation", "")))
				if target_dialects:
					progress("  reverse surface: {!r}".format(result.get("segmented", "")))
			if job.get("produce_translation"):
				progress("  translation: {!r}".format(result.get("trsl_ai", "")))
			progress("  done")
			completed += 1
		job_usage = usage_tracker.summary()
		_trace_usage(progress, job_usage, prefix="job API usage")
		if max_items is None or completed >= len(bundle["items"]):
			nrdb.set_job_status(job_id, "completed")
		summary = nrdb.summary(job_id)
		if isinstance(summary, dict):
			summary["api_usage"] = job_usage
		return summary
	except BaseException as error:
		nrdb.set_job_status(job_id, "failed", str(error))
		raise
