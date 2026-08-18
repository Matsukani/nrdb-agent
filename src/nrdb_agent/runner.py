import json
import os

from .annotator import AnnotationAgent
from .annotator_v7 import AnnotationAgentV7
from .annotator_v8 import AnnotationAgentV8
from .reverse_agent import ReverseIdAgent
from .reverse_surface_syntax_agent import SyntaxAwareReverseSurfaceAgent
from .reverse_surface_critic_agent import SurfaceCriticReverseAgent


AGENT_JSON_ATTEMPTS = 3


def run_job(nrdb, job_id, max_items=None, openai_client=None, progress=print, target_dialects=None, surface_model=None):
	surface_model = surface_model or os.environ.get("NRDB_SURFACE_MODEL")
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
	elif prompt_version == "reverse-v1" and target_dialects:
		agent_class = SyntaxAwareReverseSurfaceAgent
	elif prompt_version == "reverse-v1":
		agent_class = ReverseIdAgent
	elif prompt_version == "annotation-v8":
		agent_class = AnnotationAgentV8
	elif prompt_version == "annotation-v7":
		agent_class = AnnotationAgentV7
	else:
		agent_class = AnnotationAgent
	agent_kwargs = {"client": openai_client, "progress": progress}
	if agent_class is SurfaceCriticReverseAgent:
		agent_kwargs["surface_model_path"] = surface_model
	agent = agent_class(nrdb, job["model_name"], **agent_kwargs)
	nrdb.set_job_status(job_id, "running")
	completed = 0
	try:
		for index, item in enumerate(items, start=1):
			progress("[{}/{}] sentence {}".format(index, len(items), item["sentence_id"]))
			try:
				if prompt_version == "reverse-v1":
					progress("  reverse-v1: Japanese={!r}".format(item.get("translation_jp") or ""))
					if target_dialects:
						progress("  reverse surface: target dialect priority={}".format(job["target_dialect_ids"]))
						if surface_model:
							progress("  reverse surface: nrdb-morph critic={}".format(surface_model))
					morph = None
				else:
					progress("  morph: analyze")
					morph = nrdb.morph_analyze(item["text"], item["dialect_id"], job["annotation_schema_id"])
					progress("  morph: segmented={!r} annotation={!r}".format(morph.get("segmented", ""), morph.get("annotation", "")))
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
		if max_items is None or completed >= len(bundle["items"]):
			nrdb.set_job_status(job_id, "completed")
		return nrdb.summary(job_id)
	except BaseException as error:
		nrdb.set_job_status(job_id, "failed", str(error))
		raise
