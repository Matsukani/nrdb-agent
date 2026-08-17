from .annotator import AnnotationAgent
from .annotator_v7 import AnnotationAgentV7


def run_job(nrdb, job_id, max_items=None, openai_client=None, progress=print):
	bundle = nrdb.job_items(job_id)
	job = bundle["job"]
	items = bundle["items"]
	if max_items is not None:
		items = items[:max(0, int(max_items))]
	agent_class = AnnotationAgentV7 if job.get("prompt_version") == "annotation-v7" else AnnotationAgent
	agent = agent_class(nrdb, job["model_name"], client=openai_client, progress=progress)
	nrdb.set_job_status(job_id, "running")
	completed = 0
	try:
		for index, item in enumerate(items, start=1):
			progress("[{}/{}] sentence {}".format(index, len(items), item["sentence_id"]))
			try:
				progress("  morph: analyze")
				morph = nrdb.morph_analyze(item["text"], item["dialect_id"], job["annotation_schema_id"])
				progress("  morph: segmented={!r} annotation={!r}".format(morph.get("segmented", ""), morph.get("annotation", "")))
				result = agent.annotate(item, job, morph)
			except Exception as error:
				progress("  infrastructure failure: {}".format(error))
				raise

			progress("  save: AI result")
			nrdb.save_result(
				job_id=job_id,
				sentence_id=item["sentence_id"],
				segmented=result["segmented"],
				annotation=result["annotation"],
				trsl_ai=result.get("trsl_ai", "") if job.get("produce_translation") else "",
				decision=result["decision"],
				confidence=result["confidence"],
				evidence=result["evidence"],
				model_response_id=result.get("model_response_id"),
			)
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
