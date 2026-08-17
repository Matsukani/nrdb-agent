from .annotator import AnnotationAgent


def run_job(nrdb, job_id, max_items=None, openai_client=None, progress=print):
	bundle = nrdb.job_items(job_id)
	job = bundle["job"]
	items = bundle["items"]
	if max_items is not None:
		items = items[:max(0, int(max_items))]
	agent = AnnotationAgent(nrdb, job["model_name"], client=openai_client, progress=progress)
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
				decision=result["decision"],
				confidence=result["confidence"],
				evidence=result["evidence"],
				model_response_id=result.get("model_response_id"),
			)
			progress("  done")
			completed += 1
		if max_items is None or completed >= len(bundle["items"]):
			nrdb.set_job_status(job_id, "completed")
		return nrdb.summary(job_id)
	except BaseException as error:
		nrdb.set_job_status(job_id, "failed", str(error))
		raise
