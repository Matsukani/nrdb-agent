import os

from .http import JsonHttpClient


class NrdbClient:
	def __init__(self, agent_url=None, morph_url=None, timeout=60):
		self.agent_url = (agent_url or os.environ.get("NRDB_AGENT_URL") or "http://127.0.0.1/php/agent.php").rstrip("/")
		self.evidence_url = os.environ.get("NRDB_AGENT_EVIDENCE_URL") or self.agent_url.rsplit("/", 1)[0] + "/agent_evidence.php"
		self.morph_url = (morph_url or os.environ.get("NRDB_MORPH_URL") or "http://127.0.0.1:8765").rstrip("/")
		self.http = JsonHttpClient(timeout=timeout)

	def _agent_get(self, action, **params):
		payload = self.http.get(self.agent_url, {"action": action, **params})
		if not payload.get("success"):
			raise RuntimeError(payload.get("error") or "NRDB agent API failed")
		return payload

	def _evidence_get(self, action, **params):
		payload = self.http.get(self.evidence_url, {"action": action, **params})
		if not payload.get("success"):
			raise RuntimeError(payload.get("error") or "NRDB agent evidence API failed")
		return payload

	def _agent_post(self, action, payload):
		result = self.http.post(self.agent_url + "?action=" + action, payload)
		if not result.get("success"):
			raise RuntimeError(result.get("error") or "NRDB agent API failed")
		return result

	def create_job(self, dataset_id, mode, limit, model_name, prompt_version="annotation-v1", selection_seed=1, produce_translation=False, blind_translation=False):
		return self._agent_post("create_job", {
			"dataset_id": int(dataset_id), "mode": mode, "limit": int(limit),
			"model_name": model_name, "prompt_version": prompt_version,
			"selection_seed": int(selection_seed),
			"produce_translation": bool(produce_translation or blind_translation),
			"blind_translation": bool(blind_translation),
		})

	def jobs(self):
		return self._agent_get("jobs")["jobs"]

	def job_items(self, job_id):
		return self._agent_get("job_items", job_id=int(job_id))

	def lookup_id(self, label, annotation_schema_id):
		return self._evidence_get("lookup_id", label=label, annotation_schema_id=int(annotation_schema_id))

	def examples(self, label, annotation_schema_id, exclude_sentence_id, limit=12):
		return self._evidence_get(
			"examples", label=label, annotation_schema_id=int(annotation_schema_id),
			exclude_sentence_id=int(exclude_sentence_id), limit=int(limit),
		)

	def morph_analyze(self, text, dialect_id, annotation_schema_id):
		payload = self.http.post(self.morph_url + "/analyze", {
			"text": text,
			"target_dialect_id": int(dialect_id),
			"annotation_schema_id": int(annotation_schema_id),
			"segmentation_mode": "joint",
			"segmentation_top_k": 5,
		})
		if "error" in payload:
			raise RuntimeError(payload["error"])
		return payload

	def validate_analysis(self, text, segmented, annotation):
		return self.http.post(self.morph_url + "/validate-analysis", {
			"text": text, "segmented": segmented, "annotation": annotation,
		})

	def save_result(self, **payload):
		return self._agent_post("save_result", payload)

	def set_job_status(self, job_id, status, error_message=None):
		payload = {"job_id": int(job_id), "status": status}
		if error_message:
			payload["error_message"] = str(error_message)
		return self._agent_post("set_job_status", payload)

	def summary(self, job_id):
		return self._agent_get("summary", job_id=int(job_id))
