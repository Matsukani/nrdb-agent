import os

from .http import JsonHttpClient


class NrdbClient:
	def __init__(self, agent_url=None, morph_url=None, timeout=60):
		self.agent_url = (agent_url or os.environ.get("NRDB_AGENT_URL") or "http://127.0.0.1/php/agent.php").rstrip("/")
		base_url = self.agent_url.rsplit("/", 1)[0]
		self.workflow_url = os.environ.get("NRDB_AGENT_WORKFLOW_URL") or base_url + "/agent_workflow.php"
		self.evidence_url = os.environ.get("NRDB_AGENT_EVIDENCE_URL") or base_url + "/agent_evidence.php"
		self.constructions_url = os.environ.get("NRDB_AGENT_CONSTRUCTIONS_URL") or base_url + "/agent_constructions.php"
		self.licensed_forms_url = os.environ.get("NRDB_AGENT_LICENSED_FORMS_URL") or base_url + "/agent_licensed_forms.php"
		self.results_url = os.environ.get("NRDB_AGENT_RESULTS_URL") or base_url + "/agent_results.php"
		self.morph_eval_url = os.environ.get("NRDB_AGENT_MORPH_EVAL_URL") or base_url + "/agent_morph_eval.php"
		self.form_support_url = os.environ.get("NRDB_AGENT_FORM_SUPPORT_URL") or base_url + "/agent_form_support.php"
		self.reverse_evidence_url = os.environ.get("NRDB_AGENT_REVERSE_EVIDENCE_URL") or base_url + "/agent_reverse_evidence.php"
		self.region_dialects_url = os.environ.get("NRDB_AGENT_REGION_DIALECTS_URL") or base_url + "/agent_region_dialects.php"
		self.morph_url = (morph_url or os.environ.get("NRDB_MORPH_URL") or "http://127.0.0.1:8765").rstrip("/")
		self.http = JsonHttpClient(timeout=timeout)
		self.exclude_job_id = 0
		self.evidence_exclusion = {"datasets": [], "texts": [], "sentence_ranges": []}

	def set_evidence_exclusion(self, datasets=None, texts=None, sentence_ranges=None):
		self.evidence_exclusion = {
			"datasets": sorted({int(value) for value in (datasets or []) if int(value) > 0}),
			"texts": sorted({(int(dataset_id), int(text_id)) for dataset_id, text_id in (texts or []) if int(dataset_id) > 0 and int(text_id) > 0}),
			"sentence_ranges": sorted({(int(dataset_id), int(start), int(end)) for dataset_id, start, end in (sentence_ranges or []) if int(dataset_id) > 0 and int(start) > 0 and int(end) >= int(start)}),
		}
		return dict(self.evidence_exclusion)

	def _evidence_exclusion_params(self):
		scope = self.evidence_exclusion or {}
		return {
			"exclude_dataset_ids": ",".join(str(value) for value in scope.get("datasets", [])),
			"exclude_texts": ",".join("{}:{}".format(dataset_id, text_id) for dataset_id, text_id in scope.get("texts", [])),
			"exclude_sentence_ranges": ",".join("{}:{}:{}".format(dataset_id, start, end) for dataset_id, start, end in scope.get("sentence_ranges", [])),
		}

	def _agent_get(self, action, **params):
		payload = self.http.get(self.agent_url, {"action": action, **params})
		if not payload.get("success"):
			raise RuntimeError(payload.get("error") or "NRDB agent API failed")
		return payload

	def _workflow_get(self, action, **params):
		payload = self.http.get(self.workflow_url, {"action": action, **params})
		if not payload.get("success"):
			raise RuntimeError(payload.get("error") or "NRDB agent workflow API failed")
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

	def _workflow_post(self, action, payload):
		result = self.http.post(self.workflow_url + "?action=" + action, payload)
		if not result.get("success"):
			raise RuntimeError(result.get("error") or "NRDB agent workflow API failed")
		return result

	def create_job(self, dataset_id, mode, limit, model_name, prompt_version="annotation-v1", selection_seed=1, produce_translation=False, blind_translation=False):
		return self._agent_post("create_job", {
			"dataset_id": int(dataset_id), "mode": mode, "limit": int(limit),
			"model_name": model_name, "prompt_version": prompt_version,
			"selection_seed": int(selection_seed),
			"produce_translation": bool(produce_translation or blind_translation),
			"blind_translation": bool(blind_translation),
		})

	def create_workflow_job(self, dataset_id, task, limit, model_name, selection_seed=1,
		semantic_feedback="none", require_semantic_feedback=False, use_constructions=False,
		morphology_source="predict", needs_filter="any", scope_text_id=None,
		scope_sentence_start=None, scope_sentence_end=None, translation_evidence=None):
		payload = {
			"dataset_id": int(dataset_id), "task": str(task), "limit": int(limit),
			"model_name": str(model_name), "selection_seed": int(selection_seed),
			"semantic_feedback": str(semantic_feedback),
			"require_semantic_feedback": bool(require_semantic_feedback),
			"use_constructions": bool(use_constructions),
			"morphology_source": str(morphology_source), "needs_filter": str(needs_filter),
			"scope_text_id": scope_text_id, "scope_sentence_start": scope_sentence_start,
			"scope_sentence_end": scope_sentence_end,
		}
		if translation_evidence is not None:
			payload["translation_evidence"] = str(translation_evidence)
		return self._workflow_post("create_job", payload)

	def jobs(self):
		return self._agent_get("jobs")["jobs"]

	def job_items(self, job_id):
		return self._agent_get("job_items", job_id=int(job_id))

	def workflow_job_items(self, job_id):
		return self._workflow_get("job_items", job_id=int(job_id))

	def job_results(self, job_id):
		payload = self.http.get(self.results_url, {"job_id": int(job_id)})
		if not payload.get("success"):
			raise RuntimeError(payload.get("error") or "NRDB agent results API failed")
		return payload

	def morph_eval_rows(self, dataset_ids, page_size=500, text_internal_id=None):
		dataset_ids = sorted({int(value) for value in dataset_ids})
		if not dataset_ids:
			return []
		if text_internal_id is not None and len(dataset_ids) != 1:
			raise ValueError("text_internal_id requires exactly one dataset ID")
		rows = []
		after_id = 0
		while True:
			params = {
				"dataset_ids": ",".join(str(value) for value in dataset_ids),
				"after_id": int(after_id), "limit": int(page_size),
			}
			if text_internal_id is not None:
				params["text_internal_id"] = int(text_internal_id)
			payload = self.http.get(self.morph_eval_url, params)
			if not payload.get("success"):
				raise RuntimeError(payload.get("error") or "NRDB morph evaluation API failed")
			batch = payload.get("rows", [])
			rows.extend(batch)
			next_after_id = payload.get("next_after_id")
			if next_after_id in (None, ""):
				break
			after_id = int(next_after_id)
		return rows

	def region_dialects(self, region, annotation_schema_id):
		payload = self.http.get(self.region_dialects_url, {
			"region": str(region or "").strip(),
			"annotation_schema_id": int(annotation_schema_id),
		})
		if not payload.get("success"):
			raise RuntimeError(payload.get("error") or "NRDB region dialect lookup failed")
		return payload.get("dialects", [])

	def lookup_id(self, label, annotation_schema_id):
		return self._evidence_get("lookup_id", label=label, annotation_schema_id=int(annotation_schema_id))

	def construction_candidates(self, annotation, annotation_schema_id, region=None, dialect_id=None):
		payload = self.http.get(self.constructions_url, {
			"annotation": str(annotation or "").strip(),
			"annotation_schema_id": int(annotation_schema_id),
			"region": str(region or "").strip(),
			"dialect_id": "" if dialect_id in (None, "") else int(dialect_id),
		})
		if not payload.get("success"):
			raise RuntimeError(payload.get("error") or "NRDB construction lookup failed")
		return payload

	def licensed_forms_in_text(self, text, annotation_schema_id, region, dialect_id):
		payload = self.http.get(self.licensed_forms_url, {
			"text": str(text or "").strip(),
			"annotation_schema_id": int(annotation_schema_id),
			"region": str(region or "").strip(),
			"dialect_id": int(dialect_id),
		})
		if not payload.get("success"):
			raise RuntimeError(payload.get("error") or "NRDB licensed-form lookup failed")
		return payload

	def examples(self, label, annotation_schema_id, exclude_sentence_id, limit=12, exclude_job_id=None):
		label = str(label or "").strip()
		segment_count = len(label.split("-")) if label else 0
		if len(label) > 256 or segment_count > 8:
			reason = "corpus_examples query rejected locally: use a shorter construction with at most 8 hyphen-separated segments and at most 256 characters"
			return {"success": True, "label": "{} [QUERY_REJECTED: {}]".format(label[:160], reason), "examples": [], "query_rejected": True, "error": reason, "segment_count": segment_count}
		if exclude_job_id is None:
			exclude_job_id = self.exclude_job_id
		params = {
			"label": label, "annotation_schema_id": int(annotation_schema_id),
			"exclude_sentence_id": int(exclude_sentence_id), "exclude_job_id": int(exclude_job_id or 0),
			"limit": int(limit), **self._evidence_exclusion_params(),
		}
		return self._evidence_get("examples", **params)

	def search_japanese_evidence(self, query, annotation_schema_id, exclude_sentence_id, exclude_job_id=None, region=None, dialect_ids=None, limit=8):
		if exclude_job_id is None:
			exclude_job_id = self.exclude_job_id
		dialect_ids = dialect_ids or []
		payload = self.http.get(self.reverse_evidence_url, {
			"q": str(query or "").strip(), "annotation_schema_id": int(annotation_schema_id),
			"exclude_sentence_id": int(exclude_sentence_id), "exclude_job_id": int(exclude_job_id or 0),
			"region": str(region or "").strip(),
			"dialect_ids": ",".join(str(int(value)) for value in dialect_ids),
			"limit": int(limit),
		})
		if not payload.get("success"):
			raise RuntimeError(payload.get("error") or "NRDB reverse evidence API failed")
		return payload

	def surface_forms_for_id(self, label, annotation_schema_id, dialect_ids, region=None, exclude_job_id=None):
		if exclude_job_id is None:
			exclude_job_id = self.exclude_job_id
		payload = self.http.get(self.reverse_evidence_url, {
			"action": "surface_forms",
			"label": str(label or "").strip(),
			"annotation_schema_id": int(annotation_schema_id),
			"dialect_ids": ",".join(str(int(value)) for value in dialect_ids),
			"region": str(region or "").strip(),
			"exclude_job_id": int(exclude_job_id or 0),
		})
		if not payload.get("success"):
			raise RuntimeError(payload.get("error") or "NRDB reverse surface evidence failed")
		return payload

	def form_id_support(self, surface, candidate_id, region, annotation_schema_id, exclude_sentence_id=0):
		params = {
			"surface": surface, "candidate_id": candidate_id, "region": region,
			"annotation_schema_id": int(annotation_schema_id),
			"exclude_sentence_id": int(exclude_sentence_id or 0),
			**self._evidence_exclusion_params(),
		}
		payload = self.http.get(self.form_support_url, params)
		if not payload.get("success"):
			raise RuntimeError(payload.get("error") or "NRDB form-ID support API failed")
		return payload

	def morph_analyze(self, text, dialect_id, annotation_schema_id):
		payload = self.http.post(self.morph_url + "/analyze", {"text": text, "target_dialect_id": int(dialect_id), "annotation_schema_id": int(annotation_schema_id), "segmentation_mode": "joint", "segmentation_top_k": 5})
		if payload.get("http_status", 200) >= 500:
			raise RuntimeError(payload.get("error") or "nrdb-morph analyze failed")
		if "error" in payload:
			raise RuntimeError(payload["error"])
		return payload

	def validate_analysis(self, text, segmented, annotation):
		payload = self.http.post(self.morph_url + "/validate-analysis", {"text": text, "segmented": segmented, "annotation": annotation})
		status = int(payload.get("http_status", 200))
		if status >= 500:
			raise RuntimeError(payload.get("error") or "nrdb-morph validation failed")
		if status >= 400:
			return {"valid": False, "error": payload.get("error") or "analysis rejected by nrdb-morph", "http_status": status}
		return payload

	def save_result(self, **payload):
		return self._agent_post("save_result", payload)

	def set_job_status(self, job_id, status, error_message=None):
		payload = {"job_id": int(job_id), "status": status}
		if error_message:
			payload["error_message"] = str(error_message)
		return self._agent_post("set_job_status", payload)

	def summary(self, job_id):
		return self._agent_get("summary", job_id=int(job_id))
