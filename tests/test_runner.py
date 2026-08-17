from nrdb_agent.runner import run_job


class FakeNrdb:
	def __init__(self):
		self.statuses = []
		self.results = []

	def job_items(self, job_id):
		return {"job": {"id": job_id, "model_name": "fake", "annotation_schema_id": 2}, "items": []}

	def set_job_status(self, job_id, status, error_message=None):
		self.statuses.append(status)

	def summary(self, job_id):
		return {"success": True, "summary": {"completed": 0}}


def test_empty_job_completes_without_model_call():
	nrdb = FakeNrdb()
	result = run_job(nrdb, 7)
	assert nrdb.statuses == ["running", "completed"]
	assert result["summary"]["completed"] == 0
