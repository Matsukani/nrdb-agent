from io import StringIO
from contextlib import redirect_stdout

from nrdb_agent.cli import _print_jobs, _select_jobs


def _jobs():
	return [
		{"id": 3, "status": "completed", "dataset_id": 30, "dataset_name": "minaai_sen", "task": "translate", "model_name": "gpt-5.6-terra", "item_limit": 200, "selection_seed": 1, "created_at": "2026-08-20 14:00:00"},
		{"id": 1, "status": "completed", "dataset_id": 21, "dataset_name": "minaai_danwa", "task": "morph", "model_name": "gpt-5.6-sol", "item_limit": 30, "selection_seed": 1, "created_at": "2026-08-20 12:00:00"},
		{"id": 2, "status": "failed", "dataset_id": 21, "dataset_name": "minaai_danwa", "task": "morph", "model_name": "gpt-5.6-terra", "item_limit": 20, "selection_seed": 2, "created_at": "2026-08-20 13:00:00"},
	]


def test_jobs_are_oldest_to_newest_by_default():
	assert [job["id"] for job in _select_jobs(_jobs())] == [1, 2, 3]


def test_latest_slice_is_still_printed_oldest_to_newest():
	assert [job["id"] for job in _select_jobs(_jobs(), latest=2)] == [2, 3]
	assert [job["id"] for job in _select_jobs(_jobs(), latest=1)] == [3]


def test_short_jobs_are_one_line_each():
	stream = StringIO()
	with redirect_stdout(stream):
		_print_jobs(_select_jobs(_jobs()), short=True)
	lines = stream.getvalue().splitlines()
	assert len(lines) == 3
	assert lines[0].startswith("#1")
	assert lines[-1].startswith("#3")
	assert "translate" in lines[-1]
