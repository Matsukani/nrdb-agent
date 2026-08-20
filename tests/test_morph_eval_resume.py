import json

import pytest

from nrdb_agent.morph_eval_resumable import CHECKPOINT_FORMAT, _load_checkpoint, _verify_checkpoint
from nrdb_agent.task_agent import TaskAwareAnnotationAgent


class DummyNrdb:
	pass


class MalformedFinalAgent(TaskAwareAnnotationAgent):
	def __init__(self):
		self.progress_messages = []
		self.progress = self.progress_messages.append

	def _annotation_phase_v9(self, item, job, morph_result):
		raise ValueError("Expecting value: line 1 column 1 (char 0)")


def test_malformed_final_keeps_baseline():
	agent = MalformedFinalAgent()
	result = agent.annotate(
		{"translation_jp": ""},
		{"semantic_feedback": "none", "produce_translation": False},
		{"segmented": "a-b", "annotation": "A-B"},
	)
	assert result["segmented"] == "a-b"
	assert result["annotation"] == "A-B"
	assert result["decision"] == "uncertain"
	assert result["evidence"]["v9_fallback"]["kept_baseline"] is True


def _meta(**values):
	base = {
		"record_type": "meta",
		"format": CHECKPOINT_FORMAT,
		"morph_run": "/run",
		"train_path": "/run/train.jsonl",
		"datasets": [30],
		"cohort_sentence_ids": [10, 11, 12],
		"limit": 3,
		"seed": 1,
		"agent_model": "gpt-5.6-sol",
		"expected_morph_model": "morph-v1",
		"id_model": "/run/id.json",
		"semantic_feedback": "none",
		"require_semantic_feedback": False,
		"translation_filter": "any",
	}
	base.update(values)
	return base


def test_checkpoint_loader_preserves_completed_rows(tmp_path):
	path = tmp_path / "eval.tsv.checkpoint.jsonl"
	meta = _meta()
	row = {"record_type": "row", "row": {"sentence_id": 10, "agent_cost_usd": 0.2}}
	path.write_text(json.dumps(meta) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
	loaded_meta, rows = _load_checkpoint(path)
	_verify_checkpoint(meta, loaded_meta)
	assert [value["sentence_id"] for value in rows] == [10]


def test_checkpoint_mismatch_refuses_resume():
	expected = _meta(semantic_feedback="none")
	actual = _meta(semantic_feedback="generated")
	with pytest.raises(ValueError, match="semantic_feedback differs"):
		_verify_checkpoint(expected, actual)


def test_checkpoint_translation_filter_is_part_of_experiment_identity():
	expected = _meta(translation_filter="present")
	actual = _meta(translation_filter="any")
	with pytest.raises(ValueError, match="translation_filter differs"):
		_verify_checkpoint(expected, actual)
