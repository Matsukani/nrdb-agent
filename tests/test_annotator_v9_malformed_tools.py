from types import SimpleNamespace

import nrdb_agent.annotator_v9 as annotator_v9
from nrdb_agent.annotator_v9 import AnnotationAgentV9


class FakeAgent(AnnotationAgentV9):
	def __init__(self):
		self.model_name = "gpt-test"
		self.max_rounds = 3
		self.max_evidence_calls = 6
		self.id_critic = None
		self._shared_evidence = {"lookup": {}, "corpus": {}, "form": {}}
		self.messages = []
		self.responses = [
			SimpleNamespace(
				id="r1",
				output=[SimpleNamespace(type="function_call", name="lookup_id", arguments='{"label":"broken', call_id="c1")],
				output_text="",
			),
			SimpleNamespace(
				id="r2",
				output=[],
				output_text='{"segmented":"a","annotation":"A","trsl_ai":"","decision":"uncertain","confidence":0.5,"evidence":{"note":"kept","labels_checked":[],"example_sentence_ids":[]}}',
			),
		]

	def progress(self, message):
		self.messages.append(message)

	def _create_response(self, *args, **kwargs):
		return self.responses.pop(0)

	def _prepare_hotspots(self, morph_result, schema_id):
		return {
			"policy": "test",
			"hotspot_ids": [],
			"uncertain_surfaces": [],
			"uncertainty_reasons": {},
			"id_sequence_review": None,
			"surface_labels": {},
		}


def test_malformed_tool_arguments_are_recoverable(monkeypatch):
	monkeypatch.setattr(annotator_v9, "_response_output_as_input", lambda response: [])
	agent = FakeAgent()
	result = agent._annotation_phase_v9(
		{"sentence_id": 1, "dialect_id": 19, "dialect_region": "宮古", "text": "a"},
		{"annotation_schema_id": 2},
		{"segmented": "a", "annotation": "A", "phrases": []},
	)
	assert result["annotation"] == "A"
	assert result["decision"] == "uncertain"
	assert any("malformed/truncated tool arguments" in message for message in agent.messages)
