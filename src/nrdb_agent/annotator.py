import json
import re

from openai import OpenAI


INSTRUCTIONS = """You are the constrained NRDB morphemic annotation agent.
Your goal is to propose the most defensible segmentation and morphemic-ID annotation for one low-resource-language utterance.

Rules:
- Start from the nrdb-morph analysis supplied in the user message, but it is evidence rather than truth.
- Use lookup_id when an annotation ID's lexical/semantic grounding is unclear.
- Use corpus_examples when attested human-reviewed usage can resolve ambiguity.
- Never invent an annotation ID. Prefer IDs already proposed by nrdb-morph or confirmed by lookup/corpus evidence.
- You may change segmentation only when the evidence strongly supports it.
- Before a PROPOSED result, call validate_analysis on the complete final segmentation and annotation.
- If evidence is insufficient, return decision UNCERTAIN. If no defensible valid analysis exists, return FAILED.
- Do not ask for gold annotation and do not infer that it is available.
- Do not produce chain-of-thought. Evidence should contain only concise, auditable facts: morph rank/labels and retrieved sentence IDs or dictionary forms.

Final response must be one JSON object and no surrounding prose:
{"segmented":"...","annotation":"...","decision":"proposed|uncertain|failed","confidence":0.0,"evidence":{"note":"brief","labels_checked":[],"example_sentence_ids":[]}}
"""

TOOLS = [
	{
		"type": "function", "name": "lookup_id",
		"description": "Look up bilingual dictionary and UniCog grounding for one existing NRDB annotation ID.",
		"parameters": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"], "additionalProperties": False},
		"strict": True,
	},
	{
		"type": "function", "name": "corpus_examples",
		"description": "Retrieve human-validated corpus examples containing one annotation ID, excluding the current sentence.",
		"parameters": {"type": "object", "properties": {"label": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["label", "limit"], "additionalProperties": False},
		"strict": True,
	},
	{
		"type": "function", "name": "validate_analysis",
		"description": "Validate that a complete segmentation and annotation are structurally aligned and legal under nrdb-morph syntax.",
		"parameters": {"type": "object", "properties": {"segmented": {"type": "string"}, "annotation": {"type": "string"}}, "required": ["segmented", "annotation"], "additionalProperties": False},
		"strict": True,
	},
]


def parse_final_json(text):
	text = (text or "").strip()
	if text.startswith("```"):
		text = re.sub(r"^```(?:json)?\s*", "", text)
		text = re.sub(r"\s*```$", "", text)
	payload = json.loads(text)
	decision = str(payload.get("decision") or "").lower()
	if decision not in {"proposed", "uncertain", "failed"}:
		raise ValueError("invalid agent decision")
	confidence = float(payload.get("confidence", 0.0))
	if confidence < 0 or confidence > 1:
		raise ValueError("confidence must be between 0 and 1")
	payload["decision"] = decision
	payload["confidence"] = confidence
	payload["segmented"] = str(payload.get("segmented") or "").strip()
	payload["annotation"] = str(payload.get("annotation") or "").strip()
	payload["evidence"] = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
	return payload


class AnnotationAgent:
	def __init__(self, nrdb, model_name, client=None, max_rounds=10):
		self.nrdb = nrdb
		self.model_name = model_name
		self.client = client or OpenAI()
		self.max_rounds = int(max_rounds)

	def _tool_result(self, name, arguments, item, schema_id):
		if name == "lookup_id":
			return self.nrdb.lookup_id(arguments["label"], schema_id)
		if name == "corpus_examples":
			return self.nrdb.examples(arguments["label"], schema_id, item["sentence_id"], arguments["limit"])
		if name == "validate_analysis":
			return self.nrdb.validate_analysis(item["text"], arguments["segmented"], arguments["annotation"])
		raise ValueError("unknown tool: {}".format(name))

	def annotate(self, item, job, morph_result):
		input_payload = {
			"sentence_id": int(item["sentence_id"]),
			"dialect_id": int(item["dialect_id"]),
			"dialect_region": item.get("dialect_region"),
			"text": item["text"],
			"translation_jp": item.get("translation_jp"),
			"annotation_schema_id": int(job["annotation_schema_id"]),
			"nrdb_morph": morph_result,
		}
		response = self.client.responses.create(
			model=self.model_name,
			instructions=INSTRUCTIONS,
			input=json.dumps(input_payload, ensure_ascii=False),
			tools=TOOLS,
			store=False,
		)
		for _round in range(self.max_rounds):
			calls = [output for output in response.output if getattr(output, "type", None) == "function_call"]
			if not calls:
				result = parse_final_json(response.output_text)
				result["model_response_id"] = response.id
				return result
			outputs = []
			for call in calls:
				arguments = json.loads(call.arguments)
				tool_result = self._tool_result(call.name, arguments, item, int(job["annotation_schema_id"]))
				outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(tool_result, ensure_ascii=False)})
			response = self.client.responses.create(
				model=self.model_name,
				instructions=INSTRUCTIONS,
				previous_response_id=response.id,
				input=outputs,
				tools=TOOLS,
				store=False,
			)
		raise RuntimeError("agent exceeded maximum tool rounds")
