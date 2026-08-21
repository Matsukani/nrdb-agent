import csv
import json
from pathlib import Path

from .annotator import AnnotationAgent, _compact_tool_result, _response_output_as_input
from .reverse_id_critic import IdSequenceCritic
from .surface_critic import SurfaceModelCritic


ASR_REVIEW_FORMAT = {
	"type": "json_schema",
	"name": "nrdb_asr_nbest_selection",
	"strict": True,
	"schema": {
		"type": "object",
		"properties": {
			"selected_rank": {"type": "integer", "minimum": 1},
			"confidence": {"type": "number"},
			"note": {"type": "string"},
		},
		"required": ["selected_rank", "confidence", "note"],
		"additionalProperties": False,
	},
}

ASR_LOOKUP_TOOL = {
	"type": "function",
	"name": "lookup_id",
	"description": "Look up bilingual dictionary and NRDB grounding for one exact annotation ID when lexical identity would materially help choose between ASR hypotheses.",
	"parameters": {
		"type": "object",
		"properties": {"label": {"type": "string", "maxLength": 128}},
		"required": ["label"],
		"additionalProperties": False,
	},
	"strict": True,
}

ASR_REVIEW_INSTRUCTIONS = """You are the blind NRDB ASR n-best linguistic selector.

You receive several COMPLETE acoustic hypotheses for one Miyako utterance. Every hypothesis was produced by the same CTC beam decoder. Your task is ONLY to select one existing hypothesis rank. Never rewrite, splice, repair, normalize, or invent transcription units.

Evidence hierarchy:
- Acoustic/decoder evidence is authoritative evidence that a sequence is licensed by the audio. Do not abandon a much stronger acoustic hypothesis for a merely more frequent linguistic sequence without strong reasons.
- nrdb-morph analyses are first-pass linguistic interpretations, not truth.
- ID-sequence and surface/phonotactic critics are soft statistical evidence trained on human linguistic data. Strong surprises/disagreements matter, but rare valid forms and constructions must survive.
- Prefer a lower-ranked acoustic hypothesis when it is still acoustically plausible and receives materially stronger coherent lexical, morphological, grammatical, and surface evidence.
- Use lookup_id only for a small number of genuinely decisive lexical ambiguities. No corpus-example tools are available in this blind experiment.
- Do not infer or ask for the reference transcription, Japanese translation, oracle rank, error rate, or held-out gold data. They are intentionally absent.
- If linguistic evidence does not clearly justify changing the decoder decision, retain rank 1.
- Return only an existing rank from the candidate list.
- Do not produce chain-of-thought.

Return exactly one JSON object:
{"selected_rank":1,"confidence":0.0,"note":"brief auditable reason"}
"""


def _read_tsv(path):
	with Path(path).open("r", encoding="utf-8", newline="") as handle:
		return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path, rows, fieldnames):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
		writer.writeheader()
		writer.writerows(rows)


def _write_json(path, value):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(value, handle, ensure_ascii=False, indent=2)
		handle.write("\n")


def _split_units(value):
	return str(value or "").strip().split()


def _edit_distance(ref, hyp):
	previous = list(range(len(hyp) + 1))
	for i, ref_token in enumerate(ref, start=1):
		current = [i] + [0] * len(hyp)
		for j, hyp_token in enumerate(hyp, start=1):
			cost = 0 if ref_token == hyp_token else 1
			current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
		previous = current
	return previous[-1]


def _onset_for_display(unit):
	for value in ("tʃ", "dʒ", "ts", "dz", "ʃ", "ʒ"):
		if unit.startswith(value):
			if value in {"ts", "tʃ"}:
				return "t"
			if value in {"dz", "dʒ"}:
				return "d"
			return value
	if not unit or unit[0] in {"a", "i", "u", "e", "o", "ɿ", "ə", "ː"}:
		return ""
	return unit[0]


def compact_from_units(units):
	out = []
	for index, unit in enumerate(units):
		if str(unit).startswith("Q_"):
			next_unit = units[index + 1] if index + 1 < len(units) else ""
			onset = _onset_for_display(str(next_unit))
			out.append(onset or str(unit))
		else:
			out.append(str(unit))
	return "".join(out)


def _load_phrase_boundary_model(path):
	if not path:
		return None
	try:
		from asr_workflow.phrase_boundary import PhraseBoundaryModel
	except ImportError as error:
		raise RuntimeError(
			"--phrase-boundary-model requires the local nrdb-asr package in this environment; run: pip install -e ../asr"
		) from error
	return PhraseBoundaryModel.load(path, force_cpu=True)


def _surface_review_compact(review):
	diagnostics = []
	for value in review.get("diagnostics", []):
		if not value.get("strong_disagreement"):
			continue
		diagnostics.append({
			"label": value.get("label"),
			"previous_surface": value.get("previous_surface"),
			"next_label": value.get("next_label"),
			"generated_form": value.get("generated_form"),
			"score_gap": value.get("score_gap"),
			"suggestions": value.get("suggestions", [])[:3],
		})
	return {
		"valid_alignment": bool(review.get("valid_alignment")),
		"phonotactic_mean_log_probability": review.get("phonotactic_mean_log_probability"),
		"strong_disagreements": int(review.get("strong_disagreements", 0)),
		"diagnostics": diagnostics[:5],
	}


def _candidate_baseline_key(candidate):
	if candidate.get("analysis_error"):
		return (10**6, 10**6, 10**6, candidate["rank"])
	id_review = candidate.get("id_review", {})
	surface_review = candidate.get("surface_review", {})
	penalty = int(id_review.get("strong_surprises", 0)) + int(surface_review.get("strong_disagreements", 0))
	morph_confidence = float(candidate.get("morph_confidence") or 0.0)
	id_mean = float(id_review.get("mean_log_probability") or -100.0)
	phonotactic = float(surface_review.get("phonotactic_mean_log_probability") or -100.0)
	# Deterministic, deliberately untuned baseline: minimize strong linguistic alerts,
	# then prefer stronger morph confidence and better linguistic scores, then ASR rank.
	return (penalty, -morph_confidence, -(id_mean + phonotactic), int(candidate["rank"]))


def select_baseline_candidate(candidates):
	if not candidates:
		raise ValueError("no ASR candidates")
	return min(candidates, key=_candidate_baseline_key)


class AsrNbestSelector(AnnotationAgent):
	def __init__(self, *args, max_lookup_calls=4, **kwargs):
		super().__init__(*args, **kwargs)
		self.max_lookup_calls = int(max_lookup_calls)

	def select(self, candidates, annotation_schema_id, dialect_id, region):
		allowed = {int(value["rank"]) for value in candidates}
		payload = {
			"annotation_schema_id": int(annotation_schema_id),
			"dialect_id": int(dialect_id),
			"region": str(region or ""),
			"candidates": candidates,
		}
		base_input = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
		lookup_calls = 0
		lookup_evidence = []
		self.progress("  asr-agent: select among {} acoustic hypotheses".format(len(candidates)))
		response = self._create_response(
			base_input, ASR_REVIEW_INSTRUCTIONS, tools=[ASR_LOOKUP_TOOL], max_output_tokens=700, text_format=ASR_REVIEW_FORMAT,
		)
		for round_index in range(1, self.max_rounds + 1):
			calls = [value for value in response.output if getattr(value, "type", None) == "function_call"]
			if not calls:
				result = json.loads((response.output_text or "").strip())
				rank = int(result.get("selected_rank", 0))
				if rank not in allowed:
					raise ValueError("ASR reviewer selected unavailable rank {}".format(rank))
				confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
				return {
					"selected_rank": rank,
					"confidence": confidence,
					"note": str(result.get("note") or "").strip(),
					"lookup_evidence": lookup_evidence,
				}
			continuation = list(base_input)
			if lookup_evidence:
				continuation.append({"role": "user", "content": "Previously retrieved dictionary evidence:\n" + json.dumps(lookup_evidence[-4:], ensure_ascii=False)})
			continuation.extend(_response_output_as_input(response))
			self.progress("  asr-agent tool round {}: {} lookup(s)".format(round_index, len(calls)))
			for call in calls:
				arguments = json.loads(call.arguments)
				if call.name != "lookup_id":
					raise ValueError("unsupported ASR review tool: {}".format(call.name))
				if lookup_calls >= self.max_lookup_calls:
					compact = {"budget_exhausted": True, "message": "Dictionary lookup budget exhausted; select conservatively."}
				else:
					result = self.nrdb.lookup_id(arguments["label"], int(annotation_schema_id))
					compact = _compact_tool_result("lookup_id", result)
					lookup_calls += 1
					lookup_evidence.append({"label": arguments["label"], "result": compact})
				continuation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(compact, ensure_ascii=False)})
			response = self._create_response(
				continuation, ASR_REVIEW_INSTRUCTIONS,
				tools=[] if lookup_calls >= self.max_lookup_calls else [ASR_LOOKUP_TOOL],
				max_output_tokens=700, text_format=ASR_REVIEW_FORMAT,
			)
		raise RuntimeError("ASR reviewer exceeded maximum tool rounds")


def _analyze_candidate(nrdb, hypothesis, dialect_id, annotation_schema_id, region, id_critic, surface_critic, phrase_boundary_model=None):
	rank = int(hypothesis.get("rank", 0))
	units = hypothesis.get("units")
	if not isinstance(units, list):
		units = _split_units(hypothesis.get("unit_text"))
	units = [str(value) for value in units]
	compact = compact_from_units(units)
	spaced = compact
	if phrase_boundary_model is not None and compact:
		spaced = str(phrase_boundary_model.predict(compact, int(dialect_id))["phrased"])
	candidate = {
		"rank": rank,
		"units": units,
		"unit_text": " ".join(units),
		"compact": compact,
		"spaced": spaced,
		"combined_score": hypothesis.get("combined_score"),
		"acoustic_score": hypothesis.get("acoustic_score"),
		"lm_contribution": hypothesis.get("lm_contribution"),
		"insertion_contribution": hypothesis.get("insertion_contribution"),
		"score_delta": hypothesis.get("score_delta"),
	}
	try:
		morph = nrdb.morph_analyze(spaced, int(dialect_id), int(annotation_schema_id))
		candidate["morph_segmented"] = str(morph.get("segmented") or "")
		candidate["morph_annotation"] = str(morph.get("annotation") or "")
		candidate["morph_confidence"] = morph.get("confidence")
		if id_critic is not None and candidate["morph_annotation"]:
			candidate["id_review"] = id_critic.compact(id_critic.review(candidate["morph_annotation"], int(annotation_schema_id)))
		else:
			candidate["id_review"] = {"strong_surprises": 0, "mean_log_probability": None, "representations": {}}
		if surface_critic is not None and candidate["morph_segmented"] and candidate["morph_annotation"]:
			review = surface_critic.review(candidate["morph_segmented"], candidate["morph_annotation"], [int(dialect_id)], int(annotation_schema_id))
			candidate["surface_review"] = _surface_review_compact(review)
		else:
			candidate["surface_review"] = {"strong_disagreements": 0, "phonotactic_mean_log_probability": None, "diagnostics": []}
	except Exception as error:
		candidate["analysis_error"] = str(error)
		candidate["morph_segmented"] = ""
		candidate["morph_annotation"] = ""
		candidate["morph_confidence"] = None
		candidate["id_review"] = {"strong_surprises": 999, "mean_log_probability": None, "representations": {}}
		candidate["surface_review"] = {"strong_disagreements": 999, "phonotactic_mean_log_probability": None, "diagnostics": []}
	return candidate


def _private_metrics(row, hypotheses, baseline_rank, agent_rank):
	ref = _split_units(row.get("ref_units"))
	by_rank = {int(value.get("rank", 0)): [str(unit) for unit in value.get("units", [])] for value in hypotheses}
	if not by_rank:
		return None
	distances = {rank: _edit_distance(ref, units) for rank, units in by_rank.items()}
	top1_rank = min(by_rank)
	oracle_rank = min(distances, key=lambda rank: (distances[rank], rank))
	return {
		"ref_units": len(ref),
		"top1_rank": top1_rank,
		"top1_edits": distances[top1_rank],
		"baseline_rank": int(baseline_rank),
		"baseline_edits": distances[int(baseline_rank)],
		"agent_rank": int(agent_rank),
		"agent_edits": distances[int(agent_rank)],
		"oracle_rank": int(oracle_rank),
		"oracle_edits": distances[oracle_rank],
	}


def review_asr_predictions(
	nrdb,
	predictions_path,
	out_dir,
	annotation_schema_id,
	region,
	dialect_id,
	*,
	model_name="gpt-5.6",
	id_model_path=None,
	surface_model_path=None,
	phrase_boundary_model_path=None,
	limit=None,
	max_candidates=None,
	use_llm=True,
	openai_client=None,
	progress=print,
):
	id_critic = IdSequenceCritic(id_model_path) if id_model_path else None
	surface_critic = SurfaceModelCritic(surface_model_path) if surface_model_path else None
	phrase_boundary_model = _load_phrase_boundary_model(phrase_boundary_model_path)
	selector = AsrNbestSelector(nrdb, model_name, client=openai_client, progress=progress) if use_llm else None

	rows = _read_tsv(predictions_path)
	if limit is not None:
		rows = rows[:max(0, int(limit))]
	out_rows = []
	totals = {"ref": 0, "top1": 0, "baseline": 0, "agent": 0, "oracle": 0}
	counts = {"rank1_retained": 0, "agent_changed": 0, "improved": 0, "harmful": 0, "neutral": 0, "oracle_selected": 0, "failed_rows": 0}

	for index, row in enumerate(rows, start=1):
		row_id = row.get("row_id") or "row_{}".format(index)
		progress("[{}/{}] ASR {}".format(index, len(rows), row_id))
		try:
			hypotheses = json.loads(row.get("nbest_json") or "[]")
		except json.JSONDecodeError as error:
			raise ValueError("invalid nbest_json for {}: {}".format(row_id, error))
		if max_candidates is not None:
			hypotheses = hypotheses[:max(1, int(max_candidates))]
		if not hypotheses:
			progress("  no n-best hypotheses; skipped")
			counts["failed_rows"] += 1
			continue

		candidates = []
		for hypothesis in hypotheses:
			candidate = _analyze_candidate(
				nrdb, hypothesis, dialect_id, annotation_schema_id, region,
				id_critic, surface_critic, phrase_boundary_model,
			)
			candidates.append(candidate)
			progress("  rank {} delta={} IDs={!r} id_surprises={} surface_disagreements={}".format(
				candidate["rank"], candidate.get("score_delta"), candidate.get("morph_annotation", ""),
				candidate.get("id_review", {}).get("strong_surprises", 0),
				candidate.get("surface_review", {}).get("strong_disagreements", 0),
			))

		baseline = select_baseline_candidate(candidates)
		progress("  baseline selector: rank {}".format(baseline["rank"]))
		if selector is not None:
			selection = selector.select(candidates, annotation_schema_id, dialect_id, region)
			agent_rank = int(selection["selected_rank"])
			progress("  asr-agent: selected rank {} confidence={:.3f}".format(agent_rank, selection["confidence"]))
		else:
			selection = {"selected_rank": int(baseline["rank"]), "confidence": None, "note": "LLM disabled; deterministic baseline used", "lookup_evidence": []}
			agent_rank = int(baseline["rank"])

		metrics = _private_metrics(row, hypotheses, baseline["rank"], agent_rank)
		if metrics is None:
			counts["failed_rows"] += 1
			continue
		for key in ("ref", "top1", "baseline", "agent", "oracle"):
			metric_key = "ref_units" if key == "ref" else key + "_edits"
			totals[key] += int(metrics[metric_key])
		if agent_rank == metrics["top1_rank"]:
			counts["rank1_retained"] += 1
		else:
			counts["agent_changed"] += 1
		if metrics["agent_edits"] < metrics["top1_edits"]:
			counts["improved"] += 1
		elif metrics["agent_edits"] > metrics["top1_edits"]:
			counts["harmful"] += 1
		else:
			counts["neutral"] += 1
		if agent_rank == metrics["oracle_rank"]:
			counts["oracle_selected"] += 1

		by_rank = {int(value["rank"]): value for value in candidates}
		selected = by_rank[agent_rank]
		out_rows.append({
			"row_id": row_id,
			"top1_rank": metrics["top1_rank"],
			"baseline_rank": metrics["baseline_rank"],
			"agent_rank": agent_rank,
			"oracle_rank": metrics["oracle_rank"],
			"ref_unit_count": metrics["ref_units"],
			"top1_edits": metrics["top1_edits"],
			"baseline_edits": metrics["baseline_edits"],
			"agent_edits": metrics["agent_edits"],
			"oracle_edits": metrics["oracle_edits"],
			"agent_confidence": selection.get("confidence"),
			"agent_note": selection.get("note"),
			"selected_units": selected.get("unit_text"),
			"selected_compact": selected.get("compact"),
			"selected_spaced": selected.get("spaced"),
			"selected_segmented": selected.get("morph_segmented"),
			"selected_annotation": selected.get("morph_annotation"),
			"candidate_json": json.dumps(candidates, ensure_ascii=False, separators=(",", ":")),
		})

	ref_total = totals["ref"]
	def rate(edits):
		return edits / ref_total if ref_total else 0.0
	result_count = len(out_rows)
	top1_uer = rate(totals["top1"])
	baseline_uer = rate(totals["baseline"])
	agent_uer = rate(totals["agent"])
	oracle_uer = rate(totals["oracle"])
	headroom = top1_uer - oracle_uer
	headroom_recovered = None if headroom <= 0 else (top1_uer - agent_uer) / headroom
	baseline_headroom_recovered = None if headroom <= 0 else (top1_uer - baseline_uer) / headroom
	summary = {
		"format": "nrdb-agent.asr-review.v1",
		"predictions": str(Path(predictions_path).resolve()),
		"rows_requested": len(rows),
		"rows_scored": result_count,
		"annotation_schema_id": int(annotation_schema_id),
		"region": str(region),
		"dialect_id": int(dialect_id),
		"model_name": model_name if use_llm else None,
		"id_model_path": str(id_model_path or ""),
		"surface_model_path": str(surface_model_path or ""),
		"phrase_boundary_model_path": str(phrase_boundary_model_path or ""),
		"top1_UER": top1_uer,
		"baseline_UER": baseline_uer,
		"agent_UER": agent_uer,
		"oracle_UER": oracle_uer,
		"available_headroom": headroom,
		"baseline_headroom_recovered": baseline_headroom_recovered,
		"agent_headroom_recovered": headroom_recovered,
		"rank1_retained": counts["rank1_retained"],
		"agent_changed": counts["agent_changed"],
		"improved_rows": counts["improved"],
		"harmful_rows": counts["harmful"],
		"neutral_rows": counts["neutral"],
		"oracle_rank_selected_rows": counts["oracle_selected"],
		"failed_rows": counts["failed_rows"],
	}
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	_write_json(out_dir / "summary.json", summary)
	_write_tsv(out_dir / "asr_review.tsv", out_rows, [
		"row_id", "top1_rank", "baseline_rank", "agent_rank", "oracle_rank", "ref_unit_count",
		"top1_edits", "baseline_edits", "agent_edits", "oracle_edits", "agent_confidence", "agent_note",
		"selected_units", "selected_compact", "selected_spaced", "selected_segmented", "selected_annotation", "candidate_json",
	])
	return summary
