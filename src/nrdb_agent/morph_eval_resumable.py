import json
import os
import random
import hashlib
from pathlib import Path

from .blindness import normalize_blind_policy, normalize_evidence_scope
from .dataset_io import write_json, write_tsv
from .licensed_agent import LicensedTaskAwareAnnotationAgent
from .metrics import annotation_metrics, segmentation_metrics
from .morph_eval import _paired_summary, _result_row, _run_contract
from .policy import forward_morph_policy
from .task_agent import SEMANTIC_FEEDBACK_MODES, TaskAwareAnnotationAgent
from .usage import UsageTracker, tracked_client


CHECKPOINT_FORMAT = "nrdb-agent.morph-ceiling-checkpoint.v7"
COHORT_FORMAT = "nrdb-agent.morph-eval-cohort.v1"
TRANSLATION_FILTERS = {"any", "present", "absent"}


def _canonical_path(value):
	if value in (None, ""):
		return None
	return str(Path(value).expanduser().resolve())


def _checkpoint_path(output=None, checkpoint=None):
	if checkpoint:
		return Path(checkpoint)
	if output:
		return Path(str(output) + ".checkpoint.jsonl")
	return Path(".nrdb-agent-morph-eval.checkpoint.jsonl")


def _append_jsonl(path, value):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
		handle.flush()
		os.fsync(handle.fileno())


def _load_checkpoint(path):
	path = Path(path)
	if not path.is_file():
		return None, []
	meta = None
	rows = []
	with path.open("r", encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				value = json.loads(line)
			except json.JSONDecodeError as error:
				if line_number > 1:
					break
				raise ValueError("invalid checkpoint metadata JSON") from error
			if value.get("record_type") == "meta":
				if meta is not None:
					raise ValueError("checkpoint contains duplicate metadata records")
				meta = value
			elif value.get("record_type") == "row":
				row = value.get("row")
				if isinstance(row, dict):
					rows.append(row)
	return meta, rows


def _normalize_evidence_scope(datasets=None, texts=None, sentence_ranges=None, auto_text=None):
	return normalize_evidence_scope(datasets, texts, sentence_ranges, auto_text)


def _build_cohort(nrdb, contract, dataset_ids, limit, seed, semantic_feedback,
	require_semantic_feedback, translation_filter, text_internal_id=None):
	run_dataset_ids = set(contract["dataset_ids"])
	if dataset_ids:
		selected_dataset_ids = {int(value) for value in dataset_ids}
		unknown = sorted(selected_dataset_ids - run_dataset_ids)
		if unknown:
			raise ValueError("dataset IDs are not represented in train.jsonl: {}".format(", ".join(map(str, unknown))))
	else:
		selected_dataset_ids = run_dataset_ids
	if text_internal_id is not None and len(selected_dataset_ids) != 1:
		raise ValueError("text_internal_id requires exactly one selected dataset")
	registered = nrdb.morph_eval_rows(selected_dataset_ids, text_internal_id=text_internal_id)
	eligible = []
	unmatched_identity = 0
	for row in registered:
		example_id = str(row.get("example_id") or "").strip()
		if not example_id:
			unmatched_identity += 1
			continue
		identity = (int(row["dataset_id"]), example_id)
		if identity in contract["train_identities"]:
			continue
		has_translation = bool(str(row.get("translation_jp") or "").strip())
		if translation_filter == "present" and not has_translation:
			continue
		if translation_filter == "absent" and has_translation:
			continue
		if semantic_feedback == "existing" and require_semantic_feedback and not has_translation:
			continue
		eligible.append(row)
	eligible_pool_size = len(eligible)
	if text_internal_id is None:
		rng = random.Random(int(seed))
		rng.shuffle(eligible)
	if limit is not None:
		eligible = eligible[:max(0, int(limit))]
	if not eligible:
		raise ValueError("no eligible registered gold rows remain after training and translation filters")
	return {
		"rows": eligible,
		"dataset_ids": sorted(selected_dataset_ids),
		"registered_gold_rows": len(registered),
		"eligible_pool_size": eligible_pool_size,
		"unmatched_identity": unmatched_identity,
		"text_internal_id": None if text_internal_id is None else int(text_internal_id),
	}


def _cohort_fingerprint(rows):
	identity = [[int(row["dataset_id"]), str(row.get("example_id") or ""), int(row["sentence_id"])] for row in rows]
	return hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_frozen_cohort(path, contract):
	payload = json.loads(Path(path).read_text(encoding="utf-8"))
	if payload.get("format") != COHORT_FORMAT or not isinstance(payload.get("rows"), list):
		raise ValueError("invalid frozen morph-eval cohort: {}".format(path))
	if payload.get("train_path") != _canonical_path(contract["train_path"]):
		raise ValueError("frozen cohort belongs to a different morph training split")
	rows = payload["rows"]
	if payload.get("fingerprint") != _cohort_fingerprint(rows):
		raise ValueError("frozen cohort fingerprint mismatch")
	return {
		"rows": rows,
		"dataset_ids": sorted({int(row["dataset_id"]) for row in rows}),
		"registered_gold_rows": int(payload.get("registered_gold_rows") or len(rows)),
		"eligible_pool_size": int(payload.get("eligible_pool_size") or len(rows)),
		"unmatched_identity": int(payload.get("unmatched_identity") or 0),
		"text_internal_id": payload.get("text_internal_id"),
		"frozen_cohort": _canonical_path(path),
	}


def _write_frozen_cohort(path, contract, cohort):
	payload = {
		"format": COHORT_FORMAT, "morph_run": _canonical_path(contract["run_dir"]),
		"train_path": _canonical_path(contract["train_path"]), "dataset_ids": cohort["dataset_ids"],
		"text_internal_id": cohort.get("text_internal_id"),
		"registered_gold_rows": cohort["registered_gold_rows"], "eligible_pool_size": cohort["eligible_pool_size"],
		"unmatched_identity": cohort["unmatched_identity"], "rows": cohort["rows"],
		"fingerprint": _cohort_fingerprint(cohort["rows"]),
	}
	write_json(path, payload)
	return payload


def _checkpoint_meta(contract, cohort, model_name, limit, seed, expected_morph_model, policy,
	semantic_feedback, require_semantic_feedback, translation_filter, evidence_exclusion,
	use_licensed_forms, blind_policy):
	return {
		"record_type": "meta",
		"format": CHECKPOINT_FORMAT,
		"morph_run": _canonical_path(contract["run_dir"]),
		"train_path": _canonical_path(contract["train_path"]),
		"datasets": cohort["dataset_ids"],
		"text_internal_id": cohort.get("text_internal_id"),
		"cohort_sentence_ids": [int(row["sentence_id"]) for row in cohort["rows"]],
		"limit": None if limit is None else int(limit),
		"seed": int(seed),
		"agent_model": str(model_name),
		"expected_morph_model": str(expected_morph_model or ""),
		"forward_morph_policy": policy.manifest(),
		"cohort_fingerprint": _cohort_fingerprint(cohort["rows"]),
		"semantic_feedback": str(semantic_feedback),
		"require_semantic_feedback": bool(require_semantic_feedback),
		"translation_filter": str(translation_filter),
		"blind_policy": str(blind_policy),
		"evidence_exclusion": evidence_exclusion,
		"use_licensed_forms": bool(use_licensed_forms),
	}


def _verify_checkpoint(expected, actual):
	if not actual:
		raise ValueError("checkpoint has no metadata record")
	for key in (
		"format", "morph_run", "train_path", "datasets", "text_internal_id", "cohort_sentence_ids",
		"limit", "seed", "agent_model", "expected_morph_model", "forward_morph_policy", "cohort_fingerprint",
		"semantic_feedback", "require_semantic_feedback", "translation_filter", "evidence_exclusion",
		"blind_policy",
	):
		if actual.get(key) != expected.get(key):
			raise ValueError("checkpoint does not match this evaluation: {} differs".format(key))
	if bool(actual.get("use_licensed_forms", False)) != bool(expected.get("use_licensed_forms", False)):
		raise ValueError("checkpoint does not match this evaluation: use_licensed_forms differs")


def evaluate_morph_agent_resumable(nrdb, run_dir, model_name="gpt-5.6", limit=None, seed=1,
	dataset_ids=None, expected_morph_model=None, id_model=None, surface_model=None,
	morph_review="agent", resegmentation=False, max_segmentation_candidates=4,
	output=None, checkpoint=None, cohort_in=None, cohort_out=None,
	resume=False, semantic_feedback="none", require_semantic_feedback=False,
	translation_filter="any", text_internal_id=None, evidence_exclude_datasets=None,
	evidence_exclude_texts=None, evidence_exclude_sentence_ranges=None, use_licensed_forms=False,
	blind_policy="row",
	openai_client=None, progress=print):
	semantic_feedback = str(semantic_feedback or "none")
	translation_filter = str(translation_filter or "any")
	use_licensed_forms = bool(use_licensed_forms)
	blind_policy = normalize_blind_policy(blind_policy)
	if semantic_feedback not in SEMANTIC_FEEDBACK_MODES:
		raise ValueError("invalid semantic_feedback: {}".format(semantic_feedback))
	if translation_filter not in TRANSLATION_FILTERS:
		raise ValueError("invalid translation_filter: {}".format(translation_filter))
	if require_semantic_feedback and semantic_feedback == "none":
		raise ValueError("require_semantic_feedback cannot be used with semantic_feedback=none")
	policy = forward_morph_policy(
		review=morph_review, resegmentation=resegmentation, id_model=id_model, surface_model=surface_model,
		max_segmentation_candidates=max_segmentation_candidates, morphology_source="predict", task="morph",
	)
	if not policy.agent_review and (semantic_feedback != "none" or use_licensed_forms):
		raise ValueError("semantic feedback and licensed evidence require --morph-review agent")
	if text_internal_id is not None:
		text_internal_id = int(text_internal_id)
		if text_internal_id < 1:
			raise ValueError("text_internal_id must be positive")
	contract = _run_contract(run_dir)
	cohort = _load_frozen_cohort(cohort_in, contract) if cohort_in else _build_cohort(
		nrdb, contract, dataset_ids, limit, seed, semantic_feedback,
		require_semantic_feedback, translation_filter, text_internal_id=text_internal_id,
	)
	if cohort_out:
		_write_frozen_cohort(cohort_out, contract, cohort)
	auto_text = None
	if text_internal_id is not None:
		auto_text = (cohort["dataset_ids"][0], text_internal_id)
	evidence_exclusion = normalize_evidence_scope(
		datasets=evidence_exclude_datasets,
		texts=evidence_exclude_texts,
		sentence_ranges=evidence_exclude_sentence_ranges,
		auto_text=auto_text,
		blind_policy=blind_policy,
		cohort_rows=cohort["rows"],
	)
	nrdb.set_evidence_exclusion(
		datasets=evidence_exclusion["datasets"],
		texts=[tuple(value) for value in evidence_exclusion["texts"]],
		sentence_ranges=[tuple(value) for value in evidence_exclusion["sentence_ranges"]],
	)
	progress("blind policy: {}".format(blind_policy))
	progress("evidence exclusion: datasets={} texts={} sentence_ranges={}".format(
		evidence_exclusion["datasets"], evidence_exclusion["texts"], evidence_exclusion["sentence_ranges"],
	))
	progress("licensed forms: {}".format("on" if use_licensed_forms else "off"))
	progress("forward morphology policy: {}".format(policy.json()))
	checkpoint_path = _checkpoint_path(output=output, checkpoint=checkpoint)
	expected_meta = _checkpoint_meta(
		contract, cohort, model_name, limit, seed, expected_morph_model, policy,
		semantic_feedback, require_semantic_feedback, translation_filter, evidence_exclusion,
		use_licensed_forms, blind_policy,
	)

	existing_meta, existing_rows = _load_checkpoint(checkpoint_path)
	if existing_meta is not None:
		if not resume:
			raise ValueError(
			"checkpoint already exists: {}. Use --resume to continue it, or choose a new --output/--checkpoint.".format(checkpoint_path)
		)
		_verify_checkpoint(expected_meta, existing_meta)
	else:
		if resume:
			raise ValueError("--resume requested but checkpoint does not exist: {}".format(checkpoint_path))
		_append_jsonl(checkpoint_path, expected_meta)

	rows_by_sentence = {int(row["sentence_id"]): row for row in existing_rows}
	results = [rows_by_sentence[int(source["sentence_id"])] for source in cohort["rows"] if int(source["sentence_id"]) in rows_by_sentence]
	completed_ids = set(rows_by_sentence)
	if completed_ids:
		progress("resume: loaded {} checkpointed row(s) from {}".format(len(completed_ids), checkpoint_path))

	for index, source in enumerate(cohort["rows"], start=1):
		sentence_id = int(source["sentence_id"])
		if sentence_id in completed_ids:
			progress("[{}/{}] sentence {}: checkpointed, skipping paid inference".format(index, len(cohort["rows"]), sentence_id))
			continue

		progress("[{}/{}] sentence {}".format(index, len(cohort["rows"]), sentence_id))
		baseline = nrdb.morph_analyze(source["text"], int(source["dialect_id"]), int(source["annotation_schema_id"]))
		inference = baseline.get("inference") if isinstance(baseline, dict) else None
		inference = inference if isinstance(inference, dict) else {}
		model_id = str(inference.get("model_id") or "")
		if expected_morph_model and model_id != str(expected_morph_model):
			raise RuntimeError("deployed morph model mismatch: expected {!r}, got {!r}".format(expected_morph_model, model_id))

		baseline_metrics = annotation_metrics(baseline.get("annotation"), source["gold_annotation"])
		baseline_seg = segmentation_metrics(baseline.get("segmented"), source["gold_segmented"])
		tracker = UsageTracker()
		client = tracked_client(openai_client, tracker)
		existing_translation = str(source.get("translation_jp") or "").strip()
		item = {
			"sentence_id": sentence_id,
			"dialect_id": int(source["dialect_id"]),
			"dialect_region": source.get("dialect_region") or "",
			"text": source["text"],
			"translation_jp": existing_translation if semantic_feedback in {"existing", "auto"} else None,
		}
		job = {
			"annotation_schema_id": int(source["annotation_schema_id"]),
			"model_name": model_name,
			"prompt_version": "annotation-v9",
			"task": "morph",
			"semantic_feedback": semantic_feedback,
			"require_semantic_feedback": bool(require_semantic_feedback),
			"use_licensed_forms": use_licensed_forms,
			"morphology_source": "predict",
			"produce_translation": False,
			"blind_translation": False,
		}
		if policy.agent_review:
			agent_class = LicensedTaskAwareAnnotationAgent if use_licensed_forms else TaskAwareAnnotationAgent
			agent = agent_class(nrdb, model_name, client=client, progress=progress, morph_policy=policy)
			agent_result = agent.annotate(item, job, baseline)
		else:
			agent_result = {
				"segmented": baseline.get("segmented", ""), "annotation": baseline.get("annotation", ""),
				"decision": "proposed", "confidence": 1.0, "evidence": {"morph_review": {"mode": "none"}},
			}
		usage = tracker.summary()
		agent_metrics = annotation_metrics(agent_result.get("annotation"), source["gold_annotation"])
		agent_seg = segmentation_metrics(agent_result.get("segmented"), source["gold_segmented"])
		row = _result_row(source, baseline, agent_result, usage, baseline_metrics, agent_metrics, baseline_seg, agent_seg)
		row["semantic_feedback"] = semantic_feedback
		row["use_licensed_forms"] = int(use_licensed_forms)
		row["existing_translation_present"] = int(bool(existing_translation))
		row["internal_text_id"] = source.get("internal_text_id") or ""
		row["evidence_exclusion_json"] = json.dumps(evidence_exclusion, ensure_ascii=False, separators=(",", ":"))
		row["forward_morph_policy_json"] = policy.json()

		_append_jsonl(checkpoint_path, {"record_type": "row", "row": row})
		results.append(row)
		completed_ids.add(sentence_id)
		progress("  baseline ID {:.1f}% -> agent {:.1f}% | agree={} | cost ${:.4f} | checkpointed".format(
			100.0 * baseline_metrics["id_match_rate"],
			100.0 * agent_metrics["id_match_rate"],
			"yes" if row["baseline_agent_full_agree"] else "no",
			float(row.get("agent_cost_usd") or 0.0),
		))

	row_map = {int(row["sentence_id"]): row for row in results}
	results = [row_map[int(source["sentence_id"])] for source in cohort["rows"] if int(source["sentence_id"]) in row_map]
	summary = _paired_summary(results)
	morph_model_ids = sorted({str(row.get("baseline_model_id") or "") for row in results if row.get("baseline_model_id")})
	total_cost = sum(float(row.get("agent_cost_usd") or 0.0) for row in results)
	pricing_complete = all(bool(row.get("agent_pricing_complete")) for row in results)
	summary.update({
		"format": "nrdb-agent.morph-ceiling-eval.v8",
		"morph_run": contract["run_dir"],
		"train_rows": contract["train_rows"],
		"train_rows_missing_identity": contract["train_rows_missing_identity"],
		"datasets": cohort["dataset_ids"],
		"text_internal_id": cohort.get("text_internal_id"),
		"registered_gold_rows": cohort["registered_gold_rows"],
		"eligible_pool_after_train_exclusion": cohort["eligible_pool_size"],
		"sampled_rows": len(cohort["rows"]),
		"registered_rows_without_example_id": cohort["unmatched_identity"],
		"selection_seed": int(seed),
		"morph_model_ids": morph_model_ids,
		"expected_morph_model": expected_morph_model,
		"agent_model": model_name,
		"semantic_feedback": semantic_feedback,
		"require_semantic_feedback": bool(require_semantic_feedback),
		"translation_filter": translation_filter,
		"blind_policy": blind_policy,
		"use_licensed_forms": use_licensed_forms,
		"forward_morph_policy": policy.manifest(),
		"cohort_fingerprint": _cohort_fingerprint(cohort["rows"]),
		"cohort_in": _canonical_path(cohort_in), "cohort_out": _canonical_path(cohort_out),
		"evidence_exclusion": evidence_exclusion,
		"estimated_cost_usd": total_cost,
		"pricing_complete": pricing_complete,
		"checkpoint": str(checkpoint_path),
		"checkpointed_rows": len(results),
	})
	payload = {"summary": summary, "rows": results}
	if output:
		if Path(output).suffix.lower() == ".json":
			write_json(output, payload)
		else:
			write_tsv(output, results)
	return payload
