from copy import deepcopy

from .task_agent import TaskAwareAnnotationAgent


def morph_surfaces(morph_result):
	"""Return the decoder's current surface segments, preserving first-seen order."""
	seen = set()
	values = []
	for phrase in str((morph_result or {}).get("segmented") or "").strip().split():
		for surface in phrase.split("-"):
			surface = surface.strip()
			if surface and surface not in seen:
				seen.add(surface)
				values.append(surface)
	return values


def _licensed_parts(match):
	surface = str(match.get("matched_surface") or "").strip()
	segmented = str(match.get("form_romaji_seg") or match.get("form_kana_seg") or surface).strip()
	annotation = str(match.get("annotation") or "").strip()
	seg_parts = segmented.split("-") if segmented else []
	ann_parts = annotation.split("-") if annotation else []
	return surface, segmented, annotation, seg_parts, ann_parts


def _contains_analysis(segmented, annotation, licensed_segmented, licensed_annotation):
	need_surface = licensed_segmented.split("-")
	need_label = licensed_annotation.split("-")
	if len(need_surface) != len(need_label) or not need_surface:
		return False
	for seg_phrase, ann_phrase in zip(str(segmented or "").split(), str(annotation or "").split()):
		surfaces = seg_phrase.split("-")
		labels = ann_phrase.split("-")
		for start in range(0, len(surfaces) - len(need_surface) + 1):
			end = start + len(need_surface)
			if surfaces[start:end] == need_surface and labels[start:end] == need_label:
				return True
	return False


class LicensedTaskAwareAnnotationAgent(TaskAwareAnnotationAgent):
	"""Use licensed forms as strong, auditable grammatical evidence."""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._active_licensed = None
		self._licensed_repair_audit = None

	def _licensed_morph(self, item, job, morph_result):
		if not isinstance(morph_result, dict):
			return morph_result
		if isinstance(morph_result.get("licensed_realizations"), dict):
			return morph_result
		text = str(item.get("text") or "").strip()
		region = str(item.get("dialect_region") or "").strip()
		dialect_id = int(item.get("dialect_id") or 0)
		schema_id = int(job.get("annotation_schema_id") or 0)
		if not text or dialect_id <= 0 or schema_id <= 0:
			return morph_result
		licensed = self.nrdb.licensed_forms_in_text(
			text, schema_id, region, dialect_id, surfaces=morph_surfaces(morph_result),
		)
		value = dict(morph_result)
		value["licensed_realizations"] = licensed
		self.progress("  licensed: grammar-derived surface matches={}".format(len(licensed.get("matches", []))))
		return value

	def _repair_unknowns(self, text, morph_result):
		"""Apply only unambiguous same-dialect exact licensed analyses to '?' segments."""
		value = deepcopy(morph_result)
		matches = (value.get("licensed_realizations") or {}).get("matches", [])
		groups = {}
		for match in matches:
			surface, segmented, annotation, seg_parts, ann_parts = _licensed_parts(match)
			if (match.get("retrieval_path") != "exact_surface" or match.get("scope") != "same_dialect"
					or not surface or not annotation or len(seg_parts) != len(ann_parts)):
				continue
			groups.setdefault(surface, {}).setdefault((segmented, annotation), []).append(match)

		audit = {"policy": "licensed_exact_unknown_repair_v1", "applied_ids": [], "candidates": []}
		seg_phrases = str(value.get("segmented") or "").split()
		ann_phrases = str(value.get("annotation") or "").split()
		if len(seg_phrases) != len(ann_phrases):
			return value, audit

		for surface, analyses in groups.items():
			candidate = {"surface": surface, "analysis_count": len(analyses), "status": "ambiguous"}
			if len(analyses) != 1:
				audit["candidates"].append(candidate)
				continue
			(segmented, annotation), rows = next(iter(analyses.items()))
			locations = []
			for phrase_index, (seg_phrase, ann_phrase) in enumerate(zip(seg_phrases, ann_phrases)):
				surfaces = seg_phrase.split("-")
				labels = ann_phrase.split("-")
				if len(surfaces) != len(labels):
					continue
				locations.extend((phrase_index, index) for index, (found, label) in enumerate(zip(surfaces, labels)) if found == surface and label == "?")
			candidate.update({"generated_wordform_ids": [int(row.get("id") or 0) for row in rows], "segmented": segmented, "annotation": annotation})
			if len(locations) != 1:
				candidate["status"] = "not_one_exact_unknown_segment"
				audit["candidates"].append(candidate)
				continue
			phrase_index, segment_index = locations[0]
			new_seg_phrases = list(seg_phrases)
			new_ann_phrases = list(ann_phrases)
			surfaces = new_seg_phrases[phrase_index].split("-")
			labels = new_ann_phrases[phrase_index].split("-")
			surfaces[segment_index:segment_index + 1] = segmented.split("-")
			labels[segment_index:segment_index + 1] = annotation.split("-")
			new_seg_phrases[phrase_index] = "-".join(surfaces)
			new_ann_phrases[phrase_index] = "-".join(labels)
			proposed_segmented = " ".join(new_seg_phrases)
			proposed_annotation = " ".join(new_ann_phrases)
			validation = self.nrdb.validate_analysis(text, proposed_segmented, proposed_annotation)
			candidate["validation"] = validation
			if not validation.get("valid"):
				candidate["status"] = "validation_rejected"
				audit["candidates"].append(candidate)
				continue
			seg_phrases, ann_phrases = new_seg_phrases, new_ann_phrases
			candidate["status"] = "applied_before_llm"
			audit["applied_ids"].extend(candidate["generated_wordform_ids"])
			audit["candidates"].append(candidate)

		value["segmented"] = " ".join(seg_phrases)
		value["annotation"] = " ".join(ann_phrases)
		if audit["applied_ids"]:
			# Avoid presenting stale decoder segment details that contradict the repaired baseline.
			value["phrases"] = []
		return value, audit

	def annotate(self, item, job, morph_result):
		morph_result = self._licensed_morph(item, job, morph_result)
		self._active_licensed = morph_result.get("licensed_realizations") if isinstance(morph_result, dict) else None
		morph_result, self._licensed_repair_audit = self._repair_unknowns(item.get("text"), morph_result)
		if self._licensed_repair_audit.get("applied_ids"):
			self.progress("  licensed: applied {} exact unknown repair(s) before LLM".format(len(self._licensed_repair_audit["applied_ids"])))
		result = super().annotate(item, job, morph_result)
		if isinstance(self._active_licensed, dict):
			matches = self._active_licensed.get("matches", [])
			applied = []
			unresolved = []
			for match in matches:
				_, segmented, annotation, _, _ = _licensed_parts(match)
				identifier = int(match.get("id") or 0)
				if segmented and annotation and _contains_analysis(result.get("segmented"), result.get("annotation"), segmented, annotation):
					applied.append(identifier)
				else:
					unresolved.append(identifier)
			final_audit = deepcopy(self._licensed_repair_audit)
			rejected = set(final_audit.get("applied_ids", [])) - set(applied)
			for candidate in final_audit.get("candidates", []):
				if candidate.get("status") == "validation_rejected":
					rejected.update(candidate.get("generated_wordform_ids", []))
			final_audit.update({
				"retrieved_ids": [int(match.get("id") or 0) for match in matches],
				"final_applied_ids": applied, "rejected_ids": sorted(rejected),
				"unresolved_ids": [identifier for identifier in unresolved if identifier not in rejected],
			})
			result.setdefault("evidence", {})["licensed_realizations"] = self._active_licensed
			result["evidence"]["licensed_form_audit"] = final_audit
		return result

	def _shared_evidence_compact(self):
		value = super()._shared_evidence_compact()
		if isinstance(self._active_licensed, dict):
			value["licensed_realizations"] = self._active_licensed
		if isinstance(self._licensed_repair_audit, dict):
			value["licensed_repair_audit"] = self._licensed_repair_audit
		return value

	def _prepare_hotspots(self, morph_result, schema_id):
		value = super()._prepare_hotspots(morph_result, schema_id)
		licensed = morph_result.get("licensed_realizations") if isinstance(morph_result, dict) else None
		matches = licensed.get("matches", []) if isinstance(licensed, dict) else []
		if not matches:
			return value

		uncertain = set(value.get("uncertain_surfaces", []))
		hotspot_ids = set(value.get("hotspot_ids", []))
		reasons = dict(value.get("uncertainty_reasons", {}))
		candidates = []
		for match in matches[:24]:
			surface, segmented, annotation, _, _ = _licensed_parts(match)
			if not surface or not annotation:
				continue
			uncertain.add(surface)
			message = (
				"grammar_licensed_realization: raw={!r}; segmented={!r}; annotation={!r}; scope={}; path={}. "
				"This is licensed grammatical evidence, not an observed corpus token. An exact same-dialect match is strong positive evidence; "
				"prefer it over '?' unless validation or conflicting licensed evidence blocks it."
			).format(surface, segmented, annotation, match.get("scope") or "unknown", match.get("retrieval_path") or "unknown")
			reasons.setdefault(surface, [])
			if message not in reasons[surface]:
				reasons[surface].append(message)
			candidates.append({
				"generated_wordform_id": int(match.get("id") or 0), "surface": surface,
				"segmented": segmented, "annotation": annotation, "scope": match.get("scope"),
				"retrieval_path": match.get("retrieval_path"), "occurrences": match.get("occurrences", []),
				"matched_start": match.get("matched_start"), "matched_end": match.get("matched_end"),
			})
			for segment in annotation.replace(" ", "-").split("-"):
				for atom in segment.split(";"):
					if atom.strip():
						hotspot_ids.add(atom.strip())

		value["uncertain_surfaces"] = sorted(uncertain)
		value["uncertainty_reasons"] = reasons
		value["hotspot_ids"] = sorted(hotspot_ids)
		value["licensed_repair_candidates"] = candidates
		value["licensed_repair_audit"] = self._licensed_repair_audit
		value["policy"] = "query_uncertainty_with_licensed_realizations_v2"
		return value
