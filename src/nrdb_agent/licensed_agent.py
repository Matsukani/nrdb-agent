from .task_agent import TaskAwareAnnotationAgent


class LicensedTaskAwareAnnotationAgent(TaskAwareAnnotationAgent):
	"""Inject grammar-licensed generated forms into v9 uncertainty triage.

	Licensed forms are grammar-derived evidence, not corpus attestations. Exact
	surface matches are strong positive evidence for the supplied segmented form
	and annotation; absence from the licensed table is neutral.
	"""
	def _licensed_morph(self, item, job, morph_result):
		if not isinstance(morph_result, dict) or not job.get("use_licensed_forms"):
			return morph_result
		if isinstance(morph_result.get("licensed_realizations"), dict):
			return morph_result
		text = str(item.get("text") or "").strip()
		region = str(item.get("dialect_region") or "").strip()
		dialect_id = int(item.get("dialect_id") or 0)
		schema_id = int(job.get("annotation_schema_id") or 0)
		if not text or dialect_id <= 0 or schema_id <= 0:
			return morph_result
		licensed = self.nrdb.licensed_forms_in_text(text, schema_id, region, dialect_id)
		value = dict(morph_result)
		value["licensed_realizations"] = licensed
		self.progress("  licensed: grammar-derived surface matches={}".format(len(licensed.get("matches", []))))
		return value

	def annotate(self, item, job, morph_result):
		morph_result = self._licensed_morph(item, job, morph_result)
		result = super().annotate(item, job, morph_result)
		licensed = morph_result.get("licensed_realizations") if isinstance(morph_result, dict) else None
		if isinstance(licensed, dict):
			result.setdefault("evidence", {})["licensed_realizations"] = licensed
		return result

	def _prepare_hotspots(self, morph_result, schema_id):
		value = super()._prepare_hotspots(morph_result, schema_id)
		licensed = morph_result.get("licensed_realizations") if isinstance(morph_result, dict) else None
		matches = licensed.get("matches", []) if isinstance(licensed, dict) else []
		if not matches:
			return value

		uncertain = set(value.get("uncertain_surfaces", []))
		hotspot_ids = set(value.get("hotspot_ids", []))
		reasons = dict(value.get("uncertainty_reasons", {}))
		for match in matches[:24]:
			surface = str(match.get("matched_surface") or "").strip()
			annotation = str(match.get("annotation") or "").strip()
			segmented = str(
				match.get("form_romaji_seg") or match.get("form_kana_seg") or surface
			).strip()
			if not surface or not annotation:
				continue
			uncertain.add(surface)
			message = (
				"grammar_licensed_realization: raw={!r}; segmented={!r}; annotation={!r}; "
				"scope={}; this is licensed grammatical evidence, NOT an observed corpus token"
			).format(surface, segmented, annotation, match.get("scope") or "unknown")
			reasons.setdefault(surface, [])
			if message not in reasons[surface]:
				reasons[surface].append(message)
			for segment in annotation.replace(" ", "-").split("-"):
				for atom in segment.split(";"):
					atom = atom.strip()
					if atom:
						hotspot_ids.add(atom)

		value["uncertain_surfaces"] = sorted(uncertain)
		value["uncertainty_reasons"] = reasons
		value["hotspot_ids"] = sorted(hotspot_ids)
		value["policy"] = "query_uncertainty_with_licensed_realizations_v1"
		return value
