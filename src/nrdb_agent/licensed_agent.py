from .task_agent import TaskAwareAnnotationAgent


class LicensedTaskAwareAnnotationAgent(TaskAwareAnnotationAgent):
	"""Inject grammar-licensed generated forms into v9 uncertainty triage.

	Licensed forms are grammar-derived evidence, not corpus attestations.  Exact
	surface matches are strong positive evidence for the supplied segmented form
	and annotation; absence from the licensed table is neutral.
	"""
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
