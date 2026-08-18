class SurfaceModelCritic:
	def __init__(self, model_path=None, model=None, disagreement_margin=0.75):
		if model is None:
			try:
				from nrdb_morph.surface_realization import SurfaceRealizationModel
			except ImportError as error:
				raise RuntimeError(
					"--surface-model requires nrdb-morph in the nrdb-agent environment; run: pip install -e ../nrdb-morph"
				) from error
			model = SurfaceRealizationModel.load(model_path)
		self.model = model
		self.disagreement_margin = float(disagreement_margin)

	def _priority_suggestions(self, label, previous_surface, dialect_ids, annotation_schema_id, phrase_start):
		fallback = None
		for dialect_id in dialect_ids:
			suggestions = self.model.suggest_forms(
				label, previous_surface, int(dialect_id), int(annotation_schema_id),
				phrase_start=bool(phrase_start), top_k=3,
			)
			if not suggestions:
				continue
			if any(value.get("form_source") == "dialect" for value in suggestions):
				return suggestions, int(dialect_id), "requested_dialect"
			if fallback is None:
				fallback = (suggestions, int(dialect_id), "schema_pool")
		return fallback or ([], int(dialect_ids[0]), "none")

	def _generated_score(self, label, generated, suggestions, previous_surface, dialect_id, annotation_schema_id, phrase_start):
		forms = [generated] + [value.get("form") for value in suggestions if value.get("form")]
		ranked = self.model.rank_forms(
			label, forms, previous_surface, int(dialect_id), int(annotation_schema_id),
			phrase_start=bool(phrase_start),
		)
		by_form = {value.get("form"): value for value in ranked}
		return by_form.get(generated, {"form": generated, "score": None})

	def review(self, segmented, annotation, dialect_ids, annotation_schema_id):
		dialect_ids = [int(value) for value in dialect_ids]
		if not dialect_ids:
			raise ValueError("surface critic requires at least one target dialect")
		seg_phrases = str(segmented or "").strip().split()
		ann_phrases = str(annotation or "").strip().split()
		if len(seg_phrases) != len(ann_phrases):
			return {"valid_alignment": False, "error": "segmented/annotation phrase count mismatch", "diagnostics": [], "strong_disagreements": 0}

		diagnostics = []
		for phrase_index, (seg_phrase, ann_phrase) in enumerate(zip(seg_phrases, ann_phrases), start=1):
			surfaces = seg_phrase.split("-")
			labels = ann_phrase.split("-")
			if len(surfaces) != len(labels):
				return {"valid_alignment": False, "error": "segmented/annotation segment count mismatch in phrase {}".format(phrase_index), "diagnostics": [], "strong_disagreements": 0}
			previous = ""
			for segment_index, (surface, label) in enumerate(zip(surfaces, labels)):
				phrase_start = segment_index == 0
				suggestions, evidence_dialect, source = self._priority_suggestions(
					label, previous, dialect_ids, annotation_schema_id, phrase_start,
				)
				if suggestions:
					generated_score = self._generated_score(
						label, surface, suggestions, previous, evidence_dialect,
						annotation_schema_id, phrase_start,
					)
					top = suggestions[0]
					top_score = float(top.get("score", 0.0))
					current_score = generated_score.get("score")
					gap = None if current_score is None else top_score - float(current_score)
					top_forms = [value.get("form") for value in suggestions]
					strong = bool(surface != top.get("form") and (surface not in top_forms or (gap is not None and gap >= self.disagreement_margin)))
					diagnostics.append({
						"phrase": phrase_index,
						"segment": segment_index + 1,
						"label": label,
						"previous_surface": previous,
						"generated_form": surface,
						"generated_score": current_score,
						"suggestions": [
							{"form": value.get("form"), "score": value.get("score"), "attested_count": value.get("attested_count"), "form_source": value.get("form_source")}
							for value in suggestions
						],
						"evidence_dialect_id": evidence_dialect,
						"evidence_scope": source,
						"score_gap": gap,
						"strong_disagreement": strong,
					})
				previous = surface

		phonotactic = self.model.score_analysis(segmented, dialect_ids[0], int(annotation_schema_id), annotation)
		strong_count = sum(int(value["strong_disagreement"]) for value in diagnostics)
		return {
			"valid_alignment": True,
			"target_dialect_ids": dialect_ids,
			"annotation_schema_id": int(annotation_schema_id),
			"phonotactic_mean_log_probability": phonotactic.get("mean_log_probability"),
			"diagnostics": diagnostics,
			"strong_disagreements": strong_count,
		}
