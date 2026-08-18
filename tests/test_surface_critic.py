from nrdb_agent.surface_critic import SurfaceModelCritic


class FakeSurfaceModel:
	def suggest_forms(self, label, previous_surface, dialect_id, annotation_schema_id, phrase_start=False, top_k=3):
		if label == "acc" and previous_surface == "umat" and dialect_id == 19:
			return [
				{"form": "tsu", "score": -0.1, "attested_count": 34, "form_source": "dialect"},
				{"form": "u:", "score": -3.0, "attested_count": 186, "form_source": "dialect"},
			]
		return [{"form": "umat", "score": -0.1, "attested_count": 20, "form_source": "dialect"}] if label == "馬tn" else []

	def rank_forms(self, label, forms, previous_surface, dialect_id, annotation_schema_id, phrase_start=False):
		scores = {"tsu": -0.1, "u:": -3.0, "umat": -0.1}
		return [{"form": form, "score": scores.get(form, -4.0)} for form in forms]

	def score_analysis(self, segmented, dialect_id, annotation_schema_id, annotation=None):
		return {"mean_log_probability": -1.0 if segmented == "umat-tsu" else -3.0}


class FakeBidirectionalSurfaceModel:
	next_label_forms = {"enabled": True}

	def suggest_forms(self, label, previous_surface, dialect_id, annotation_schema_id, phrase_start=False, top_k=3, next_label=None):
		if label == "食kv":
			if next_label == "ipf":
				return [
					{"form": "fai", "score": -0.1, "attested_count": 12, "form_source": "dialect", "right_context_log_probability": -0.05, "right_context_source": "dialect"},
					{"form": "fo:", "score": -3.2, "attested_count": 90, "form_source": "dialect", "right_context_log_probability": -4.0, "right_context_source": "dialect"},
				]
			return [{"form": "fo:", "score": -0.1, "attested_count": 90, "form_source": "dialect"}]
		if label == "ipf":
			return [{"form": "ju:", "score": -0.1, "attested_count": 20, "form_source": "dialect"}]
		return []

	def rank_forms(self, label, forms, previous_surface, dialect_id, annotation_schema_id, phrase_start=False, next_label=None):
		if label == "食kv" and next_label == "ipf":
			scores = {"fai": -0.1, "fo:": -3.2}
		else:
			scores = {"ju:": -0.1, "u:": -3.0}
		return [{"form": form, "score": scores.get(form, -4.0)} for form in forms]

	def score_analysis(self, segmented, dialect_id, annotation_schema_id, annotation=None):
		return {"mean_log_probability": -1.0 if segmented == "fai-ju:" else -2.0}


def test_surface_critic_flags_conditioned_allomorph():
	critic = SurfaceModelCritic(model=FakeSurfaceModel(), disagreement_margin=0.75)
	review = critic.review("umat-u:", "馬tn-acc", [19, 22], 2)
	assert review["valid_alignment"] is True
	assert review["strong_disagreements"] == 1
	diagnostic = next(value for value in review["diagnostics"] if value["label"] == "acc")
	assert diagnostic["generated_form"] == "u:"
	assert diagnostic["suggestions"][0]["form"] == "tsu"
	assert diagnostic["evidence_dialect_id"] == 19
	assert diagnostic["strong_disagreement"] is True


def test_surface_critic_uses_next_id_to_flag_stem_allomorph():
	critic = SurfaceModelCritic(model=FakeBidirectionalSurfaceModel(), disagreement_margin=0.75)
	review = critic.review("fo:-ju:", "食kv-ipf", [19], 2)
	assert review["surface_model_bidirectional"] is True
	assert review["strong_disagreements"] == 1
	diagnostic = next(value for value in review["diagnostics"] if value["label"] == "食kv")
	assert diagnostic["next_label"] == "ipf"
	assert diagnostic["generated_form"] == "fo:"
	assert diagnostic["suggestions"][0]["form"] == "fai"
	assert diagnostic["suggestions"][0]["right_context_source"] == "dialect"
	assert diagnostic["strong_disagreement"] is True
