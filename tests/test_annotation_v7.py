from nrdb_agent.annotator_v7 import AnnotationAgentV7, V7_TRANSLATION_INSTRUCTIONS


class FakeNrdb:
	def lookup_id(self, label, schema_id):
		data = {
			"river_like_id": {"meaning_jp": "井戸"},
			"house_id": {"meaning_jp": "家"},
		}
		entry = data.get(label)
		return {
			"label": label,
			"lexical_entries": [] if entry is None else [{
				"form1": label,
				"form2": None,
				"meaning_jp": entry["meaning_jp"],
				"pos": "n",
				"dialect_name": "test",
			}],
			"local": None,
			"global": None,
		}


def test_v7_translation_treats_id_spelling_as_opaque():
	assert "annotation labels are IDENTIFIERS, not glosses" in V7_TRANSLATION_INSTRUCTIONS
	assert "Never infer lexical meaning from kanji, spelling" in V7_TRANSLATION_INSTRUCTIONS
	assert "Dictionary meaning_jp/explanation_jp outrank" in V7_TRANSLATION_INSTRUCTIONS


def test_v7_batch_grounding_uses_dictionary_meanings():
	agent = AnnotationAgentV7(FakeNrdb(), "test-model", client=object())
	result = agent._ground_lexical_ids(["river_like_id", "house_id", "unknown_id"], 2)
	assert len(result["labels"]) == 3
	assert result["labels"][0]["grounded"] is True
	assert result["labels"][0]["lexical_entries"][0]["meaning_jp"] == "井戸"
	assert result["labels"][1]["lexical_entries"][0]["meaning_jp"] == "家"
	assert result["labels"][2]["grounded"] is False
