from pathlib import Path

from nrdb_agent.dataset_io import item_matches_needs, load_tsv, output_row, write_tsv


def test_tsv_normalizes_existing_morphology_and_translation(tmp_path: Path):
	path = tmp_path / "input.tsv"
	path.write_text(
		"id\ttrsc2\tdialect_id\ttrsc2_seg\tannotation_r\ttrsl\n"
		"x1\taga za\t19\taga-za\t東an-top:1\t東だ\n"
		"x2\tndza\t19\t\t\tどこ\n",
		encoding="utf-8",
	)
	bundle = load_tsv(path, 2, "宮古")
	assert bundle["annotation_schema_id"] == 2
	assert bundle["region"] == "宮古"
	assert bundle["items"][0]["existing_annotation"] == "東an-top:1"
	assert bundle["items"][0]["translation_jp"] == "東だ"
	assert item_matches_needs(bundle["items"][0], "annotation") is False
	assert item_matches_needs(bundle["items"][1], "annotation") is True
	assert item_matches_needs(bundle["items"][1], "translation") is False


def test_tsv_output_preserves_source_columns_and_appends_ai(tmp_path: Path):
	item = {"_original": {"id": "x1", "trsc2": "aga"}}
	row = output_row(item, result={
		"segmented": "aga", "annotation": "東an", "translation": "東",
		"decision": "proposed", "confidence": 0.9, "estimated_cost_usd": 0.0123,
		"model": "gpt-5.6-terra", "evidence": {"ok": True},
	})
	path = tmp_path / "out.tsv"
	write_tsv(path, [row])
	text = path.read_text(encoding="utf-8")
	assert "trsc2" in text
	assert "ai_annotation" in text
	assert "東an" in text
	assert "0.012300" in text
