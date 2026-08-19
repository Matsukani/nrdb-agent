from nrdb_agent.reverse_agent import REVERSE_TOOLS, SEARCH_JAPANESE_BATCH_TOOL


def _contains_key(value, key):
	if isinstance(value, dict):
		return key in value or any(_contains_key(item, key) for item in value.values())
	if isinstance(value, list):
		return any(_contains_key(item, key) for item in value)
	return False


def test_reverse_batch_schema_avoids_unsupported_unique_items():
	assert _contains_key(SEARCH_JAPANESE_BATCH_TOOL, "uniqueItems") is False


def test_reverse_function_schemas_avoid_unsupported_unique_items():
	for tool in REVERSE_TOOLS:
		assert _contains_key(tool, "uniqueItems") is False
