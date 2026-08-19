from types import SimpleNamespace

from nrdb_agent.usage import UsageTracker, canonical_model, price_usage


def _response(model, input_tokens=1000, cached_tokens=0, output_tokens=100, reasoning_tokens=0):
	return SimpleNamespace(
		id="resp_test",
		model=model,
		usage=SimpleNamespace(
			input_tokens=input_tokens,
			input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
			output_tokens=output_tokens,
			output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
			total_tokens=input_tokens + output_tokens,
		),
	)


def test_gpt56_alias_prices_as_sol():
	assert canonical_model("gpt-5.6") == "gpt-5.6-sol"
	priced = price_usage("gpt-5.6", 1000, 0, 0, 100)
	assert round(priced["estimated_cost_usd"], 6) == 0.008


def test_terra_and_luna_current_prices():
	terra = price_usage("gpt-5.6-terra", 1000, 0, 0, 100)
	luna = price_usage("gpt-5.6-luna", 1000, 0, 0, 100)
	assert round(terra["estimated_cost_usd"], 6) == 0.0032
	assert round(luna["estimated_cost_usd"], 6) == 0.00032


def test_cached_input_discount_is_counted():
	priced = price_usage("gpt-5.6-sol", 2000, 1000, 0, 100)
	assert round(priced["estimated_cost_usd"], 6) == 0.0085


def test_tracker_groups_usage_by_stage_and_model():
	tracker = UsageTracker()
	tracker.record(
		_response("gpt-5.6-sol", input_tokens=1000, output_tokens=100),
		requested_model="gpt-5.6",
		instructions="You are the constrained NRDB morphemic annotation agent.",
	)
	tracker.record(
		_response("gpt-5.6-sol", input_tokens=500, output_tokens=50),
		requested_model="gpt-5.6",
		instructions="You are the constrained NRDB Japanese translation phase.",
	)
	summary = tracker.summary()
	assert summary["totals"]["requests"] == 2
	assert summary["totals"]["input_tokens"] == 1500
	assert summary["by_stage"]["annotation"]["requests"] == 1
	assert summary["by_stage"]["translation"]["requests"] == 1
	assert summary["by_model"]["gpt-5.6-sol"]["requests"] == 2
	assert summary["pricing_complete"] is True
