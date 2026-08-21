from collections import defaultdict


PRICING_AS_OF = "2026-07-30"
PRICING_SOURCE = "OpenAI API standard text-token pricing"
LONG_CONTEXT_THRESHOLD = 272000

# USD per 1M text tokens. Current GPT-5.6 standard API rates after the
# 2026-07-30 Terra/Luna price reduction. Cache writes are 1.25x uncached input.
MODEL_PRICING = {
	"gpt-5.6-sol": {"input": 5.00, "cached_input": 0.50, "cache_write": 6.25, "output": 30.00},
	"gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "cache_write": 2.50, "output": 12.00},
	"gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "cache_write": 0.25, "output": 1.20},
}


def canonical_model(model):
	value = str(model or "").lower()
	if value == "gpt-5.6" or value.startswith("gpt-5.6-sol"):
		return "gpt-5.6-sol"
	if value.startswith("gpt-5.6-terra"):
		return "gpt-5.6-terra"
	if value.startswith("gpt-5.6-luna"):
		return "gpt-5.6-luna"
	return value


def _value(obj, name, default=0):
	if obj is None:
		return default
	if isinstance(obj, dict):
		return obj.get(name, default)
	return getattr(obj, name, default)


def _int_value(obj, *names):
	for name in names:
		value = _value(obj, name, None)
		if value is not None:
			try:
				return int(value)
			except (TypeError, ValueError):
				pass
	return 0


def classify_stage(instructions):
	text = str(instructions or "").lower()
	if "grammatical-id analyst" in text:
		return "id_analysis"
	if "semantic consistency reviewer" in text:
		return "semantic_review"
	if "japanese translation phase" in text:
		return "translation"
	if "japanese-to-miyako id agent" in text:
		return "reverse_id_planning"
	if "grammatical id-sequence reviewer" in text:
		return "reverse_id_review"
	if "surface" in text and "japanese-to-miyako" in text:
		return "reverse_surface"
	if "asr" in text and ("selector" in text or "review" in text):
		return "asr_review"
	if "morphemic annotation agent" in text:
		return "annotation"
	return "other"


def price_usage(model, input_tokens, cached_tokens, cache_write_tokens, output_tokens):
	canonical = canonical_model(model)
	rates = MODEL_PRICING.get(canonical)
	if rates is None:
		return None
	input_tokens = max(0, int(input_tokens))
	cached_tokens = max(0, min(input_tokens, int(cached_tokens)))
	cache_write_tokens = max(0, min(input_tokens - cached_tokens, int(cache_write_tokens)))
	uncached_tokens = max(0, input_tokens - cached_tokens - cache_write_tokens)
	output_tokens = max(0, int(output_tokens))
	long_context = input_tokens > LONG_CONTEXT_THRESHOLD
	input_multiplier = 2.0 if long_context else 1.0
	output_multiplier = 1.5 if long_context else 1.0
	cost = (
		uncached_tokens * rates["input"] * input_multiplier
		+ cached_tokens * rates["cached_input"] * input_multiplier
		+ cache_write_tokens * rates["cache_write"] * input_multiplier
		+ output_tokens * rates["output"] * output_multiplier
	) / 1000000.0
	return {"canonical_model": canonical, "uncached_input_tokens": uncached_tokens, "long_context": long_context, "estimated_cost_usd": cost}


class UsageTracker:
	def __init__(self):
		self.calls = []

	def snapshot(self):
		return len(self.calls)

	def record(self, response, requested_model=None, instructions=None):
		usage = getattr(response, "usage", None)
		if usage is None:
			return None
		input_details = _value(usage, "input_tokens_details", {})
		output_details = _value(usage, "output_tokens_details", {})
		input_tokens = _int_value(usage, "input_tokens", "prompt_tokens")
		cached_tokens = _int_value(input_details, "cached_tokens")
		cache_write_tokens = _int_value(input_details, "cache_write_tokens", "prompt_cache_write_tokens")
		output_tokens = _int_value(usage, "output_tokens", "completion_tokens")
		reasoning_tokens = _int_value(output_details, "reasoning_tokens")
		total_tokens = _int_value(usage, "total_tokens") or input_tokens + output_tokens
		actual_model = str(getattr(response, "model", None) or requested_model or "")
		priced = price_usage(actual_model, input_tokens, cached_tokens, cache_write_tokens, output_tokens)
		row = {
			"response_id": getattr(response, "id", None), "requested_model": str(requested_model or ""),
			"model": actual_model, "canonical_model": canonical_model(actual_model), "stage": classify_stage(instructions),
			"input_tokens": input_tokens, "cached_input_tokens": cached_tokens, "cache_write_tokens": cache_write_tokens,
			"output_tokens": output_tokens, "reasoning_tokens": reasoning_tokens, "total_tokens": total_tokens,
			"long_context": bool(priced and priced["long_context"]),
			"estimated_cost_usd": priced["estimated_cost_usd"] if priced else None,
		}
		self.calls.append(row)
		return row

	def summary(self, since=0):
		calls = self.calls[int(since):]
		totals = {"requests": len(calls), "input_tokens": 0, "cached_input_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
		unknown_pricing = False
		stage_rows = defaultdict(list)
		model_rows = defaultdict(list)
		for row in calls:
			for key in ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
				totals[key] += int(row.get(key) or 0)
			cost = row.get("estimated_cost_usd")
			if cost is None:
				unknown_pricing = True
			else:
				totals["estimated_cost_usd"] += float(cost)
			stage_rows[row.get("stage") or "other"].append(row)
			model_rows[row.get("canonical_model") or "unknown"].append(row)

		def aggregate(rows):
			return {
				"requests": len(rows), "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
				"cached_input_tokens": sum(int(row.get("cached_input_tokens") or 0) for row in rows),
				"output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
				"reasoning_tokens": sum(int(row.get("reasoning_tokens") or 0) for row in rows),
				"estimated_cost_usd": sum(float(row.get("estimated_cost_usd") or 0.0) for row in rows),
			}

		return {
			"format": "nrdb-agent.api-usage.v1", "pricing_as_of": PRICING_AS_OF, "pricing_source": PRICING_SOURCE,
			"pricing_complete": not unknown_pricing, "totals": totals,
			"by_stage": {key: aggregate(rows) for key, rows in sorted(stage_rows.items())},
			"by_model": {key: aggregate(rows) for key, rows in sorted(model_rows.items())}, "calls": calls,
		}


class _TrackedResponses:
	def __init__(self, client_getter, tracker):
		self._client_getter = client_getter
		self._tracker = tracker

	def create(self, **kwargs):
		response = self._client_getter().responses.create(**kwargs)
		self._tracker.record(response, requested_model=kwargs.get("model"), instructions=kwargs.get("instructions"))
		return response


class TrackedOpenAIClient:
	def __init__(self, client, tracker):
		self._client = client
		self._resolved_client = None
		self.responses = _TrackedResponses(self._get_client, tracker)

	def _get_client(self):
		if self._client is not None:
			return self._client
		if self._resolved_client is None:
			from openai import OpenAI
			self._resolved_client = OpenAI()
		return self._resolved_client

	def __getattr__(self, name):
		return getattr(self._get_client(), name)


def tracked_client(client, tracker):
	if isinstance(client, TrackedOpenAIClient):
		return client
	return TrackedOpenAIClient(client, tracker)
