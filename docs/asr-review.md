# Blind ASR n-best review

`nrdb-agent asr-review` evaluates whether NRDB linguistic reasoning can select a better complete transcription from an existing ASR n-best beam without changing the ASR model or inventing audio content.

The experiment is intentionally selection-only:

```text
ASR n-best
  -> compact trsc2 reconstruction
  -> optional phrase-boundary model
  -> nrdb-morph analysis
  -> ID-sequence critic
  -> surface/allomorph + phonotactic critic
  -> deterministic linguistic baseline
  -> blind LLM selector
  -> post-selection evaluation against hidden reference
```

The LLM receives no reference transcription, Japanese translation, oracle rank, UER, or gold annotation. `lookup_id` is the only online NRDB evidence tool in v1. Corpus-example and form-support tools are deliberately disabled until ASR evaluation rows carry an exact NRDB sentence identifier that can be excluded from retrieval.

## Inputs

First produce n-best predictions with `nrdb-asr`:

```bash
nrdb-asr eval-manifest \
  --model-dir PATH/TO/best_model \
  --manifest PATH/TO/test.tsv \
  --out-dir PATH/TO/eval-nbest10 \
  --wav-column cut_audio_path \
  --ref-column unit_label \
  --decoder beam \
  --beam-width 25 \
  --beam-top-k 25 \
  --lm-path PATH/TO/lm.json \
  --lm-alpha VALUE \
  --nbest 10
```

Then review those hypotheses from the `nrdb-agent` environment:

```bash
export NRDB_ID_MODEL=../nrdb-morph/training-runs/.../id_sequence_model.json
export NRDB_SURFACE_MODEL=../nrdb-morph/training-runs/.../surface_model_v2.json

nrdb-agent asr-review PATH/TO/eval-nbest10/predictions.tsv \
  --out-dir PATH/TO/asr-review10 \
  --annotation-schema 2 \
  --region 宮古 \
  --dialect 19 \
  --limit 10
```

Use `--no-llm` to run only the deterministic linguistic baseline. This is useful for a full-test-set zero-API-cost pass before running the LLM selector.

An optional `--phrase-boundary-model PATH` can be supplied. This requires the local `nrdb-asr` package to be importable in the agent environment. Without it, each compact n-best hypothesis is passed to nrdb-morph as one phrase.

## Outputs

`summary.json` reports:

- top-1 UER;
- deterministic baseline UER;
- agent-selected UER;
- oracle UER within the reviewed n-best list;
- available oracle headroom;
- fraction of that headroom recovered by baseline and agent;
- rank-1 retention/change counts;
- improved, harmful and neutral selection counts;
- number of exact oracle-rank selections.

`asr_review.tsv` records selected ranks, edit counts, selected transcription/analysis, and complete candidate diagnostics for audit.

The deterministic baseline is deliberately untuned. It first minimizes the total number of strong ID-sequence surprises plus strong surface disagreements, then uses morph confidence and the combined ID/surface plausibility scores as tie-breakers. It is included as a cheap non-LLM comparison, not as an optimized decoder.
