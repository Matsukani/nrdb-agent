# Translation discrepancy discovery

This workflow freezes a reproducible sample of human-translated NRDB sentences, generates blind translations without grammatical constructions, identifies semantic discrepancies, and then reruns the same sample with constructions enabled.

The artifacts are local JSON files. They do not modify NRDB, the gold corpus, or `annotation_constructions`.

## 1. Freeze a discovery cohort

The default selection matches any requested exact morpheme ID. `--require-all` requires every requested ID. Only rows with a gold Japanese translation are eligible.

```bash
nrdb-agent discrepancy-create adv foc 'ppt>2' \
	--dataset-id 30 \
	--dataset-id 31 \
	--annotation-schema 2 \
	--region 宮古 \
	--limit 50 \
	--min-morphemes 4 \
	--seed 42 \
	--output discovery.json
```

Selection uses the human gold annotation, but generation never receives the gold translation. The frozen artifact records the sampled sentence identities and complete evaluation inputs.

## 2. Generate and discover discrepancies

```bash
nrdb-agent discrepancy-run discovery.json \
	--translation-model gpt-5.6-luna \
	--discrepancy-model gpt-5.6-terra \
	--output baseline.json
```

`--translation-model` controls blind translation generation. `--discrepancy-model` independently controls semantic evaluation against gold. Generation uses the normal nrdb-agent ID-critic configuration: the deployed NRDB morph analysis plus `NRDB_ID_MODEL` when that environment variable is configured. This workflow deliberately provides no separate ID-model override.

The discrepancy judge accepts semantic paraphrases and separately identifies target-morpheme errors, unrelated lexical differences, free gold translations, and missing-context cases. The baseline summary ranks `morphemes_to_analyse` by attributed severity and records recurring discrepancy types, candidate patterns, and examples. Translation and discrepancy-model usage and estimated costs are reported separately and together.

## 3. Check construction repair

After reviewed grammatical records have been imported and enabled:

```bash
nrdb-agent discrepancy-check baseline.json \
	--discrepancy-model gpt-5.6-sol \
	--output repaired.json
```

The check reruns exactly the frozen baseline rows with constructions enabled. It defaults to the baseline translation model and rejects a different `--translation-model`, since changing the generator would confound the construction comparison. The discrepancy model may be changed because it judges the baseline and construction-assisted translations together in the same repair comparison.

Each command writes after every completed row, preserving paid work if a later row fails. Failed rows remain explicit in the artifact and are excluded from outcome counts.
