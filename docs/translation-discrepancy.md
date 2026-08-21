# Translation discrepancy discovery

This workflow freezes a reproducible sample of human-translated NRDB sentences, generates blind translations without grammatical constructions, identifies semantic discrepancies, and then reruns the same sample with constructions enabled.

The artifacts are local JSON files. They do not modify NRDB, the gold corpus, or `annotation_constructions`.

## 1. Freeze a discovery cohort

Every requested exact morpheme ID is sampled independently. `--limit 50` means up to 50 rows for each ID, not 50 rows from their union. A sentence containing several requested IDs is retained as a separate target-specific assignment for each ID. `--require-all` is the exceptional intersection mode: it requires every requested ID and retains the resulting sentences under each requested target. Only rows with a gold Japanese translation are eligible. With no `--dataset-id`, the cohort pools every sentence, text, and `lxs` row matching `--annotation-schema` and `--region`.

```bash
nrdb-agent discrepancy-create adv foc 'ppt>2' \
	--annotation-schema 2 \
	--region 宮古 \
	--limit 50 \
	--min-morphemes 4 \
	--gold-morph \
	--seed 42 \
	--output discovery.json
```

Dataset scope is optional. One option accepts several IDs, and the option may also be repeated:

```bash
--dataset-id 21 29 31 28 30
```

```bash
--dataset-id 21 29 --dataset-id 31 28 30
```

Selection uses the human gold annotation, but generation never receives the gold translation. The frozen artifact records the sampled sentence identities and complete evaluation inputs.

`--gold-morph` restricts creation to rows carrying both gold segmentation and gold annotation and records that requirement in the artifact. Use it for the first experiment, where grammatical translation is evaluated independently of automatic morphology.

## 2. Generate and discover discrepancies

```bash
nrdb-agent discrepancy-run discovery.json \
	--translation-model gpt-5.6-luna \
	--discrepancy-model gpt-5.6-terra \
	--gold-morph \
	--output baseline.json
```

`--translation-model` controls blind translation generation. `--discrepancy-model` independently controls semantic evaluation against gold. Without `--gold-morph`, generation uses the normal nrdb-agent ID-critic configuration: the deployed NRDB morph analysis plus `NRDB_ID_MODEL` when that environment variable is configured. This workflow deliberately provides no separate ID-model override.

With `--gold-morph`, generation skips NRDB morph prediction and the ID critic and translates from the frozen human segmentation and annotation. The option is accepted only for a cohort created with `discrepancy-create --gold-morph`. `discrepancy-check` automatically inherits the baseline morphology mode, so baseline and construction-assisted translations always use the same morphology.

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

## List local experiments

Discrepancy experiments are local audited JSON artifacts rather than database jobs. List the artifacts in the current directory with:

```bash
nrdb-agent discrepancy-list
```

Useful variants are:

```bash
nrdb-agent discrepancy-list --latest 5
nrdb-agent discrepancy-list --directory experiments --recursive
nrdb-agent discrepancy-list --json
```

The listing reports stage, status, target IDs, schema/region scope, row count, models, cost, modification time, and path.
