# Morphology and translation workflows

`nrdb-agent` separates the workflow into independent axes:

1. **source**: registered NRDB dataset, portable `_meta_`/`_cf_` XLSX, TSV, or direct one-line input;
2. **task/output**: `morph`, `translate`, `morph-translate`, or `reverse`;
3. **semantic feedback for morphology**: `none`, `generated`, `existing`, or `auto`;
4. **morphology source**: `predict`, `existing`, or `auto`;
5. **selection/output**: NRDB scope/missingness or local TSV/JSON output.

The central rule is that **producing a Japanese translation and using Japanese as semantic feedback are different decisions**. A translation may be generated internally to review morphology without becoming an output, and a requested translation may be produced without reopening morphology.

## Semantic feedback

`--semantic-feedback` applies while predicted morphology is still reviewable:

- `none`: morphology is reviewed only from nrdb-morph, NRDB lexical/corpus evidence, critics, and annotation constraints;
- `generated`: generate a dictionary-grounded Japanese translation internally, then use it for a semantic consistency review of morphology;
- `existing`: use the Japanese translation already present in the dataset as semantic evidence;
- `auto`: use an existing translation when available, otherwise generate one internally.

`--require-semantic-feedback` makes the selected source mandatory. It is most useful with `existing` when an experiment must contain only rows with human/gold Japanese translations.

If generated semantic feedback revises morphology and Japanese translation is also a requested output, the final Japanese is regenerated from the revised morphology so annotation and translation cannot diverge.

## Task semantics

### `morph`

Produce Miyako segmentation and morphemic annotation only. Semantic feedback is independent:

```bash
# A. Morph + NRDB/LLM evidence only
nrdb-agent process data.xlsx --task morph --semantic-feedback none

# B. Morph + internally generated Japanese semantic feedback
nrdb-agent process data.xlsx --task morph --semantic-feedback generated

# C. Morph + existing data translation semantic feedback
nrdb-agent process data.xlsx --task morph --semantic-feedback existing --require-semantic-feedback
```

In B, the generated Japanese is internal evidence and `ai_translation` remains empty because the requested task is morphology only.

### `translate`

Produce Japanese from morphology.

- `--morphology-source existing`: require existing segmentation + annotation, validate them, freeze them, skip nrdb-morph prediction, and run only dictionary-grounded Japanese translation;
- `--morphology-source auto`: use existing morphology when present; otherwise predict it;
- `--morphology-source predict`: always infer morphology before translation.

Semantic feedback remains independent. For example, `--task translate --morphology-source predict --semantic-feedback none` translates the final predicted morphology but does not use generated Japanese to revise it. `--semantic-feedback generated` performs the semantic review before producing the final translation.

An existing human Japanese translation used as feedback is never exposed to the Japanese generation phase itself.

### `morph-translate`

Produce both finalized morphology and Japanese. `--semantic-feedback none|generated|existing|auto` independently controls whether Japanese semantics may revise morphology.

### `reverse`

Japanese -> Miyako IDs -> Miyako surface realization. Semantic feedback is not applicable. `--dialects` / `--target-dialects` supplies ordered target-dialect preference. Existing `NRDB_ID_MODEL` and `NRDB_SURFACE_MODEL` critics remain available.

## Registered NRDB jobs

Create a production morphology job on rows lacking annotation:

```bash
nrdb-agent create \
  --dataset-id 21 \
  --task morph \
  --semantic-feedback none \
  --needs annotation \
  --model gpt-5.6-terra
```

Use existing human translations as morphology constraints, but do not generate translation output:

```bash
nrdb-agent create \
  --dataset-id 21 \
  --task morph \
  --semantic-feedback existing \
  --morphology-source predict
```

Require such translations:

```bash
nrdb-agent create \
  --dataset-id 21 \
  --task morph \
  --semantic-feedback existing \
  --require-semantic-feedback
```

Translate only from existing/gold morphology:

```bash
nrdb-agent create \
  --dataset-id 21 \
  --task translate \
  --morphology-source existing \
  --semantic-feedback none
```

Select rows where annotation or translation is absent:

```bash
nrdb-agent create --dataset-id 21 --task morph-translate --needs either
```

Select one internal text from a text dataset:

```bash
nrdb-agent create --dataset-id 31 --task morph --text-id 34
```

Select an internal sentence/lxs ID interval:

```bash
nrdb-agent create --dataset-id 30 --task morph --sentence-id 12:21
```

Run normally:

```bash
nrdb-agent run JOB_ID
```

Export audited results:

```bash
nrdb-agent results JOB_ID --output results.tsv
```

## Portable XLSX

Portable workbooks are first-class inputs. `nrdb-agent` delegates workbook interpretation to `nrdb-morph.job.import_annotation_job`, so `_meta_`, `_cf_`, dialect resolution, annotation schema, region, enabled components, and existing morphology retain the same meaning as elsewhere in the NRDB suite.

The active Python environment must contain the local `nrdb-morph` package.

Morph all unannotated rows:

```bash
nrdb-agent process data.xlsx \
  --task morph \
  --semantic-feedback none \
  --needs annotation \
  --model gpt-5.6-terra \
  --output analyzed.tsv
```

For a portable lexicon workbook with more than one annotation component, select it explicitly:

```bash
nrdb-agent process lexicon.xlsx --component lxs --task morph --output lxs.tsv
```

Translation only from existing morphology:

```bash
nrdb-agent process data.xlsx \
  --task translate \
  --morphology-source existing \
  --semantic-feedback none \
  --output translations.tsv
```

## TSV

TSV uses common aliases for the canonical fields. Supported aliases include:

- source: `text`, `sentence_trsc2`, `trsc2`, `sentence`;
- dialect: `dialect_id`, `target_dialect_id`;
- segmentation: `segmented`, `sentence_trsc2_segmented`, `trsc2_seg`;
- annotation: `annotation`, `sentence_annotation`, `annotation_r`;
- Japanese: `translation_jp`, `translation`, `trsl`.

Because TSV has no `_meta_`, schema and region are supplied explicitly; `--dialect` can provide a fallback when rows lack a dialect column.

```bash
nrdb-agent process input.tsv \
  --task morph-translate \
  --semantic-feedback generated \
  --annotation-schema 2 \
  --region 宮古 \
  --dialect 19 \
  --output output.tsv
```

The TSV output preserves original columns and appends:

- `ai_segmented`
- `ai_annotation`
- `ai_translation`
- `ai_decision`
- `ai_confidence`
- `ai_cost_usd`
- `ai_model`
- `ai_error`
- `ai_evidence_json`

## Direct one-line demo

Miyako -> Japanese defaults to the full demonstrated pipeline with **generated semantic feedback**:

```bash
nrdb-agent translate 'aga za ndza...' \
  --target japanese \
  --annotation-schema 2 \
  --region 宮古
```

This is equivalent to adding:

```bash
--semantic-feedback generated
```

For ablations you can switch it off:

```bash
--semantic-feedback none
```

or supply an existing translation strictly as morphology evidence:

```bash
--semantic-feedback existing \
--existing-translation '東はどこにあるのか。'
```

The supplied existing translation is hidden from final Japanese generation.

## Morph ceiling evaluation

`nrdb-agent-morph-eval` supports the same semantic-feedback axis and durable checkpoints. To compare the three morphology conditions on the **same translation-present cohort**, keep run, model, seed, limit, and `--translation-filter present` fixed:

```bash
# A. Morph only
nrdb-agent-morph-eval RUN \
  --semantic-feedback none \
  --translation-filter present \
  --seed 4 --limit 20 --output eval_none.tsv

# B. Generated Japanese semantic feedback
nrdb-agent-morph-eval RUN \
  --semantic-feedback generated \
  --translation-filter present \
  --seed 4 --limit 20 --output eval_generated.tsv

# C. Existing human/data translation semantic feedback
nrdb-agent-morph-eval RUN \
  --semantic-feedback existing \
  --require-semantic-feedback \
  --translation-filter present \
  --seed 4 --limit 20 --output eval_existing.tsv
```

`--translation-filter any|present|absent` is independent of semantic feedback. It exists so ablation conditions can be evaluated on identical cohorts. Checkpoint metadata includes both controls, preventing accidental cross-condition resume.

## Missingness filters

`--needs` is independent of the requested task:

- `any`: no missingness filter;
- `annotation`: morphology absent;
- `translation`: Japanese translation absent;
- `either`: at least one absent;
- `both`: both absent.

For NRDB text datasets `--text-id` refers to `text_sentence_meta.text_id`. `--sentence-id` refers to the internal `ex_sen_lx.id` and therefore works for sentence and lxs rows as well as text utterances.

## Legacy jobs

The old `--mode blind_gold|unannotated`, `--prompt-version`, `--translate`, `--blind-translation`, and hidden `--translation-evidence` compatibility path remain accepted. New workflows should use `--semantic-feedback` explicitly.
