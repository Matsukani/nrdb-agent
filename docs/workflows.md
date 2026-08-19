# Morphology and translation workflows

`nrdb-agent` separates five concerns that were previously entangled in one annotation job:

1. **source**: registered NRDB dataset, portable `_meta_`/`_cf_` XLSX, or TSV;
2. **task**: `morph`, `translate`, `morph-translate`, or `reverse`;
3. **translation evidence** for morphology: `ignore`, `use`, or `required`;
4. **morphology source**: `predict`, `existing`, or `auto`;
5. **selection/output**: NRDB scope/missingness or local TSV/JSON output.

## Task semantics

### `morph`

Produce Miyako segmentation and morphemic annotation only.

- `--translation-evidence ignore`: do not expose an existing Japanese translation.
- `--translation-evidence use`: if a human Japanese translation exists, use it only in a final semantic consistency review of the proposed morphology.
- `--translation-evidence required`: as above, but fail/skip rows without a human translation.

No Japanese translation is generated.

### `translate`

Produce Japanese from morphology.

- `--morphology-source existing`: require existing segmentation + annotation, validate them, freeze them, skip nrdb-morph prediction, and run only dictionary-grounded Japanese translation.
- `--morphology-source auto`: use existing morphology when present; otherwise predict it.
- `--morphology-source predict`: always infer morphology before translation.

An existing human Japanese translation is never exposed to the Japanese generation phase, so translation-only evaluation does not leak the target.

### `morph-translate`

Infer/review morphology and then produce Japanese from the finalized morphology. With `--translation-evidence use|required`, a human translation constrains morphology before the generated Japanese phase.

### `reverse`

Japanese -> Miyako IDs -> Miyako surface realization. `--dialects` / `--target-dialects` supplies ordered target-dialect preference. Existing `NRDB_ID_MODEL` and `NRDB_SURFACE_MODEL` critics remain available.

## Registered NRDB jobs

Create a production morphology job on rows lacking annotation:

```bash
nrdb-agent create \
  --dataset-id 21 \
  --task morph \
  --needs annotation \
  --model gpt-5.6-terra
```

Use existing human translations as morphology constraints, but do not generate translations:

```bash
nrdb-agent create \
  --dataset-id 21 \
  --task morph \
  --translation-evidence use \
  --morphology-source predict
```

Translate only from existing/gold morphology:

```bash
nrdb-agent create \
  --dataset-id 21 \
  --task translate \
  --morphology-source existing
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

## Missingness filters

`--needs` is independent of the requested task:

- `any`: no missingness filter;
- `annotation`: morphology absent;
- `translation`: Japanese translation absent;
- `either`: at least one absent;
- `both`: both absent.

For NRDB text datasets `--text-id` refers to `text_sentence_meta.text_id`. `--sentence-id` refers to the internal `ex_sen_lx.id` and therefore works for sentence and lxs rows as well as text utterances.

## Legacy jobs

The old `--mode blind_gold|unannotated`, `--prompt-version`, `--translate`, and `--blind-translation` flags remain accepted. Supplying one of those flags creates a legacy job through the original API. New jobs use the task-based workflow endpoint.
