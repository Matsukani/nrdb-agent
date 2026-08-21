# Morphology and translation workflows

`nrdb-agent` separates the workflow into independent axes:

1. **source**: registered NRDB dataset, portable `_meta_`/`_cf_` XLSX, TSV, or direct one-line input;
2. **task/output**: `morph`, `translate`, `morph-translate`, or `reverse`;
3. **semantic feedback for morphology**: `none`, `generated`, `existing`, or `auto`;
4. **morphology source**: `predict`, `existing`, or `auto`;
5. **constructional evidence for Japanese translation**: off by default, enabled with `--constructions`;
6. **morph review**: raw nrdb-morph (`--morph-review none`) or agent review (`--morph-review agent`);
7. **optional capabilities**: ID critic, surface critic, and resegmentation are independent and disabled unless explicitly passed;
8. **selection/output**: NRDB scope/missingness or local TSV/JSON output.

The central rule is that **producing a Japanese translation and using Japanese as semantic feedback are different decisions**. A translation may be generated internally to review morphology without becoming an output, and a requested translation may be produced without reopening morphology.

## Semantic feedback

`--semantic-feedback` applies while predicted morphology is still reviewable:

- `none`: morphology is reviewed only from nrdb-morph, NRDB lexical/corpus evidence, critics, and annotation constraints;
- `generated`: generate a dictionary-grounded Japanese translation internally, then use it for a semantic consistency review of morphology;
- `existing`: use the Japanese translation already present in the dataset as semantic evidence;
- `auto`: use an existing translation when available, otherwise generate one internally.

`--require-semantic-feedback` makes the selected source mandatory. It is most useful with `existing` when an experiment must contain only rows with human/gold Japanese translations.

If generated semantic feedback revises morphology and Japanese translation is also a requested output, the final Japanese is regenerated from the revised morphology so annotation and translation cannot diverge.

## Curated constructional evidence

`--constructions` adds an explicit grammatical pass before Miyako -> Japanese generation. It is orthogonal to semantic feedback and morphology source.

The backward-compatible NRDB table `annotation_constructions` stores schema/region/dialect-scoped grammatical-knowledge rows with:

- `entry_type`: `morpheme` for the default function of one exact ID, or `construction` for a contextual specialization;
- `trigger_id`: one exact atomic annotation ID used for cheap candidate retrieval;
- `pattern`: a lightweight pattern over the frozen NRDB annotation;
- `meaning_jp`: the construction-level Japanese interpretation;
- `realization_jp`: a strong Japanese realization hint;
- `note`, `priority`, and `enabled`.

An exact trigger hit makes a `morpheme` policy applicable: the translator must consult it but must still choose a contextual realization rather than substitute `realization_jp` mechanically. A trigger hit retrieves a `construction` row as a candidate only: the translator must verify that the full pattern fits the frozen annotation before using it. When it fits, the curated construction meaning specializes and may outrank the default morpheme or atom-by-atom reading. The translation audit records every consulted morpheme row and classifies every retrieved construction row as applied or rejected.

Until NRDB has an explicit local morph-ID classification table, translation uses a deliberately small namespace policy: `l:` is local lexical, `exp:` expressive, `intj:` interjective, and every other schema-local ID is provisionally grammatical, including `dm:` demonstratives. Global IDs retain semantically opaque stems but expose their final `n/v/a/o` structural POS code for construction-placeholder matching; global IDs remain eligible construction anchors.

The first resource is intentionally hand-curated and rerunnable from NRDB's:

```text
sql/constructions/miyako_translation.sql
```

No extra LLM tool round is spent on the grammatical pass: NRDB retrieves rows deterministically before the first translation response. Retrieved rows are preserved in translation evidence for audit.

Candidate rows can be researched from CPS and exported for human review with [`nrdb-agent id-analysis`](id-analysis.md). That workflow never writes directly to the curated table.

## Licensed generated forms

`--licensed` treats human-licensed `generated_wordforms` rows as grammatical evidence for forward morphology. Retrieval has two ordered paths: exact lookup against the current decoder surface segments, followed by containment lookup against the unsegmented source text for cases where the decoder boundary is wrong. Returned evidence includes its retrieval path and source-text offsets.

`--resegmentation` explicitly grants annotation-v9 one bounded batch of up to four alternative segmentations through the current nrdb-morph model with those boundaries fixed. Without that flag, segmentation boundaries are frozen and the tool is not exposed to the agent. Alternatives must reconstruct the observed source exactly and preserve phrase spaces. A final boundary change must equal a successfully tested candidate and pass structural validation, otherwise the host restores the morph baseline. `--surface-model PATH` independently adds comparative surface compatibility evidence; it never silently enables resegmentation. Gold/existing boundaries are always authoritative.

An exact, same-dialect licensed form may deterministically replace one `?` decoder segment only when its licensed segmentation and annotation align, no competing licensed analysis exists, and the complete revised analysis passes `nrdb-morph` structural validation. Regional matches, containment matches, ambiguous matches, and already analysed segments are supplied to the agent as structured candidates instead of being silently rewritten. Licensed evidence is also carried into semantic review. `evidence.licensed_form_audit` records the pre-LLM repair decision and which retrieved forms occur in the final analysis.

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

Add `--constructions` when curated constructional evidence should participate in Japanese interpretation:

```bash
nrdb-agent process data.xlsx --task translate --morphology-source existing --constructions
```

An existing human Japanese translation used as feedback is never exposed to the Japanese generation phase itself.

### `morph-translate`

Produce both finalized morphology and Japanese. `--semantic-feedback none|generated|existing|auto` independently controls whether Japanese semantics may revise morphology. `--constructions` independently controls curated constructional evidence whenever Japanese is generated.

### `reverse`

Japanese -> Miyako IDs -> Miyako surface realization. Semantic feedback and `--constructions` are not applicable. `--dialects` / `--target-dialects` supplies ordered target-dialect preference. Critics are enabled only by explicit `--id-model PATH` and `--surface-model PATH` options.

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

Translate only from existing/gold morphology with constructional evidence:

```bash
nrdb-agent create \
  --dataset-id 30 \
  --task translate \
  --morphology-source existing \
  --semantic-feedback none \
  --constructions \
  --needs translation \
  --model gpt-5.6-terra
```

Select one internal text from a text dataset:

```bash
nrdb-agent create --dataset-id 31 --task morph --text-id 34
```

`--text-id` means `texts_list.text_internal_id`, scoped by dataset; the database mapping to sentence rows is handled internally.

Select an internal sentence/lxs ID interval:

```bash
nrdb-agent create --dataset-id 30 --task morph --sentence-id 12:21
```

Run normally:

```bash
nrdb-agent run JOB_ID
```

Creation freezes the complete forward-morphology policy in the NRDB job: review mode, resegmentation capability, candidate limit, critic paths, and critic file hashes when available. `run` replays that policy and rejects morphology-policy overrides.

Portable `process` output includes the untouched `morph_segmented`/`morph_annotation` baseline, final agent analysis, and serialized policy.

## Orthogonal morph evaluation

`nrdb-agent-morph-eval` exposes the same axes. Freeze selected rows once and reuse them across ablations:

```bash
nrdb-agent-morph-eval RUN --limit 100 --cohort-out cohort.json --morph-review none --output raw.json
nrdb-agent-morph-eval RUN --cohort-in cohort.json --morph-review agent --output agent.json
nrdb-agent-morph-eval RUN --cohort-in cohort.json --morph-review agent --surface-model surface.json --output surface.json
nrdb-agent-morph-eval RUN --cohort-in cohort.json --morph-review agent --resegmentation --surface-model surface.json --output reseg.json
```

The cohort has a content fingerprint; checkpoints freeze that fingerprint and the full policy manifest. A mismatched resume is rejected.

Export audited results:

```bash
nrdb-agent results JOB_ID --output results.tsv
```

## Portable XLSX

Portable workbooks are first-class inputs. `nrdb-agent` delegates workbook interpretation to `nrdb-morph.job.import_annotation_job`, so `_meta_`, `_cf_`, dialect resolution, annotation schema, region, enabled components, and existing morphology retain the same meaning as elsewhere in the NRDB suite.

Translation only from existing morphology with constructions:

```bash
nrdb-agent process data.xlsx \
  --task translate \
  --morphology-source existing \
  --semantic-feedback none \
  --constructions \
  --output translations.tsv
```

## Direct one-line demo

Miyako -> Japanese defaults to generated semantic feedback, while constructional evidence is opt-in. To isolate the construction effect, compare the same input with semantic feedback disabled in both runs:

```bash
nrdb-agent translate 'SOURCE' \
  --target japanese \
  --annotation-schema 2 \
  --region 宮古 \
  --model gpt-5.6-terra \
  --semantic-feedback none
```

versus:

```bash
nrdb-agent translate 'SOURCE' \
  --target japanese \
  --annotation-schema 2 \
  --region 宮古 \
  --model gpt-5.6-terra \
  --semantic-feedback none \
  --constructions
```

With `--verbose`, construction-aware runs report `translation-v7: construction pass candidates=N`.

## Missingness filters

`--needs` is independent of the requested task:

- `any`: no missingness filter;
- `annotation`: morphology absent;
- `translation`: Japanese translation absent;
- `either`: at least one absent;
- `both`: both absent.

## Legacy jobs

The old `--mode blind_gold|unannotated`, `--prompt-version`, `--translate`, `--blind-translation`, and hidden `--translation-evidence` compatibility path remain accepted. New workflows should use the orthogonal controls explicitly.

For `nrdb-agent-morph-eval`, corpus evidence is row-blind by default: the currently scored sentence is excluded from corpus examples and form-ID support. Use `--blind-policy cohort` to exclude every sampled evaluation sentence from corpus-backed evidence for the complete run. The selected policy and resulting exclusion ranges are stored in the durable checkpoint and cannot change on resume. Explicit `--exclude-dataset`, `--exclude-text`, and `--exclude-sentences` restrictions are combined with the selected policy. Dictionaries and human-curated licensed forms remain available as declared knowledge sources.
