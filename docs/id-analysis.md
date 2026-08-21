# Corpus-based grammatical ID analysis

`nrdb-agent id-analysis` turns CPS evidence for exact morphemic IDs into reviewable candidate rows for NRDB's `annotation_constructions` table. It can also test strongly supported combinatorial patterns against POS-tagged lexicon hosts and produce a separate review queue of missing licensed wordforms. It is a corpus-analysis and typing aid, not an autonomous grammar writer.

## Workflow

Create one job for one or several exact atomic IDs:

```bash
nrdb-agent id-analysis create adv foc 'ppt>2' \
	--annotation-schema 2 \
	--region 宮古 \
	--provider-dataset-id 21 29 \
	--probe-limit 24 \
	--probe-seed 42 \
	--note 'foc=foc is a focus particle. Devise a Japanese policy that may omit it when Japanese has no natural overt counterpart.' \
	--model gpt-5.6-terra
```

Quote IDs containing shell metacharacters such as `>`.

Run the job and write the candidate table:

```bash
nrdb-agent id-analysis run JOB_ID --output id-analysis-JOB_ID.tsv
```

When provider datasets were selected, this also writes `id-analysis-JOB_ID-licensed.tsv`. Use `--licensed-output PATH` to choose the companion filename explicitly.

Inspect the stored audit result or recover its TSV later:

```bash
nrdb-agent id-analysis show JOB_ID --json
nrdb-agent id-analysis show JOB_ID --output id-analysis-JOB_ID.tsv
nrdb-agent id-analysis show JOB_ID --licensed-output id-analysis-JOB_ID-licensed.tsv
nrdb-agent id-analysis list
```

`--instructions` supplies guidance shared by every target. Repeated `--note 'ID=TEXT'` values supply ID-specific expert guidance. Evidence can be restricted with repeated `--source txt|sen|lxs`, repeated `--dataset-id`, `--region`, and `--dialect`.

The two dataset scopes are deliberately different:

- `--dataset-id` restricts attested CPS evidence.
- `--provider-dataset-id ID [ID ...]` selects lexicon datasets that supply annotated wordforms, POS/morphological categories, and meanings for productive tests. The option may be repeated.

Provider sampling is deterministic for a target and `--probe-seed`. The run first tests a small attested gold subset, then the agent infers a strong formal host pattern from exact same-phrase evidence and uses provider POS and meaning data to state both its morphological and semantic scope and select credible productive test lexemes. `--probe-limit` caps all morphology-model calls per target; when a productive pattern exists, up to eight calls are reserved for attested controls and the remainder for synthetic probes.

## CPS evidence contract

For every target ID, NRDB returns:

- exact schema-qualified token, phrase, sentence, dataset, dialect, and translated-token counts;
- schema-local ID metadata;
- existing curated grammar rows with the same trigger;
- exact content-only same-phrase patterns and attested segmented forms;
- corpus-native, phrase-bounded, control-free anchored bigrams and trigrams;
- short examples balanced toward translated attestations and distinct phrase patterns;
- complete source/segmentation/annotation/Japanese sentence evidence and provenance.
- when requested, a deterministic sample of provider lexemes with source dataset, dialect, segmented forms, annotation, exact POS, first Japanese meaning, and explanation.

The Japanese tier is sentence-level evidence, not a morpheme alignment. The prompt requires recurring contrasts before assigning a Japanese realization to the target. N-grams rank recurrent environments but cannot by themselves establish a construction because they do not retain every segmentation/conflation boundary. Construction patterns must be checked against exact phrase annotations.

## Output

Each supported ID produces exactly one `morpheme` candidate and at most four `construction` candidates. The TSV begins with the database-facing columns:

```text
annotation_schema_id region dialect_id entry_type name trigger_id pattern
meaning_jp realization_jp note priority enabled
```

Audit columns follow with linguistic names, confidence, evidence counts, selected example keys, exact example JSON, expert notes, and warnings.

The companion licensed-form TSV starts with the existing `generated_wordforms` update fields and leaves `generated_id` blank. Rows have `license_status=candidate`, never `licensed`. Audit columns record the productive hypothesis, provider lexeme and meaning, semantic scope, expected analysis, actual model output, validator result, and licensed-table check. Before ingestion, a human must approve the linguistic combination, assign a dataset-local `generated_id`, and remove or otherwise handle the review-only columns required by the chosen update workflow.

`name` is always a stable English `lower_snake_case` identifier scoped by the exact target ID. Morpheme names are derived deterministically from `linguistic_name_en`, for example `focus_particle_foc` and `potential_ppt_2`; `linguistic_name_jp` remains the Japanese display label.

All candidates have `enabled=0`. The agent cannot insert them into `annotation_constructions`. Example selection is also constrained: the model returns CPS evidence keys, and `nrdb-agent` reconstructs the actual source, annotation, and translation from NRDB. Unknown or invented keys are discarded and recorded as warnings.

Productive morphology hypotheses are constrained in the same way: the model may cite only exact same-phrase patterns and lexeme IDs present in the frozen evidence. `nrdb-agent` reconstructs the form, validates the proposed segmentation/annotation, checks the current licensed table, and runs the current morphology model. A wordform becomes a candidate only if the expected analysis is structurally valid, the model does not return it exactly, and no exact licensed row already exists. This first version probes simple overt segment concatenation only; fusion, stem alternation, sandhi repair, and zero exponence remain outside scope.

## Interpretation policy

- `morpheme`: the exact target ID licenses its general grammatical function and Japanese realization policy.
- `construction`: a trigger hit retrieves a candidate, but the complete pattern must match before translation uses it.
- A matching construction specializes and may override the general morpheme policy.
- Recurrent co-occurrence is not sufficient for construction status; specialized meaning or realization must also be supported.
- Jobs with no exact CPS attestations complete without proposing a row for that ID.

The full JSON result, evidence snapshot, API usage, and TSV are stored with the job for later review and reproducibility.
