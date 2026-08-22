# nrdb-agent

Constrained, auditable AI agents for NRDB. The first workflow performs morphemic annotation jobs without overwriting human annotation.

## Design

NRDB owns jobs, corpus data, lexical grounding, human annotations, results, and the deployed model registry. `nrdb-agent` owns production morphology/translation execution and LLM reasoning. It talks to NRDB through loopback-only PHP endpoints and to the dedicated `nrdb-morph` installation through its local HTTP service.

All production inputs converge on one `ExecutionRequest` contract:

- `create` records a scoped NRDB job and immutable execution policy; `run` executes it and stores results;
- `process` reads portable XLSX/TSV data and exports local results without creating a database job;
- `translate` creates one direct request without creating a database job;
- `execute` accepts one complete `nrdb-agent.execution-request.v1` JSON object for programmatic callers.

`create`, `process`, and `translate` share the same execution policy. `--nrdb-evidence
enabled|none` controls access to NRDB linguistic evidence (default: `enabled`), while
`--morphology-source none|predict|existing|auto` independently controls the morphology
input. Invalid combinations are rejected before an item or registered job is executed.

Evaluation, frozen cohorts, ablations and discrepancy experiments belong to the separate `nrdb-exp` repository.

Corpus evidence is retrieved from NRDB's current v2 annotation index. Atomic IDs, conflated segments such as `A;cvb`, and segment sequences such as `A-dat` are supported.

The agent cannot invent ordinary database IDs, modify UniCog, alter dictionaries, or overwrite `sentence_annotation` or `translation_jp`. The reserved productive `n:` namespace is the explicit Japanese lexical-reservoir exception in reverse translation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY='...'
export NRDB_AGENT_URL='http://127.0.0.1/php/agent.php'
export NRDB_MORPH_URL='http://127.0.0.1:8765'
```

`NRDB_AGENT_URL` must resolve to the same host because the NRDB endpoint is intentionally loopback-only.

## Direct translation demo

Translate arbitrary Miyako to Japanese without creating an annotation job:

```bash
nrdb-agent translate 'miyako text here' \
	--target japanese \
	--annotation-schema 2 \
	--region 宮古
```

The command resolves a representative schema-scoped dialect from NRDB for the morph request. The deployed nrdb-morph model is selected by the existing NRDB/nrdb-morph service architecture; there is no morph-model selector in `nrdb-agent`.

Translate arbitrary Japanese to Miyako with an ordered target-dialect preference:

```bash
nrdb-agent translate '日本語の文' \
	--target miyako \
	--annotation-schema 2 \
	--region 宮古 \
	--dialects 19,22,14 \
	--surface-model ../nrdb-morph/path/to/surface_model.json
```

Reverse translation runs Japanese-to-ID reasoning, dialect-scoped trsc2 surface retrieval, annotation-syntax enforcement, and an optional explicitly selected nrdb-morph allomorph/phonotactic critic. Critics are never silently loaded from environment variables. Add `--json` to print the complete audit result instead of the compact demo output.

## Registered production job

```bash
nrdb-agent create --dataset-id 27 --task morph --needs annotation --limit 20 --model gpt-5.6
nrdb-agent list
nrdb-agent run JOB_ID
nrdb-agent show JOB_ID
```

Create a translation job from existing morphology:

```bash
nrdb-agent create --dataset-id 27 --task translate --morphology-source existing \
	--semantic-feedback none --constructions --needs translation --limit 20
```

The job record freezes the exact morphology-review, critic and resegmentation policy. `run` cannot silently override it.

## Corpus-based grammatical ID analysis

Create one auditable job for several exact annotation IDs, optionally supplying expert guidance for individual IDs:

```bash
nrdb-agent id-analysis create adv foc 'ppt>2' \
	--annotation-schema 2 --region 宮古 \
	--note 'foc=Focus particle; devise a natural Japanese policy, including licensed omission.' \
	--model gpt-5.6-terra
nrdb-agent id-analysis run JOB_ID --output id-analysis-JOB_ID.tsv
```

The agent analyzes exact CPS occurrences, same-phrase patterns, anchored corpus-native N-grams, and Japanese sentence translations. It proposes disabled `morpheme` and `construction` rows for human review; it never inserts them into NRDB. See [docs/id-analysis.md](docs/id-analysis.md).
