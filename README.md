# nrdb-agent

Constrained, auditable AI agents for NRDB. The first workflow performs morphemic annotation jobs without overwriting human annotation.

## Design

NRDB owns jobs, corpus data, lexical grounding, gold annotations, and results. `nrdb-agent` owns only orchestration and LLM reasoning. It talks to NRDB through the loopback-only `public/php/agent.php` endpoint and to `nrdb-morph` through its local HTTP service.

Initial job modes:

- `blind_gold`: evaluate on already human-annotated material. Gold annotation is never returned in the work item; it is revealed only after the AI result has been stored and scored.
- `unannotated`: propose annotations for rows with blank human annotation. Results remain separate from the human annotation.

Translation options:

- `--translate`: also generate a Japanese translation and store it as `trsl_ai`; an existing human `translation_jp` may still be used as annotation evidence.
- `--blind-translation`: implies translation, withholds `translation_jp` from the agent, and stores the human translation only after submission as `gold_translation_jp` for later evaluation.

Corpus evidence is retrieved from NRDB's current v2 annotation index. Atomic IDs, conflated segments such as `A;cvb`, and segment sequences such as `A-dat` are supported.

The agent cannot invent database IDs, modify UniCog, alter dictionaries, or overwrite `sentence_annotation` or `translation_jp`.

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

## First blind experiment

```bash
nrdb-agent create --dataset-id 27 --mode blind_gold --limit 20 --model gpt-5.6
nrdb-agent list
nrdb-agent run JOB_ID
nrdb-agent show JOB_ID
```

With translation enabled but human translation still available as annotation evidence:

```bash
nrdb-agent create --dataset-id 27 --mode blind_gold --limit 20 --model gpt-5.6 --translate
```

For true blind translation:

```bash
nrdb-agent create --dataset-id 27 --mode blind_gold --limit 20 --model gpt-5.6 --blind-translation
nrdb-agent run JOB_ID
nrdb-agent show JOB_ID
```

Then scale to 500 only after inspecting small runs:

```bash
nrdb-agent create --dataset-id 27 --mode blind_gold --limit 500 --model gpt-5.6 --seed 2 --blind-translation
nrdb-agent run JOB_ID
nrdb-agent show JOB_ID
```

The running agent does not learn online from gold mismatches. Blind results are retained as evaluation evidence for explicit later revisions to prompts, tools, or models.
