# TMDB Movie Agent

An agentic movie discovery and analysis system over the [TMDB 5000 Movie Dataset][kaggle].
An LLM plans and narrates; deterministic components decide and compute. Every number and
record in an answer comes from pandas or an exact vector search — never from the model.

Built to the SiliconExpert AI Engineer technical assignment
([`docs/SiliconExpert_AI_Engineer_Technical_Assignment_Movies.pdf`](docs/SiliconExpert_AI_Engineer_Technical_Assignment_Movies.pdf)).

[kaggle]: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

---

## Setup

Two commands before the app runs. The second one is the price of the build/serve split
([ADR-0005](docs/decisions/ADR-0005-offline-build-pandas-repository.md)) — it is the
design's weakest ergonomic point, and it exists because embedding ~4,800 documents cannot
happen on every Streamlit rerun.

```bash
# 1. environment (Python 3.12)
uv venv --python 3.12
uv pip install -e ".[dev]"          # or: uv pip install -r requirements.txt

# 2. dataset + vector index  (~10 min: the first run downloads a ~130 MB model)
python scripts/build_index.py

# 3. run
cp .env.example .env                # set LLM_API_KEY
streamlit run app.py
```

Both CSVs must be in `data/`. If they are missing, download them from
[Kaggle][kaggle] — `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.

### Configuration

Everything has a working default except `LLM_API_KEY`. See
[`.env.example`](.env.example), which is **generated** from the settings model
(`python scripts/gen_env_example.py`) so it cannot drift from the code.

| Variable | Default | Notes |
|---|---|---|
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Swap for a vLLM URL; no code change |
| `LLM_API_KEY` | — | Required for the agent only |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Must support native tool calling |
| `EMBEDDING_PROVIDER` | `sentence_transformers` | or `openai_compatible` |
| `EMBEDDING_BASE_URL` | — | **Independent of `LLM_BASE_URL`.** OpenRouter serves no `/v1/embeddings` — point this at OpenAI, vLLM, Ollama or LM Studio |
| `VECTOR_BACKEND` | `numpy` | or `qdrant` — see [Vector backends](#vector-backends) |
| `QDRANT_URL` | — | Unset means embedded Qdrant in `<ARTIFACTS_DIR>/qdrant`; set it to use a server |

**The deterministic half of the system needs no API key at all.** Preprocessing, structured
search, fuzzy title matching and the local embedding index all run without one; validation
is layered so the app refuses to start only on the paths that genuinely need a credential
([ADR-0015](docs/decisions/ADR-0015-config-secrets-layered-validation.md)).

### Vector backends

Vector search sits behind one protocol with two implementations, chosen by
`VECTOR_BACKEND`:

| | `numpy` (default) | `qdrant` |
|---|---|---|
| Where vectors live | `artifacts/embeddings.npy`, one in-process array | a Qdrant collection |
| Who applies filters | a boolean mask over row order | Qdrant payload filters, engine-side |
| Deployment | nothing to run | embedded by default; a server when `QDRANT_URL` is set |
| Needs | nothing beyond numpy | `qdrant-client` |

```bash
# load the existing vectors into an embedded Qdrant collection (no re-embedding)
python scripts/build_index.py --skip-embeddings --with-qdrant

# then either export VECTOR_BACKEND=qdrant, or set it in .env
```

Both backends must answer **identically** — the numpy mask is the definition of what a
filter means, and the Qdrant payload filter is a second implementation of that same
definition. A drift between them would not raise; it would quietly return plausible
movies that violate a constraint the user asked for. So it is measured, not assumed:

```bash
python scripts/benchmark_vector_backends.py     # exits non-zero on any disagreement
```

Across eight filter shapes (membership, case folding, numeric ranges, `between`,
cast-or-director, empty pools) the two agree exactly: identical candidate pools,
identical top-k, identical ordering, scores within 2.5e-07. `tests/test_vector_backends.py`
asserts the same thing in CI.

On latency, embedded Qdrant is **~180x slower** than the numpy index at this corpus size,
and the shape of the gap is instructive: numpy gets *faster* as a filter narrows, while
embedded Qdrant gets *slower* — local mode evaluates payload conditions in Python and
ignores payload indexes entirely, so every filtered query walks the collection. A served
Qdrant applies those filters through indexed structures and would need re-measuring. Full
numbers: [`artifacts/vector_backend_benchmark.md`](artifacts/vector_backend_benchmark.md).

That is why `numpy` remains the default. `qdrant` exists because ADR-0006 named its own
expiry condition — O(N·d) per query does not bend — and this is that exit, wired and
verified before it is needed rather than during an incident. Embedded mode takes an
exclusive lock on its directory, so the app and a script cannot both hold it; concurrent
readers need a server.

### Tests

```bash
pytest                # 137 tests, no network, ~25s
pytest -m live        # opt-in: a couple of real turns against the configured provider
```

The default run makes **no network calls**. Run `-m live` after any LangChain/LangGraph
upgrade — the fake chat model encodes the library's message contract, so a version bump
could break the integration while the suite stays green
([ADR-0024](docs/decisions/ADR-0024-testing-fake-chat-model.md)).

---

## What it does

Five tools with deliberately non-overlapping boundaries:

| Tool | Handles | Does **not** handle |
|---|---|---|
| `structured_search` | Filters, sorting, aggregation, counts, numeric comparisons | Approximate titles; plot/theme queries |
| `fuzzy_movie_search` | *Title string → movie identity* ("Intersteler" → Interstellar) | Finding similar movies |
| `semantic_search` | *Meaning → movies*, plus hybrid metadata pre-filtering | Resolving a title the user typed |
| `movie_details` | The full record for one movie id | Search of any kind |
| `rag_answer` | Grounded prose over an explicit id list | Retrieval |

The pair most easily confused is fuzzy vs semantic — the PDF files "what is similar to lord
of the rings?" under both. The rule: **fuzzy maps strings to a movie, semantic maps meaning
to movies.** That query is a two-tool chain, not one tool doing both jobs.

### Try these

```
What are the 10 most common genres?
Show the 10 highest-rated science fiction movies with at least 1,000 votes
Tell me about Intersteler
I want a movie about someone trying to survive alone on another planet
Find me a funny science-fiction movie from after 2010 that is under 2 hours
How many movies have Christopher Nolan as director?
What movies are similar to lord of the rings?      ← asks which one you meant
```

Multi-turn:

```
Show me science fiction movies from after 2010
Only show ones rated above 7.5
Tell me more about the first one
```

---

## How it works

```
START → plan → agent ⇄ tools → synthesize → ground → END
          │       │       │
          │       │       └→ clarify ─(interrupt)→ resumed next turn → agent
          └→ smalltalk → END
```

A LangGraph `StateGraph`. The `plan` node emits a typed plan *before* any tool runs — the
tool sequence, extracted filters, and a one-line rationale shown in the UI. The
`agent ⇄ tools` cycle executes. If a title match comes back ambiguous, a conditional edge
routes to a real `interrupt`: the turn pauses, the checkpointer holds the state, and your
next message **resumes that turn** rather than starting a new one. `ground` checks the
finished answer for movies that were never retrieved.

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Project layout

```
app.py                      Streamlit UI
scripts/build_index.py      Offline preprocessing + embedding build (+ --with-qdrant)
scripts/benchmark_vector_backends.py   numpy vs Qdrant: agreement and latency
scripts/test_qdrant.py      Drive the Qdrant collection: real queries, filters, latency
scripts/profile_data.py     Raw-CSV data analysis report
scripts/profile_documents.py           Document length statistics
scripts/build_sqlite.py     Side utility: movies.parquet -> artifacts/movies.db
src/movieagent/
  config.py                 pydantic-settings, layered validation
  data/                     preprocessing, repository, filter DSL   ← framework-free
  retrieval/                documents, vector backends, fuzzy, coverage ← framework-free
  tools/                    the five tools, pure functions           ← framework-free
  llm/                      chat model factory, sanitizer, embeddings
  agent/                    graph, state, nodes, prompts, trace, grounding
  ui/                       result and trace rendering
tests/                      137 tests
docs/                       requirements, architecture, ADRs, agent documentation
```

Import direction is one-way and **enforced by a test**: `langgraph`/`langchain*` may be
imported only under `agent/` and `llm/`, and `streamlit` only under `ui/`. That containment
is why swapping the agent runtime (ADR-0001 → ADR-0020) touched one package rather than the
whole system.

---

## Documentation

| Document | What it is |
|---|---|
| [`docs/AGENT_DOCUMENTATION.md`](docs/AGENT_DOCUMENTATION.md) | **Deliverable 3** — data decisions, agent design, search, RAG pipeline, memory, worked examples, limitations |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The system as built: diagrams, state model, trust boundaries, extension points, rejected architectures |
| [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) | Every requirement extracted from the PDF, numbered and testable |
| [`docs/TRACEABILITY.md`](docs/TRACEABILITY.md) | Requirement → component → test → ADR |
| [`docs/PATTERNS.md`](docs/PATTERNS.md) | Every design pattern used, and the simpler alternative passed over |
| [`docs/decisions/`](docs/decisions/) | 26 ADRs, including two reversals and their evidence |
| [`docs/OPEN-QUESTIONS.md`](docs/OPEN-QUESTIONS.md) | Ambiguities in the brief and how each was resolved |

---

## Known limitations

Stated here rather than only in the conclusions, because they will be visible in use.

1. **The dataset ends around 2016.** No film released after mid-2017 exists here. Asked
   about a recent movie the agent says so rather than answering from memory — but a
   fluent query about an absent topic (*"the invasion of Ukraine in 2022"*) returns
   confident-looking films and nothing detects it.
2. **Grounding catches invented *movies*, not invented *numbers*.** A wrong runtime in
   prose passes the check. Numbers in tables come from the tool payload, so the exposure is
   limited to narration.
3. **Retrieval finds themes, not famous films.** "A hacker discovers reality is a
   simulation" returns *Primer*, *The Thirteenth Floor* and *eXistenZ* — all correct — while
   *The Matrix* ranks ~113th, because its stored overview describes Neo and Morpheus and
   never says "simulated reality". Three such cases are pinned as `xfail` tests.
4. **Fuzzy thresholds rest on 20 hand-chosen cases**, not observed queries.
5. **One prior result set is remembered**, so "go back to that earlier list" fails.
6. **Two LLM calls minimum per turn** — the planning stage costs latency on simple queries.
7. **Conversations do not survive a restart.** Durable checkpointing is one line away
   (`MemorySaver` → `SqliteSaver`) and deliberately not enabled.

---

## Deliberately not built

MCP interface, REST API, web-search fallback, and hosted deployment — all out of scope by
agreement, each with its reasoning and a revisit trigger in
[ADR-0017](docs/decisions/ADR-0017-deliberate-non-decisions.md). Web search additionally
conflicts with the "never invent movie metadata" requirement unless a full provenance model
is built, and half-doing it would be worse than not doing it.

## Version control

This tree is not a git repository — version control was left to the repository owner. The
[ADR index](docs/decisions/README.md) ends with a 19-step commit plan mapped to the ADRs, so
the "meaningful commits" intent stays recoverable. A `.gitignore` is included and ready to
adopt.
