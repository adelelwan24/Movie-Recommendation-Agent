# TMDB Movie Agent

An agentic movie discovery and analysis system over the [TMDB 5000 Movie Dataset][kaggle].
An LLM plans and narrates; deterministic components decide and compute. Every number and
record in an answer comes from pandas or an exact vector search — never from the model.

Built to the SiliconExpert AI Engineer technical assignment. The repository review and
requirement assessment are in [`docs/`](docs/README.md).

[kaggle]: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

---

## Quickstart

From a fresh clone to a running app. Steps 1-4 need **no API key**; only the agent itself does.

**Prerequisites**

| | |
|---|---|
| Python | 3.12 (pinned `>=3.12,<3.13`) |
| Package manager | [`uv`](https://docs.astral.sh/uv/) recommended; plain `pip` works |
| Disk | ~3 GB — torch is ~2 GB, the embedding model ~130 MB, the artifacts ~12 MB |
| Network | Needed once, to download the embedding model. Everything after that runs offline |
| API key | Only for the agent (OpenRouter by default). Preprocessing, structured search, fuzzy matching and the vector index all work without one |

### 1. Put the dataset in `data/`

Download the [TMDB 5000 Movie Dataset][kaggle] from Kaggle and unzip both CSVs into `data/`:

```bash
ls data/
# tmdb_5000_credits.csv    (~39 MB, 4,803 rows)
# tmdb_5000_movies.csv     (~5.5 MB, 4,803 rows)
```

Filenames must match exactly — the build script looks for those two names.

### 2. Create the environment

```bash
uv venv --python 3.12
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

uv pip install -e ".[dev]"          # or: uv pip install -r requirements.txt
```

`requirements.txt` is a fully resolved lock of the versions this was tested against;
`pyproject.toml` holds the intended ranges. Use the lock if you want the exact tree.

### 3. Configure

```bash
cp .env.example .env                # Windows: copy .env.example .env
```

Then set one value in `.env`:

```ini
LLM_API_KEY=sk-or-...               # OpenRouter key, or any OpenAI-compatible provider
```

Everything else has a working default. `.env.example` is **generated** from the settings model
(`python scripts/gen_env_example.py`), so it cannot drift from the code.

> **Watch `DATA_DIR` and `ARTIFACTS_DIR`.** Both default to this checkout and can be left
> unset. If you do set them — or copy a `.env` from another clone — make sure they point at
> *this* directory. An absolute path left over from elsewhere makes the app silently read
> another folder's artifacts, which looks like the build "not taking effect".

### 4. Build the artifacts

This is the step that turns the two CSVs into everything the app loads:

```bash
python scripts/build_index.py
```

It reads both CSVs, joins them, normalises the columns, builds one semantic document per
movie, embeds all 4,803 of them, and writes a manifest recording exactly what it used.
**Expect roughly 5-15 minutes on CPU the first time** — most of it the one-off model download
plus the embedding pass. Later rebuilds skip the download.

On success it prints the real counts, not estimates:

```
Build complete.
  movies processed          4803
  joined from               4803 movies / 4803 credits
  unmatched on join         0 movies, 0 credits
  missing release_date      1
  missing runtime           37
  budget 0 -> unknown       1037
  revenue 0 -> unknown      1427
  no director in crew       30
  no keywords               412
  embedding dimension       384
  artifacts                 <repo>/artifacts
  elapsed                   ...s

Next:  streamlit run app.py
```

Four files land in `artifacts/`:

| File | Size | What it is |
|---|---|---|
| `movies.parquet` | 2.9 MB | The processed dataset — 28 columns, nullable dtypes, real dates. Parquet because a CSV round-trip would undo the unknown-is-not-zero semantics |
| `documents.parquet` | 1.9 MB | One labelled semantic document per movie (mean 95 words) |
| `embeddings.npy` | 7.4 MB | 4,803 x 384 float32, L2-normalised |
| `manifest.json` | 809 B | Provenance: source SHA-256s, row count, embedding model + dimension, template version, build report |

The manifest is what makes a stale artifact fail **loudly**. If the CSVs, the preprocessing
version or the embedding model change, loading raises with instructions instead of quietly
answering from an outdated index.

Useful flags:

```bash
python scripts/build_index.py --force             # rebuild even if up to date
python scripts/build_index.py --skip-embeddings   # dataset only, while iterating on preprocessing
python scripts/build_index.py --with-qdrant       # also load the vectors into Qdrant
```

### 5. Verify the build (optional)

```bash
pytest                                  # 186 pass, no network, ~40s
python scripts/profile_data.py          # profiles the raw CSVs   -> artifacts/data_analysis.md
python scripts/profile_documents.py     # document length stats   -> artifacts/document_stats.md
```

### 6. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501. The sidebar should read **"4,803 movies · 4,803 vectors"** — if it
does, the artifacts loaded and the agent is ready. Try one of the sample questions in the
sidebar, or ask something like *"Find me a funny science-fiction movie from after 2010 that is
under 2 hours"*.

### Optional extras

```bash
# Load the existing vectors into an embedded Qdrant collection (no re-embedding),
# then set VECTOR_BACKEND=qdrant in .env
python scripts/build_index.py --skip-embeddings --with-qdrant

# A queryable SQLite copy of the dataset, on the side
python scripts/build_sqlite.py            # -> artifacts/movies.db
python scripts/build_sqlite.py --samples  # worked example queries
```

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Source CSV not found: .../tmdb_5000_movies.csv` | The dataset is not in `data/` | Download both CSVs from [Kaggle][kaggle] (step 1) |
| `The dataset has not been built yet` | No artifacts | Run `python scripts/build_index.py` |
| `The vector index was built with embedding model X` | `EMBEDDING_MODEL` changed after the build | `python scripts/build_index.py --force` |
| `Artifacts do not match the source CSVs` | The CSVs changed | Rebuild with `--force` |
| `LLM_API_KEY is not set` | The agent needs a chat model | Set it in `.env`. Structured search, fuzzy matching and index building do not need one |
| App shows stale or missing data after a rebuild | `DATA_DIR` / `ARTIFACTS_DIR` in `.env` point elsewhere | Unset them, or point them at this checkout |
| `The embedded Qdrant store ... is locked by another process` | Embedded Qdrant is single-process | Stop the app or the other script; for concurrent access run a server and set `QDRANT_URL` |
| `Port 8501 is already in use` | Another Streamlit instance | `streamlit run app.py --server.port 8502` |
| First run stalls on "Loading weights" | One-off ~130 MB model download | Wait; set `HF_TOKEN` if you hit Hugging Face rate limits |
| `pytest` errors with `PermissionError ... pytest-of-<user>` | Environment: pytest cannot scan its temp root | `pytest --basetemp=.pytest-tmp` |

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
is layered so the app refuses to start only on the paths that genuinely need a credential.

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
pytest                # 189 collected: 186 pass + 3 documented xfails, no network
pytest -m live        # opt-in: a couple of real turns against the configured provider
```

The default run makes **no network calls**. Run `-m live` after any LangChain/LangGraph
upgrade — the fake chat model encodes the library's message contract, so a version bump
could break the integration while the suite stays green.

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

Full detail: [`docs/AGENT_DOCUMENTATION.md`](docs/AGENT_DOCUMENTATION.md).

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
tests/                      unit, graph, retrieval, structural and backend-parity tests
docs/                       design guide, architecture review and interview Q&A
```

Import direction is one-way and **enforced by a test**: `langgraph`/`langchain*` may be
imported only under `agent/` and `llm/`, and `streamlit` only under `ui/`. That containment
is why swapping the agent runtime (ADR-0001 → ADR-0020) touched one package rather than the
whole system.

---

## Documentation

| Document | What it is |
|---|---|
| [`docs/TECHNICAL_DESIGN.md`](docs/TECHNICAL_DESIGN.md) | The design document: data decisions, agent design, search, RAG, memory, 14 worked examples |
| [`docs/AGENT_DOCUMENTATION.md`](docs/AGENT_DOCUMENTATION.md) | Technical design, repository guide, actual checked outputs and limitations |
| [`docs/LANGGRAPH_WORKFLOW.md`](docs/LANGGRAPH_WORKFLOW.md) | Graph topology, state channels and reducers, node-by-node walkthrough |
| [`docs/CLASS_REFERENCE.md`](docs/CLASS_REFERENCE.md) | Every class and model: what it represents and how the pieces connect |
| [`docs/ARCHITECTURE_REVIEW.md`](docs/ARCHITECTURE_REVIEW.md) | Senior-engineer validation against the assignment, with prioritized findings |
| [`docs/INTERVIEW_QA.md`](docs/INTERVIEW_QA.md) | Architecture and AI-engineering interview questions with sample answers |

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

## Not currently built

MCP interface, REST API, and web-search fallback are optional bonuses and are not
implemented. A hosted deployment is also absent; the assignment's deliverables table
requires a live Streamlit URL, so deployment remains a submission blocker. Web search
would additionally require a provenance model to preserve the "never invent movie
metadata" guarantee.

## Version control

This tree is a Git repository. At review time the active branch is `vector_db`; review the
history and current working-tree changes before preparing the private submission.
