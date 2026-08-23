# TMDB movie agent: technical design and repository guide

## 1. What this system is

This repository implements a Streamlit movie-discovery agent over the 4,803-row TMDB 5000 dataset. The LLM interprets a request and chooses tools, while deterministic Python code performs filtering, counting, title matching, vector retrieval, and record lookup.

The central design rule is sound: **the LLM decides what operation is needed, but dataset components decide the facts**. This is the right architecture for questions where fluent language understanding and exact numeric answers are both required.

The runtime flow is:

```text
CSV files
  -> offline preprocessing
  -> Parquet records + semantic documents + embeddings + manifest
  -> read-only runtime
  -> LangGraph planner/agent
  -> typed tools
  -> grounded answer + visible trace
  -> Streamlit UI
```

The graph itself is:

```text
START -> plan -> agent <-> tools -> synthesize -> ground -> END
          |                     |
          -> smalltalk -> END    -> clarify -> interrupt/resume -> agent
```

## 2. Repository map

| Path | Responsibility |
| --- | --- |
| `app.py` | Streamlit chat UI, session handling, result/trace rendering |
| `scripts/build_index.py` | Offline CSV join, preprocessing, document creation, embeddings, optional Qdrant build |
| `scripts/profile_*.py` | Dataset and document-length analysis |
| `scripts/benchmark_vector_backends.py` | Numpy/Qdrant result-parity and latency benchmark |
| `src/movieagent/config.py` | Typed settings and paths |
| `src/movieagent/data/` | Schema, preprocessing, query DSL, repository, artifact manifest |
| `src/movieagent/retrieval/` | Fuzzy matcher, semantic documents, lexical coverage, Numpy and Qdrant indexes |
| `src/movieagent/tools/` | Five framework-independent tools and their typed result contract |
| `src/movieagent/llm/` | Chat/embedding adapters, fake model, reasoning metadata sanitizer |
| `src/movieagent/agent/` | LangGraph state, planner, prompts, tool bindings, grounding check, trace |
| `src/movieagent/ui/` | Streamlit rendering and answer-table cleanup |
| `tests/` | Unit, graph, retrieval, structural, UI, and backend-parity tests |
| `artifacts/` | Built Parquet data, embeddings, Qdrant store, manifest, and analysis reports |

Dependencies point inward: UI and LangGraph depend on tools; tools depend on data and retrieval; data/retrieval do not depend on Streamlit or the agent framework. This makes deterministic logic independently testable and keeps framework replacement localized.

## 3. Data pipeline

### Input and join

`tmdb_5000_movies.csv.id` is inner-joined one-to-one with `tmdb_5000_credits.csv.movie_id`. The current manifest records:

- 4,803 movies and 4,803 credits rows read;
- 4,803 joined rows;
- zero unmatched rows;
- zero malformed JSON values.

An inner join is acceptable for this dataset because the build report proves there is no loss. In a changing production source, a left join plus an explicit missing-credit status would be safer.

### Normalization and missing data

The preprocessing layer:

- parses genres, keywords, production companies, production countries, spoken languages, cast, and crew JSON;
- keeps all cast names for filtering and the top-billed cast for embeddings/generation;
- extracts every credited director, including co-directors;
- parses release dates and derives nullable release years;
- stores numeric fields with nullable Pandas dtypes;
- converts runtime values at or below zero to unknown;
- treats `budget == 0` and `revenue == 0` as unknown, with explicit `*_known` flags;
- preserves title, original title, tagline, overview, language, status, and homepage.

Current quality counts are:

| Condition | Count |
| --- | ---: |
| Missing release date | 1 |
| Missing/invalid runtime | 37 |
| Empty overview reported by the build | 7 |
| Zero budget treated as unknown | 1,037 |
| Zero revenue treated as unknown | 1,427 |
| No director | 30 |
| No keywords | 412 |

The build stores SHA-256 hashes, preprocessing version, embedding model, vector dimension, document-template version, row count, and quality report in `artifacts/manifest.json`. Runtime loading rejects stale or mismatched artifacts.

### Search-field allocation

| Purpose | Fields |
| --- | --- |
| Structured filters/aggregates | genres, keywords, companies, countries, languages, cast, directors, year, rating, votes, popularity, runtime, budget, revenue |
| Fuzzy identity resolution | normalized title and original title |
| Embedded semantic text | title/year, distinct original title, tagline, genres, up to 20 keywords, directors, top cast, overview |
| Grounded generation | selected structured records, including overview and known metadata |

One document per movie is appropriate. Documents average about 95 words and do not benefit from arbitrary chunking.

## 4. Search and tool design

### Structured search

The LLM produces a validated `SearchQuery`, not SQL. The closed DSL supports numeric comparisons, year bounds, categorical membership, people/director filters, sorting, limits, and aggregations. `MovieRepository` converts the query to deterministic Pandas/Numpy masks.

This is safer than generated SQL for a small fixed dataset: invalid fields are schema errors, categorical values are checked against the real vocabulary, filters can be shown in the UI, and queries can be merged across turns.

Important semantic limitation: list values currently mean **any value matches**. For example, `genres=["Science Fiction", "Comedy"]` means Science Fiction **or** Comedy, not both. This matters for the assignment's “funny science-fiction” hybrid example and is discussed in the architecture review.

### Fuzzy title search

Titles are case-folded, de-accented, stripped of punctuation, whitespace-normalized, and normalized around leading English articles. Both display and original titles are indexed.

RapidFuzz edit-distance scoring uses confidence bands:

- exact normalized title: accept;
- score at least 82: accept, unless there is a near tie;
- score 65-81: ambiguous;
- score below 65: not found;
- top-two difference at most 3 points: ambiguous regardless of score;
- fewer than four normalized characters: exact matches only.

The matcher returns a typed `match`, `ambiguous`, or `not_found` outcome. Ambiguity triggers a LangGraph interrupt, and the user's ordinal/title response is resolved deterministically.

### Semantic and hybrid retrieval

The default embedding model is `BAAI/bge-small-en-v1.5` with a query-only retrieval prefix. Embeddings are L2-normalized, making cosine similarity a dot product.

The default index is exact Numpy search over 4,803 x 384 float vectors. This is a good corpus-size decision: it is simple, fast, deterministic, and easy to pre-filter. Qdrant is available behind the same protocol for scale-out and has parity tests against the Numpy definition.

Hybrid retrieval applies a `SearchQuery` restriction before vector ranking. This guarantees returned results come from the candidate pool defined by the filter implementation. It does not, however, correct an incorrectly modeled filter such as the current same-field OR issue.

Low-confidence handling combines a low cosine backstop with lexical coverage against the movie-document vocabulary. It catches gibberish well but cannot detect a fluent query about a topic absent from this 2016-era corpus.

### Tool boundaries

| Tool | Owns | Must not own |
| --- | --- | --- |
| `structured_search` | exact filtering, sorting, counts, aggregates | approximate titles, themes |
| `fuzzy_movie_search` | title text to movie identity | details, similar movies |
| `semantic_search` | themes, plots, moods, similar-to, hybrid ranking | exact title resolution |
| `movie_details` | one complete stored record by movie ID | discovery |
| `rag_answer` | prose grounded in an explicit ID set | retrieval |

Every tool returns `status`, `message`, references, payload, and metadata. Full artifacts feed tables and traces, while a smaller JSON summary is sent to the agent model.

## 5. Agent, grounding, and memory

The planning model emits a typed `Plan` containing intent, tool sequence, short rationale, filters, reference resolutions, and whether the query starts a fresh topic. The plan is visible, but execution is still model-driven; the graph records deviations rather than enforcing the planned sequence.

Grounding has three layers:

1. `rag_answer` accepts explicit movie IDs and builds context only from their stored records.
2. prompts forbid adding movies or metadata not returned by tools;
3. a post-hoc checker warns when answer text appears to mention a movie absent from tool artifacts.

The third layer detects unsupported entities, not incorrect attributes. A wrong runtime for a retrieved movie can pass it.

Memory is keyed by a Streamlit-generated thread ID and stored in LangGraph's in-memory checkpointer. Persisted state includes:

- a bounded message window;
- the active structured query;
- the previous result references and total count;
- the selected movie ID;
- the current plan and answer.

Follow-up filters merge with the active query. A `fresh_topic` plan resets old filters. “The first one” is resolved to an ID from stored result references. State survives Streamlit reruns within the process but not a process restart.

## 6. User interface and observability

The Streamlit app displays:

- chat history;
- structured tables and movie-detail cards from tool artifacts;
- selected plan and one-line rationale;
- tool calls, arguments, outcomes, and filters;
- semantic retrieval scores and document text;
- carried conversational filters;
- grounding and plan-deviation warnings;
- duration and token usage when available.

The design correctly avoids exposing hidden chain-of-thought. Provider reasoning fields are removed at the model boundary with metadata allowlists, and the trace type contains no reasoning field.

## 7. Actual checked outputs

These are direct tool outputs from the current artifacts on 22 August 2026; they do not depend on an LLM inventing a number.

| Query | Observed output |
| --- | --- |
| Ten most common genres | Drama 2,297; Comedy 1,722; Thriller 1,274; Action 1,154; Romance 894; Adventure 790; Crime 696; Science Fiction 535; Horror 519; Family 513 |
| Top ten movies by revenue | Avatar, Titanic, The Avengers, Jurassic World, Furious 7, Avengers: Age of Ultron, Frozen, Iron Man 3, Minions, Captain America: Civil War; 1,427 unknown-revenue rows excluded |
| Movies rated above 8 | 50 matches; 25 displayed by default |
| Movies longer than 150 minutes | 171 matches |
| “Intersteler” | Interstellar (2014), score 87 |
| “Someone trying to survive alone on another planet” | Eight retrieved; The Martian ranks second. Top result is What Planet Are You From?, showing imperfect precision |
| “Time travel and changing the past” | Time Changer, The Butterfly Effect, Timeline, Project Almanac, About Time, Somewhere in Time, X-Men: Days of Future Past, Timecrimes |
| Multi-turn sci-fi flow | Science Fiction after 2010 produces more than 25 matches; adding rating above 7.5 produces 8; the first result can then be resolved by stored ID |
| “flurble qqqzzz” | `low_confidence`, zero results shown, lexical coverage 0.0 |
| Hybrid “funny science-fiction after 2010 under 120 minutes” | Pool 417 and includes single-genre matches such as Crazy, Stupid, Love. and Elysium; this exposes the same-field OR bug |

The weakest semantic examples are abstract-theme queries. “Dark and psychological, a person losing their grip on reality” returned Phenomenon first, and known expected failures show that The Matrix, The Shawshank Redemption, and Inception-neighbor expectations are sensitive to how TMDB describes each film.

## 8. Running and testing

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
python scripts/build_index.py
streamlit run app.py
pytest
```

Before running, remove machine-specific `DATA_DIR` and `ARTIFACTS_DIR` overrides or point them at this repository. The checked-in local `.env` currently references an older sibling directory; the architecture review explains the impact.

With paths overridden to this V2 repository and a writable Pytest temporary directory, the current suite reports **186 passed, 3 expected failures, and 2 warnings**. The expected failures intentionally pin known semantic-retrieval weaknesses.

## 9. Current limitations

- no deployed review URL, despite the assignment making deployment mandatory;
- hybrid same-field constraints cannot express “has all genres”;
- the graph does not enforce semantic search followed by `rag_answer`;
- configured tool-iteration and wall-clock guards are not effective;
- memory is in-process and stores only one prior result set;
- the grounding checker cannot validate attribute values;
- lexical coverage does not detect fluent out-of-domain questions;
- fuzzy thresholds are calibrated on a small handcrafted set;
- embedded Qdrant is single-process and slower than Numpy at this scale;
- no REST API, MCP interface, or web fallback, which are optional bonuses.

The prioritized remediation plan is in [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md).
