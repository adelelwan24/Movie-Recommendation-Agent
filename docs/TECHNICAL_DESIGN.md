# Technical design document

Movie discovery agent over the TMDB 5000 dataset. This document explains the engineering
decisions — what was chosen, what was rejected, and what it costs.

Every number below was produced by running the system, not estimated. Section 6's outputs come
from direct tool invocation and from real graph runs against a scripted model; the reproduction
commands are in the appendix.

---

## 1. Data decisions

### 1.1 Source files

| File | Rows | Columns | Size |
|---|---|---|---|
| `tmdb_5000_movies.csv` | 4,803 | 20 | 5.5 MB |
| `tmdb_5000_credits.csv` | 4,803 | 4 | 39.1 MB |

The credits file is seven times larger from four columns, because two of them (`cast`, `crew`)
hold 236,000 nested JSON records between them.

### 1.2 Movies processed: 4,803

All of them. No row is dropped for missing data — a film with an unknown budget is still a film,
and dropping it would silently change every count the system reports.

### 1.3 Join strategy

```python
merged = movies.merge(credits, on="id", how="inner", validate="one_to_one")
```

Two deliberate choices:

- **`how="inner"`.** A movie without credits cannot produce a semantic document (no director, no
  cast) or answer a person query. Rather than carry half-populated rows, the join defines the
  working set. In practice this costs nothing: **0 unmatched on either side, 4,803 joined.**
- **`validate="one_to_one"`.** A duplicated `id` on either side would silently multiply rows and
  inflate every aggregate. Pandas raises instead. This is a guard against a class of bug that is
  otherwise invisible — the counts stay plausible while being wrong.

The credits file's key is `movie_id`, renamed to `id` before the merge.

### 1.4 JSON fields parsed

Seven columns arrive as JSON-encoded strings. All are parsed in
[`preprocess.py`](../src/movieagent/data/preprocess.py):

| Column | Source | Shape | Extracted as |
|---|---|---|---|
| `genres` | movies | `[{id, name}]` | `genre_names: list[str]` |
| `keywords` | movies | `[{id, name}]` | `keyword_names: list[str]` |
| `production_companies` | movies | `[{name, id}]` | `company_names: list[str]` |
| `production_countries` | movies | `[{iso_3166_1, name}]` | `country_names: list[str]` |
| `spoken_languages` | movies | `[{iso_639_1, name}]` | `language_names: list[str]` |
| `cast` | credits | `[{cast_id, character, credit_id, gender, id, name, order}]` | `top_cast` (5) + `full_cast` |
| `crew` | credits | `[{credit_id, department, gender, id, job, name}]` | `directors: list[str]` |

**Malformed JSON yields an empty list and is counted, never raised and never silently dropped** —
the count lands in the manifest. This dataset has zero malformed values, but the counter is the
difference between "we know it was clean" and "we assume it was".

Three extraction decisions worth stating:

- **`top_cast` sorts by the `order` field** rather than trusting file order. The CSV is usually
  already ordered, but "usually" is not a guarantee and billing order is the entire point of the
  field. Cut at 5 — enough to identify a film, short enough not to dominate its document.
- **`full_cast` is kept separately** so structured search can answer "which films star X" over the
  complete cast (106,257 credits) while the embedded document stays lean.
- **`directors` is a list, not a scalar.** Co-directed films are real — the Coens, the Wachowskis —
  and collapsing to one name loses data. 308 films (6.4%) have more than one director; the maximum
  is 21 (*Paris, je t'aime*).

Only `name` is retained from the id/name pairs. TMDB's internal ids are not a stable public
vocabulary, and the raw analysis found `production_companies` has 5,047 distinct ids for 5,017
distinct names — the same company under multiple ids. Keying on names avoids importing that
inconsistency.

### 1.5 Missing-value handling

The governing rule: **unknown is unknown, never zero.** Every numeric column uses a pandas nullable
dtype (`Int64`, `Float64`), and there is no `fillna(0)` anywhere in the pipeline.

| Field | Missing | Treatment |
|---|---|---|
| `release_date` | 1 | `NaT`; `release_year` is `NA` |
| `runtime` | 37 | `Int64` NA. **A runtime of 0 is a data-entry artifact, not a zero-length film**, so `<= 0` is masked to NA |
| `overview` | 4 | Empty string; the document omits the line entirely |
| `budget` | 1,037 zeros | `NA` + `budget_known: bool` |
| `revenue` | 1,427 zeros | `NA` + `revenue_known: bool` |
| `directors` | 30 | Empty list |
| `keywords` | 412 | Empty list |
| `tagline` | 844 | Omitted from the document |
| `homepage` | 3,091 | Metadata only; never used in retrieval |

**Why budget and revenue get companion flags.** Roughly a fifth of the dataset has a zero here, and
it means "not recorded", not "$0". Treating those as zero would place them at the bottom of every
"lowest revenue" sort and corrupt every average. NA excludes them from comparisons automatically;
the `_known` flag lets the UI say "unknown" rather than rendering `$0`.

The consequence is visible in ranking behavior: `SearchResult.excluded_unknown` reports how many
rows were dropped from a sort because the sort field was unknown, and the tool message states it —
*"312 movies match; showing 25. 41 excluded from the ranking because revenue is unknown for them."*

**Parquet is what makes this survive a reload.** A CSV round-trip would silently undo the nullable
dtypes, the real dates, and the NA-versus-zero distinction — which is why the artifact format is
part of the design and not an implementation detail.

### 1.6 Field allocation — the central data decision

Every field is assigned to exactly one job, on one principle:

> **Embed what is fuzzy and thematic. Filter what is exact and categorical.**

| Field | Structured | Fuzzy | Semantic doc | Generation context |
|---|---|---|---|---|
| `title` | sort/display | ✅ normalized | ✅ header | ✅ |
| `original_title` | — | ✅ normalized | ✅ when it differs | ✅ |
| `overview` | — | — | ✅ | ✅ |
| `tagline` | — | — | ✅ | ✅ |
| `genres` | ✅ membership, group-by | — | ✅ | ✅ |
| `keywords` | ✅ membership, group-by | — | ✅ (capped at 20) | ✅ (capped at 15) |
| `cast` | ✅ full cast membership | — | ✅ top 5 only | ✅ top 5 |
| `crew` → `directors` | ✅ membership, group-by | — | ✅ | ✅ |
| `production_companies` | ✅ membership, group-by | — | ❌ | ✅ |
| `production_countries` | ✅ membership, group-by | — | ❌ | ✅ |
| `spoken_languages` | ✅ membership, group-by | — | ❌ | ✅ |
| `vote_average` | ✅ compare, sort, avg | — | ❌ | ✅ |
| `vote_count` | ✅ compare, sort | — | ❌ | ✅ |
| `budget` / `revenue` | ✅ compare, sort, sum | — | ❌ | ✅ with known-flags |
| `runtime` | ✅ compare, sort | — | ❌ | ✅ |
| `release_date` / `release_year` | ✅ range, group-by | — | ✅ year in header | ✅ |
| `popularity` | ✅ compare, sort | — | ❌ | ✅ |
| `status`, `homepage` | display | — | ❌ | ✅ |

**Why numerics are not embedded.** "Movies rated above 8.0" is a comparison, not a similarity. An
embedded `Rating: 8.1` produces a vector that is *near* `Rating: 8.0` and also near `Rating: 6.1`,
because the model is reading digits as tokens. Vector search over numbers is a worse implementation
of a `>` operator, so numbers go to pandas.

**Why companies and countries are not embedded.** "Films by Pixar" is an exact set membership. A
vector search would return films that *feel* like Pixar films — which is a different, unrequested
question.

**Why keywords *are* embedded.** They carry TMDB's own thematic vocabulary — "dystopia", "time
travel", "loss of loved one" — which is frequently absent from the overview prose. This is the
field that makes theme-shaped queries work at all.

Two costs, recorded rather than hidden:

1. The field labels repeat across all 4,803 documents, raising the *floor* of pairwise similarity.
   This is why the low-confidence threshold is 0.35 and not an intuitive 0.5 — an intuitive value
   would reject good matches.
2. TMDB keyword coverage is uneven. Films with no keywords and a short overview produce thin
   documents that rarely retrieve — a systematic recall gap correlating with obscurity. Measured:
   documents range from **9 to 233 words** (mean 95); the four shortest are exactly the four films
   with no overview.

### 1.7 What is preserved for generation

`movie_details` and `rag_answer` build their context from the **complete stored record** — all 28
columns, including everything excluded from the embedded document. Generation therefore sees
companies, countries, languages, full cast, and every numeric field, with unknowns marked
explicitly as `unknown` rather than omitted or zeroed.

This is deliberate asymmetry: the embedding decides *which* films are relevant; the record decides
*what can be said* about them.

---

## 2. Agent design

### 2.1 How routing works

Routing is a two-stage decision, and neither stage is left to the model's discretion alone.

**Stage 1 — the plan.** One structured LLM call before any tool runs, returning a typed `Plan`
(intent, steps, rationale, extracted filters, resolved movie ids, `refines_previous`, `needs_tools`).
Because it is a schema and not free text, the tool choice is inspectable, the rationale is capped
at 240 characters, and conversational references are resolved to concrete ids *before* execution —
so tools never receive "the first one" and stay independently testable.

**Stage 2 — outcome-driven edges.** After each tool batch, `_after_tools` routes on
`ToolResult.status`:

| Status | Routing |
|---|---|
| `AMBIGUOUS` | → `clarify`, which calls LangGraph's `interrupt()` and pauses the turn |
| everything else | → `agent` for another pass, or `synthesize` |

**This is the reason the graph is hand-built rather than `create_react_agent`.** The prebuilt agent
has no planning stage and cannot cleanly interrupt on a tool *result*. The requirement — run the
fuzzy matcher, and if three candidates fall within three points, stop and ask the user — is a
conditional edge reading a tool's status, and that edge only exists if you own the edges.

Full topology, state channels and reducers: [LANGGRAPH_WORKFLOW.md](LANGGRAPH_WORKFLOW.md).

### 2.2 The five tools and their boundaries

| Tool | Answers | Explicitly does **not** |
|---|---|---|
| `structured_search` | Counts, aggregations, numeric comparisons, year ranges, genre/company/country/language/person filters, sorting, top-N | Resolve misspelled titles; answer plot/theme questions |
| `fuzzy_movie_search` | "Which film did the user mean?" from a typed title | Search by plot or filter by attributes |
| `semantic_search` | Plot, theme, mood, "similar to X" — with optional hard filters | Resolve a specific typed title; count or aggregate |
| `movie_details` | One complete record, given a numeric id | Search; it takes an id, never a title |
| `rag_answer` | Prose grounded in specific movie ids | Choose which movies to talk about |

The boundaries are enforced structurally, not just described:

- `movie_details` accepts an `int`, so it *cannot* be called with a title.
- `rag_answer` requires a non-empty `movie_ids` list and rejects an empty one as `INVALID_INPUT` —
  it cannot retrieve, so it cannot introduce a film nobody selected.
- `structured_search` takes a closed `SearchQuery`, so an unknown field is a validation error the
  agent can act on rather than a silent empty result.

### 2.3 When multiple tools are invoked

Chaining is the normal case, not an exception. Three shapes:

**Sequential dependency** — the output of one is the input of the next.

```
"Tell me about Intersteler"
  fuzzy_movie_search("Intersteler")  → id 157336 (score 87)
  movie_details(157336)              → full record
```

**Filter plus retrieval** — a structured pre-filter narrows the pool, then semantic ranking orders
it (see §4.4).

**Retrieval plus generation** — `semantic_search` selects the films, `rag_answer` writes prose
grounded only in those ids.

**Recovery** — an `EMPTY` result flows back to `agent`, which can relax a constraint and retry.
The plan is a statement of intent, not a script; the trace records plan *and* actual and marks the
difference, so a deviation is visible rather than hidden.

### 2.4 Worked example — *"How many movies have Christopher Nolan as director?"*

```
plan     → intent: count films by a director
           tools: [structured_search]
           filters: {directors: ["Christopher Nolan"]}
structured_search
  ├─ repo.validate(query)   "Christopher Nolan" ∈ directors vocabulary ✓
  ├─ repo.mask_for(query)   term index → 8 rows
  └─ SearchResult(total=8, rows=8)
answer   → "8 movies match; showing 8."
```

**A deliberate deviation from the brief.** The suggested flow is *entity/fuzzy resolution →
structured search*, but `fuzzy_movie_search` resolves **titles only**. Person-name resolution
happens inside `MovieRepository.validate()`, which checks every name against the real cast and
director vocabularies and attaches a suggestion on a near-miss — `unknown person 'Cristopher
Nolan'; closest match is 'Christopher Nolan'`. That returns `INVALID_INPUT` with an actionable
correction the agent retries with.

Two reasons for putting it there rather than in a tool: a person filter can name several people at
once, and it is *one field among many* in a query — routing the whole query through a title matcher
to fix one field would be the wrong shape. It also means the same guard protects genres, companies,
countries and languages, which have the identical failure mode ("Sci-Fi" vs "Science Fiction").

### 2.5 Worked example — *"Find me a dark science-fiction movie about artificial intelligence"*

```
plan     → intent: thematic search within a genre
           tools: [semantic_search, rag_answer]
           filters: {genres: ["Science Fiction"]}
semantic_search
  ├─ vocabulary.coverage("dark science fiction about artificial intelligence") = 1.0  ✓
  ├─ Restriction.from_query(repo, filters)      pool: 377 of 4,803
  ├─ embedder.embed_query(...)                  384-d, BGE query prefix applied
  └─ index.search(vector, k=8, restriction)     mask applied BEFORE ranking
       0.6722  Dark City (1998)
       0.6716  I, Robot (2004)
       0.6716  A Scanner Darkly (2006)
       0.6689  A.I. Artificial Intelligence (2001)
rag_answer(question, movie_ids=[...])           prose from those records only
```

The ordering matters: coverage is checked before embedding, and the filter is applied before
ranking. Both are refusals to do expensive work on a question that cannot be answered well.

### 2.6 Why this architecture

| Considered | Rejected because |
|---|---|
| Model-generated SQL | Unbounded surface; a wrong-but-valid query returns confident nonsense. The closed `SearchQuery` DSL makes an invalid request a *validation error the agent can fix* |
| `create_react_agent` | No planning stage; cannot interrupt on a tool result — the clarification requirement is unimplementable |
| Pure RAG over everything | "How many movies per genre" is a `groupby`. Answering it by embedding similarity is wrong in a way that is hard to detect |
| Post-filtering after retrieval | Retrieve 50, discard non-matching, return 3 of a requested 10 — indistinguishable from genuine scarcity |
| One "search" tool | Collapses four different failure modes into one empty list |

---

## 3. Search system

### 3.1 Structured search

Everything runs through one closed DSL — `SearchQuery` — which does three jobs: the structured
request, the semantic pre-filter, and the unit of conversational memory.

**Filtering.** `MovieRepository.mask_for()` builds a single boolean row mask:

- **Numeric conditions** — `gt`, `gte`, `lt`, `lte`, `eq`, `between` over seven fields. Results go
  through `.fillna(False)`, so **a null never satisfies a comparison** (an unknown budget is not
  "under $100M").
- **Membership** — genres, keywords, companies, countries, languages, directors, resolved through a
  prebuilt case-folded term index (term → row positions) rather than a scan.
- **`people`** — the DSL's only disjunction, matching cast **or** director.
- **Year range** — `year_from` / `year_to` as numeric conditions on `release_year`.

Multi-condition queries are ANDed. There is no OR across fields and no nested boolean groups —
a deliberate expressive ceiling, revisited only if real queries need it.

**Sorting.** Rows with an unknown sort value are **excluded from the ranking**, not sorted to one
end, and the count is reported. "Top 10 by revenue" must not be shaped by films whose revenue is
unknown. Sorts use `kind="mergesort"` for stability, so equal values keep a deterministic order.

**Aggregation.** List fields are exploded first, so a film in three genres counts once per genre —
which is what "how many movies in each genre" means. Three metrics: `count`, `avg_rating`,
`sum_revenue`. The revenue sum relies on NA-skipping so unknown revenue is not added as zero.

**The true total always travels with capped rows.** `SearchResult` carries `total` (the real match
count) beside `rows` (capped at `limit`), so a truncated view can never pass for the complete
answer.

**Vocabulary validation.** Pydantic validates *structure*; only the data knows whether "Sci-Fi" is a
genre. Every value in a membership filter is checked against the real vocabulary before execution,
with a fuzzy suggestion attached on a near-miss. The suggestion scorer is
`max(WRatio, token_set_ratio, partial_ratio)` — `partial_ratio` is what carries abbreviations
("sci-fi" scores 67 against "science fiction" where `token_set_ratio` gives 29). The cutoff is a
loose 65: this is advisory text inside an error the agent will retry, so a wrong suggestion costs
one retry while a missing one costs the user an unexplained empty result.

### 3.2 Fuzzy title search

**Normalization** ([`normalize_title`](../src/movieagent/data/preprocess.py)) — casefold, strip
diacritics (NFKD), drop punctuation, collapse whitespace, and move a leading English article to the
end, so `the dark knight` and `dark knight, the` converge. Both `title` and `original_title` are
normalized and indexed, because non-English films are frequently searched by their native name.
*Known limitation, recorded rather than accepted silently:* the article rule is English-centric and
does nothing for `Le`/`La`/`Der`.

**Similarity.** Plain `fuzz.ratio`, with `token_set_ratio` admitted **only when the query's
significant tokens are a subset of the candidate's**:

```python
base = float(fuzz.ratio(query, choice))
query_tokens = set(query.split()) - _GATE_STOPWORDS
if query_tokens and query_tokens <= set(choice.split()):
    return max(base, float(fuzz.token_set_ratio(query, choice)))
return base
```

This replaced an earlier `max(WRatio, token_set_ratio)` that inflated garbage: *"qqqzzz nonexistent
film"* scored 86 against *An Alan Smithee Film: Burn, Hollywood, Burn* because both contain "film".
The gate is a **containment test, not a similarity test** — it asks whether the query is a fragment
*of* this title, and only then permits fragment-friendly scoring. It recovers the partial-title case
(`"lord of the rings"` scores 56 under `ratio` alone, below the floor) without the false positives.

**Thresholds** — calibrated against a measured 14-case match set and a 6-case refusal set, not
guessed:

| Condition | Outcome |
|---|---|
| Exact normalized match, unambiguous | `MATCH`, score 100 (short-circuit) |
| ≥ 82 (`accept`) | `MATCH` |
| 65–82 | `AMBIGUOUS` — return candidates and ask |
| < 65 (`ambiguous`) | `NOT_FOUND` |
| Query shorter than 4 characters | `NOT_FOUND` — at that length an edit-distance score says nothing |

Correct matches score 86–100 (lowest: "Titanik" → "Titanic" at 86); unmatched input scores 33–59.
82 and 65 sit inside that gap with margin on both sides.

**Ambiguity.** One rule matters more than the thresholds: **if the top two candidates are within
`tie_margin` (3 points), the result is `AMBIGUOUS` regardless of score.** The dangerous failure is
not a low score — it is returning film A at 90 when film B scored 89. In calibration it was this
check, not the floor, that caught franchise queries where three films all score 100.

`AMBIGUOUS` propagates as a tool status, becomes a graph interrupt, and reaches the user as a
numbered choice. Their reply is resolved **deterministically** by `resolve_choice` — ordinal, bare
number, or title. Having just told the user the system will not guess, resolving their answer with
another model call would reintroduce the guess one step later.

---

## 4. RAG pipeline

### 4.1 Document construction

One curated, labelled document per movie — no chunking. Overviews are 1–3 sentences; splitting them
would produce fragments with less context than the whole.

```
Title: Interstellar (2014)
Tagline: Mankind was born on Earth. It was never meant to die here.
Genres: Adventure, Drama, Science Fiction
Keywords: saving the world, artificial intelligence, father son relationship, nasa,
          expedition, wormhole, space travel, famine, black hole, dystopia, ...
Director: Christopher Nolan
Starring: Matthew McConaughey, Jessica Chastain, Anne Hathaway, Michael Caine, Casey Affleck
Overview: Interstellar chronicles the adventures of a group of explorers who make use of a
          newly discovered wormhole to surpass the limitations on human space travel ...
```

Empty fields are **omitted**, not emitted as a bare `Field:` label — a blank label is noise every
document would share. Keywords are capped at 20 to stop keyword-rich films from drowning their own
overview. The template is versioned (`DOCUMENT_TEMPLATE_VERSION`) and stamped into the manifest,
because a template change invalidates every embedding built with the old one.

Measured corpus: 4,803 documents, **95.4 words / 624 characters mean**, range 9–233 words, p99 187.
Nothing approaches the model's token limit, so nothing is truncated.

### 4.2 Embedding model

`BAAI/bge-small-en-v1.5` — 384 dimensions, runs locally via `sentence-transformers`.

Chosen as the default despite pulling ~2 GB of torch, because a default that might not run for the
reader is not a default: no API key, and offline after the first weight download. An
OpenAI-compatible backend (`/v1/embeddings`: OpenAI, vLLM, Ollama, LM Studio) is one setting away.

**Documents and queries are embedded through separate methods** — BGE requires an instruction
prefix on *queries only* (`"Represent this sentence for searching relevant passages: "`).
Collapsing both into one `embed()` is the standard way to silently lose retrieval quality with BGE
models, so the asymmetry lives inside the backend where no call site has to remember it.

Vectors are L2-normalized at build time, making cosine similarity a plain dot product.

**The manifest enforces consistency.** It records the model id and dimension; loading with a
different embedding model configured raises `ArtifactStaleError` before any query runs. Querying a
`bge-small` index with `text-embedding-3-small` vectors is meaningless, and if the dimensions
happened to match it would fail *silently*.

### 4.3 Vector index

Two interchangeable backends behind one protocol (`VECTOR_BACKEND`):

| | `numpy` (default) | `qdrant` |
|---|---|---|
| Storage | `embeddings.npy`, one in-process array (7 MB) | Qdrant collection, embedded or served |
| Filtering | Boolean mask over row order | Payload filters, engine-side |
| Deployment | Nothing to run | Embedded by default; `QDRANT_URL` for a server |

4,803 × 384 floats is ~7 MB; one matmul searches the whole corpus in **0.35 ms median**.
Approximate search would buy nothing and cost test stability.

Qdrant exists because the numpy index has a stated expiry condition — O(N·d) per query does not
bend — and the exit is wired and verified *before* it is needed. Measured across eight filter
shapes, the two backends agree **exactly**: identical pools, identical top-k, identical ordering,
scores within 2.5e-07. Embedded Qdrant is currently ~180× slower, because local mode evaluates
payload conditions in Python and ignores payload indexes; that is a property of the embedded
deployment, not of the engine. Full numbers:
[`vector_backend_benchmark.md`](../artifacts/vector_backend_benchmark.md).

### 4.4 Retrieval strategy, top-k and metadata filtering

**Pre-filtering, not post-filtering** — the single most important retrieval decision.

```
SearchQuery ──► Restriction ──┬─► numpy: boolean mask, applied before scoring
                              └─► qdrant: payload filter, applied during traversal
```

The mask restricts the candidate set *before* ranking, which guarantees k constraint-satisfying
results whenever k exist. Post-filtering (retrieve 50, then discard) returns 3 when you asked for
10 and looks like scarcity rather than filtering — a silent failure this shape cannot produce.

**Top-k = 8** by default (`TOP_K`), tunable per call. Eight is enough for the model to have real
choice when writing an answer and few enough that a weak eighth result does not dilute the context.

**Pool size is always reported.** Ranking 8 films out of a pool of 12 is close to arbitrary, and
the message says so: *"Ranked within a filtered pool of 17 movies. That pool is small, so the
ordering is weakly meaningful."* Honesty about confidence is cheaper than a confident-looking order.

**No reranking.** A cross-encoder pass was considered and rejected: at 4,803 documents with exact
cosine search, the bottleneck is not candidate ordering but document *thinness* (§1.6). Reranking
the same weak documents more precisely does not fix a recall gap. It would also add a second model
to the offline-by-default path. If retrieval quality becomes the binding constraint, enriching the
document template is the higher-yield change.

**Two guards run before retrieval:**

1. **Lexical coverage** — the fraction of a query's content words that appear anywhere in the corpus
   vocabulary (~33,500 words). Below 0.4, the tool returns `LOW_CONFIDENCE` without embedding
   anything. This is checked *first* because if a query is not made of words this corpus uses, no
   amount of vector geometry will reveal that afterwards.
2. **Similarity floor (0.35)** — a backstop only. Measurement showed an absolute cosine threshold
   *cannot* separate real queries from gibberish here: genuine queries top out at 0.575–0.730 and
   gibberish reaches 0.566–0.629 — overlapping populations. The floor catches catastrophic
   mismatch and nothing subtler; lexical coverage is the signal that actually works.

### 4.5 Context construction

`rag_answer` builds a **closed context from movie ids**, not from free text. For each id it renders
the complete stored record — genres, director, cast, dates, runtime, rating, votes, budget, revenue,
tagline, keywords, overview — with unknowns written as the literal string `unknown`.

The structural property: **there is no code path by which an unretrieved movie can enter the
prompt.** The context is built from ids, and the ids come from a tool that ran this turn.

### 4.6 Generation model

Chat and tool-calling default to `openai/gpt-4o-mini` via OpenRouter, temperature 0. Configured
independently of the embedding backend, because OpenRouter serves no `/v1/embeddings` endpoint.
Any OpenAI-compatible endpoint (including a local vLLM) is a base-URL change with no code edit.

The model is wrapped in `SanitizedChatOpenAI`, which strips reasoning content at the boundary. It is
*subclassed* rather than wrapped in a `RunnableLambda`, so `bind_tools` and `with_structured_output`
return bindings around the sanitized object and the guarantee holds through the whole graph.

### 4.7 Hallucination mitigation

Four layers, weakest to strongest — and the strongest is architectural, not textual.

1. **Numbers never pass through the model.** Result tables render from the tool payload
   (`artifact()`), while the model sees only a compact summary (`summary_for_model()`). A
   fabricated number cannot reach a table because the model does not write the table. The UI also
   strips markdown tables from answer prose, since any table there is a duplicate of one rendered
   from data.
2. **Closed context.** `rag_answer` is id-scoped (§4.5).
3. **Prompt constraints.** "Use ONLY the records below… Never state a fact that is not in the
   records… If a field is marked unknown, say it is unknown."
4. **Post-hoc grounding check** (`ground` node) — extracts title-shaped entities from the answer and
   flags any that no tool returned. Its scope is stated honestly: it catches invented **entities**,
   not invented **attributes** (if the payload says 169 minutes and the answer says 195, nothing
   fires), and common-word titles like *Up* or *Her* are largely invisible to it. It is **advisory**
   — it flags, never rewrites, because a silently edited answer is worse than a flagged one.

---

## 5. Multi-turn memory

### 5.1 What persists, and why it is structured

The requirement names filters, result sets and selected movies — all three are **structured values,
not prose**. Storing them as text and asking a language model to recover them is a lossy round-trip.
Storing them as typed channels makes *"tell me about the first one"* an array index.

| Channel | Type | Purpose | Update rule |
|---|---|---|---|
| `active_query` | `SearchQuery \| None` | Filters carried across turns | Merged only when the turn refines the previous one; otherwise replaced |
| `last_results` | `list[MovieRef]` | The previous result set — the referent for ordinals | Overwritten each turn that returns refs |
| `last_result_total` | `int \| None` | True match count, which may exceed the shown rows | Overwritten with the set |
| `selected_movie_id` | `int \| None` | The film under discussion | Set by the planner, a clarification, or a single-result turn |
| `messages` | `list[AnyMessage]` | Conversational tone and phrasing | Append, trimmed to a window of 8 |
| `question`, `plan`, `answer`, `deviations` | scalars | Turn-scoped working values | Cleared at the start of each turn |

### 5.2 What is deliberately **not** persisted

**Result rows.** `MemorySaver` retains every checkpoint for the process lifetime, so anything put in
state is retained with it. Only `MovieRef` — `{movie_id, title, year}` — survives; rows, document
text and tool artifacts live in the turn's `Trace` and are discarded after rendering.

This has a design consequence beyond memory usage: **what travels forward is the filter object, not
the rendered table.** A follow-up re-queries the dataset rather than filtering a display-capped
list, so a 25-row display cap can never silently truncate a follow-up.

### 5.3 Filter merge rules

Implemented once in `resolve_active_query`:

- **`refines_previous` false (the default) → drop what was carried.** Only a message that adjusts the
  previous result set ("only the ones above 7.5") inherits its filters; a question with its own
  subject starts clean.
- New condition on an already-constrained field → **replaces** it ("above 7.5" then "above 8" gives
  `> 8`, not both).
- New non-empty list filter → **replaces** its counterpart rather than unioning. "Actually,
  comedies" means comedies, not sci-fi *and* comedies.
- Nothing new → carry unchanged.

This lives in the plan node rather than in a channel reducer, for a concrete reason: LangGraph's
`BinaryOperatorAggregate` assigns `values[0]` directly while a channel is still `MISSING`, applying
the operator only from the second write onward. As a reducer, the first-turn write bypassed the
merge entirely. Computing the value in the node means **what is written is already correct**.

### 5.4 The message window

Raw messages are trimmed to 8 at write time, with two protections: the window never trims the turn
in progress (a seven-tool turn produces 15 messages on its own), and an `AIMessage` carrying
`tool_calls` is never separated from its `ToolMessage` replies, or the provider rejects the
conversation.

Anything older than the window is gone unless it survived as structured state — and that failure is
*visible* (the agent says it does not know what you mean) rather than silent.

### 5.5 Measured behavior

Real three-turn run (scripted model, real graph — §6, Q11):

```
turn 1  "Show me science fiction movies from after 2010"
        active_query : ['release_year >= 2010', "genre in ['Science Fiction']"]
        last_results : John Carter (2012), Avengers: Age of Ultron (2015), Man of Steel (2013), ...
        total        : 159

turn 2  "only the ones rated above 7.5"
        active_query : ['vote_average >= 7.5', 'release_year >= 2010', "genre in ['Science Fiction']"]
                       ← the new condition layered onto the carried filters
        last_results : X-Men: Days of Future Past (2014), Edge of Tomorrow (2014), ...

turn 3  "tell me about the first one"
        'the first one' → MovieRef(movie_id=127585, title='X-Men: Days of Future Past', year=2014)
        tools used     : ['movie_details']
```

Turn 3 involves no model recall: the ordinal is an index into `last_results`, so it either works or
returns `None`.

---

## 6. Example queries and results

All outputs below are real. Deterministic tools were invoked directly; the multi-turn example ran
through the real graph with a scripted model (reducers, edges, `ToolNode`, checkpointer all live).

### Q1 — Aggregation: *"How many movies are there in each genre?"*

```
status : ok
message: count by genre_names (top 10) over 10 groups.

  Drama            2297      Adventure         790
  Comedy           1722      Crime             696
  Thriller         1274      Science Fiction   535
  Action           1154      Horror            519
  Romance           894      Family            513
```

Sums to more than 4,803 because the list field is exploded — a film in three genres counts in three.

### Q2 — Entity + count: *"How many movies have Christopher Nolan as director?"*

```
status : ok
message: 8 movies match; showing 8.

  The Dark Knight Rises (2012) 7.6     Batman Begins (2005)  7.5
  The Dark Knight (2008)       8.2     Insomnia (2002)       6.8
  Interstellar (2014)          8.1     The Prestige (2006)   8.0
  Inception (2010)             8.1     Memento (2000)        8.1
```

### Q3 — Multi-condition filter + sort: *"Well-rated sci-fi since 2010, under two hours"*

`genres=["Science Fiction"]`, `year_from=2010`, `runtime < 120`, `vote_count >= 500`,
sorted by rating:

```
status : ok
message: 64 movies match; showing 10.

  Edge of Tomorrow (2014)               113 min  7.6  (4,858 votes)
  Ex Machina (2015)                     108 min  7.6  (4,737)
  I Origins (2014)                      106 min  7.5  (1,063)
  Gravity (2013)                         91 min  7.3  (5,751)
  Source Code (2011)                     93 min  7.1  (2,699)
  Ant-Man (2015)                        117 min  7.0  (5,880)
  Rise of the Planet of the Apes (2011) 105 min  7.0  (4,347)
```

Four conditions across three fields, ANDed, with the true total (64) reported beside the ten shown.

### Q4 — Sorting with unknowns: *"Highest-revenue films with a budget over $150M"*

```
status : ok
message: 103 movies match; showing 5.

  Avatar (2009) · Titanic (1997) · The Avengers (2012) · Furious 7 (2015) · Avengers: Age of Ultron (2015)
```

The 1,037 films with unknown budgets never enter the candidate set — a NULL does not satisfy
`> 150000000`. Had the sort field been unknown for any matching row, the count would have been
reported as `excluded_unknown`.

### Q5 — Validation: *"Show me Sci-Fi movies"*

```
status : invalid_input
message: The arguments were rejected: unknown genre 'Sci-Fi'; closest match is 'Science Fiction'
```

The failure mode this exists for: SQL would have answered "0 rows" with total confidence.

### Q6 — Fuzzy lookup: *"Tell me about Intersteler"*

```
status : ok
message: 'Intersteler' resolves to Interstellar (2014) (matched at 87).
candidates: Interstellar (87.0) · Winter's Tale (75.0) · The Intern (66.7)
```

87 clears the accept band (82), and the runner-up is 12 points behind — outside the tie margin.

### Q7 — Fuzzy ambiguity: *"lord of the rings"*

```
status : ambiguous
message: 'lord of the rings' could be several movies -- 'The Lord of the Rings: The Fellowship of
         the Ring' (100) and 'The Lord of the Rings: The Return of the King' (100) are within 3
         points -- refusing to guess.

  The Lord of the Rings: The Fellowship of the Ring (2001)  100.0
  The Lord of the Rings: The Return of the King (2003)      100.0
  The Lord of the Rings: The Two Towers (2002)              100.0
  Rise of the Guardians (2012)                               68.4
```

Three films at 100 via the subset gate. The tie-margin rule fires, the turn interrupts, and the user
chooses. By contrast `"the matrix"` is an *exact* normalized match on one film and short-circuits
to `MATCH` at 100 without asking.

### Q8 — Record lookup: `movie_details(157336)`

```
title            Interstellar          vote_average  8.1
release_date     2014-11-05            vote_count    10,867
runtime_minutes  169                   budget_usd    165,000,000 (known)
genres           Adventure, Drama, Science Fiction
director         Christopher Nolan     revenue_usd   675,120,017
top_cast         Matthew McConaughey, Jessica Chastain, Anne Hathaway, Michael Caine, Casey Affleck
unknown fields   []
```

### Q9 — Semantic: *"someone trying to survive alone on another planet"*

```
lexical coverage 1.0 · pool 4,803 · top score 0.6218

  0.6218  What Planet Are You From? (2000)     0.5976  Planet 51 (2009)
  0.6127  The Martian (2015)                   0.5974  Silent Running (1972)
  0.6090  I Am Legend (2007)                   0.5971  Seeking a Friend for the End of the World
  0.6002  Meet Dave (2008)                     0.5951  E.T. the Extra-Terrestrial (1982)
```

*The Martian* is the right answer and ranks second. The film above it is a comedy that shares
surface vocabulary — a real weakness of a labelled-template embedding, discussed in Q12.

### Q10 — Hybrid: *"dark science fiction about artificial intelligence"* + genre filter

```
message: Retrieved 8 movies matching 'dark science fiction about artificial intelligence'.
         Ranked within a filtered pool of 377 movies.

  0.6722  Dark City (1998)                 0.6661  Idiocracy (2006)
  0.6716  I, Robot (2004)                  0.6615  Transcendence (2014)
  0.6716  A Scanner Darkly (2006)          0.6565  TRON: Legacy (2010)
  0.6689  A.I. Artificial Intelligence     0.6477  The Matrix Reloaded (2003)
```

The pool drops from 4,803 to 377 *before* ranking, so all eight results satisfy the genre and
vote-count constraints by construction.

### Q11 — Multi-turn: filters carry, ordinals resolve

See §5.5 — three turns, filters layered (`vote_average >= 7.5` added to the carried genre and year
constraints), and `"the first one"` resolved to `MovieRef(127585, 'X-Men: Days of Future Past')`
without a model call.

### Q12 — **A query that does not work well**: *"movies similar to Inception"*

```
status : ok
message: Retrieved 8 movies similar to Inception (2010).

  0.8001  The Score (2001)                 0.7878  The Jacket (2005)
  0.7989  Trance (2013)                    0.7859  The Terminator (1984)
  0.7978  10 Cloverfield Lane (2016)       0.7851  Light Sleeper (1992)
  0.7893  Wristcutters: A Love Story       0.7849  The Muse (1999)
```

**This is a weak result.** *Trance* and *The Jacket* are defensible; *The Muse* and *Wristcutters:
A Love Story* are not films anyone would offer someone who liked *Inception*.

**Why it happens.** Three causes compound:

1. **The labelled template raises the similarity floor.** Every document shares `Title:`,
   `Genres:`, `Keywords:`, `Director:`, `Starring:`, so no two documents are ever truly dissimilar.
   Note the scores: the *worst* of these eight is 0.785, while the best result for a genuine text
   query (Q9) is 0.622. Document-to-document similarity is compressed into a narrow high band where
   small differences decide the ranking.
2. **`similar_to` uses the seed's full document vector**, which is dominated by cast and keyword
   lists rather than by what makes *Inception* distinctive. Films sharing generic thriller
   vocabulary score highly.
3. **No genre or era constraint is implied.** The tool ranks the entire corpus; nothing prevents a
   1999 comedy from surfacing.

**What would fix it, in order of expected yield:** (a) constrain `similar_to` by the seed's own
genres by default — a one-line pre-filter using existing machinery; (b) weight the overview more
heavily than the metadata lines, or embed overview and metadata separately and combine scores;
(c) a cross-encoder rerank over the top 50. This is recorded as a known limitation rather than
presented as working.

### Q13 — **A second failure mode**: *"Which director has the best average rating?"*

```
status : ok
message: avg_rating by directors (top 10) over 10 groups.

  Gary Sinyor      10.0     Tim McCanlies    8.45
  Rohit Jugraj      9.5     John Cromwell     8.4
  Lance Hool        9.3     W.S. Van Dyke     8.4
  Floyd Mutrux      8.5     Damien Chazelle   8.3
```

The arithmetic is correct and the answer is useless. Every name in the top three directed **one**
film with a handful of votes. The DSL has no `HAVING` clause — no way to express "at least 5 films"
or "at least 1,000 votes per film" — so an aggregation cannot filter on its own output. Pre-filtering
by `vote_count` would help but is not equivalent: it filters *films*, not *directors*.

**The honest characterisation:** this is a structural limit of the closed DSL, which was chosen over
model-generated SQL precisely because it is closed. Adding `min_group_size` to `AggregateSpec` would
fix this specific case; the general problem — aggregations that need to filter on aggregates — needs
either a HAVING equivalent or a second query pass.

### Q14 — Gibberish rejection

```
status : low_confidence
message: 'qwertyuiop zxcvbnm flurble wizzlewop' does not read like a description of anything in
         this dataset -- 'qwertyuiop', 'zxcvbnm', 'flurble', 'wizzlewop' do not appear anywhere
         in the ~4,800 movie documents. Ask the user to rephrase.
payload: {'lexical_coverage': 0.0, 'shown': 0}
```

Rejected before embedding. The similarity floor alone would not have caught this — that same
gibberish scores above 0.35 against real films.

---

## Appendix — reproduction

```bash
# Build the dataset, embeddings and manifest (~10 min on first run)
python scripts/build_index.py

# Optional: load the vectors into an embedded Qdrant collection
python scripts/build_index.py --skip-embeddings --with-qdrant

# Verify the two vector backends agree, and measure both
python scripts/benchmark_vector_backends.py

# Dataset profiling (section 1 numbers)
python scripts/profile_data.py          # → artifacts/data_analysis.md
python scripts/profile_documents.py     # → artifacts/document_stats.md

# Tests: 186 passing, no network required
pytest
```

**Related documents**

| Document | Covers |
|---|---|
| [LANGGRAPH_WORKFLOW.md](LANGGRAPH_WORKFLOW.md) | Graph topology, state channels, reducers, node-by-node logic |
| [CLASS_REFERENCE.md](CLASS_REFERENCE.md) | Every class and model, and how they connect |
| [AGENT_DOCUMENTATION.md](AGENT_DOCUMENTATION.md) | Repository walkthrough |
| [`artifacts/data_analysis.md`](../artifacts/data_analysis.md) | Raw-CSV profiling output |
| [`artifacts/vector_backend_benchmark.md`](../artifacts/vector_backend_benchmark.md) | numpy vs Qdrant agreement and latency |
