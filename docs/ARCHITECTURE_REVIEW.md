# Senior AI engineering architecture review

## Executive verdict

The approach is **architecturally strong but not submission-ready**.

The repository demonstrates senior-level instincts in deterministic/LLM separation, typed tool contracts, offline artifacts, stale-index checks, explicit ambiguity handling, pre-filtered vector search, observable traces, and offline testing. These choices fit the assignment better than a generic “LLM over CSV” solution.

Three functional gaps prevent a clean acceptance:

1. hybrid multi-genre filters use OR when the example requires AND;
2. semantic retrieval is not guaranteed to feed a grounded generation step;
3. the advertised tool-iteration and timeout guards are ineffective.

Submission readiness is also blocked by the absent deployment URL and, before this review, absent deliverable documentation. The repository configuration currently points at an older project directory, so a default test run can use the wrong artifacts.

My assessment:

| Area | Assessment |
| --- | --- |
| Core architecture | Strong |
| Deterministic correctness | Strong for single-value filters and aggregates |
| Fuzzy matching | Strong, with sensible refusal behavior |
| Semantic retrieval | Good baseline, measured limitations |
| Hybrid retrieval | Conceptually correct pre-filtering, incorrect same-field conjunction semantics |
| Agent orchestration | Good topology, insufficient enforcement and guards |
| Grounding | Good layered intent, incomplete guarantees |
| Memory | Meets the canonical demo in scripted tests; process-local and LLM-dependent |
| Testing | Broad offline suite, but important gaps and stale documentation |
| Deliverable completeness | Incomplete |

## Assignment compliance

| Requirement | Status | Evidence / concern |
| --- | --- | --- |
| Streamlit natural-language demo | Implemented locally | `app.py`; no deployment URL |
| Agent-based routing | Implemented | Typed planner plus LangGraph agent/tool loop |
| Structured search | Implemented | Closed `SearchQuery` DSL and deterministic repository |
| Fuzzy title matching | Implemented | RapidFuzz, thresholds, ties, interrupt/resume |
| Semantic retrieval | Implemented | BGE embeddings and Numpy/Qdrant backends |
| Grounded LLM answers | Partial | `rag_answer` is grounded, but the graph does not require it after semantic retrieval |
| Multi-turn memory | Implemented with limitations | Typed filters/results/selection in `MemorySaver`; scripted three-turn flow passes |
| Visible tool selection and context | Implemented, one inaccurate caption | Trace shows tools/filters/documents; displayed documents are not necessarily the generation context |
| Graceful failures/empty results | Mostly implemented | Typed outcomes and UI handling; wall-clock timeout is not enforced |
| Hybrid search bonus | Partial / incorrect for the example | Pre-filtering is real, but multiple genres are OR-ed |
| Tests | Strong breadth | 186 pass and 3 known xfails with correct local paths; live provider behavior is opt-in |
| README/setup | Partial | Useful content, but stale counts/links and machine-specific paths existed |
| Agent technical document | Added by this review | See `AGENT_DOCUMENTATION.md` |
| Deployed app | Missing | README explicitly treats deployment as not built |
| Private Git repository | Repository exists | Current branch has a small commit history; privacy cannot be verified locally |

## What should be kept

### Deterministic facts behind a typed boundary

`SearchQuery` is preferable to model-generated SQL here. It validates fields and value shapes, produces displayable filters, supports state merging, and keeps arithmetic outside the LLM. The repository also distinguishes the true match total from capped display rows, preventing “showing 25” from becoming “there are 25.”

### One curated semantic document per movie

The document template uses title, year, alternate title, tagline, genre, keywords, director, top cast, and overview. At roughly 95 words per movie, one document is the correct granularity. Splitting these records would fragment context and add no useful recall.

### Exact search at this corpus size

The Numpy backend is a pragmatic default. About 7 MB of normalized vectors can be searched exactly in memory, making results deterministic and hybrid masks simple. The Qdrant adapter is a credible scale-out seam, and backend-parity tests address the biggest migration risk: silently different filter semantics.

### Explicit ambiguity rather than best-match guessing

The fuzzy matcher treats ambiguity as a normal outcome. The top-two margin rule is especially good because a high score is not enough when two franchise titles are nearly tied. A real graph interrupt preserves the tool result and resumes the same turn.

### Artifact integrity and observable operation

The manifest guards against source, preprocessing, embedding-model, and dimension drift. Full tool artifacts stay out of the model prompt while feeding tables and traces. Provider reasoning metadata is removed before graph state, which is a good response to the “do not expose chain-of-thought” requirement.

## Blocking findings

### P0 - Hybrid same-field constraints have the wrong meaning

In [repository.py](../src/movieagent/data/repository.py), `_membership_mask` unions all requested values. Therefore `genres=["Science Fiction", "Comedy"]` means “Science Fiction OR Comedy.” The Qdrant translation intentionally mirrors the same behavior with `MatchAny`.

That is valid for a user asking for “horror or thriller,” but invalid for “funny science-fiction,” where both concepts are constraints. An actual checked run produced a pool of 417 and returned Comedy-only *Crazy, Stupid, Love.* and Science-Fiction-only *Elysium*.

Recommended fix:

- model membership explicitly, for example `MembershipCondition(field, values, mode="any"|"all")`;
- use `all` for genre conjunctions and `any` only when the user states alternatives;
- implement equivalent Pandas and Qdrant semantics;
- add an acceptance test asserting every hybrid result has both Comedy and Science Fiction, is after 2010, and is under 120 minutes.

Do not merely change all list filters to AND. `people`, countries, and user-stated alternatives still need OR semantics.

### P0 - The semantic-to-RAG chain is advisory, not guaranteed

`semantic_search` places full documents in the tool artifact, but `ToolResult.summary_for_model()` sends only a compact status and movie references to the executor. The documents are visible to the UI, not to the executor model.

The executor may call `rag_answer`, which builds grounded context from the selected records. However, nothing enforces that sequence. In fact, the semantic graph test plans `semantic_search -> rag_answer` but scripts the model to answer immediately after `semantic_search`, and the test passes.

Consequences:

- a semantic answer can be generated from titles/years rather than retrieved plot context;
- the UI says the displayed documents were “handed to the model,” which is not generally true;
- plan adherence is reported only after the fact.

Recommended fix:

- make semantic retrieval deterministically route to a grounded-answer node, or have `semantic_search` return an ID set consumed automatically by that node;
- treat an agent prose response after semantic retrieval as incomplete until grounded synthesis has run;
- record the exact generation context separately from the retrieval document;
- update the UI label to distinguish “retrieval document” from “generation context.”

### P0 - Loop and time guards are advertised but not effective

In [graph.py](../src/movieagent/agent/graph.py), `_after_tools` calculates `iterations = state.tool_iterations + 1` but never writes it back. `tool_iterations` remains zero, so `MAX_TOOL_ITERATIONS` does not stop a loop. The recursion limit is the only effective loop backstop.

`TURN_TIMEOUT_S` is configured and documented but not referenced by execution code. A slow provider call can therefore exceed the claimed wall-clock budget.

Recommended fix:

- add a node/update that increments `tool_iterations` after every tool batch;
- route to a clear terminal failure when the cap is reached;
- enforce provider/request timeouts and, if a whole-turn budget is required, calculate remaining time before each model/tool call;
- add tests that deliberately produce more calls than the cap and use a slow fake model.

## High-priority findings

### P1 - Local configuration can silently select another repository

The checked-in `.env` and `.env.bak` set `DATA_DIR` and `ARTIFACTS_DIR` to `D:\Docs\Current\Movie Recommendation Agent`, not this V2 workspace. `.env.example` also renders absolute machine paths.

The first default test run therefore reached an old Qdrant store and produced 34 setup errors. After overriding the paths to this repository and using a writable temporary directory, the result was 186 passed and 3 expected failures.

Recommended fix:

- do not commit `.env` or `.env.bak`; rotate any exposed credential as a separate security check;
- omit path variables from `.env.example` so portable code defaults are used, or set relative paths such as `data` and `artifacts`;
- make the app display resolved artifact paths;
- add a startup guard that the manifest/source paths belong to the expected project when running in development.

### P1 - Planner output is not execution policy

The typed plan is valuable for visibility, but the free ReAct loop may ignore, reorder, or skip steps. Deviations are shown after execution rather than prevented. For hard requirements such as fuzzy -> details and semantic -> RAG, deterministic conditional edges are safer.

Recommended approach: keep the LLM planner for intent and argument extraction, then compile its plan into a small allowed workflow. Use the free tool loop only for recoverable correction, such as an invalid vocabulary value.

### P1 - Person fuzzy resolution is not integrated

`FuzzyTitleMatcher.resolve_person()` exists and is unit-tested but no tool or graph path calls it. Misspelled people currently rely on structured-query vocabulary validation returning a suggestion and the LLM retrying correctly.

Either expose a general `entity_resolver` for titles/people or remove the dead method and document validation/retry as the intended design. The assignment's Nolan example explicitly invites entity resolution before structured aggregation.

### P1 - Grounding validates entities, not attributes

The post-hoc checker can warn about an unreturned movie title, but cannot detect a wrong year, runtime, rating, budget, or cast member attached to an allowed movie. This is openly documented in code and should remain explicit in the submission.

For stronger guarantees, generate factual fields directly from structured payload templates and use the LLM only for connective prose, or validate claims against a fact table before display.

### P1 - Mandatory deployment is absent

The assignment requires a live deployed Streamlit demo. This is not an optional bonus in the deliverables table, even though deployment also appears in the bonus list. The candidate should clarify the ambiguity with the reviewer, but safest is to deploy and provide the URL.

## Medium-priority observations

- `SUM_REVENUE` uses Pandas' default sum behavior; an all-unknown group can become zero. Use `sum(min_count=1)` and surface unknown coverage.
- `similarity_floor` is intentionally weak, while lexical coverage can reject rare but valid wording and cannot detect fluent absent topics. A calibrated evaluation set or cross-encoder confidence model would be stronger.
- one previous result set is stored, so “go back to the earlier list” cannot work reliably.
- in-memory checkpoints disappear on restart and can grow for long-running multi-user deployments.
- the test suite has three strict expected failures, which is honest, but the README previously reported 137 tests rather than the current 189 collected.
- the local embedding stack is operationally heavy for a 4,803-record demo. It is defensible for offline operation, but a smaller runtime/container plan should be stated for deployment.
- exact vector similarity has no reranker. For abstract-theme queries, a cross-encoder or lightweight LLM reranker over the top 30-50 could improve precision.

## Recommended order of work

1. Fix membership semantics and add a strict hybrid acceptance test.
2. Make semantic retrieval deterministically flow into grounded generation; correct the UI context label.
3. Persist/enforce iteration counts and implement the timeout contract.
4. remove machine-specific `.env` paths and verify a clean-clone setup.
5. deploy the Streamlit app and run live provider smoke tests.
6. integrate person/entity resolution or simplify the claim.
7. strengthen attribute grounding and semantic evaluation if time remains.

After steps 1-5, this would be a credible senior-level submission. Before them, the architecture is promising but several visible guarantees are stronger than the implementation.
