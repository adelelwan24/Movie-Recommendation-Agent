# Vector backend benchmark - numpy vs Qdrant
`scripts/benchmark_vector_backends.py` - 200 queries per scenario, k=8, corpus of 4,803 documents.
**The two backends agree exactly on every scenario**: identical candidate pools, identical top-k, identical ordering. The Qdrant payload filters reproduce `MovieRepository.mask_for` exactly, so ADR-0011's pre-filter guarantee survives the swap.

## Agreement
| scenario | pool (numpy) | pool (qdrant) | pools agree | recall@8 | identical ordering | max score delta |
| --- | --- | --- | --- | --- | --- | --- |
| unfiltered | 4,803 | 4,803 | yes | 1.000 | 1.000 | 2.38e-07 |
| single genre | 2,297 | 2,297 | yes | 1.000 | 1.000 | 2.38e-07 |
| genre + year + runtime | 93 | 93 | yes | 1.000 | 1.000 | 2.52e-07 |
| numeric only (nulls must not match) | 181 | 181 | yes | 1.000 | 1.000 | 1.92e-07 |
| budget between (unknown != 0) | 964 | 964 | yes | 1.000 | 1.000 | 2.38e-07 |
| person (cast OR director) | 14 | 14 | yes | 1.000 | 1.000 | 3.25e-07 |
| multi-value membership | 1,441 | 1,441 | yes | 1.000 | 1.000 | 2.38e-07 |
| narrow pool | 17 | 17 | yes | 1.000 | 1.000 | 2.17e-07 |

`recall@k` is the share of numpy's top-k that Qdrant also returned; `identical ordering` is the stricter test that the ranked lists match position for position.

## Latency per query
| scenario | numpy median (ms) | numpy p95 (ms) | qdrant median (ms) | qdrant p95 (ms) | ratio |
| --- | --- | --- | --- | --- | --- |
| unfiltered | 0.78 | 1.26 | 8.73 | 11.92 | 11.1x |
| single genre | 2.40 | 4.39 | 79.14 | 186.55 | 33.0x |
| genre + year + runtime | 0.13 | 0.24 | 112.86 | 314.86 | 873.9x |
| numeric only (nulls must not match) | 0.16 | 0.35 | 73.33 | 181.12 | 455.1x |
| budget between (unknown != 0) | 0.74 | 1.86 | 66.64 | 162.48 | 90.4x |
| person (cast OR director) | 0.10 | 0.26 | 313.37 | 681.67 | 3165.4x |
| multi-value membership | 2.16 | 4.17 | 107.32 | 141.45 | 49.6x |
| narrow pool | 0.10 | 0.19 | 89.73 | 236.43 | 875.4x |

- numpy, all scenarios: median 0.38 ms
- qdrant (embedded), all scenarios: median 95.20 ms - 247x slower

### Reading these numbers
The gap is not a property of Qdrant, and the shape of it says why: numpy gets **faster** as a filter narrows, because fewer rows get scored. Embedded Qdrant gets **slower**. The narrowest filters are its *slowest* scenarios -- 'person (cast OR director)' admits 14 candidates and takes 313 ms.

That inversion is the embedded engine's cost model. Local mode evaluates payload conditions in Python and **ignores payload indexes entirely** -- it warns as much at build time -- so every filtered query walks the collection. A served Qdrant applies the same filters through indexed structures during traversal, so these latencies would have to be re-measured against a real instance before concluding anything about the engine itself.

Concretely, at this corpus size numpy is the faster backend by a wide margin and remains the sensible default. Embedded Qdrant buys the operational shape -- a real store, server-side filters, a one-setting move to a cluster -- and costs tens of milliseconds per query, which is still small beside an LLM turn but no longer free. The case for switching is the corpus outgrowing an O(N*d) matmul per query, which is the expiry condition ADR-0006 set for itself. This benchmark is not that case.
