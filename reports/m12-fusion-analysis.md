# M12: why rank fusion helps at K=5 and hurts at K=10

The frozen comparison left one unexplained result. Paired against vector retrieval,
hybrid gains **+5.0 percentage points at K=5** (3 wins, 36 ties, 1 loss) and loses
**7.5 points at K=10** (0 wins, 37 ties, 3 losses). Same graph, same index, same
questions, opposite sign.

This is a complete explanation of that flip, derived from the saved records of
`experiments/runs/m11-research-comparison-02`. It is an error analysis of one
development run on unreviewed draft labels, not a general claim about rank fusion.

## The arithmetic sets the stage

Reciprocal rank fusion scores a chunk `1 / (rank_constant + rank)` per method and sums.
With `rank_constant = 60`:

| rank | score |
| ---: | ---: |
| 1 | 0.01639 |
| 5 | 0.01538 |
| 6 | 0.01515 |
| 10 | 0.01429 |
| 20 | 0.01250 |

A chunk found **only** by graph at graph rank `g` outranks a chunk found **only** by
vector at vector rank `v` whenever `g < v`. Both methods contribute up to 20 candidates,
so graph ranks 1…n compete position-for-position with vector ranks 1…n and the fused list
interleaves the two.

Measured across the 35 questions where graph produced any candidate, hybrid's top ten
contains on average **4.20 graph-only chunks and 5.80 vector-derived chunks**. Fusion is
not reranking vector's list; it is handing graph roughly half the slots.

## That makes the cutoff decisive

Whether handing graph half the slots helps depends entirely on where the needed evidence
sits in vector's own ranking. The seven questions that moved show both sides cleanly.

### K=5 wins: graph rescues evidence vector buried

| Question | Needed chunk at vector rank | Hybrid rank | Source |
| --- | ---: | ---: | --- |
| `paper-d11` | 9 | **2** | graph |
| `paper-d15` | 7 | **1** | graph |
| `paper-d19` | 10 | **1** | graph |

In each case vector *had* the right chunk but ranked it 7th to 10th — unreachable at a
cutoff of 5. Graph ranked it highly, RRF promoted it to position 1 or 2, and the question
flipped from incomplete to complete. Graph's contribution was not new information; it was
a better position for information vector already held.

### K=5 loss: the one case where graph displaced a winner

| Question | Needed chunk at vector rank | Hybrid rank |
| --- | ---: | ---: |
| `paper-d13` | 3 | **absent** |

The only way fusion can hurt at K=5 is by displacing evidence already inside vector's top
five. That happened once in 40.

### K=10 losses: there is nothing left to rescue, so injection is pure cost

| Question | Needed chunk at vector rank | Hybrid rank | Graph-only chunks promoted | Vector chunks displaced |
| --- | ---: | ---: | ---: | ---: |
| `paper-d21` | 6 | absent | 5 | 5 |
| `paper-g01` | 8 | absent | 5 | 5 |
| `paper-g02` | 6 | absent | 4 | 4 |

Every needed chunk sat at vector rank 6 or 8 — *inside* the top ten. Vector reaches it
unaided. Graph then injected four or five chunks that displaced exactly the tail positions
where that evidence lived.

## The finding

**Fusion with a sparse secondary signal helps only when the cutoff is tighter than the
primary retriever's reach.**

The graph's useful contribution here lives in a narrow band: evidence vector ranks roughly
7th to 10th. At K=5 that band is excluded by the budget, so promoting it is almost free
upside — three wins against one loss. At K=10 the budget already covers that band, so
there is nothing to rescue, and the same promotion mechanism evicts correct tail hits
instead. The sign flip is not a property of graph quality. It is the interaction between a
fixed fusion rule and a cutoff.

Two practical consequences, if this work continues:

- **Fusion weight should depend on K.** A rule that gives graph half the slots is
  defensible at K=5 and counterproductive at K=10. `graph_candidates` is already
  configurable; making it a function of the cutoff is the obvious ablation.
- **This is a ranking problem, not a coverage problem.** It is the same conclusion the
  multi-hop diagnostic reached from the other direction. Vector held the right chunk in
  all six questions that moved; only its position was wrong.

## Limits

Seven questions moved out of 40, across 3 wins and 4 losses. These counts are too small to
support a significance claim, and the report makes none. The 8 multi-document questions are
unaffected — every method scores zero on them at both cutoffs — so nothing here addresses
the project's central question.

All 40 labels remain unreviewed draft development annotations with zero held-out questions,
and the graph under analysis holds 111 assertions from an extractor that recovers 2 of 6
sampled draft facts. A denser or reviewed graph could shift every number above. What is
established is the mechanism, not its magnitude.

## Ablation: the sign flip is not tunable with the available parameters

The analysis above suggested an obvious fix — give graph fewer slots at larger cutoffs.
Four additional verified runs on bundle `m10-expanded-04` tested that and a second idea.
**Both predictions failed**, and the failures are more informative than a success would
have been.

| Variant | K=5 delta | K=5 W/T/L | K=10 delta | K=10 W/T/L |
| --- | ---: | ---: | ---: | ---: |
| `graph_candidates = 5` | +2.5 pp | 2/37/1 | **−7.5 pp** | 0/37/3 |
| `graph_candidates = 10` | +5.0 pp | 3/36/1 | **−7.5 pp** | 0/37/3 |
| `graph_candidates = 20` (default) | +5.0 pp | 3/36/1 | **−7.5 pp** | 0/37/3 |
| `rank_constant = 6` | +5.0 pp | 3/36/1 | **−7.5 pp** | 0/37/3 |
| `rank_constant = 60` (default) | +5.0 pp | 3/36/1 | **−7.5 pp** | 0/37/3 |
| `rank_constant = 600` | +5.0 pp | 3/36/1 | **−7.5 pp** | 0/37/3 |

Runs: `m12-fusion-graphcand-5`, `m12-fusion-graphcand-10`, `m12-fusion-rankc-6`,
`m12-fusion-rankc-600`, each 240 records with offline verification passed.

### Why the candidate window cannot help

`score = 1 / (c + rank)`. For a graph-only chunk at graph rank `g` against a vector-only
chunk at vector rank `v`, `1/(c+g) > 1/(c+v)` exactly when `g < v`. The displacement is
performed by graph's **top** candidates, which survive any window of five or more.
Shrinking the window only trims graph's tail, which never reached the fused top ten.
Shrinking it to five does not fix K=10 and costs half the K=5 gain, because it starts
discarding the useful promotions too.

### Why rank_constant cannot help either

In that same inequality, `c` cancels. It only matters for chunks found by **both**
methods, whose score is `1/(c+g) + 1/(c+v)`: as `c` grows, each term approaches `1/c`, so
a two-method chunk approaches `2/c` against a one-method chunk's `1/c` and agreement comes
to dominate rank position. That reasoning is correct and irrelevant here, because the
decisive comparisons contain no agreement. Measured on all three K=10 losses:

- the needed chunk was **vector-only** in every case — graph never found it;
- of the 13 displacing chunks across the three questions, **0 were in vector's top 20**.

Vector and graph retrieve largely disjoint candidates: across the 35 questions where graph
produced anything, they share **69 candidates in total**, and 12 questions share none.
RRF's central mechanism is rewarding agreement between methods, and here there is almost
no agreement to reward, so it degenerates into interleaving two unrelated lists.

### What this actually implies

The sign flip cannot be removed by configuration. Every knob in `[benchmark.hybrid]`
leaves it at exactly −7.5 pp. Fixing it needs a capability the fusion rule does not have:
a **per-method weight**, so graph's contribution can be scaled down relative to vector
rather than merely truncated. That is a code change and a new experiment, not an ablation,
and it is deliberately not attempted here.

It also sharpens the earlier conclusion. Graph is not a weak version of vector; it is
retrieving a nearly disjoint part of the corpus. On this corpus that disjoint part is
mostly unhelpful, but the disjointness is why unweighted fusion is the wrong tool — it
spends half the budget on the other list regardless of how good that list is.

### Limits specific to the ablation

Six configurations, one corpus, 40 unreviewed draft questions, one sparse graph of 111
assertions. The invariance of the K=10 result across every parameter is a strong signal
about this fusion rule's structure, and says nothing about fusion rules in general. No
significance test is implied by any number above.
