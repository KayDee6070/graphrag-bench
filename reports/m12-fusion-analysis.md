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
