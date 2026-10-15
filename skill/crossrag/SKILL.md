---
name: crossrag
description: Creative and discovery retrieval using experimental semantic geometry. Use when asked to write unexpectedly coherent fiction, poetic metaphors, surprising copy, or to brainstorm cross-domain inventions and unusual research connections with CrossRAG; compare similarity, orthogonal-residual, oriented wedge (exterior-product) and random retrieval using an existing concept corpus. Runs local Python scripts when execution is available and labels unverified hypotheses clearly.
---

# CrossRAG

Use the numerical retrieval engine before generating text. The core is adapted from CrossRAG Lab v0.1; do not claim a wedge operation directly generates prose or that creativity improvements are proved.

## Execution

1. Determine **creative** (fiction, metaphor, evocative copy) or **discovery** (brainstorming, research connections) from the request. When ambiguous, default to creative for writing and discovery for ideas.
2. Extract **at least two context concepts** from the request, retaining their exact order for wedge mode. Do not fabricate user facts.
3. Use the bundled `scripts/run.py` from this skill directory. Install `requirements.txt` (`numpy`) only if needed/authorised; default hash encoder is offline but **is not a semantic embedding**. Prefer `--encoder semantic` only if `sentence-transformers` and the model are locally available; first-time downloads require permission/network. Avoid downloading silently or causing paid API calls.
4. Run `--method wedge` and `--method similar` as a baseline on **the same corpus**. Consider `orthogonal` if alternative directions are wanted. The experimental mode comparison is meaningful only with a common encoder, corpus, and generation conditions.
5. Use the retrieved entries as *inspiration*, not evidence or sentences to reproduce. Record source terms and numerical scores separately from your own interpretive claims.
6. For creative work: retain recognisable scene logic, introduce ideally one surprising relational shift, and make a finished piece of prose. For discovery: state 3 hypotheses, bridges to the source concepts, what may falsify each, and one minimal test. Do not label speculative analogies as discoveries.
7. When reporting research comparisons, be clear that ranks are heuristic and do not establish superiority over vanilla RAG. When retrieval scripts cannot run, say so, and label any language-only mock-up explicitly; never claim mathematical retrieval occurred.

## CLI

```bash
python scripts/run.py --context '犬、カレー、夕暮れ、帰宅' --mode creative --method wedge --limit 5
python scripts/run.py --context '記憶、法線、検索' --mode discovery --method orthogonal --limit 5
python scripts/run.py --context '犬、カレー' --mode creative --method similar --encoder semantic
python scripts/run.py --context '犬、カレー' --corpus references/example-corpus.json
```

The script prints structured UTF-8 JSON with retrieved motifs, `cosine`, `orthogonal`, `wedge` and ranked scores. The LLM then generates the final artefact. Use `--corpus` to point to a user-supplied JSON collection of triples `["concept1", "concept2", "inspiration"]` or objects with `a`, `b`, `line`.

Read `references/methodology.md` for formulae, limitations, and evaluation guidance. Read `references/modes.md` for output patterns.
