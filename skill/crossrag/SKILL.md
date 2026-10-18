---
name: crossrag
description: Find experimental concept-corpus motifs for creative writing, surprising metaphors, copy, or cross-domain ideation, then use them as inspiration. Use when a user wants an unexpected connection or wants to compare similarity, orthogonal-residual, oriented wedge, and random retrieval. Runs local Python when available; retrieved motifs and scores are heuristic, not facts or proof.
---

# CrossRAG

## Purpose and limits

Use CrossRAG when the user wants a fresh starting point for fiction, poetry, metaphor, copy, or cross-domain ideas. It searches an existing concept corpus from several numerical viewpoints and passes selected motifs to the host model as inspiration. Local Python performs retrieval; the host model writes the final text.

This is an experimental ideation aid. It does not fact-check, establish causality, prove that a connection is useful, or show improved creativity or retrieval quality. A retrieved motif or rank is not evidence. The default character n-gram hash is lexical and offline; it is not a semantic embedding. Never say a search ran if the script did not run.

## First run

If Python and NumPy are ready, run this from the skill directory:

```bash
python scripts/run.py --context '犬、カレー、夕暮れ' --method similar
```

This uses the bundled 22-item example corpus and offline `hash` encoder as a simple similarity baseline. Read the JSON `selected` entries as candidate motifs, not finished prose. Choose one to three motifs, preserve a coherent scene or argument, and write original text without copying retrieved lines. The example corpus is only a demonstration; use a suitable user corpus for real work.

For exploration or research ideas, run `--mode discovery`. Present connections as hypotheses with uncertainty and a small test; do not present them as discoveries. If the user only wants ideas, avoid making them learn every scoring option before returning a useful first result.

## Execution

1. Determine **creative** (fiction, metaphor, evocative copy) or **discovery** (brainstorming, research connections) from the request. When ambiguous, default to creative for writing and discovery for ideas.
2. Extract **at least two context concepts** from the request, retaining their exact order for wedge mode. Do not fabricate user facts.
3. Use the bundled `scripts/run.py` from this skill directory. If `numpy` is missing, tell the user that `requirements.txt` installs it; do not install dependencies without authorisation. Default `hash` needs no model download, runs offline, and **is not a semantic embedding**. Use `--encoder semantic` only when `sentence-transformers` and the chosen model are already available or the user authorises setup; never download silently or cause paid API calls.
4. Run `--method wedge` and `--method similar` as a baseline on **the same corpus**. Consider `orthogonal` if alternative directions are wanted. The experimental mode comparison is meaningful only with a common encoder, corpus, and generation conditions.
   Inspect the returned `warnings`. With the default hash encoder, short Japanese terms can make the wedge component zero for every candidate; then wedge contributes nothing to the ranking. Report that warning plainly. Do not claim the selected motifs came from wedge geometry when the component is degenerate.
5. Use the retrieved entries as *inspiration*, not evidence or sentences to reproduce. Record source terms and numerical scores separately from your own interpretive claims.
6. For creative work: retain recognisable scene logic, introduce ideally one surprising relational shift, and make a finished piece of prose. For discovery: state 3 hypotheses, bridges to the source concepts, what may falsify each, and one minimal test. Do not label speculative analogies as discoveries.
7. When reporting research comparisons, be clear that ranks are heuristic and do not establish superiority over vanilla RAG. When retrieval scripts cannot run, say so, and label any language-only mock-up explicitly; never claim mathematical retrieval occurred.

## Host compatibility

Keep this directory intact when installing it. Codex, Gemini CLI, Grok Build, and Pi can discover the user-level `~/.agents/skills/crossrag/` location; Claude Code uses `~/.claude/skills/crossrag/`. The repository README has copy commands, host-specific activation examples, and links to current official discovery documentation. These paths are documented compatibility targets; live host recognition has not been tested here.

## CLI

```bash
python scripts/run.py --context '犬、カレー、夕暮れ、帰宅' --mode creative --method wedge --limit 5
python scripts/run.py --context '記憶、法線、検索' --mode discovery --method orthogonal --limit 5
python scripts/run.py --context '犬、カレー' --mode creative --method similar --encoder semantic
python scripts/run.py --context '犬、カレー' --corpus references/example-corpus.json
```

The script prints structured UTF-8 JSON with retrieved motifs, `cosine`, `orthogonal`, `wedge` and ranked scores. The LLM then generates the final artefact. Use `--corpus` to point to a user-supplied JSON collection of triples `["concept1", "concept2", "inspiration"]` or objects with `a`, `b`, `line`.

Read `references/methodology.md` for formulae, limitations, and evaluation guidance. Read `references/modes.md` for output patterns.

## third-concept (experimental)

Use `python scripts/third_concept_cli.py --A 'concept A' --B 'concept B' --corpus path/to/corpus.json --encoder hash --limit 5 --seed 0` to rank a supplied candidate pool. Use `--emit-prompt` (optionally `--with-math`) to prepare an external hypothesis prompt, then import the resulting file with `--hypotheses path/to/hypotheses.json`. Compare `similar`, `orthogonal`, `wedge`, `random`, `third-concept`, `mean`, and `third-concept-no-wedge` with `python scripts/compare_third_concept.py --A 'concept A' --B 'concept B' --corpus path/to/corpus.json --format markdown`. Preserve every output candidate's `origin` label (`retrieved`, `math`, `llm_hypothesis`, or `idea`); all outputs remain unverified and must not be presented as facts, proofs, or discoveries.

`third_concept_cli.py` requires `--limit` to be at least 1. The three third-concept ranking variants exclude corpus entries that match or contain either input, or have raw cosine similarity of at least 0.98 to an input; comparison baselines remain the unfiltered `engine.run()` results and expose `contains_input` on each selected entry.
