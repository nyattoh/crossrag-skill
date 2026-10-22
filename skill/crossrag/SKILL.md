---
name: crossrag
description: Explore conversation or explicitly supplied concept corpora with single-metric geometry (similarity, orthogonal-residual, or oriented wedge), normal-ray, or random-orthogonal search. Draw uniformly from a separately generated host-LLM knowledge pool in a context-free mode. Bundled demo data is never used implicitly.
license: MIT
compatibility: Requires Python 3.10+ and NumPy for local retrieval. Semantic mode additionally needs sentence-transformers and a locally available model.
---

# CrossRAG

## Purpose and limits

CrossRAG supports two separate inputs. Geometric modes retrieve from conversation utterances or an explicitly supplied corpus. The `llm-pool` mode draws concepts uniformly from a separately generated host-LLM knowledge pool.

- **Materials are explicit**: Retrieval requires `--conversation`, `--corpus`, `--demo-corpus`, or `--llm-pool`. The built-in 22-item `DEFAULT_CORPUS` is used only with explicit `--demo-corpus` and is labelled `corpus_source: "demo"`. `references/example-corpus.json` is a separate example file; when supplied with `--corpus`, it is labelled `corpus_source: "explicit"`.
- **LLM knowledge is a separate mode**: `--llm-pool` accepts a JSON array of strings and does not use context, a geometric mode, or an encoder. Its `generation_model` is optional metadata. Do not claim RAG retrieval unless the host confirms that retrieval was used.
- **Single-metric mathematical ranking**: In `engine.py` / `run.py`, each mode ranks strictly by one metric (`similar` = `cosine`, `orthogonal` = `orthogonal_residual`, `wedge` = `gram_bivector_cos`). No composite score, cosine-band target, novelty weight, or coherence weight is used.
- **Encoder disclosure**: Always display the `encoder` field from the output. The default `hash` encoder (`hash-ngram（文字特徴・意味理解ではない）`) is a character n-gram hash representation and **is not a semantic embedding**.
- Never say a search ran if the script did not run.

## Reporting policy (Strict Raw Output Default)

When the user asks to run CrossRAG (e.g. "CrossRAGして"):

1. **Default output is raw results only**. For geometric retrieval, return the structured JSON or a plain table with `context`, `corpus_source`, `encoder`, `mode`, `ranking_metric`, `selected`, `warnings`, and `note`. Each selected row contains `index`, `a`, `b`, `line`, `rank_value`, `ranking_metric`, `cosine`, `orthogonal`, and `wedge`.
   For `--llm-pool`, return `mode`, `corpus_source`, `knowledge_basis`, `encoder`, `ranking_metric`, `seed`, `generation_model`, `selected`, and `total`. Each selected concept includes `index`, `concept`, `rank_value`, and `ranking_metric`. `index` is zero-based in the deduplicated list. Do not add a context field because the pool draw does not use context.
2. **No unrequested interpretation**: Unless the user explicitly asks for interpretation, prose generation, metaphorical explanation, or hypothesis writing, **do NOT add any interpretation, psychological reading, metaphorical commentary, or creative prose**.

## First run

Run from the skill directory with a conversation log or explicit corpus:

```bash
python scripts/run.py --context '犬、カレー、夕暮れ' --conversation path/to/conversation.json --mode orthogonal
```

For an explicit smoke test using the demo corpus, you must pass `--demo-corpus`:

```bash
python scripts/run.py --context '犬、カレー、夕暮れ' --mode similar --demo-corpus
```

Omitting `--conversation`, `--corpus`, `--demo-corpus`, and `--llm-pool` raises a CLI error (`exit code 2`). The third-concept and comparison CLIs also require an explicit corpus, demo flag, or LLM pool for retrieval.

## Geometric execution

1. Extract **at least two context concepts** from the user's request or conversation, retaining their exact order for `wedge` mode. Do not fabricate user facts.
2. Prepare the candidate pool from **conversation data** (`--conversation <path>`) or an **explicit user corpus** (`--corpus <path>`). Only use `--demo-corpus` if the user explicitly asks for a demo/smoke test.
3. Run `scripts/run.py` from this skill directory. If `numpy` is missing, tell the user that `requirements.txt` installs it; do not install dependencies without authorisation. Default `hash` needs no model download, runs offline, and **is not a semantic embedding**. Use `--encoder semantic` only when `sentence-transformers` and the chosen model are already available or the user authorises setup; never download silently or cause paid API calls.
4. Select `--mode` (`similar`, `orthogonal`, or `wedge`). Each mode orders strictly by its single metric (`rank_value`):
   - `similar` → `cosine`
   - `orthogonal` → `orthogonal_residual`
   - `wedge` → `gram_bivector_cos`
   Inspect the returned `warnings`. With the default hash encoder, short Japanese terms can make the wedge component zero for every candidate; report that warning plainly.
5. **Follow the Reporting policy**: For geometric modes, return only the raw JSON or metric table with context, source, encoder, ranking metric, selected rows, and warnings. For `--llm-pool`, return the separate draw schema described above. Only perform downstream writing or hypothesis framing if the user explicitly requests it.

## Host compatibility

Keep this directory intact when installing it; relative script and reference paths depend on that layout. Codex, Gemini CLI, Grok Build, and Pi support the shared user-level `~/.agents/skills/crossrag/` location. Claude Code uses `~/.claude/skills/crossrag/` and does not document `~/.agents/skills/` as a discovery root. `agents/openai.yaml` provides optional Codex UI metadata; the other hosts use `SKILL.md` and its supporting files.

Use each host's own check and invocation when needed: Codex `/skills` then `$crossrag`; Claude Code `/skills` then `/crossrag`; Gemini CLI `gemini skills list` or `/skills list`, then ask in natural language for a task matching the description (the official CLI guide does not document per-skill slash invocation); Grok Build `/skills` then `/crossrag`; Pi `/skill:crossrag`. Reload changed skills with a Codex restart, Claude Code `/reload-skills` when a new top-level skills directory was added mid-session, Gemini `/skills reload`, or Pi `/reload`; Grok Build can be restarted if a new skill does not appear. These are documented discovery and activation methods, not a claim that every host runtime has been exercised.

## CLI

```bash
python scripts/run.py --context '記憶、法線、検索' --conversation path/to/conversation.json --mode orthogonal --limit 5
python scripts/run.py --context '犬、カレー、夕暮れ、帰宅' --corpus path/to/corpus.json --mode wedge --limit 5
python scripts/run.py --context '犬、カレー' --mode similar --encoder hash --demo-corpus
python scripts/run.py --context '犬、カレー' --corpus references/example-corpus.json --mode orthogonal
```

The script prints structured UTF-8 JSON with `context`, `mode`, `ranking_metric`, `corpus_source` (`conversation`, `explicit`, or `demo`), `encoder`, `selected` (each item containing `index`, `a`, `b`, `line`, `rank_value`, `ranking_metric`, `cosine`, `orthogonal`, `wedge`), `total`, `warnings`, and `note`.

## LLM knowledge pool

Use this mode when the user asks for concepts from the host model's general knowledge rather than conversation retrieval. Regenerate the pool for every new request and vary the fields as well as the concepts. Reuse a saved pool only when explicitly reproducing a previous draw.

1. Start a fresh host request that does not inherit the parent conversation, transcript, or conversation-derived context. If the host cannot guarantee that isolation, state the limitation and do not describe the pool as context-free.
2. Use this fixed prompt without adding context terms: `From your general learned knowledge, list 40 distinct concepts across varied fields. Choose a fresh mix of fields and concepts for this request. Return only a JSON array of non-empty strings. Do not use conversation history or user-provided context. Do not claim external retrieval unless you can confirm it was used.` Save only the returned JSON array as a UTF-8 file. This workflow requests model knowledge; it does not prove which training or retrieval sources produced an item.
3. Run `python scripts/run.py --llm-pool path/to/pool.json --limit 5`. Add `--seed 17` to reproduce a draw. Add `--llm-pool-model model-id` only when the generation model is known. The optional `--context` argument has no effect on this mode.
4. The file must be a UTF-8 JSON array of non-empty strings. CrossRAG strips surrounding whitespace, removes duplicates after Unicode NFKC, case-folding, and whitespace normalisation, then requires at least 10 unique concepts. If fewer remain, CrossRAG exits with code 2. `--limit` must not exceed the unique pool size.
5. Without `--seed`, CrossRAG creates a fresh 53-bit seed for each run and returns it. With a seed, it selects the first `k` indices from `numpy.random.default_rng(seed).permutation(pool_size)`, where `k` is `--limit`. Reproducing the order requires the same pool and NumPy version. Never derive the seed from context.
6. The result uses `corpus_source: "llm_knowledge"`, `knowledge_basis: "host_llm_general_knowledge"`, `ranking_metric: "seeded_uniform_draw"`, `encoder: "none"`, and `rank_value` values from 1 through `k`. These labels are distinct from geometric ranking metrics. Do not describe the draw as relevance, similarity, novelty, or quality ranking.
7. Only after selection, combine the selected concepts with context if the user asks. Keep that later host request separate from pool generation and the draw.

The same `--llm-pool` and `--llm-pool-model` options are available in `third_concept_cli.py` and `compare_third_concept.py`. They return the same context-free draw schema and reject geometric or comparison options. `--emit-prompt` without `--with-math` returns a prompt without reading a corpus. `--emit-prompt --with-math` reads an explicitly selected corpus and includes geometric candidates. It labels demo-derived prompts with `corpus_source: demo`.

Read `references/methodology.md` for formulae, limitations, and evaluation guidance. Read `references/modes.md` for mode details.

## third-concept (experimental)

Use `python scripts/third_concept_cli.py --A 'concept A' --B 'concept B' --corpus path/to/corpus.json --encoder hash --limit 5 --seed 0` to rank a supplied candidate pool. Use `--emit-prompt` (optionally `--with-math`) to prepare an external hypothesis prompt, then import the resulting file with `--hypotheses path/to/hypotheses.json`. Compare `similar`, `orthogonal`, `wedge`, `third-concept`, `mean`, and `third-concept-no-wedge` with `python scripts/compare_third_concept.py --A 'concept A' --B 'concept B' --corpus path/to/corpus.json --format markdown`. Preserve every output candidate's `origin` label (`retrieved`, `math`, `llm_hypothesis`, or `idea`); all outputs remain unverified and must not be presented as facts, proofs, or discoveries.

`third_concept_cli.py` requires `--limit` to be at least 1. The three third-concept ranking variants exclude corpus entries that match or contain either input, or have raw cosine similarity of at least 0.98 to an input; comparison baselines remain the unfiltered `engine.run()` results and expose `contains_input` on each selected entry.

## normal-ray (experimental)

The default `residual-cluster` method uses an explicit source article C. Example:

```bash
python scripts/third_concept_cli.py --A 'concept A' --B 'concept B' --C 'source article' --corpus references/example-corpus.json --encoder hash --search-mode ray --ray-radius 0.25 --ray-start-distance 0.25 --ray-cluster-cosine 0.7 --ray-min-cluster-size 2 --limit 5
```

The CLI equivalent for Japanese text is `python scripts/third_concept_cli.py --A "概念A" --B "概念B" --C "出典記事" --corpus references/example-corpus.json --encoder hash --search-mode ray --ray-radius 0.25 --ray-start-distance 0.25 --ray-cluster-cosine 0.7 --ray-min-cluster-size 2 --limit 5`. `--C` is required for the default method. `--ray-method {residual-cluster,pca-cross}` defaults to `residual-cluster`. In residual-cluster mode, `--no-center`, `--vocab`, `--ray-cluster-gap`, and `--ray-min-set-size` are rejected with exit code 2; `--C`, `--ray-cluster-cosine`, and `--ray-min-cluster-size` are rejected with `pca-cross` (exit code 2). The old PCA cross-product ray is available only with `--ray-method pca-cross`; it retains its former centred PCA, A×B orientation, axial-gap set grouping, and exclusion behaviour. Calls to ray mode without C now fail; add `--ray-method pca-cross` to reproduce the former behaviour.

A, B, C and every corpus candidate must be embedded in one shared coordinate space: all are encoded with the single `encoder.encode_passage` path, using A, B and C as input text and each candidate as its formatted corpus-row text (for example, `aとb。line` for pair rows). With role-prefixed encoders, any known leading `query: ` or `passage: ` marker is removed before encoding, then the encoder applies the same passage prefix to every vector; prefix-free encoders add no role prefix. Original corpus text is retained in hit provenance. `encode_query` is never used in this mode. Use one encoder, one encoding path, and one prefix policy for all vectors. This mode intentionally uses only `encode_passage` for A, B, C and all candidates. Let `a`, `b` and `c` be the resulting raw coordinates. The affine source plane is `S = c + span(a-c, b-c)` and the ray origin is `c`; vectors are not centred. For each corpus passage vector `x`, set `V = span(a-c, b-c)` and `r(x) = (x-c) - P_V(x-c)`, which is orthogonal to `V`. No explicit centring is needed because the residual is translation invariant. A residual with norm at most `min_residual_norm` (default `1e-9`) is treated as in-plane: its axial distance is set exactly to `0.0`, its radial distance is therefore `||x-c||`, and it is unusable for clustering but remains eligible for retrieval geometry. This threshold defines in-plane membership, so even `0 < ||r|| <= min_residual_norm` counts as in-plane; a start distance below that scale (for example, `1e-10`) can therefore differ from the exact mathematical `t`. For example, exact duplicate embeddings of A, B or C are reported with `--ray-start-distance 0` when their radial distance is at most `--ray-radius`, and never with the default start distance `0.25`.

Usable candidates are clustered in original corpus-index order with deterministic leader clustering. The lowest-index unassigned candidate seeds a cluster; each later unassigned candidate joins when its residual cosine to the seed direction is at least `--ray-cluster-cosine` (default `0.7`, range `(0,1]`), with a floating-point tolerance of `1e-12`; thus `--ray-cluster-cosine 1` groups numerically parallel residuals. Membership is intentionally order-dependent but reproducible. Thresholds are uncalibrated across encoders. Clusters smaller than `--ray-min-cluster-size` (default `2`) produce no ray. A ray direction is the normalised arithmetic mean of member raw residuals. Output records `mean_residual`, `resultant_norm`, and `coherence = ||sum r|| / sum ||r||`. A zero resultant produces no direction or ray and increments `clustering.zero_resultant_cluster_count`.

For every non-excluded passage, clustered or not, retrieval uses `u = x-c`, `t = u·n`, and full-space `radial = ||u - t n||`. A hit is reported when `t >= --ray-start-distance` (default `0.25`) and `radial <= --ray-radius` (default `0.25`), inclusively. Forward and radial tests compare floating-point values exactly at their inclusive boundaries, so a row whose true `t` or radial distance equals a threshold may fall on either side because of rounding. Near-zero residuals are treated as in-plane when their norm is at most `min_residual_norm`: their axial distance is set exactly to `0.0` and radial distance is `||x-c||`. Thus exact duplicate embeddings of A, B or C are reported with `--ray-start-distance 0` when radial distance is at most `--ray-radius`, and never with the default start distance `0.25`. This threshold defines in-plane membership, so even `0 < ||r|| <= min_residual_norm` counts as in-plane; a start distance below that scale (for example, `1e-10`) can differ from the exact mathematical `t`. The absolute `min_residual_norm` threshold assumes roughly unit-scale embeddings, as with the bundled hash and normalised encoders; at very large coordinate scales, rounding in an in-plane row's residual can exceed it, so the in-plane rule may not apply. The rule for text identical to A, B or C also assumes the encoder returns bit-identical vectors for identical text; this has not been verified for semantic encoders such as E5. Hits are ordered by radial distance and then corpus index; the same passage may occur under multiple rays. `in_cluster` marks whether a hit helped define that ray. Cluster members need not be near the ray in full-space distance: their component within plane S contributes to `radial`, and clustering does not constrain it. `--limit` keeps the rays with the largest `member_count`, breaking ties by lowest seed index; `truncated_ray_count` reports omitted rays.

Residual-cluster has no content or similarity filters. Only rows matching explicit `--exclude` terms are removed; matching uses normalised substring matching on row values. `clustering.excluded_count` counts only these removals. Lexically or semantically close passages, including C's own text if it is in the corpus, can be clustered or returned when geometry permits. For `s1 >= s2`, the singular values of `[a-c; b-c]`, the plane is unavailable when the embedding dimension is below 3 (`embedding_dimension_below_three`), when `s1 <= 1e-12` (`source_anchors_coincide`), or when `s2 <= 1e-6 * s1` (`plane_rank_deficient`). Non-finite or non-representable anchors raise a validation error (exit 2) rather than a plane status; a near-zero residual is excluded from clustering only, not retrieval. Meaning, similarity, ranking, and relevance are never scored. Unrelated passages are valid geometric hits; no items are generated, and no hit means an empty result.

Statuses are `hits_found`, `no_hits` (rays exist but no passage lies in a tube), `no_residual_clusters` (no corpus rows remain after `--exclude`, no usable residuals, no cluster reaches `--ray-min-cluster-size`, or only zero-resultant clusters remain; the last case is possible only when `--ray-cluster-cosine` is about `1e-12` or smaller, given the `1e-12` tolerance), and `plane_unavailable`, with reason `embedding_dimension_below_three`, `source_anchors_coincide`, or `plane_rank_deficient`. Output includes `mode: "normal-ray"`, `method: "affine_residual_cluster"`, `input{A,B,C}`, `params`, `plane{origin,basis,singular_values}`, `clustering` counts and `unusable_corpus_indices`, `rays[{ray_id, seed_corpus_index, member_corpus_indices, member_count, direction, mean_residual, resultant_norm, coherence, hit_count, hits[{corpus_index,a,b,line,axial_distance,radial_distance,in_cluster}]}]`, `warnings`, and `disclaimer`. Corpus indices refer to original row order.

Only explicit `--exclude` terms remove passages in the default method. The output records `params.encoding_path = "encode_passage"`. Distances are not calibrated across encoders. The ray direction is estimated from the supplied corpus distribution; it is not an A×B cross-product direction or a semantic fact. This method has not been evaluated for usefulness.

## random-orthogonal (experimental)

Use this mode to explore reproducible random directions in the orthogonal complement of the affine subspace defined by repeated context inputs. It returns existing corpus rows with provenance; it does not generate retrieved rows or judge semantic relevance or quality.

```bash
python scripts/third_concept_cli.py --search-mode random-orthogonal --context 'rain' --context 'clock' --corpus references/example-corpus.json --seed 7 --ray-count 8 --serendipity 60
```

Repeat `--context` at least twice. The first context embedding is the origin. Encode all context and corpus inputs through `encode_passage`, then use SVD to determine the rank of the affine span. Draw each direction from a seeded Gaussian vector and project it into the orthogonal complement. The mode-specific options are `--context`, `--seed`, `--ray-count`, `--serendipity`, and `--law`; do not mix options from the legacy `ray` or other search modes. Legacy modes retain their existing arguments and behaviour.

`--serendipity` is an integer in the inclusive range 0–100 and defaults to 60. Its exact tone bands are 0–33 `grounded` (a modest leap), 34–66 `playful` (a strange law), and 67–100 `absurd` (a thoroughly nonsensical law). Let `s = serendipity / 100`; unless explicitly overridden by a caller, use radius `0.25 * (0.5 + s)` and cosine threshold `0.85 - 0.35 * s`. The knob controls geometric exploration, not an estimate of surprise or usefulness.

`--ray-cluster-cosine` explicitly overrides the serendipity-mapped cosine threshold in this mode. The output records both `params.mapped_cosine_threshold` and the effective value in `params.cosine_threshold`; `params.cosine_threshold_source` identifies the winner as `serendipity_mapping` or `explicit_override`. For example, add `--ray-cluster-cosine 0.5` to use 0.5 while retaining the mapped value for comparison.

Return deterministic geometry clusters in `selected_clusters` and `combined_clusters`, with real corpus rows and their source provenance. Report the forward ray parameter `t`, origin-relative radial distance, and `ray_distance` separately. Rows behind the origin carry `behind_origin`; do not describe them as forward hits. When the rank is valid and at least one non-excluded candidate row exists but no row enters any ray tube, you must return the nearest real row as an explicitly labelled singleton fallback; it is not a tube hit. Return an honest unavailable state when the corpus is empty, every candidate is excluded, or an invalid or unavailable rank prevents forming a valid orthogonal complement.

Without `--law`, preserve the JSON-serialisable `awaiting` state and include the exact instruction `この世界が成立する奇妙な法則を一つ作れ` in the host prompt. The active host model creates the law and the caller repeats the same arguments with `--law`; the Python script makes no LLM/API call. Label any fictional host-generated law example as fictional and never describe it as retrieved content or an actual run.

In this awaiting state, `host_prompt` is a JSON object, not only a prompt string. It exposes `contexts`, `selected_clusters`, `tone`, `tone_instruction`, and `guardrails` as structured fields alongside the `prompt` string; `required_directive` records the required host instruction. The prompt string is supplied for hosts that consume a single text prompt, while the structured fields remain available to hosts that accept typed context.

Document errors for invalid or insufficient context, an unavailable orthogonal complement, and empty/no-hit geometry distinctly. Do not infer semantic relevance, retrieval quality, creativity, or superiority from these geometric results.
