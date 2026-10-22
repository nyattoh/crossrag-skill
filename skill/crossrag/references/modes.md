# CrossRAG modes

## Default reporting policy (Raw mathematical results only)
- Input: 2+ context concepts plus conversation data (`--conversation`) or an explicit corpus (`--corpus`).
- Retrieve using one of the three single-metric modes (`similar`, `orthogonal`, `wedge`).
- Default output: structured JSON or plain table showing `context`, `corpus_source`, `encoder`, `mode`, `ranking_metric`, `selected` rows (`index`, `a`, `b`, `line`, `rank_value`, `ranking_metric`, `cosine`, `orthogonal`, `wedge`), `warnings`, and `note`.
- Do NOT add interpretation, psychological reading, metaphorical commentary, or creative writing unless the user explicitly asks for it.

## Single-metric retrieval modes (`run.py` / `engine.py`)
- **similar** (`ranking_metric: "cosine"`): ranks strictly by cosine similarity to the averaged context anchor vector.
- **orthogonal** (`ranking_metric: "orthogonal_residual"`): ranks strictly by projection residual norm from the span of context anchors.
- **wedge** (`ranking_metric: "gram_bivector_cos"`): ranks strictly by oriented bivector cosine between the first two context terms and the candidate pair.

No composite `score`, cosine-band target, novelty weight, or coherence weight is used. The legacy `random` mode is removed.

## LLM knowledge pool draw

`--llm-pool <path>` is separate from geometric retrieval. It reads a JSON array of non-empty strings, removes normalised duplicates, requires at least 10 unique concepts, and samples exactly `--limit` concepts uniformly without replacement. `--limit` cannot exceed the pool size.

Generate a new pool for every new request, varying its fields and concepts. Reuse a saved pool only to reproduce a previous draw. Generate the pool in a fresh host request without parent conversation or context terms. The CLI does not generate concepts or call an LLM. A host that cannot guarantee this isolation must report that limitation. Do not claim RAG was used unless the host confirms retrieval.

The output reports `mode: "llm-pool"`, `corpus_source: "llm_knowledge"`, `knowledge_basis: "host_llm_general_knowledge"`, `ranking_metric: "seeded_uniform_draw"`, `encoder: "none"`, the effective seed, optional `generation_model` metadata, `selected` concepts with one-based `rank_value`, and `total` unique concepts. Each selected `index` is zero-based in the deduplicated list. A missing seed uses a fresh 53-bit random seed, which is exactly representable in JSON consumers that use IEEE-754 numbers. A supplied seed reproduces the same draw from the same pool and NumPy version. Do not derive the seed from context. The context does not affect the draw, and no hash or semantic encoder is loaded.

Do not compare this draw's `rank_value` with geometric metrics. It records sample order only. After the draw, combine concepts with context in a separate host request only if the user asks. Existing geometric `--law` processing remains a separate workflow.

## Corpus source requirements

`run.py` retrieval requires exactly one of `--conversation`, `--corpus`, `--demo-corpus`, or `--llm-pool`. `third_concept_cli.py` and `compare_third_concept.py` retrieval require exactly one of `--corpus`, `--demo-corpus`, or `--llm-pool`. The demo corpus is an explicit opt-in and returns `corpus_source: "demo"`. Prompt-only `third_concept_cli.py --emit-prompt` does not retrieve a corpus.

## Optional downstream tasks (only when explicitly requested)
When the user explicitly asks for creative writing or hypothesis generation on top of the retrieved results:
- Keep raw retrieval numbers and corpus provenance separate from host-generated prose or hypotheses.
- Do not treat embedding similarities, residuals, or wedge orientation as factual evidence or proof of causality.
- For method comparisons, maintain equal corpus, encoder, prompt, model, and sampling controls across retrieval modes.

## Random-orthogonal (experimental)

This mode explores reproducible random rays orthogonal to the affine span of repeated context inputs. Repeat `--context` at least twice; the first context embedding is the origin. Encode context and corpus rows with the same `encode_passage` path. Determine the affine-span rank with SVD, draw seeded Gaussian vectors, and project them into its orthogonal complement. Do not use `encode_query` in this mode.

The mode-specific CLI options are `--context`, `--seed`, `--ray-count`, `--serendipity`, and `--law`. Do not mix options from legacy modes; legacy mode options and behaviour remain unchanged. `--serendipity` is an integer from 0 to 100 and defaults to 60. Its exact tone bands are:

| Integer range | Tone | Description |
| --- | --- | --- |
| 0–33 | `grounded` | A modest leap |
| 34–66 | `playful` | A strange law |
| 67–100 | `absurd` | A thoroughly nonsensical law |

Let `s = serendipity / 100`; map it to radius `0.25 * (0.5 + s)` and cosine threshold `0.85 - 0.35 * s`, unless explicitly overridden. This is a geometry control, not a score of semantic surprise or usefulness.

In this mode, `--ray-cluster-cosine` explicitly overrides the mapped cosine threshold. The JSON records the mapped threshold as `params.mapped_cosine_threshold`, the effective threshold as `params.cosine_threshold`, and the winning source as `params.cosine_threshold_source` (`serendipity_mapping` or `explicit_override`). For example, `--ray-cluster-cosine 0.5` makes 0.5 effective while retaining the mapped value in output.

Return actual corpus rows with source provenance. Deterministic clusters are represented by `selected_clusters` and `combined_clusters`. For origin `o`, candidate vector `x`, and unit ray direction `n`, let `u = x - o` and `t = u · n`. `radial_distance = ||u - t n||` is the distance to the infinite axis, while `ray_distance = ||u - max(t, 0)n||` is the distance to the forward half-line from the origin. For a candidate behind the origin (`t < 0`), `ray_distance` includes the offset back to the ray endpoint and therefore need not equal `radial_distance`. Mark rows behind the origin with `behind_origin`. When the rank is valid and at least one non-excluded candidate row exists but no row enters any ray tube, you must return the nearest real row as an explicitly labelled singleton fallback; it is not a tube hit. Return an honest unavailable state when the corpus is empty, every candidate is excluded, or an invalid or unavailable rank prevents forming a valid orthogonal complement.

Without `--law`, return a JSON-serialisable `awaiting` state and a host prompt containing the exact instruction `この世界が成立する奇妙な法則を一つ作れ`. The host model generates the law, then the same search is repeated with `--law`; the script does not make an LLM/API call. Label any illustrative host-generated law as fictional and separate it from retrieved rows and actual runs.

During the awaiting state, `host_prompt` is a JSON object. It exposes `contexts`, `selected_clusters`, `tone`, `tone_instruction`, and `guardrails` as structured fields as well as the text `prompt`; `required_directive` carries the required host instruction. This lets a host consume either the complete prompt string or the individual structured values without changing the generation instruction.

Describe errors for insufficient or invalid context, unavailable orthogonal complement, and empty/no-hit geometry distinctly. These results are geometric candidates only: do not claim semantic relevance, quality, or improved creativity.
