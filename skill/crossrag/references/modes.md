# CrossRAG modes

## Creative
- Input: 2+ concepts plus optional genre, length, tone.
- Retrieve with wedge and similar baseline using identical input.
- Output: complete original story/poem/copy, followed (when useful) by a short note explaining which unexpected concept bridge was chosen; no raw scores unless requested.
- Avoid a sequence of unconnected bizarre details. Prefer a plausible scene and one unconventional transformation.

## Discovery
- Input: two or more domains, problem statement, optional constraints.
- Retrieve and identify at least three possible conceptual bridges.
- Output for each idea: proposition, why it may work, what evidence is missing, falsifying or discriminating test. Clearly mark speculative ideas.
- Do not treat top-ranked embedding similarities or wedge orientation as evidence of real-world causality.

## Shared evaluation
For legacy creative and discovery comparisons, maintain equal corpus, encoder, prompt, model and sampling controls across retrieval methods. Blind-rate coherence, surprise, specificity and usefulness separately, and include a no-retrieval baseline when possible. This guidance does not apply to `random-orthogonal`: acceptance for that mode is based on tests of mathematical behaviour. `--serendipity` (デタラメ度) expresses creative intent and tone; it is not a quality or retrieval metric.

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
