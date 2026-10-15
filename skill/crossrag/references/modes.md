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
Maintain equal corpus, encoder, prompt, model and sampling controls across retrieval methods. Blind-rate coherence, surprise, specificity and usefulness separately. Include a no-retrieval baseline when possible.
