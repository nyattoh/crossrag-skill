# Geometry and methodological boundaries

The baseline `engine.py` is copied from CrossRAG Lab v0.1. For each corpus item, its candidate full sentence is embedded; its concept pair is embedded separately.

- **similar**: cosine between averaged context anchors and candidate text.
- **orthogonal**: projection residual from the span of context anchors, blended with cosine-band and coherence weights.
- **wedge**: cosine of oriented bivectors `a∧b` and `u∧v` computed without materialising d-by-d arrays: `((a·u)(b·v)-(a·v)(b·u)) / (||a∧b|| ||u∧v||)` for unit vectors. The *first two* context terms determine orientation. Swapping term order flips the sign; this does NOT entail a natural linguistic direction.
- **random**: deterministic hash-based ranking used as a control.

These ranking formulae are **heuristics** with no established relation to story quality. HashEncoder is a reproducible character n-gram representation, not a semantic model. For actual semantic tests use a compatible embedding model (e.g. multilingual E5) and observe its query/passage conventions: the copied v0.1 prototype encodes strings symmetrically without such prefixes, so production-quality retrieval is not established. Model downloads are optional.

A wedge compares oriented planes; it does not itself decode text or discover a hidden concept. The LLM consumes the retrieved corpus motifs and writes the final text. Factual questions require separate evidence sources; use this tool to propose ideas, not certify them.

Source project: local CrossRAG Lab v0.1. See its original README for additional method notes. Background: https://arxiv.org/abs/1603.09320 and https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
