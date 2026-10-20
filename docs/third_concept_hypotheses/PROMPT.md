# Prompts and hashes

Generator: claudeAgent / claude-sonnet-5-5 / effort medium / 2026-10-18

## yogurt_ume.json

sha256: `73023769a7c14c8f279f05383e33d5f31508dfeae4723caa85c3fc358e7f0bc6`

```
あなたは発想支援のための仮説生成役です。概念A「ヨーグルト」と概念B「梅シロップ」の組み合わせから生じうる、予想外の第三概念Cの候補を5〜8個挙げてください。
条件:
- AまたはB単独からは容易に連想できず、AとBの組み合わせに依存するものを優先する。
- 一般論・既存研究の要約・並列説明・助言の列挙は避け、具体的な現象・製品・仕組み・実験方法の形にする。
- 各候補は検証されていない仮説として書く。確立した事実のように書かない。
- 各候補について concept, kind(llm_hypothesis|idea), explanation, relation_to_A_alone, relation_to_B_alone, why_requires_AB, uncertainty, falsification_or_check, llm_self_assessment(unexpectedness, coherence, emergence, testability を1〜5) を与える。
- 出力は JSON のみ: {"hypotheses": [...]}
```

## insomnia_conditioning.json

sha256: `8e5ce3d74c1b5d48208056e0852865a9c38123265bcd6464579d8dd28038dda3`

```
あなたは発想支援のための仮説生成役です。概念A「入眠障害」と概念B「条件反射」の組み合わせから生じうる、予想外の第三概念Cの候補を5〜8個挙げてください。
条件:
- AまたはB単独からは容易に連想できず、AとBの組み合わせに依存するものを優先する。
- 一般論・既存研究の要約・並列説明・助言の列挙は避け、具体的な現象・製品・仕組み・実験方法の形にする。
- 各候補は検証されていない仮説として書く。確立した事実のように書かない。
- 各候補について concept, kind(llm_hypothesis|idea), explanation, relation_to_A_alone, relation_to_B_alone, why_requires_AB, uncertainty, falsification_or_check, llm_self_assessment(unexpectedness, coherence, emergence, testability を1〜5) を与える。
- 出力は JSON のみ: {"hypotheses": [...]}
```

## h2o_trajexa.json

sha256: `11a07a7aa0d81b2a9692bf5aeec28c9cea4dddc5fe0abc53daf9ae96c9ef6b25`

```
あなたは発想支援のための仮説生成役です。概念A「H2O MEMORY」と概念B「TRAJEXA」の組み合わせから生じうる、予想外の第三概念Cの候補を5〜8個挙げてください。
条件:
- AまたはB単独からは容易に連想できず、AとBの組み合わせに依存するものを優先する。
- 一般論・既存研究の要約・並列説明・助言の列挙は避け、具体的な現象・製品・仕組み・実験方法の形にする。
- 各候補は検証されていない仮説として書く。確立した事実のように書かない。
- 各候補について concept, kind(llm_hypothesis|idea), explanation, relation_to_A_alone, relation_to_B_alone, why_requires_AB, uncertainty, falsification_or_check, llm_self_assessment(unexpectedness, coherence, emergence, testability を1〜5) を与える。
- 出力は JSON のみ: {"hypotheses": [...]}
```

## Files read (pair 3 only)

Search under D:\develop (excluding crossrag-skill) found the names in the TRAJEXA project worktree. Read:
- D:\develop\worktrees\home-offer-layout\docs\existing-methodology.md (lines 60-89)
- D:\develop\worktrees\home-offer-layout\CONTEXT.md (lines 1-40)
- D:\develop\worktrees\home-offer-layout\README.md (lines 1-30)

Grep snippets (not full reads) also surfaced lines from docs/ip-hygiene.md, docs/problem.md and DESIGN.md in the same worktree.
