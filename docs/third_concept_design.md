# third-concept モード 設計（PoC）

状態: 設計確定（PM）。実装は子タスク。数値はすべて探索用ヒューリスティックであり、科学的正しさを示さない。

> 注意: 評価用の既知事例 C はこの文書に書かない（仮説生成役がこの文書を読むため）。既知事例はテスト内で符号化して保持する。

## 1. 目的と前提

入力 A・B から、A 単独・B 単独では説明しにくい第三概念 C の候補を出す。

- 楔積 a∧b は「A と B が張る向き付き平面」を表すだけで、C を一意に生成しない（高次元では直交補空間が多次元）。
- したがって数学部分は **C の生成ではなく、与えられた候補集合上での位置づけ（選別）** を行う。新しい概念名を作るのは LLM（外部ファイル）だけ。
- 既存 4 方式（similar / orthogonal / wedge / random）と `engine.run()`・`run.py` は **変更しない**。

## 2. 役割分担

| 区分 (`origin`) | 誰が作るか | 主張できること |
|---|---|---|
| `retrieved` | コーパス検索 | 「指定コーパスにこの記述が存在する」ことだけ。記述内容の真偽は未確認 |
| `math` | 候補語彙プール上の幾何スコア | 「埋め込み空間でこの位置にある」ことだけ。意味的妥当性は未検証 |
| `llm_hypothesis` | 外部 LLM（仮説ファイル） | LLM が説明付きで推論した仮説。未検証 |
| `idea` | 外部 LLM（仮説ファイル） | 説明の連鎖が弱い着想。未検証 |

LLM ステップ（説明文の生成・自己評価）は本体に組み込まず、`--emit-prompt` でプロンプトを出力し、外部で作った仮説 JSON を `--hypotheses` で読み込む。仮説ファイルなしでもオフラインで完結する。

## 3. 数学部分

### 3.1 エンコーダと接頭辞

`ConceptEncoder`（scripts/third_concept.py）が `encode_query(texts)` / `encode_passage(texts)` / `encode(texts)` を持つ。

| 対象 | e5 系モデル（model_id に `e5` を含む） | HashEncoder |
|---|---|---|
| A, B, 候補概念語（短い概念同士の対称比較） | `"query: "` | なし |
| コーパス項目 `f'{a}と{b}。{line}'` | `"passage: "` | なし |
| `encode()`（engine.run() 互換の入口） | `"query: "`（run() は query/passage を区別しないため） | なし |

- 既に接頭辞が付いた文字列に二重付与しない。
- 出力 `encoder` に `name`, `model_id`, `prefix`（`{"query":"query: ","passage":"passage: "}` または `null`）, `applied_to` を記録する。
- 意味埋め込みは `SentenceEncoder` の内部モデル（`.model`）を再利用し、テストではモデルを差し替え可能にする（入力記録用のフェイク）。
- ネットワークを使わない: 実行時は `HF_HUB_OFFLINE=1` を推奨。

### 3.2 中心化

e5 の埋め込みはコサインが全体に高く偏る（異方性）。幾何量は、A・B・候補全体の平均 μ を引いて正規化した x̃ = normalise(x − μ) で計算する（`center=True` 既定、params に記録）。

### 3.3 幾何量（候補 c ごと）

- ρ = ã·b̃。平面 span(a,b) の正規直交基底 e1 = ã, e2 = normalise(b̃ − ρã)。‖b̃ − ρã‖ < 1e-6 なら平面が退化 → warnings に記録し emergence = 0。
- s_A = c̃·ã, s_B = c̃·b̃（A 単独・B 単独との関係）
- p = √((c̃·e1)² + (c̃·e2)²) … a∧b が表す平面への射影長（**外積の寄与その1: 平面**）
- r = √(max(0, 1 − p²)) … 平面からの残差 = ‖a∧b∧c‖ / (‖a∧b‖‖c‖)（**外積の寄与その2: 三重積の体積**）
- emergence = max(0, p² − max(s_A², s_B²))（s_A > 0 かつ s_B > 0 のときのみ、他は 0）
  - 平面との関係のうち、A 軸・B 軸のどちらか単独では説明できない分。
  - 上界は (1 − ρ)/2（二等分方向かつ p = 1）。emergence_norm = clip(emergence / ((1 − ρ)/2), 0, 1)。
- balance = 2·p·r ∈ [0,1] … 半分は AB 平面で説明でき、半分は新しい方向（p = r = 1/√2 で最大）
- unexpectedness_h = clip(1 − max(s_A, s_B, 0), 0, 1)
- coherence_h = p（平面との関係の強さ。説明の整合性そのものではない）
- **score_third = 0.5·emergence_norm + 0.5·balance**

### 3.4 比較対象（単純ベースラインとアブレーション）

- `mean`: s_m = c̃·m̃（m = normalise(ã + b̃)）で降順。単純ベクトル平均検索。
- `third-concept-no-wedge`: score = 2·s_m·√(1 − s_m²)。平面・emergence を使わず、平均方向の 1 次元だけで同じ帯域を取る。**third-concept との差が「外積（平面・三重積）情報の寄与」の検証対象**。

### 3.5 候補集合と除外

- `math` 候補の語彙プール = コーパスの a・b 語（重複除去・出現順）＋任意の `--vocab`（JSON 文字列配列）。
- `retrieved` 候補 = コーパス項目（passage 表現）を同じ score_third で順位付け。
- 除外: NFKC・小文字化・空白除去後に A/B と一致、または一方が他方を含む語。生埋め込みのコサイン ≥ 0.98 も除外。除外数を params に記録。
- 決定的: 同点は元の順序で解決。`seed` は random ベースラインと評価シートのシャッフルにのみ使う（params に記録）。

## 4. 出力 JSON（`third_concept_cli.py`）

```json
{
  "mode": "third-concept",
  "input": {"A": "...", "B": "..."},
  "encoder": {"name": "...", "model_id": "...", "prefix": {"query": "query: ", "passage": "passage: "}, "applied_to": {"query": ["A", "B", "concept candidates"], "passage": ["corpus entries"]}},
  "corpus_id": "sha256 先頭16桁",
  "corpus_size": 51,
  "params": {"limit": 5, "seed": 0, "center": true, "weights": {"emergence_norm": 0.5, "balance": 0.5}, "excluded_inputs": 0, "vocab_size": 0, "hypotheses_sha256": null},
  "origin_legend": {"retrieved": "...", "math": "...", "llm_hypothesis": "...", "idea": "..."},
  "candidates": [
    {
      "id": "math-1",
      "concept": "...",
      "origin": "math",
      "generation_method": "third-concept score over vocabulary pool (centered e5 query embeddings)",
      "relation_to_A_alone": "s_A=0.12（中心化コサイン）",
      "relation_to_B_alone": "s_B=0.08",
      "why_requires_AB": "p²=0.30 が単独最大 0.014 を上回る（emergence_norm=0.41）。意味的必然性は未検証",
      "uncertainty": "...",
      "falsification_or_check": "...",
      "scores": {"heuristic_math": {"s_A": 0.0, "s_B": 0.0, "plane_projection": 0.0, "residual": 0.0, "emergence_norm": 0.0, "balance": 0.0, "unexpectedness_h": 0.0, "coherence_h": 0.0, "score_third": 0.0}, "llm_self_assessment": null},
      "source": {}
    }
  ],
  "warnings": [],
  "disclaimer": "すべての候補は未検証。retrieved はコーパス内の存在のみ、math は埋め込み空間上の位置のみ、llm_hypothesis / idea は LLM の推論であり、いずれも事実・発見として扱わない。"
}
```

- `corpus_id`: ファイル指定時はファイルバイト列の sha256、組み込みコーパスは `json.dumps(rows, ensure_ascii=False)` の UTF-8 の sha256。先頭 16 桁。
- `retrieved` の `source` は `{corpus_index, a, b, line}`、`llm_hypothesis`/`idea` の `source` は `{hypotheses_file, generator}`。
- テスト容易性（Testability）は数学では評価できないため `heuristic_math` に含めない。`llm_self_assessment` に LLM の自己評価（1–5）がある場合のみ入る（独立評価ではないと legend に明記）。
- 禁止: `verified` / `proven` / `confirmed` / `証明済` / `検証済` を含むキーや肯定的な値。

### CLI

```
python scripts/third_concept_cli.py --A ヨーグルト --B 梅シロップ [--corpus path] [--encoder hash|semantic] [--model intfloat/multilingual-e5-small] [--limit 5] [--seed 0] [--vocab path] [--hypotheses path] [--no-center]
python scripts/third_concept_cli.py --A ... --B ... --emit-prompt [--with-math]
```

stdout は UTF-8 固定（run.py と同じ）。

### 仮説ファイル（外部 LLM が作成、入力）

```json
{
  "input": {"A": "...", "B": "..."},
  "generator": {"provider": "...", "model": "...", "effort": "...", "prompt_sha256": "...", "date": "YYYY-MM-DD"},
  "hypotheses": [
    {"concept": "...", "kind": "llm_hypothesis" | "idea",
     "explanation": "...", "relation_to_A_alone": "...", "relation_to_B_alone": "...",
     "why_requires_AB": "...", "uncertainty": "...", "falsification_or_check": "...",
     "llm_self_assessment": {"unexpectedness": 1-5, "coherence": 1-5, "emergence": 1-5, "testability": 1-5}}
  ]
}
```

- `input` が CLI の A・B と一致しなければエラー。`concept` が A/B と一致（3.5 の除外規則）するものは除外し warnings に記録。
- 各仮説の `concept` も同じエンコーダで埋め込み、`heuristic_math` を付ける（LLM の説明の正しさとは無関係）。

### 仮説生成プロンプト（`--emit-prompt` が出す固定テンプレート）

```
あなたは発想支援のための仮説生成役です。概念A「{A}」と概念B「{B}」の組み合わせから生じうる、予想外の第三概念Cの候補を5〜8個挙げてください。
条件:
- AまたはB単独からは容易に連想できず、AとBの組み合わせに依存するものを優先する。
- 一般論・既存研究の要約・並列説明・助言の列挙は避け、具体的な現象・製品・仕組み・実験方法の形にする。
- 各候補は検証されていない仮説として書く。確立した事実のように書かない。
- 各候補について concept, kind(llm_hypothesis|idea), explanation, relation_to_A_alone, relation_to_B_alone, why_requires_AB, uncertainty, falsification_or_check, llm_self_assessment(unexpectedness, coherence, emergence, testability を1〜5) を与える。
- 出力は JSON のみ: {"hypotheses": [...]}
```

`--with-math` のときだけ末尾に「参考: 埋め込み空間上の候補語（意味的妥当性は未検証）: …」を追記する。

## 5. 比較スクリプト（`compare_third_concept.py`）

- 同じ A・B・コーパス・エンコーダオブジェクト・limit で、`similar / orthogonal / wedge / random`（`engine.run()` をそのまま呼ぶ）と `third-concept`（retrieved 項目）を出す。追加で `mean` と `third-concept-no-wedge` も出す。
- `engine.run()` は context を空白でも分割するため、A・B 内の空白は全方式で `_` に置換し、`input_normalization` に記録する（例: `H2O MEMORY` → `H2O_MEMORY`）。
- 比較指標は全方式の項目を **共通表現**（A,B: query、項目: passage、中心化）で再計算: s_A, s_B, plane_projection, residual, emergence_norm, score_third, 方式間の Jaccard 重なり。
- third-concept は自分の目的関数で選ぶので、score_third 等の平均で勝つのは循環的。表には載せるが優劣の根拠にしないと明記。
- 出力: JSON と Markdown 表。`--sheet-dir` 指定時、方式名を伏せてシャッフルした人手評価シート（Markdown、4 観点 1–5 の空欄）と対応キー（JSON）を書く。

## 6. ファイル所有と成果物の置き場所

| ファイル | 所有 |
|---|---|
| scripts/third_concept.py, scripts/third_concept_cli.py, tests/test_third_concept.py, tests/fixtures/*, docs/third_concept_tdd_log.md | T1 |
| docs/third_concept_hypotheses/*.json, docs/third_concept_hypotheses/PROMPT.md | T2 |
| scripts/compare_third_concept.py, tests/test_compare_third_concept.py, docs/third_concept_report.md, SKILL.md（追記のみ）, docs/third_concept_tdd_log.md（T1 完了後に追記） | T3 |

- 実コーパスを使った実行結果・評価シートは、コーパス本文を含むためリポジトリに入れず `D:\inbox\crossrag-third-concept\` に置き、レポートには要約表・パス・sha256 のみ載せる。
- engine.py / run.py / 既存テスト / .vscode/ / skill.zip は変更しない。

## 7. 既知の限界（設計時点）

- 数学部分は候補プールの外にある概念を出せない。実コーパス（工学・記録系 51 件）では食品系の A・B に対して語彙が合わず、math 候補は的外れになりうる。それはそのまま報告する。
- emergence は「平面上で二等分方向に近い」ことを測るため、平均ベクトル検索と相関しうる。差はアブレーションで測る。
- 4 観点の数値は代理指標であり、Coherence と Testability の本当の評価は人手（ブラインド）が必要。PoC では評価シート生成までで、評価は未実施。
