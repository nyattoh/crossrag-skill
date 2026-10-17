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

e5 の埋め込みはコサインが全体に高く偏る（異方性）。幾何量は、平均 μ を引いて正規化した x̃ = normalise(x − μ) で計算する（`center=True` 既定、params に記録）。

μ は 1 回の探索につき 1 回だけ計算し、全 origin で共有する（R1-M1 を受けて改訂）: μ = mean({A, B} ∪ 語彙プール全体（除外前、query）∪ `--vocab`（query）∪ コーパス項目全体（passage））。LLM 仮説は μ に含めず、同じ μ で射影だけする（仮説の件数や内容が他候補のスコアを変えないため）。比較スクリプトも同じ関数で同じ μ を使う。

### 3.3 幾何量（候補 c ごと）

- ρ = ã·b̃。平面 span(a,b) の正規直交基底 e1 = ã, e2 = normalise(b̃ − ρã)。‖b̃ − ρã‖ < 1e-6 なら平面が退化 → warnings に記録し emergence = 0。
- s_A = c̃·ã, s_B = c̃·b̃（A 単独・B 単独との関係）
- p = √((c̃·e1)² + (c̃·e2)²) … a∧b が表す平面への射影長（**外積の寄与その1: 平面**）
- r = √(max(0, 1 − p²)) … 平面からの残差 = ‖a∧b∧c‖ / (‖a∧b‖‖c‖)（**外積の寄与その2: 三重積の体積**）
- emergence = max(0, p² − max(s_A², s_B²))（s_A > 0 かつ s_B > 0 のときのみ、他は 0）
  - 平面との関係のうち、A 軸・B 軸のどちらか単独では説明できない分。
  - 上界は (1 − ρ)/2（二等分方向かつ p = 1）。emergence_norm = clip(emergence / ((1 − ρ)/2), 0, 1)。
- balance = 2·p·r ∈ [0,1] … 半分は AB 平面で説明でき、半分は新しい方向（p = r = 1/√2 で最大）
  - 平面への射影が A+B の側を向く（s_mean > 0）ときだけ数え、それ以外は 0。A・B の両方と逆向きの候補が上位に来るのを防ぐ。
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

## 8. v3: 外積項の比率化と文入力（設計確定・PM）

### 8.1 動機（事実）

実データ（58 節、e5-small、A・B は単語、4 組）で emergence の最大値は 0.002、上位 5 件は 4 組すべてで balance 単独の順位と一致した。類似度が ±0.1 程度だと emergence ≈ 類似度² となり、類似度にほぼ比例する balance に埋もれる。

数学的事実: ‖a∧b∧c‖ = ‖a∧b‖·r（r は平面からの残差）。候補を 1 つずつ評価する限り、3 重外積の大きさは r と同じ情報しか持たない。外積に固有なのは **平面内での向き**（射影 q が A の線・B の線のどちらにも寄らず、その間にあるか）である。v3 はこの向きを大きさから切り離して測る。

### 8.2 新しい量（`geometry()` に追加するキー。既存キーの意味・値は変えない）

記号は 3.3 と同じ（ã, b̃, c̃ は中心化単位ベクトル、ρ = ã·b̃、p = 平面への射影長）。非退化のとき、射影は q⃗ = α ã + β b̃ と一意に書ける:

- α = (s_A − ρ s_B)/(1 − ρ²), β = (s_B − ρ s_A)/(1 − ρ²)
- `between_AB` = 非退化 ∧ α ≥ −1e-12 ∧ β ≥ −1e-12 ∧ p > 1e-12（射影が A の線と B の線のなす錐の中＝「間」にある。線上を含む）
- `emergence_ratio` = 1 − max(s_A², s_B²)/p²（`between_AB` ∧ s_A > 0 ∧ s_B > 0 のとき。それ以外 0。[0,1] に clip）
  - この条件下で emergence_ratio = emergence / p² = 1 − max(cos²(q,ã), cos²(q,b̃))。q の平面内の角度だけで決まり、c の大きさにも面外成分にも依存しない。
- `emergence_ratio_norm` = clip(emergence_ratio / ((1 − ρ)/2), 0, 1)（上界 ≤ 1e-12 なら 0）
- `p_below_min` = p < p_min（p_min は 8.3）
- `score_ratio` = 0.5·(0 if p_below_min else emergence_ratio_norm) + 0.5·balance
- 方式名 `third-concept-ratio` → `score_ratio`。旧 `third-concept`（`score_third`）はそのまま残し、方式名で区別する。重み 0.5/0.5 は旧式と同じにして、違いを「外積の項」だけにする。

数学的性質（テストで固定）:

1. 角度表現との一致（既知ベクトル）。
2. 平面内の向きを保ったまま c の大きさ・面外成分を変えても emergence_ratio は不変（emergence は p² 倍で変わる）。
3. A の線上で 0。二等分線上で (1 − ρ)/2（norm = 1）。錐の中では二等分線が最大（ρ ∈ (−1, 1) の全域で成り立つ）。
4. **錐ゲートの理由**: s_A > 0 ∧ s_B > 0 だけを条件にすると、ρ > 0.5 のとき錐の外（A の線の外側に回り、B と弱く正）で 1 − max(cos²) が最大 ρ² に達し、二等分線の値 (1 − ρ)/2 を上回る（ρ = 0.8 で 0.64 対 0.10）。これは「A と B の間」ではなく「A から B と反対側へ回った向き」であり、目的に反する。よって錐の外は 0 とする。0 ≤ ρ ≤ 0.5 でも錐の外は最大 ρ² の値を取るため、ゲートは ρ によらず適用する。結果として emergence_ratio = emergence/p² は錐の中でだけ成り立つ（既存 emergence_norm は錐ゲートを持たず、ρ > 0.5 で同じ問題を抱える。既存の意味を変えないため修正しない）。
5. 退化平面・s_A ≤ 0・s_B ≤ 0 では 0。p < p_min では emergence_ratio は報告するが score_ratio には入れない（= 0.5·balance）。

### 8.3 ノイズ対策: 帰無分布に基づく p の下限（採用）と、採らなかった案

p が小さいと q の向きは埋め込みのノイズでほぼ決まる。比率は大きさを捨てるので、ノイズの向きも同じ重みで数えてしまう。

- **採用: p の下限 p_min を帰無分布から決める。** 既定 `auto` = シャッフル帰無分布での p の 95 パーセンタイル。p_min 未満の候補は「平面成分が無関係なベクトルと区別できない」（片側 5% 水準の目安）ので、その向きを score に使わない。閾値の単位がモデル・コーパスごとに自動で合う。`--p-min <float>` で上書きでき、出力に値と出所（`auto:shuffled-p95` / `user`）を記録する。
- 不採用: p で重み付け（p·ratio_norm など）。大きさへの依存を戻すため、v2 で外積項が類似度に埋もれた構造を再現する。外積項の寄与を測る目的に反する。
- 不採用: 固定の絶対閾値（例 p ≥ 0.1）。埋め込みモデル・次元・中心化で p の尺度が変わり、根拠がない。
- 帰無分布は p_min の決定と評価指標（8.5）に使う。パーセンタイルそのものはスコアに入れない。

帰無分布の作り方（`null_geometry()`、seed で再現）:

- `random`: `np.random.default_rng(seed)` で標準正規 (n, d) を引き、行を正規化した方向。
- `shuffled`: 中心化単位化したコーパス passage ベクトルを順に巡回し、各々の座標を `np.random.default_rng(seed + 1)` の置換で並べ替える（各ベクトルの座標値の分布を保ち、A・B との意味的な対応だけを壊す）。テキストの再エンコードはしない。
- どちらも A・B と同じ中心化単位ベクトル ã, b̃ に対して `geometry(ã, b̃, null, mu=None)` で p と emergence_ratio を計算する（null を二重に中心化しない）。
- 既定 n = 1000。n ≥ 1 の整数、コーパスが空なら ValueError。
- パーセンタイルは中央順位: 100·(#(null < x) + 0.5·#(null = x))/n。emergence_ratio の帰無分布は 0 に大きな塊を持つため、≤ 方式だと 0 の候補が高く見える。

既知の限界: random 方向の q の向きは平面内で一様になるため、emergence_ratio の random 帰無分布は p に依存しない固定の形（角度一様）になる。シャッフルも高次元では近い。95 パーセンタイルは候補ごとの多重比較を補正しない目安であり、有意性検定ではない。

### 8.4 文入力と除外

- A・B は単語でも短い文でもよい。third-concept 系（`third-concept`, `third-concept-ratio`, `mean`, `third-concept-no-wedge`）は A・B を `encode_query` に渡すので、e5 では文にも `"query: "` が付く（二重付与なし）。
- `--input-kind sentence`（比較スクリプト）では third-concept 系は A・B を正規化せず原文のまま使う。`engine.run()` の 4 方式は context を区切り文字で分割するため、従来どおり `[、,，/／\s]+ → _` で正規化した文字列を使い、`input_normalization.applies_to` に記録する。`word`（既定）は従来どおり全方式で正規化した文字列を使う。
- 文入力では A の文に含まれる中心語が、部分一致の除外（候補が A に含まれる／A が候補に含まれる）では外れない。（これは本文の長い候補行の話。文に含まれる短いラベル・語彙の扱いは 8.8 を参照）`--exclude TERM`（複数回指定可）で除外語を明示する。
  - 正規化は既存と同じ（NFKC・casefold・空白除去）。空文字・空白だけの語は ValueError。
  - 判定は片方向: 正規化した TERM が候補文字列（行の a・b・line、語彙プールの語、仮説の concept）に **含まれる** なら除外。既存の A・B 双方向判定と生コサイン ≥ 0.98 は残し、論理和で除外する。
  - `exclusion_mask(space, A, B, exclude)` が行ごとの bool を返し、`rank_entries(..., excluded=mask)` は渡された mask をそのまま使う（長さ不一致は ValueError）。
- 単語と文の同条件比較（`--A-sentence` / `--B-sentence` を `--A` / `--B` と併用）:
  - μ: 両方の実行で A, B, A_sentence, B_sentence の 4 つを query テキストに含めて同じ μ にする。
  - 候補集合: 単語側・文側の除外 mask の論理和を両方の third-concept 系に適用し、同じ候補集合で順位を付ける。
  - encoder・コーパス・limit・seed・exclude・null_samples は共通。p_min は `auto` なら各平面で同じ手順で計算し、両方の値を記録する（`--p-min` で共通値に固定できる）。
  - 方式ごとに、単語側と文側の上位 k の Jaccard と、共通候補集合での全順位の Kendall τ_b を出す。

### 8.5 評価指標（`compare_third_concept.py`、出力は JSON）

既存の出力キーと値は変えない（ペアコーパス・単語入力・既定値のとき）。新しい内容は最上位キー `evaluation` に入れる。

- `evaluation.methods["third-concept-ratio"]`: 既存方式と同じ形の selected / means。
- `evaluation.jaccard`: 新方式を含む全方式の上位 k の Jaccard。
- `evaluation.wedge_ablation`: `third-concept` と `third-concept-ratio` それぞれについて、外積の項を外したスコア（= balance）との Kendall τ_b、`third-concept-no-wedge` との τ_b、上位 k の Jaccard、上位 k 列が完全一致か。τ は除外後の全候補で計算し、stdlib で O(n²)、tie は τ_b、分母 0 なら null。
- `evaluation.null`: n、seed、p の 50/95 パーセンタイル（random / shuffled）、p_min、各方式の上位 k の emergence_ratio と p の帰無分布パーセンタイル（平均と候補ごと）、除外後全候補の emergence_ratio の最大・中央値、p_below_min の割合、between_AB の割合。
- 本文だけのコーパス（文字列または `{"line": ...}` のみ）: `engine.run()` はペアラベルが必要なので使えない。`similar`（中心化しない A+B 平均方向とのコサイン）と `random`（`default_rng(seed)` の置換）を共有空間で計算し、`orthogonal` / `wedge` は省いて notes に書く。`evaluation.baseline_source` = `engine.run` / `shared-space`。
- `--aggregate-only`: コーパス本文（a・b・line）を出力に含めない。selected は corpus_index と数値だけになる。JSON 専用（`--format markdown` と併用不可）。入力 A・B の文字列は残る（文入力に個人的な内容を使う場合は注意）。
- 結果が既存手法を上回らなくても、そのまま出力する。third-concept 系の score 平均で勝つことは循環的であり、優越性の証拠にしない（既存 notes を維持）。

### 8.6 何を数学で、何を LLM でやるか

| 区分 | 数学（このリポジトリのコード） | LLM（外部、仮説ファイル経由） |
|---|---|---|
| 候補の生成 | しない。コーパス・語彙プール・仮説ファイルにある候補の順位付けだけ | 新しい概念名・説明・反証方法を作る |
| 「A と B の両方が要る」か | 平面内の向き（emergence_ratio、錐ゲート）と平面成分の大きさ（p, p_min）で測る代理指標 | 意味的な必然性の説明（why_requires_AB）。未検証 |
| ノイズ・偶然 | 帰無分布（random / shuffled）とパーセンタイル | 扱わない |
| 品質（意外性・整合性・検証可能性） | 測らない | 自己評価のみ（独立評価ではない）。本当の評価は人手のブラインド評価 |

### 8.7 v3 の既知の限界

- 比率は p と独立なので、p_min を超えたばかりの候補と大きな p の候補を同じ重みで扱う。p_min の付近の順位は不安定でありうる。
- 錐ゲートは「A と B の正の組み合わせ」を要求する。A と B の差（αã − βb̃）で表される概念は 0 になる。
- 帰無分布は埋め込み空間の形だけを使う。意味の偶然一致（コーパスの話題の偏り）は表さない。
- 文入力では engine.run() の 4 方式が正規化した文を使うため、第三概念系と入力表現が完全には一致しない（記録して報告する）。

### 8.8 実装上の補足（統合時に PM が追記）

- `explore(method="third-concept-ratio")` では仮説候補を score_ratio の降順に並べる。`third-concept`（既定）は従来どおり仮説ファイルの順を保つ。方式によって並び順の扱いが異なる点に注意する。
- explore の `params.weights` は旧式（emergence_norm 0.5 / balance 0.5）の記録のまま。ratio 方式の重みは 8.2 の定義（emergence_ratio_norm 0.5 / balance 0.5、p_min 未満は前者 0）による。
- `rank_entries()` の p_min の既定値は 0（下限なし）。auto 下限は `explore()` と比較スクリプトで計算する。
- 文入力の除外（レビュー round 1 の minor 5 への対応。仕様として明記する）: 既存の双方向判定は文入力でも残す。そのため、ペアラベル・語彙・仮説 concept が A・B の **文に含まれる** 語なら除外される（例: A = 「湯気が窓を曇らせる」なら a = 「窓」の行は除外）。文に含まれる語は A の内容そのものなので、第三概念の候補から外すのを妥当とする。paired 比較では両側の除外 mask の論理和をとるため、この除外は単語側の候補集合にも及ぶ（同じ候補集合で比べるための意図した挙動）。文に含まれない中心語は `--exclude` で明示する。
- `rank_entries(excluded=...)` に mask を渡すと、`exclude` 引数は使われない（mask が優先される）。
- 集計の平均は、対象が 0 件のとき 0.0 になる。実測の 0 と見分けるには `eligible_count` と selected の件数を併せて見る。
- `explore()` は既定の third-concept 方式でも帰無分布を計算する。`heuristic_ratio.p_below_min` を出すためで、そのぶん 1 回あたり数千回の幾何計算が増える。
- 1 文字のラベル（例: 「光」）は多くの文に部分一致するため、文入力では候補がまとめて除外されうる。除外の規模は `evaluation.eligible_count` / `excluded_count` で確認する。

### 8.9 token 方式 `third-concept-token-joint-v2`

`compare_third_concept.py` の control-pair calibration は、各候補文を文脈付き token embedding に変換し、新方式 `third-concept-token-joint-v2` で別に順位付けする。既存の `third-concept` / `third-concept-ratio` の定義や結果は変えない。`sentence_transformers` が `output_value="token_embeddings"` を提供しない encoder では、候補文全体の embedding 1 本へ退避する。この経路は token 単位の評価ではないので、encoder と集約モードを出力で確認する。

候補行 i の token vector をそれぞれ単位化して算術平均し、候補方向 `c_i` とする。A/B は既存の query encoder で得た方向 `a0`, `b0` を単位化する。中心 `mu` は、単位化した `a0`, `b0` と全 eligible 候補の `c_i` の単純平均とし、同じ `mu` を3群から引いて再単位化した `a`, `b`, `c_i` で以下を計算する。候補文内の token 順は平均に影響せず、token を max pooling しない。

- `side_A = max(0, a·c)`, `side_B = max(0, b·c)`。
- `G = [[1, a·b], [a·b, 1]]`, `z = G⁻¹[a·c, b·c]ᵀ` とし、`z` の2成分を A/B 基底で展開した射影を `q` とする。`plane_support = clip(||q||² / ||c||², 0, 1)`。
- `z_A,z_B ≥ -1e-12` かつ `max(z_A,0)+max(z_B,0)>1e-12` のときだけ正の錐内とする。その錐内で `balance = 2·min(max(z_A,0),max(z_B,0)) / (max(z_A,0)+max(z_B,0))`、`between_support = plane_support·(0.5+0.5·balance)`。錐外では 0。
- `token_score = [((1+side_A)/2)·((1+side_B)/2)·between_support·plane_support]^(1/4)`。
- 文内 token は単位化後の平均方向1本へまとめてからこの量を計算する（`mean-vector-geometric-v2`）。複数 token の句を使う場合も同じ集約規則。

直交する単位 A/B と `mu=0` の既知ベクトルでは、`A+B` 二等分方向は `token_score=0.92388`、A 軸のみ・B 軸のみは各 `0.707107`、AB 平面に直交するノイズとゼロベクトルは 0 になる。スコアは正の倍率変更に不変で、A/B を交換しても総スコアは不変。共線アンカーは平面が退化するため 0。これは幾何上の診断量であり、単独ベクトルを明示文脈なしに埋めたときの「意味上の組合せ」を証明しない。文脈付き token vector でも意味的な必然性は測らず、人手評価が必要。

### 8.10 ペア入替較正と帰無分布

- `--control-pair LEFT|RIGHT` は明示対照ペアを複数回指定する。`--control-words <json-file>` は JSON 文字列配列から順不同の全組合せを作り、`--control-samples N --seed S` で非復元抽出する。指定ペアと語リストは同時に指定でき、明示ペアを評価する。
- target と各 control は同じ候補コーパス・encoder・`input_kind`・`exclude`・`p_min`・`null_samples` で実行する。legacy `valid_rate` は除外後候補全体で `between_AB` が真となる割合。legacy top-k の有効率と token 方式の joint-supported 件数・割合、平均スコア、各支持成分、top-k を別々に出す。
- pair percentile は control 値に対する中央順位 `100·(#(control < target)+0.5·#(control=target))/N`。descending rank は `1 + #(control > target)`（同順位は同じ competition rank）。これは指定された語彙・コーパスでの記述的比較であり、p 値でも有意差検定でもない。
- controls は語彙選択に依存し、語を共有する対照同士は独立とは限らない。語の抽象度・頻度の差、コーパスの話題偏り、ペアの順序による方向性も残る。target が controls と交換可能である保証はない。少数の percentile/rank から母集団の有意性を主張しない。多重比較補正は行わない。
- 既存 random/shuffled null と `p_min` は幾何上の目安として残す。埋め込み・話題の偏りを再現するペア入替 null の代わりではなく、有意性判定に使わない。
- `--aggregate-only` は JSON から corpus `a` / `b` / `line` を再帰的に除く。候補は `corpus_index` で参照する。入力 A/B の文字列は出力に残るため、必要なら匿名化した入力を渡す。

### 8.11 モデルと接頭辞 policy

両 CLI の `--model MODEL_ID` は semantic encoder のモデル ID を選ぶ。既定 `--prefix-policy auto` は model ID に `e5` を含む場合 `query: ` / `passage: ` を付け、それ以外は接頭辞なしにする。`--prefix-policy e5` と `--prefix-policy prefix-free` で明示的に上書きできる。既に role prefix の付いたテキストへの二重付与はしない。`auto` の未知 ID は prefix-free とするが、そのモデル固有の指示形式が必要かは判定しないため、利用者がモデル仕様を確認して policy を指定する。HashEncoder には影響しない。

### 8.12 合成評価と境界

テストの fake encoder は、陽性（A+B の語を含む）、A のみ、B のみ、無関係、English/config 風の異質文、A/B の文字列を含まない組合せ例に決めたベクトルを割り当て、旧 `emergence_ratio` と token joint score を同じ既知幾何で比較する。語を含まない組合せ例は、fake encoder 上で陽性と同じベクトルなら同じ順位になる。これは embedding が与えた幾何の帰結であり、意味を発見した証拠ではない。token embedding の実モデルでの挙動、大きいモデルの比較、利用者データの結果は親が別途実行する。
