# Third-concept PoC 評価報告

## 1. 結論

【事実】独立レビュー round 1 の指摘（M1/M2 ほか）を修正した後の v2 出力を `D:\inbox\crossrag-third-concept\v2\` に生成した。対象は実コーパス51件、3ケース、hash/semantic の6条件である。人手評価は未実施であり、方式間の品質差は判断できない。

【数学的結果】共有平均 μ で再計算した全306項目のうち、`emergence_norm=0` は284件、全306件が0.01以下だった。`score_third` と `0.5×balance` の最大差は0.003629で、このコーパスでは emergence の寄与は小さい。

【事実】【数学的結果】third-concept の上位5件集合は、6条件すべてで mean とも similar とも同一ではない。Jaccard は0または0.111111だった。ただし third-concept は自分の目的関数で選ぶため、同じ `score_third` で高くなることは循環的であり、品質上の優位を示さない。

【LLM仮説】外部生成器による17候補はすべて未検証である。共有 μ に変えた結果、仮説候補の数理値も旧出力から変わったが、値は説明の真偽、新規性、安全性、実装可能性を証明しない。

## 2. レビュー指摘と修正

【事実】独立レビュー round 1 の M1/M2 と付随指摘を次のように修正した。以前の `t1`、`t3`、`t4` 出力と、それらから得た数値は本報告の v2 出力により superseded（置換済み）である。

- 【事実】M1: `shared_geometry_space()` を追加し、1探索につき μ を1回だけ計算するようにした。μ は `{A,B}`、除外前のコーパス語彙、`--vocab`、全コーパス passage を含み、仮説候補を含まない。`explore()`、`rank_entries()`、比較の共通指標が同じ関数と μ を使う。同一概念を `--vocab` と仮説で与えたときの `heuristic_math` 一致、仮説追加で既存仮説の値が変わらないことを回帰試験で確認した。
- 【事実】M2: `rank_entries()` の third-concept / mean / third-concept-no-wedge に、正規化文字列の一致・包含と生コサイン0.98以上の入力除外を適用した。baseline 4方式は `engine.run()` の未除外結果を維持し、各選択項目へ `contains_input` を付けた。
- 【事実】`limit>=1`、除外総数の limit 非依存化、入力分割と同じ正規化規則の記録、評価シートへの入力A/B表示、`emergence_norm=0` 時の説明、近退化平面、bool自己評価拒否、dictコーパス検証、hash時の `applied_to=null` を実装した。
- 【事実】仮説内の禁止語は否定文でも厳格に拒否する方針を維持した。仮説候補の重複エンコード費用は残る。

## 3. 数理的な役割分担

【事実】semantic encoder は `intfloat/multilingual-e5-small` を使い、A・B・短い候補概念に `query: `、コーパス項目に `passage: ` を付ける。hash encoder は再現可能な文字 n-gram 表現であり、意味埋め込みではなく、接頭辞も付けない。baseline の `engine.run()` は semantic でも全テキストを `encode()` に渡すため、全て `query: ` になる。

【数学的結果】中心化では、1探索につき共有する `μ = mean({A,B} ∪ 語彙プール全体 ∪ --vocab ∪ コーパス passage 全体)` を引いて正規化する。LLM仮説は μ に含めず、同じ μ で射影だけを行う。

【数学的結果】`a∧b` は `span(A,B)` の向き付き平面であり、第三概念 C を生成しない。`plane_projection=p`、`residual=r=√(1-p²)`、正の `s_A,s_B` を要件とする `emergence_norm`、`balance=2pr` を計算し、`score_third=0.5×emergence_norm+0.5×balance` とする。これらは探索用ヒューリスティックである。

## 4. v2 比較結果

【事実】以下は各方式の上位5件について、共有 μ の共通表現で再計算した平均である。`J(T,M)` は third-concept と mean、`J(T,S)` は third-concept と similar の上位集合 Jaccard である。

| 根拠 | ケース | encoder | third E/R | mean E/R | similar E/R | J(T,M) | J(T,S) |
|---|---|---|---:|---:|---:|---:|---:|
| 【事実】【数学的結果】 | ヨーグルト×梅シロップ | hash | 0.000096 / 0.990986 | 0.000684 / 0.994717 | 0.000792 / 0.994860 | 0.111111 | 0.111111 |
| 【事実】【数学的結果】 | ヨーグルト×梅シロップ | semantic | 0 / 0.970135 | 0 / 0.996675 | 0 / 0.992293 | 0 | 0 |
| 【事実】【数学的結果】 | 入眠障害×条件反射 | hash | 0.001452 / 0.987947 | 0.002483 / 0.993552 | 0.002483 / 0.993552 | 0.111111 | 0.111111 |
| 【事実】【数学的結果】 | 入眠障害×条件反射 | semantic | 0 / 0.978505 | 0 / 0.989377 | 0 / 0.990605 | 0.111111 | 0.111111 |
| 【事実】【数学的結果】 | H2O MEMORY×TRAJEXA | hash | 0 / 0.991278 | 0.000621 / 0.997436 | 0.000259 / 0.990265 | 0 | 0 |
| 【事実】【数学的結果】 | H2O MEMORY×TRAJEXA | semantic | 0 / 0.968805 | 0 / 0.991262 | 0 / 0.986901 | 0 | 0 |

【事実】比較スクリプトは engine.run() の分割規則に合わせて入力を正規化するため、H2O ケースの上表は `H2O_MEMORY` で計算している。探索 CLI（§5・§6 の数値）は元の表記 `H2O MEMORY` のまま計算するので、H2O ケースだけは探索の retrieved 上位5件と比較の third-concept 上位5件が一致しない（例: hash で [29,36,9,15,5] と [29,9,15,36,41]）。他の2ケースは一致する。

【事実】食品ケースでは、third-concept の上位候補も食品・香味の既知事例に到達していない。実コーパスの語彙領域が合わず、選択集合が baseline と異なること自体を品質差とは解釈できない。

【事実】mean と third-concept-no-wedge の上位5件は6条件すべてで同一だった。hash 3条件では similar・orthogonal・wedge も集合として同一だった（順位は一部異なる）。

## 5. PM観察の再確認

【数学的結果】観察 (a)「emergence がほぼ0」は v2 でも支持される。`emergence_norm=0` は食品/hash 43/51、入眠/hash 43/51、H2O/hash 45/51、semantic 3条件はいずれも51/51だった。最大値は順に0.002314、0.007258、0.001311、0、0、0である（探索 CLI の表現。H2O を比較スクリプトの `H2O_MEMORY` 表現で再計算すると hash は 44/51・最大0.001809 で、結論は変わらない）。

【数学的結果】全306項目で `score_third` と `0.5×balance` の最大差は0.003629だった。`score_third` と `plane_projection` の平均絶対差は条件別に0.000318〜0.004084、平面射影の最大値は0.269504だった。この範囲では `p` が小さく、`0.5×balance=p√(1-p²)≈p` となる。

【数学的結果】観察 (b)「mean と no-wedge が同順位」は v2 でも支持される。全項目の `s_mean` は `[-0.264234, 0.200092]` で `|s_mean|<1/√2` にあり、`2s_mean√(1-s_mean²)` は単調増加する範囲だった。除外後の上位5件は6条件すべてで一致した。

【数学的結果】観察 (c)「既存 wedge は C を生成しない」は実装上変わらず支持される。既存 wedge は入力対とコーパス対の bivector cosine を順位に使うだけで、概念語 C の生成・復号を行わない。

## 6. ヨーグルト×梅シロップの仮説スコア

【事実】【数学的結果】列は `s_A / s_B / p / r / emergence_norm / balance / unexpectedness_h / coherence_h / score_third`。左がhash、右がsemanticである。

| 根拠 | 候補 | hash | semantic |
|---|---|---|---|
| 【LLM仮説】【数学的結果】 | 乳酸菌の浸透圧ショック耐性試験 | -0.018233 / 0.011852 / 0.023682 / 0.999720 / 0 / 0.047350 / 0.988148 / 0.023682 / 0.023675 | 0.078121 / 0.043201 / 0.083613 / 0.996498 / 0.002160 / 0.166640 / 0.921879 / 0.083613 / 0.084400 |
| 【LLM仮説】【数学的結果】 | 糖クリーム分離ゾーン | -0.041192 / 0.037469 / 0.060997 / 0.998138 / 0 / 0.121767 / 0.962531 / 0.060997 / 0.060883 | 0.149395 / 0.143186 / 0.190712 / 0.981646 / 0.034173 / 0.374424 / 0.850605 / 0.190712 / 0.204298 |
| 【LLM仮説】【数学的結果】 | 酸味マスキング味覚順応実験 | 0.002470 / 0.100245 / 0.101286 / 0.994857 / 0.000504 / 0.201531 / 0.899755 / 0.101286 / 0.101018 | 0.142627 / 0.132713 / 0.179582 / 0.983743 / 0.028956 / 0.353326 / 0.857373 / 0.179582 / 0.191141 |
| 【着想】【数学的結果】 | 梅果実の再発酵種床 | -0.014647 / 0.010473 / 0.019654 / 0.999807 / 0 / 0.039301 / 0.989527 / 0.019654 / 0.019650 | 0.136849 / 0.339781 / 0.348563 / 0.937285 / 0.014701 / 0.653407 / 0.660219 / 0.348563 / 0.334054 |
| 【LLM仮説】【数学的結果】 | ペクチン・カゼイン共存ゲル粘度曲線 | -0.051034 / 0.076364 / 0.100093 / 0.994978 / 0 / 0.199180 / 0.923636 / 0.100093 / 0.099590 | 0.066281 / 0.049361 / 0.076500 / 0.997070 / 0.003548 / 0.152552 / 0.933719 / 0.076500 / 0.078050 |
| 【着想】【数学的結果】 | 酸糖による発酵停止スイッチ | -0.106960 / 0.006292 / 0.109734 / 0.993961 / 0 / 0.218143 / 0.993708 / 0.109734 / 0.109071 | 0.173174 / 0.094373 / 0.184848 / 0.982767 / 0.010164 / 0.363326 / 0.826826 / 0.184848 / 0.186745 |

【LLM仮説】上表の候補は耐性試験、界面層、味覚順応、再発酵、混合ゲル、発酵停止を扱うが、安全性・実在性・新規性はいずれも未確認である。

## 7. 実行と成果物

【事実】作業ディレクトリは `D:\develop\works\crossrag-skill\skill\crossrag`、コーパスは `D:\develop\works\crossrag\crossrag_past_engineering_corpus.json`、`corpus_id=26636196c50a5d08`、51件である。semantic は全て `HF_HUB_OFFLINE=1` で実行した。24コマンドの終了コードは全て0だった。

### 7.1 CLIコマンド

```powershell
python scripts/third_concept_cli.py --A '<A>' --B '<B>' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 > 'D:\inbox\crossrag-third-concept\v2\<case>_hash.json'
$env:HF_HUB_OFFLINE='1'; python scripts/third_concept_cli.py --A '<A>' --B '<B>' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder semantic --model 'intfloat/multilingual-e5-small' --limit 5 --seed 0 > 'D:\inbox\crossrag-third-concept\v2\<case>_semantic.json'
python scripts/third_concept_cli.py --A '<A>' --B '<B>' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\<hyp>.json' > 'D:\inbox\crossrag-third-concept\v2\<case>_hash_hyp.json'
$env:HF_HUB_OFFLINE='1'; python scripts/third_concept_cli.py --A '<A>' --B '<B>' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder semantic --model 'intfloat/multilingual-e5-small' --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\<hyp>.json' > 'D:\inbox\crossrag-third-concept\v2\<case>_semantic_hyp.json'
```

置換値は `(case,A,B,hyp)` の順に `(yogurt_ume,ヨーグルト,梅シロップ,yogurt_ume)`、`(insomnia_conditioning,入眠障害,条件反射,insomnia_conditioning)`、`(h2o_trajexa,H2O MEMORY,TRAJEXA,h2o_trajexa)` である。

### 7.2 比較コマンド

```powershell
python scripts/compare_third_concept.py --A '<A>' --B '<B>' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --format <json|markdown> > 'D:\inbox\crossrag-third-concept\v2\<case>_hash_compare.<json|md>'
$env:HF_HUB_OFFLINE='1'; python scripts/compare_third_concept.py --A '<A>' --B '<B>' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder semantic --model 'intfloat/multilingual-e5-small' --limit 5 --seed 0 --format <json|markdown> --sheet-dir 'D:\inbox\crossrag-third-concept\v2\sheets\<case>' > 'D:\inbox\crossrag-third-concept\v2\<case>_semantic_compare.<json|md>'
```

### 7.3 終了コードとSHA-256

| 成果物 | Exit | SHA-256 |
|---|---:|---|
| `yogurt_ume_hash.json` | 0 | `7a0a9da0726eef1df53da62483c62403d580072c7b4ed4906c4fcc574a137745` |
| `yogurt_ume_hash_hyp.json` | 0 | `e9898114ae5b092636fff7cdcb301e698d046e7e5e395321a64b68e0b8bef484` |
| `yogurt_ume_semantic.json` | 0 | `a1e47249c801f3ccfb653abcc667532c48789be8e64dfe3486a684935c95108d` |
| `yogurt_ume_semantic_hyp.json` | 0 | `9aa18ece4e02dcc839c1d99680f5e7b6a78d7c319a5443c262c3d84424b956ff` |
| `insomnia_conditioning_hash.json` | 0 | `1011098e16e0c476b8db32df84f3113abbe9386e30c25ced0f4b8e22e8c63275` |
| `insomnia_conditioning_hash_hyp.json` | 0 | `889b29fd95136d3a92809b11e887aa18570fb38f85daa77215dd73ba7c18aa11` |
| `insomnia_conditioning_semantic.json` | 0 | `2000280f05225e063218f356498dc22d37334b33ed534c37e25ad4dff6847e70` |
| `insomnia_conditioning_semantic_hyp.json` | 0 | `d107d67c555617bec994d33c8dbb641479812e80b7148481c8e03104666ce2a9` |
| `h2o_trajexa_hash.json` | 0 | `1f85d3ee4913293ee26c95616073ec7c8f3126fafdf315011b20a8b43077ca57` |
| `h2o_trajexa_hash_hyp.json` | 0 | `da62fe66695fd7ba7a3df06a0cdd0417cba60592ff3c841f3af0415e9c1aa3ca` |
| `h2o_trajexa_semantic.json` | 0 | `14724357eb7ceccb3bd2e00a9151f388a130305a9235746811b7a4e0fa50ade6` |
| `h2o_trajexa_semantic_hyp.json` | 0 | `478e54ee55c295f7c073929e4d70ec2c67c6192942917ec75091f28c6e89588d` |
| `yogurt_ume_hash_compare.json` | 0 | `e333f378b0d7e91072e6b5c2009d1cb81b92bc90011c6eec24201a2279d3b7a4` |
| `yogurt_ume_hash_compare.md` | 0 | `3ac450f35d14aa95d5dd99b3c09649092ba8bd9926cb9c1462a9b0f3684c6830` |
| `yogurt_ume_semantic_compare.json` | 0 | `93830c0192f565765ce2c492f42ce83fe5c533adc3c2047358f768da1bd46c7f` |
| `yogurt_ume_semantic_compare.md` | 0 | `556d2816521e2183a90c687d01af6a075fee2ef7a3216b6805d234d42a7c822e` |
| `insomnia_conditioning_hash_compare.json` | 0 | `90f95ba685f02eb8eda092f4a8fe3b90541e3e556ff59fe25c85e863c7e3d023` |
| `insomnia_conditioning_hash_compare.md` | 0 | `7687b938956de058832555d55ca763fb4e748d4eb4071c005f78b1f209590033` |
| `insomnia_conditioning_semantic_compare.json` | 0 | `20b4415c990c48dbf8665c88e419c25535c0b6a18d62583836536b336b9632d9` |
| `insomnia_conditioning_semantic_compare.md` | 0 | `97eeaa6b0e7d9414d3c5ef9d335029649465050be6c7f5433a43f3ee62d5b556` |
| `h2o_trajexa_hash_compare.json` | 0 | `4c4c2b08cfb33199fd34d4b209d956ba4451252a540e6d944f4a73d12dafcaeb` |
| `h2o_trajexa_hash_compare.md` | 0 | `de8bdd2d0e31e6ac2c59ebcf98ba0866a8c747ba903d402d5a1f0f3daca10009` |
| `h2o_trajexa_semantic_compare.json` | 0 | `7da4215f293645263debb2ed4c570e69077aed3ae8ce97f9678f2d4f3766e80d` |
| `h2o_trajexa_semantic_compare.md` | 0 | `6a98f3b900c78de06f7b75845c6da727443188f3d9156cf8d1b8ccecfdf694ea` |

評価シートと対応キーのSHA-256は、yogurt が `7057699826cc6db81ddafde1a3c099db3c58c0eaa32a580e9cc8b936d3ffc504` / `ff2c43acdca63c0c5a767d32bf141b7c024e886665d96e414a194f07d9feda33`、insomnia が `309f8a4fcb2cb39e0ca51ad8afceb179c61bb8b6f06a7a3b859b58b8f78bac74` / `4a3032bcc5a8cc5fd0d19c18869d98e2659c60c7c28756a4379cccfdb633f134`、H2O が `b8469b776c8f388f9d6aa4c6e556f4eb174adfe43f203c48f58eaa1324f89d01` / `83001a96a554ba07e04608330570fc50c3ce8b98bea8ef09681c0cf36b1b1646` である（各組は `blind_sheet.md` / `blind_key.json`）。

## 8. 検証と限界

【事実】TDDのREDは終了コード1、40件中13失敗だった。修正後は `python -m unittest discover -s tests -v` が終了コード0、40件成功となり、既存 legacy snapshot も成功した。`python -m py_compile scripts/third_concept.py scripts/third_concept_cli.py scripts/compare_third_concept.py` と `git diff --check` も終了コード0だった。

【事実】semantic 実行では `embeddings.position_ids | UNEXPECTED` のロード報告が出た。各対象コマンドは終了コード0でJSON/Markdownを生成したが、このロード報告の影響は独立に評価していない。

【事実】実コーパスは工学・記録系で、食品・睡眠の語彙領域と一致しない。数学部分は候補プール外の概念を作れない。hash は意味モデルではなく、semantic の中心化手法もPoC上の設計である。

【事実】baseline は入力除外を行わず、semantic では passage/query 接頭辞も third-concept と異なる。`contains_input` は文字列一致・包含の表示であり、baseline の順位を変更しない。`seed` は評価シートのシャッフルにだけ使われ、`engine.run()` の random は hash ベースである。

【事実】ブラインド人手評価は未実施であり、unexpectedness、coherence、emergence、testability の品質差、既知事例以外の有用性、再現性は判断できない。

【着想】次の最小評価は、方式キーを開く前に複数評価者が v2 のブラインドシートを採点し、評価者間一致、方式別分布、ケース別の失敗類型を報告することである。

## 9. v3: 外積項の比率化・帰無分布・文入力（実装済み、実データ実行済み）

【事実】v2 の実データ 4 組では emergence の最大値が 0.002 で、上位 5 件は balance 単独の順位と一致した。v3 は外積に固有の情報（平面内の向き）を大きさから切り離して測り、その寄与を測定できるようにする。定義・根拠の詳細は design §8。

### 9.1 新しい量

【数学的結果】非退化平面で、射影を q⃗ = αã + βb̃（α = (s_A − ρs_B)/(1 − ρ²), β = (s_B − ρs_A)/(1 − ρ²)）と書く。

- `between_AB`: α ≥ 0 ∧ β ≥ 0（射影が A の線と B の線のなす錐の中）。
- `emergence_ratio` = 1 − max(s_A², s_B²)/p² = 1 − max(cos²(q,ã), cos²(q,b̃))（between_AB ∧ s_A > 0 ∧ s_B > 0 のとき。他は 0）。錐の中では emergence/p² に等しい。
- `emergence_ratio_norm` = emergence_ratio / ((1 − ρ)/2)。二等分線で 1。
- `score_ratio` = 0.5·(p ≥ p_min なら emergence_ratio_norm、未満なら 0) + 0.5·balance。方式名 `third-concept-ratio`。旧 `score_third` と `emergence_norm` は変更しない。

【数学的結果】テストで固定した性質: 角度表現との一致、c の大きさ・面外成分に対する不変性、A の線上で 0、二等分線で (1 − ρ)/2 かつ錐の中の最大（ρ = −0.6, 0, 0.3, 0.8）、退化平面・s_A ≤ 0・s_B ≤ 0 で 0、p < p_min で score_ratio = 0.5·balance。

【数学的結果】錐ゲートの理由: s_A > 0 ∧ s_B > 0 だけでは、ρ > 0.5 のとき錐の外で 1 − max(cos²) が最大 ρ² になり、二等分線の (1 − ρ)/2 を上回る（ρ = 0.8 で 0.64 対 0.10）。これは「A と B の間」ではないので 0 とした。既存の emergence_norm は同じ問題を持つが、意味を変えないために修正していない。

### 9.2 ノイズ対策

【事実】採用したのは、帰無分布で決める p の下限 `p_min` である。既定は `auto` で、座標をシャッフルした passage ベクトル（seed + 1）での p の 95 パーセンタイルを使う。p で重み付けする案は大きさへの依存が戻るため採らなかった。固定の絶対閾値は尺度がモデルに依存するため採らなかった。比較スクリプトは、各方式の上位 k の emergence_ratio と p について、ランダム方向（seed）とシャッフルの帰無分布に対する中央順位パーセンタイルを出す。

【限界】ランダム方向での emergence_ratio の帰無分布は、平面内の角度が一様になることから決まる固定の形で、p に依存しない。95 パーセンタイルは多重比較を補正しない目安であり、有意性検定ではない。

### 9.3 文入力と除外

【事実】A・B は短い文でもよい。e5 では文にも `query: ` が 1 回だけ付く（テストで固定）。比較スクリプトの `--input-kind sentence` は、third-concept 系に原文を渡し、engine.run() の 4 方式には正規化した文を渡す（`input_normalization.applies_to` に記録）。`--exclude TERM` は、NFKC・casefold・空白除去で正規化した TERM が候補（a・b・line、語彙、仮説の concept）に含まれていれば除外する。判定は片方向で、既存の A・B 双方向判定と生コサイン ≥ 0.98 の除外も残す。`--A-sentence/--B-sentence` を指定すると、単語側と文側を同じ μ・同じ候補集合（両側の除外 mask の論理和）で比べ、方式ごとに上位 k の Jaccard と Kendall τ_b を出す。

### 9.4 数学と LLM の分担

【事実】数学部分は候補を生成しない。コーパス・語彙・仮説ファイルにある候補を、平面内の向き（emergence_ratio）、平面成分の大きさ（p, p_min）、帰無分布で位置づけるだけである。概念名・説明・反証方法を作るのは外部の LLM で、その出力は未検証の仮説として扱う。品質（意外性・整合性・検証可能性）は数学部分では測らず、人手のブラインド評価が必要である。

### 9.5 実データの結果（2026-10-18、親が手元で実行）

【事実】コーパスは利用者の私的な文章（日本語の夢の記録 2 件を読点で 58 節に分割、本文なしの形式）。本文は個人情報を含むためリポジトリに入れず、ここには集計値だけを載せる。エンコーダは e5-small、中心化あり、`p_min` は auto（shuffled の p 95 パーセンタイル）、帰無分布は 1000 サンプル・seed 0。入力の中心語（各組の A・B の単語）は単語入力・文入力の両方で `--exclude` した。文入力は、夢の本文をなぞらないよう辞書的な定義文にした（例: 「リウマチは関節が痛んで腫れる病気だ」）。

| 組 | 入力 | ρ(中心化) | 候補 | s_A>0 ∧ s_B>0 | between_AB | 有効（ratio>0 ∧ p≥p_min） | p_min | p 最大 | 帰無（random）の有効率 | τ_b(score_ratio, balance) | 外積の項で上位 5 件が変化 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ハンバーガー×鍋スープ | 単語 | 0.414 | 49 | 2 | 2 | 0 | 0.126 | 0.197 | 0.014 | 1.000 | なし |
| ハンバーガー×鍋スープ | 文 | 0.471 | 49 | 0 | 0 | 0 | 0.125 | 0.177 | 0.007 | 1.000 | なし |
| 夢×リウマチ | 単語 | 0.325 | 51 | 4 | 0 | 0 | 0.121 | 0.237 | 0.016 | 1.000 | なし |
| 夢×リウマチ | 文 | 0.312 | 51 | 3 | 2 | 1 | 0.124 | 0.273 | 0.015 | 0.990 | なし |
| 注射×寄付 | 単語 | 0.474 | 56 | 3 | 1 | 0 | 0.125 | 0.211 | 0.009 | 1.000 | なし |
| 注射×寄付 | 文 | 0.335 | 56 | 4 | 3 | 0 | 0.125 | 0.217 | 0.006 | 1.000 | なし |
| 漫画家×編集者 | 単語 | 0.524 | 54 | 5 | 1 | 0 | 0.129 | 0.202 | 0.011 | 1.000 | なし |
| 漫画家×編集者 | 文 | 0.606 | 54 | 3 | 3 | 0 | 0.128 | 0.213 | 0.007 | 1.000 | なし |

【数学的結果】8 条件・合計 418 候補のうち、有効な候補は 1 件（夢×リウマチ・文入力。ratio_norm 0.132、p 0.138 で p_min 0.124 をわずかに上回る）。有効率は 0〜2% で、ランダム方向の帰無分布の有効率（0.6〜1.6%）と区別できない。外積の項は 8 条件すべてで上位 5 件を変えなかった。文入力は単語入力より p の最大値を少し上げた組があるが（夢×リウマチで 0.237 → 0.273）、結論は変わらない。

【着想】唯一の有効候補は、入力語を含まないまま、両方の概念に関わる内容だった（夢の中の関節の病気）。目的とする種類の候補ではあるが、1 件で信号も弱く、偶然と区別できない。

### 9.5b 陽性対照（合成データ）

【事実】外積の量が「A と B の両方を必要とする文」を見分けられるかを、合成コーパスで確かめた。各組で、A だけの文 3 件・B だけの文 3 件・A と B の組合せ文 3 件（例: 「A と B を組み合わせてみた」）・共通の無関係な文 10 件、計 19 件を作った。条件は 9.5 と同じ（除外なし）。

| 組 | 組合せ文の between_AB | 組合せ文の有効数 | 組合せ文の ratio 平均 | A だけ・B だけ・無関係の ratio | 組合せ文の平均順位（score_ratio / balance / s_mean / ratio 単独） |
|---|---|---|---|---|---|
| ハンバーガー×鍋スープ | 3/3 | 3 | 0.258 | すべて 0 | 2 / 2 / 2 / 2 |
| 夢×リウマチ | 3/3 | 3 | 0.140 | すべて 0 | 2 / 2 / 2 / 2 |
| 注射×寄付 | 2/3 | 2 | 0.011 | すべて 0 | 2 / 2 / 2 / 4 |
| 漫画家×編集者 | 2/3 | 1 | 0.018 | すべて 0 | 4 / 4 / 2 / 4 |

（平均順位は 19 件中。組合せ文 3 件が 1〜3 位なら 2。）

【数学的結果】emergence_ratio が正になったのは組合せ文だけで（12 件中 10 件が between_AB、9 件が有効）、A だけ・B だけ・無関係な文 64 件では 1 件も正にならなかった。この量は「両方を必要とするか」の判定としては機能する。一方、順位付けでは単純な平均方向の類似度（s_mean）でも組合せ文が上位に来ており、外積の項は順位を改善しなかった。

【限界】陽性対照の組合せ文は A・B の語をそのまま含むため、語彙的に容易である。目的の「A・B を含まない第三概念」の検出力は、この対照では測れない。合成文はテンプレートで作ったもので、1 回・4 組だけの結果である。

### 9.5c 結論

【数学的結果】外積に固有の量（平面内の向き）は、A と B の両方にまたがる候補を、帰無分布並みの誤検出率で判定できる。しかし今回の私的コーパス（58 節）には、入力語を除いたうえで両方にまたがる候補がほぼなく、取り出す対象自体が帰無分布と区別できなかった。順位付けへの寄与は、実データでも陽性対照でも確認できなかった。

【着想】外積の量は順位付けよりも「両方を必要とする」というラベルやフィルタとして使う方が合う可能性がある。候補を取り出せるかどうかは、コーパスの大きさと内容の幅に強く依存すると考えられる。検証には、より大きく多様なコーパスと、人手のブラインド評価が必要である。

結果が既存手法を上回らなくても、そのまま記入する。

### 9.6 実行コマンド例

```bash
cd skill/crossrag
# 単語入力（集計のみ）
python scripts/compare_third_concept.py --A <語A> --B <語B> --corpus <corpus.json> --encoder semantic --limit 5 --seed 0 --aggregate-only
# 文入力 + 中心語の除外
python scripts/compare_third_concept.py --A "<文A>" --B "<文B>" --input-kind sentence --exclude <中心語A> --exclude <中心語B> --corpus <corpus.json> --encoder semantic --aggregate-only
# 単語と文の同条件比較
python scripts/compare_third_concept.py --A <語A> --B <語B> --A-sentence "<文A>" --B-sentence "<文B>" --exclude <中心語A> --corpus <corpus.json> --encoder semantic --null-samples 1000 --aggregate-only
# 候補の抽出（新方式）
python scripts/third_concept_cli.py --A "<文A>" --B "<文B>" --corpus <corpus.json> --encoder semantic --method third-concept-ratio --exclude <中心語A>
```

`--aggregate-only` はコーパス本文（a・b・line）を出力から外すが、入力 A・B の文字列は残る。
