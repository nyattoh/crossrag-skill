# Third-concept PoC 評価報告

## 1. 目的と結論の要約

【事実】失敗・限界を先に述べると、人手評価は未実施であり、方式間の品質差は判断できない。実コーパスは工学・記録系51件で、食品の「ヨーグルト×梅シロップ」には語彙領域が合わず、数学候補と検索候補は既知事例に一致しなかった。

【数学的結果】6条件（3ケース×hash/semantic）のコーパス全306項目では、`emergence_norm` が厳密に0の項目が274件、0.01以下が304件だった。この条件では `score_third` はほぼ `0.5 × balance`、さらに平面射影 `p` が小さいためほぼ `p` となり、third-concept は実質的に A–B 平面へ近い項目を選ぶ挙動だった。

【数学的結果】mean と third-concept-no-wedge は6条件すべてで上位5件が完全一致した。全項目の `s_m` は `[-0.188680, 0.180069]` に入り、`2s_m√(1-s_m²)` が単調増加する範囲内だったためである。

【事実】既存 wedge は入力2語の bivector とコーパス各行の `a×b` bivector の cosine を比較する検索であり、第三概念 C を生成しない。hash の3ケースでは similar・orthogonal・wedge の上位5件が同一だった。

【LLM仮説】外部生成器は17候補を作ったが、内容はすべて未検証である。入眠ケースには具体的な装置・機構・実験案がある一方、既存の睡眠介入の説明や助言に近い候補も残った。H2O MEMORY×TRAJEXA には故障試験や比較実験がある一方、既存概念の要約・並列配置に近い候補も残った。

## 2. 数理的な役割分担

【事実】semantic encoder は `intfloat/multilingual-e5-small` を使い、A・B・短い候補概念に `query: `、コーパス項目に `passage: ` を付ける。hash encoder は再現可能な文字 n-gram 表現であり、意味埋め込みではなく、接頭辞も付けない。既存 `run.py` 系の baseline は全テキストを `encode()` に渡すため、e5 ではすべて `query: ` になる。

【数学的結果】中心化ありの既定設定では、A・B・候補集合の埋め込み平均 `μ` を引き、各ベクトルを正規化する。これは e5 埋め込みの異方性を緩和するためのPoC上の選択であり、中心化の正しさや最適性を示すものではない。

【数学的結果】中心化後の A・B を `a,b` とすると、`a∧b` が表すものは `span(A,B)` の向き付き平面である。平面の正規直交基底を `e1=a`, `e2=normalise(b-(a·b)a)` とし、候補 `c` の平面射影長を `p=√((c·e1)²+(c·e2)²)` とする。

【数学的結果】平面からの残差は `r=√(1-p²)=‖a∧b∧c‖/(‖a∧b‖‖c‖)` であり、正規化した trivector volume に相当する。

【数学的結果】`emergence_norm` は、平面内成分のうち A 軸または B 軸の単独成分を超える量 `max(0,p²-max(s_A²,s_B²))` を、上界 `(1-a·b)/2` で正規化したヒューリスティックである。ただし `s_A>0` かつ `s_B>0` のときだけ非0となる。

【数学的結果】`balance=2pr` は平面成分と残差成分の釣り合いを測り、`p=r=1/√2` で最大1となる。`score_third=0.5×emergence_norm+0.5×balance` であり、重みは未調整の固定値である。

【事実】LLM の役割は、候補名、説明、A単独/B単独との関係、AB依存性、不確実性、反証方法、自己評価を作ることである。数学の役割は、与えられたコーパス・語彙プール・LLM候補を埋め込み空間上に位置づけることだけであり、候補プール外の概念を作れない。

【数学的結果】`a∧b` 単独から C は得られない。`a∧b` は平面を表すだけで、高次元空間の直交補空間には多数の方向があり、どの概念語に対応するかを決める復号規則も持たない。

## 3. 比較結果

【事実】以下は T3 の実コーパス出力で、各方式の上位5件について共通表現で再計算した平均値である。`J(third)` は当該方式と third-concept の上位5件集合の Jaccard 係数である。項目本文は載せず、`a×b` のみを示す。

### 3.1 ヨーグルト × 梅シロップ — hash

| 根拠 | 方式 | 上位5件の `a×b` | emergence_norm | residual | plane_projection | score_third | J(third) |
|---|---|---|---:|---:|---:|---:|---:|
| 【事実】【数学的結果】 | similar | 作曲×休符; 裁判×反対尋問; 多視点×観測の独立性; 舞台×観客; 取引×信頼の伝達 | 0.000593 | 0.995696 | 0.083425 | 0.083217 | 0.111111 |
| 【事実】【数学的結果】 | orthogonal | 作曲×休符; 裁判×反対尋問; 多視点×観測の独立性; 取引×信頼の伝達; 舞台×観客 | 0.000593 | 0.995696 | 0.083425 | 0.083217 | 0.111111 |
| 【事実】【数学的結果】 | wedge | 作曲×休符; 裁判×反対尋問; 多視点×観測の独立性; 舞台×観客; 取引×信頼の伝達 | 0.000593 | 0.995696 | 0.083425 | 0.083217 | 0.111111 |
| 【事実】【数学的結果】 | random | 市場×発明の採用; 反実仮想×選択の機会; レシピ×再現性; 作曲×休符; 取引×信頼の伝達 | 0.000306 | 0.994982 | 0.092845 | 0.092409 | 0.111111 |
| 【事実】【数学的結果】 | third-concept | 航海×戻れる経路; 盲点探索×証拠台帳; 作曲×休符; 教育×試す機会; 協同設計×人間の主導権 | 0.000027 | 0.989790 | 0.142224 | 0.140773 | 1.000000 |
| 【事実】【数学的結果】 | mean | 作曲×休符; 多視点×観測の独立性; 手話×空間文法; 舞台×観客; 裁判×反対尋問 | 0.000317 | 0.995392 | 0.089182 | 0.088812 | 0.111111 |
| 【事実】【数学的結果】 | third-concept-no-wedge | 作曲×休符; 多視点×観測の独立性; 手話×空間文法; 舞台×観客; 裁判×反対尋問 | 0.000317 | 0.995392 | 0.089182 | 0.088812 | 0.111111 |

### 3.2 ヨーグルト × 梅シロップ — semantic

| 根拠 | 方式 | 上位5件の `a×b` | emergence_norm | residual | plane_projection | score_third | J(third) |
|---|---|---|---:|---:|---:|---:|---:|
| 【事実】【数学的結果】 | similar | 沈黙×選択肢; 植物×休眠; レシピ×再現性; 生態系×攪乱; 灯台×見えない岸 | 0.001336 | 0.997117 | 0.056551 | 0.056873 | 0.111111 |
| 【事実】【数学的結果】 | orthogonal | 贈与×価値の計測; 市場×発明の採用; 翻訳×失われる含意; 舞台×観客; 作曲×休符 | 0.000000 | 0.998061 | 0.051687 | 0.051519 | 0.000000 |
| 【事実】【数学的結果】 | wedge | 地図×未踏領域; 工場の安全×非常停止; 設計図×現場作業; ソフトウェアテスト×敗北条件; 翻訳×失われる含意 | 0.000000 | 0.993658 | 0.103549 | 0.102670 | 0.111111 |
| 【事実】【数学的結果】 | random | 市場×発明の採用; 反実仮想×選択の機会; レシピ×再現性; 作曲×休符; 取引×信頼の伝達 | 0.000000 | 0.997111 | 0.064393 | 0.064109 | 0.000000 |
| 【事実】【数学的結果】 | third-concept | 設計図×現場作業; 労働×暗黙の調整; 観測者×観測対象; 植物×休眠; 障害記録×直した理由 | 0.001329 | 0.987630 | 0.155919 | 0.154612 | 1.000000 |
| 【事実】【数学的結果】 | mean | 植物×休眠; 博物館の収蔵×来歴; 手話×空間文法; 古文書解読×証拠の空白; 生態系×攪乱 | 0.003175 | 0.996316 | 0.074576 | 0.075740 | 0.111111 |
| 【事実】【数学的結果】 | third-concept-no-wedge | 植物×休眠; 博物館の収蔵×来歴; 手話×空間文法; 古文書解読×証拠の空白; 生態系×攪乱 | 0.003175 | 0.996316 | 0.074576 | 0.075740 | 0.111111 |

【事実】食品ケースでは、どの方式の上位項目も食品・香味の既知事例に到達せず、third-concept が品質面で優れるとは判断できない。

### 3.3 入眠障害 × 条件反射 — hash

| 根拠 | 方式 | 上位5件の `a×b` | emergence_norm | residual | plane_projection | score_third | J(third) |
|---|---|---|---:|---:|---:|---:|---:|
| 【事実】【数学的結果】 | similar | 都市計画×誰も通らない道; レシピ×再現性; 航海×戻れる経路; 可逆記憶×人間の訂正権; 贈与×価値の計測 | 0.000951 | 0.995129 | 0.090420 | 0.090293 | 0.111111 |
| 【事実】【数学的結果】 | orthogonal | 都市計画×誰も通らない道; レシピ×再現性; 航海×戻れる経路; 可逆記憶×人間の訂正権; 贈与×価値の計測 | 0.000951 | 0.995129 | 0.090420 | 0.090293 | 0.111111 |
| 【事実】【数学的結果】 | wedge | 都市計画×誰も通らない道; レシピ×再現性; 航海×戻れる経路; 可逆記憶×人間の訂正権; 贈与×価値の計測 | 0.000951 | 0.995129 | 0.090420 | 0.090293 | 0.111111 |
| 【事実】【数学的結果】 | random | 古文書解読×証拠の空白; 沈黙×選択肢; 労働×暗黙の調整; 写真のネガ×未写像; 盲点探索×証拠台帳 | 0.000433 | 0.995482 | 0.088165 | 0.087883 | 0.000000 |
| 【事実】【数学的結果】 | third-concept | 多視点×観測の独立性; 都市計画×誰も通らない道; 香り×言葉にならない差; セマンティックHTML×支援技術; 触覚×非言語プロトコル | 0.000419 | 0.987398 | 0.157353 | 0.155536 | 1.000000 |
| 【事実】【数学的結果】 | mean | 都市計画×誰も通らない道; レシピ×再現性; 可逆記憶×人間の訂正権; 航海×戻れる経路; 盲点探索×証拠台帳 | 0.001172 | 0.995081 | 0.091472 | 0.091451 | 0.111111 |
| 【事実】【数学的結果】 | third-concept-no-wedge | 都市計画×誰も通らない道; レシピ×再現性; 可逆記憶×人間の訂正権; 航海×戻れる経路; 盲点探索×証拠台帳 | 0.001172 | 0.995081 | 0.091472 | 0.091451 | 0.111111 |

### 3.4 入眠障害 × 条件反射 — semantic

| 根拠 | 方式 | 上位5件の `a×b` | emergence_norm | residual | plane_projection | score_third | J(third) |
|---|---|---|---:|---:|---:|---:|---:|
| 【事実】【数学的結果】 | similar | 植物×休眠; 看護×説明できない変化; 沈黙×選択肢; ソフトウェアテスト×敗北条件; 身体感覚×環境 | 0.003494 | 0.987510 | 0.139604 | 0.138803 | 0.250000 |
| 【事実】【数学的結果】 | orthogonal | 舞台×観客; 博物館の収蔵×来歴; 贈与×価値の計測; 分散システム×同意; 写真のネガ×未写像 | 0.000000 | 0.996330 | 0.073907 | 0.073502 | 0.000000 |
| 【事実】【数学的結果】 | wedge | 能力の時間変化×観測条件; ソフトウェアテスト×敗北条件; 写真のネガ×未写像; 分散システム×同意; 沈黙×選択肢 | 0.000000 | 0.993225 | 0.101369 | 0.100312 | 0.111111 |
| 【事実】【数学的結果】 | random | 古文書解読×証拠の空白; 沈黙×選択肢; 労働×暗黙の調整; 写真のネガ×未写像; 盲点探索×証拠台帳 | 0.000000 | 0.997374 | 0.067560 | 0.067331 | 0.000000 |
| 【事実】【数学的結果】 | third-concept | 植物×休眠; 能力の時間変化×観測条件; 記録×反論; 取引×信頼の伝達; 看護×説明できない変化 | 0.003205 | 0.979631 | 0.198099 | 0.195440 | 1.000000 |
| 【事実】【数学的結果】 | mean | 植物×休眠; 看護×説明できない変化; 能力の時間変化×観測条件; 身体感覚×環境; 外積×モデルに見えないもの | 0.003713 | 0.985235 | 0.156416 | 0.155201 | 0.428571 |
| 【事実】【数学的結果】 | third-concept-no-wedge | 植物×休眠; 看護×説明できない変化; 能力の時間変化×観測条件; 身体感覚×環境; 外積×モデルに見えないもの | 0.003713 | 0.985235 | 0.156416 | 0.155201 | 0.428571 |

【事実】入眠ケースの semantic では third-concept と mean/no-wedge の重なりが3/7（Jaccard 0.428571）あり、完全に異なる検索結果ではない。内容の品質差は人手評価前には判断できない。

### 3.5 H2O MEMORY × TRAJEXA — hash

| 根拠 | 方式 | 上位5件の `a×b` | emergence_norm | residual | plane_projection | score_third | J(third) |
|---|---|---|---:|---:|---:|---:|---:|
| 【事実】【数学的結果】 | similar | 能力の時間変化×観測条件; 生態系×攪乱; 触覚×非言語プロトコル; 設計図×現場作業; 失敗ログ×因果介入 | 0.000128 | 0.990577 | 0.106850 | 0.104705 | 0.111111 |
| 【事実】【数学的結果】 | orthogonal | 能力の時間変化×観測条件; 生態系×攪乱; 触覚×非言語プロトコル; 設計図×現場作業; 失敗ログ×因果介入 | 0.000128 | 0.990577 | 0.106850 | 0.104705 | 0.111111 |
| 【事実】【数学的結果】 | wedge | 能力の時間変化×観測条件; 生態系×攪乱; 触覚×非言語プロトコル; 設計図×現場作業; 失敗ログ×因果介入 | 0.000128 | 0.990577 | 0.106850 | 0.104705 | 0.111111 |
| 【事実】【数学的結果】 | random | Agentオーケストレーション×高負荷時の相互保護; 灯台×見えない岸; 外積×モデルに見えないもの; 能力の時間変化×観測条件; 舞台×観客 | 0.000000 | 0.990425 | 0.106589 | 0.104347 | 0.111111 |
| 【事実】【数学的結果】 | third-concept | 能力の時間変化×観測条件; ソフトウェアテスト×敗北条件; 教育×試す機会; 反実仮想×選択の機会; 博物館の収蔵×来歴 | 0.000000 | 0.983878 | 0.170800 | 0.167475 | 1.000000 |
| 【事実】【数学的結果】 | mean | 能力の時間変化×観測条件; 観測者×観測対象; 生態系×攪乱; 触覚×非言語プロトコル; 設計図×現場作業 | 0.001421 | 0.989643 | 0.120165 | 0.118562 | 0.111111 |
| 【事実】【数学的結果】 | third-concept-no-wedge | 能力の時間変化×観測条件; 観測者×観測対象; 生態系×攪乱; 触覚×非言語プロトコル; 設計図×現場作業 | 0.001421 | 0.989643 | 0.120165 | 0.118562 | 0.111111 |

### 3.6 H2O MEMORY × TRAJEXA — semantic

| 根拠 | 方式 | 上位5件の `a×b` | emergence_norm | residual | plane_projection | score_third | J(third) |
|---|---|---|---:|---:|---:|---:|---:|
| 【事実】【数学的結果】 | similar | 能力の時間変化×観測条件; 沈黙×選択肢; 生態系×攪乱; 身体感覚×環境; 記憶×忘却の権利 | 0.006861 | 0.989337 | 0.109871 | 0.110616 | 0.111111 |
| 【事実】【数学的結果】 | orthogonal | 古文書解読×証拠の空白; 博物館の収蔵×来歴; 幾何学的検索×意外性の評価; 贈与×価値の計測; 舞台×観客 | 0.000000 | 0.996468 | 0.081985 | 0.081672 | 0.000000 |
| 【事実】【数学的結果】 | wedge | 記憶×忘却の権利; 触覚×非言語プロトコル; 反実仮想×選択の機会; 建築の余白×人の回遊; 可逆記憶×人間の訂正権 | 0.000000 | 0.995985 | 0.085735 | 0.085343 | 0.000000 |
| 【事実】【数学的結果】 | random | Agentオーケストレーション×高負荷時の相互保護; 灯台×見えない岸; 外積×モデルに見えないもの; 能力の時間変化×観測条件; 舞台×観客 | 0.006861 | 0.988226 | 0.124773 | 0.125370 | 0.111111 |
| 【事実】【数学的結果】 | third-concept | 能力の時間変化×観測条件; 市場×発明の採用; 協同設計×人間の主導権; ソフトウェアテスト×敗北条件; 保険×不確実性の配分 | 0.006861 | 0.983439 | 0.170215 | 0.170015 | 1.000000 |
| 【事実】【数学的結果】 | mean | 能力の時間変化×観測条件; 身体感覚×環境; 可逆記憶×人間の訂正権; 気象予報×外れた理由; 植物×休眠 | 0.006861 | 0.987665 | 0.133860 | 0.134409 | 0.111111 |
| 【事実】【数学的結果】 | third-concept-no-wedge | 能力の時間変化×観測条件; 身体感覚×環境; 可逆記憶×人間の訂正権; 気象予報×外れた理由; 植物×休眠 | 0.006861 | 0.987665 | 0.133860 | 0.134409 | 0.111111 |

【数学的結果】third-concept は6条件すべてで自分の目的関数 `score_third` の選択平均が最も高いが、これは目的関数で選んだ結果を同じ目的関数で測る循環であり、品質上の優位を示さない。

【事実】比較結果で明確に「差がない」のは、全6条件の mean と no-wedge、ならびにhash 3条件の similar・orthogonal・wedge である。差がある行についても、人手評価なしでは優劣を判断できない。

## 4. 外積の寄与の確認

【数学的結果】PM観察 (a) は支持される。条件別の `emergence_norm=0` は、食品/hash 44/51、入眠/hash 44/51、H2O/hash 45/51、食品/semantic 46/51、入眠/semantic 45/51、H2O/semantic 50/51だった。最大値は順に 0.001424、0.002096、0.006855、0.009197、0.011335、0.034307 である。

【数学的結果】全306項目で `score_third` と `0.5×balance` の最大差は0.017154、`score_third` と `plane_projection` の平均絶対差は条件別に0.000351〜0.000700だった。平面射影の最大値も条件別に0.156166〜0.290787で小さく、`0.5×balance=p√(1-p²)≈p` が成立するため、この実コーパスではほぼ平面近接順位になった。

【数学的結果】PM観察 (b) は支持される。`score_no_wedge=2s_m√(1-s_m²)` の導関数は `2(1-2s_m²)/√(1-s_m²)` で、`|s_m|<1/√2` では単調増加する。6条件の全項目は `s_m∈[-0.188680,0.180069]` にあり、mean と no-wedge の上位5件は各条件で同一、相互 Jaccard はすべて1.0だった。

【数学的結果】PM観察 (c) は支持される。既存 wedge は `cos(a∧b,u∧v)` を入力対 `(a,b)` とコーパス対 `(u,v)` の内積から計算し、候補行の順位に使う。概念語 C を作る処理や復号処理はない。

【事実】semantic wedge の上位行は、食品では「地図×未踏領域」など、入眠では「能力の時間変化×観測条件」など、H2O/TRAJEXAでは「記憶×忘却の権利」などの既存コーパス対である。hash では wedge 項が順位に寄与せず、3ケースとも similar・orthogonal と同じ行になった。

【数学的結果】したがって、このPoCで観測できた外積由来の差は、Cの生成ではなく、third-concept の平面射影・三重積残差によるコーパス行の並べ替えである。その並べ替えが人に有用かは未評価である。

## 5. LLM 仮説の扱い

【事実】以下の `E/B/S` は `emergence_norm / balance / score_third` で、左がhash、右がsemanticである。`U/C/E/T` は生成器自身の unexpectedness/coherence/emergence/testability 自己評価であり、独立評価ではない。

### 5.1 ヨーグルト × 梅シロップ

| 根拠 | origin | 候補 | hash E/B/S | semantic E/B/S | LLM自己評価 U/C/E/T |
|---|---|---|---|---|---|
| 【LLM仮説】【数学的結果】 | llm_hypothesis | 乳酸菌の浸透圧ショック耐性試験 | 0/0.484946/0.242473 | 0/0.656527/0.328263 | 4/3/4/5 |
| 【LLM仮説】【数学的結果】 | llm_hypothesis | 糖クリーム分離ゾーン | 0/0.381460/0.190730 | 0/0.501878/0.250939 | 4/3/4/5 |
| 【LLM仮説】【数学的結果】 | llm_hypothesis | 酸味マスキング味覚順応実験 | 0/0.267343/0.133671 | 0/0.533322/0.266661 | 3/4/3/5 |
| 【LLM仮説】【数学的結果】 | idea | 梅果実の再発酵種床 | 0/0.480483/0.240241 | 0/0.390650/0.195325 | 3/2/3/3 |
| 【LLM仮説】【数学的結果】 | llm_hypothesis | ペクチン・カゼイン共存ゲル粘度曲線 | 0/0.559985/0.279993 | 0/0.627910/0.313955 | 3/4/4/5 |
| 【LLM仮説】【数学的結果】 | idea | 酸糖による発酵停止スイッチ | 0/0.616723/0.308362 | 0/0.741926/0.370963 | 2/3/2/4 |

【LLM仮説】6候補は耐性試験、界面層形成、味覚順応、再発酵、混合ゲル、発酵停止という具体的な現象・製品・実験方向を持つ。ただし安全性・実在性・新規性はいずれも未確認である。

【数学的結果】6候補すべてで `emergence_norm=0` であり、数理スコアは LLM のAB依存説明を支持していない。`score_third` は balance だけで決まった。

### 5.2 入眠障害 × 条件反射

| 根拠 | origin | 候補 | hash E/B/S | semantic E/B/S | LLM自己評価 U/C/E/T |
|---|---|---|---|---|---|
| 【LLM仮説】【数学的結果】 | llm_hypothesis | 就床前ルーティンの条件刺激化 | 0/0.566124/0.283062 | 0/0.671377/0.335689 | 3/3/4/5 |
| 【LLM仮説】【数学的結果】 | llm_hypothesis | 寝床=覚醒の逆条件づけ | 0/0.408564/0.204282 | 0/0.591947/0.295973 | 2/5/2/5 |
| 【LLM仮説】【数学的結果】 | llm_hypothesis | 生体指標ゲート式の入眠合図 | 0/0.557154/0.278577 | 0/0.605692/0.302846 | 4/3/4/4 |
| 【LLM仮説】【数学的結果】 | llm_hypothesis | 入眠努力の条件刺激化 | 0/0.272243/0.136121 | 0/0.099822/0.049911 | 3/4/3/4 |
| 【LLM仮説】【数学的結果】 | idea | 記憶再活性化型の入眠合図 | 0/0.452793/0.226397 | 0/0.711708/0.355854 | 3/2/3/3 |
| 【LLM仮説】【数学的結果】 | idea | 夜間覚醒時の部分強化 | 0/0.399510/0.199755 | 0/0.601807/0.300903 | 3/3/3/3 |

【LLM仮説】チャーターが求める具体的な製品・機構・実験という方向には、「生体指標ゲート式の入眠合図」と各候補の比較試験が比較的沿う。「就床前ルーティン」も対提示条件を操作する実験案になっている。

【LLM仮説】一方、「寝床=覚醒の逆条件づけ」は既存の刺激制御法の説明に近く、「夜間覚醒時の部分強化」は睡眠上の助言へ寄りやすい。「記憶再活性化型」は仮説ファイル自身が入眠前とは機構が異なると認めており、第三概念として弱い。ケース2は全候補が十分にチャーターを満たしたとはいえない。

【数学的結果】6候補すべてで `emergence_norm=0` であり、数理スコアは balance だけで決まった。semantic の最高 `score_third=0.355854` は「記憶再活性化型」だが、この順位は内容妥当性を意味しない。

### 5.3 H2O MEMORY × TRAJEXA

| 根拠 | origin | 候補 | hash E/B/S | semantic E/B/S | LLM自己評価 U/C/E/T |
|---|---|---|---|---|---|
| 【LLM仮説】【数学的結果】 | llm_hypothesis | 遷移ログの忘却可能な正本化 | 0/0.521936/0.260968 | 0/0.730458/0.365229 | 3/4/4/5 |
| 【LLM仮説】【数学的結果】 | llm_hypothesis | 由来ラベル付き気づきの再活性化 | 0/0.543506/0.271753 | 0/0.691842/0.345921 | 3/4/3/4 |
| 【LLM仮説】【数学的結果】 | llm_hypothesis | 回答再訪の再活性化遅延プローブ | 0/0.633535/0.316768 | 0/0.689425/0.344712 | 4/3/4/5 |
| 【LLM仮説】【数学的結果】 | idea | 認識段階×保存温度の二軸マップ | 0/0.515403/0.257701 | 0/0.443401/0.221700 | 3/2/3/3 |
| 【LLM仮説】【数学的結果】 | idea | 負の結果の公開テンプレート | 0/0.342857/0.171429 | 0/0.526794/0.263397 | 2/4/2/4 |

【LLM仮説】「遷移ログの忘却可能な正本化」は削除後の再出現という故障試験を持ち、「回答再訪の再活性化遅延プローブ」は手掛かり表示の有無を比較する実験を持つため、具体的な仕組み・実験に近い。

【LLM仮説】「由来ラベル付き気づき」は既存原則の適用説明に近く、「認識段階×保存温度」は既存の二軸を並べた表示案、「負の結果の公開テンプレート」は既存原則のテンプレート化である。ケース3が避けるべき既存作業の要約・並列説明を十分に脱していない候補が3件あり、チャーター達成は部分的である。

【数学的結果】5候補すべてで `emergence_norm=0` であり、数理スコアは balance だけで決まった。semantic の最高 `score_third=0.365229` は「遷移ログの忘却可能な正本化」だが、内容の新規性や実装可能性を示す値ではない。

## 6. 既知事例との照合

【事実】評価上の既知事例は「ヨーグルト×梅シロップからバナナ様の香味」である。`test_known_answer_does_not_leak` は「バナナ」「ばなな」「banana」を符号化して保持し、合成コーパス、scripts、生成プロンプト、CLI出力、および環境変数で指定された評価コーパスに含まれないことを検査する（`skill/crossrag/tests/test_third_concept.py` 176–199行）。今回の全26テストで同テストは通過した。

【事実】`docs/third_concept_hypotheses/PROMPT.md` の食品プロンプトには既知事例、バナナ、香り、風味を示す語がない。生成器に渡した固定プロンプトも同ファイルに記録されている。

【事実】`yogurt_ume.json` の6候補について「バナナ」「ばなな」「banana」「香り」「aroma」「風味」「flavour」「flavor」を検索した結果は0件だった。6候補のいずれも既知事例に一致しない。

【事実】実コーパスおよびCLI出力への既知語漏洩はテスト対象だが、今回のレポートは既知事例を明示して評価するため、仮説生成入力とは分離して作成した。

## 7. 限界

【事実】実コーパスは工学・記録系で、食品・睡眠の語彙領域と一致しない。数学部分はコーパスの `a`・`b` と任意追加語彙、または与えられたLLM候補の範囲外に概念を作れない。

【事実】hash encoder は文字 n-gram 表現であり意味モデルではない。semantic encoder も e5 の異方性を持ち、A・B・候補全体の平均で中心化する現在の選択はPoCの一案にすぎない。

【事実】`emergence_norm` と `balance` の重みは0.5/0.5に固定され、調整・外部校正されていない。LLM仮説は `claudeAgent / claude-sonnet-5-5 / effort medium` の単一生成器だけである。

【事実】ケース3の生成器は `PROMPT.md` の記録どおり `D:\develop\worktrees\home-offer-layout\README.md` ほかプロジェクト資料を読んだ。既知解には盲検だったが、プロジェクト文脈には盲検ではない。

【事実】baseline の `engine.run()` は `encode()` を使うため、semantic 比較でも候補本文と候補対を含む全テキストに `query: ` を付ける。一方、third-concept と共通評価はコーパス本文に `passage: ` を使う。この接頭辞差は比較上の交絡である。

【事実】semantic 実行では3回とも `embeddings.position_ids | UNEXPECTED` のロード報告が出た。各実行は終了コード0でJSONを生成したが、このロード報告の影響を独立に評価していない。

【事実】ブラインド人手評価は未実施であり、unexpectedness、coherence、emergence、testability の品質差、既知事例以外の有用性、再現性は判断できない。

## 8. 未検証事項と人手評価

【事実】T3 のブラインド評価シートは `D:\inbox\crossrag-third-concept\t3\sheets\yoghurt_ume\`、`sleep_conditioning\`、`h2o_trajexa\` にある。各シートはコーパス行だけを対象とし、方式名とスコアを伏せている。評価状態は未実施である。

【着想】第1段階として、各シートを複数評価者が independent に採点し、方式キーを開く前に unexpectedness、coherence、emergence、testability を1–5で記録する。評価者間一致、方式別分布、ケース別の失敗理由を報告対象にする。

【着想】第2段階として、数学候補語とLLM仮説を同じ候補プールに混ぜ、origin と方式を伏せた別シートを作る。短い概念名だけでは判断できないため、全候補に同じ長さの「AB依存説明」「不確実性」「反証方法」を付け、既知事例は採点後にだけ照合する。

【着想】第2シートでは、(1) 単なるA/Bの言い換え、(2) 既存作業の要約、(3) 助言の列挙、(4) 具体的な製品・機構・実験、を別々に分類し、数値評点と失敗類型を混同しない。

【事実】これらの人手評価はまだ行っていない。

## 9. 再現手順

【事実】すべて `D:\develop\works\crossrag-skill\skill\crossrag` を作業ディレクトリとして実行した。実コーパスは `D:\develop\works\crossrag\crossrag_past_engineering_corpus.json`、`corpus_id=26636196c50a5d08`、51件である。

【事実】比較JSONの再現コマンドは次のとおりである。

```powershell
python scripts/compare_third_concept.py --A 'ヨーグルト' --B '梅シロップ' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --format json > 'D:\inbox\crossrag-third-concept\t3\yoghurt_ume_hash.json'
$env:HF_HUB_OFFLINE='1'; python scripts/compare_third_concept.py --A 'ヨーグルト' --B '梅シロップ' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder semantic --model 'intfloat/multilingual-e5-small' --limit 5 --seed 0 --format json --sheet-dir 'D:\inbox\crossrag-third-concept\t3\sheets\yoghurt_ume' > 'D:\inbox\crossrag-third-concept\t3\yoghurt_ume_semantic.json'
python scripts/compare_third_concept.py --A '入眠障害' --B '条件反射' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --format json > 'D:\inbox\crossrag-third-concept\t3\sleep_conditioning_hash.json'
$env:HF_HUB_OFFLINE='1'; python scripts/compare_third_concept.py --A '入眠障害' --B '条件反射' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder semantic --model 'intfloat/multilingual-e5-small' --limit 5 --seed 0 --format json --sheet-dir 'D:\inbox\crossrag-third-concept\t3\sheets\sleep_conditioning' > 'D:\inbox\crossrag-third-concept\t3\sleep_conditioning_semantic.json'
python scripts/compare_third_concept.py --A 'H2O MEMORY' --B 'TRAJEXA' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --format json > 'D:\inbox\crossrag-third-concept\t3\h2o_trajexa_hash.json'
$env:HF_HUB_OFFLINE='1'; python scripts/compare_third_concept.py --A 'H2O MEMORY' --B 'TRAJEXA' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder semantic --model 'intfloat/multilingual-e5-small' --limit 5 --seed 0 --format json --sheet-dir 'D:\inbox\crossrag-third-concept\t3\sheets\h2o_trajexa' > 'D:\inbox\crossrag-third-concept\t3\h2o_trajexa_semantic.json'
```

【事実】仮説付きJSONの再現コマンドは次のとおりで、6件とも終了コード0だった。

```powershell
python scripts/third_concept_cli.py --A 'ヨーグルト' --B '梅シロップ' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\yogurt_ume.json' > 'D:\inbox\crossrag-third-concept\t4\yogurt_ume_hash_hyp.json'
$env:HF_HUB_OFFLINE='1'; python scripts/third_concept_cli.py --A 'ヨーグルト' --B '梅シロップ' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder semantic --model 'intfloat/multilingual-e5-small' --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\yogurt_ume.json' > 'D:\inbox\crossrag-third-concept\t4\yogurt_ume_semantic_hyp.json'
python scripts/third_concept_cli.py --A '入眠障害' --B '条件反射' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\insomnia_conditioning.json' > 'D:\inbox\crossrag-third-concept\t4\insomnia_conditioning_hash_hyp.json'
$env:HF_HUB_OFFLINE='1'; python scripts/third_concept_cli.py --A '入眠障害' --B '条件反射' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder semantic --model 'intfloat/multilingual-e5-small' --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\insomnia_conditioning.json' > 'D:\inbox\crossrag-third-concept\t4\insomnia_conditioning_semantic_hyp.json'
python scripts/third_concept_cli.py --A 'H2O MEMORY' --B 'TRAJEXA' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\h2o_trajexa.json' > 'D:\inbox\crossrag-third-concept\t4\h2o_trajexa_hash_hyp.json'
$env:HF_HUB_OFFLINE='1'; python scripts/third_concept_cli.py --A 'H2O MEMORY' --B 'TRAJEXA' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder semantic --model 'intfloat/multilingual-e5-small' --limit 5 --seed 0 --hypotheses '..\..\docs\third_concept_hypotheses\h2o_trajexa.json' > 'D:\inbox\crossrag-third-concept\t4\h2o_trajexa_semantic_hyp.json'
```

【事実】プロンプト出力の再現コマンドは次のとおりで、6件とも終了コード0だった。`--with-math` はhash候補を参照した。

```powershell
python scripts/third_concept_cli.py --A 'ヨーグルト' --B '梅シロップ' --emit-prompt > 'D:\inbox\crossrag-third-concept\t4\yogurt_ume_prompt.txt'
python scripts/third_concept_cli.py --A 'ヨーグルト' --B '梅シロップ' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --emit-prompt --with-math > 'D:\inbox\crossrag-third-concept\t4\yogurt_ume_prompt_with_math.txt'
python scripts/third_concept_cli.py --A '入眠障害' --B '条件反射' --emit-prompt > 'D:\inbox\crossrag-third-concept\t4\insomnia_conditioning_prompt.txt'
python scripts/third_concept_cli.py --A '入眠障害' --B '条件反射' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --emit-prompt --with-math > 'D:\inbox\crossrag-third-concept\t4\insomnia_conditioning_prompt_with_math.txt'
python scripts/third_concept_cli.py --A 'H2O MEMORY' --B 'TRAJEXA' --emit-prompt > 'D:\inbox\crossrag-third-concept\t4\h2o_trajexa_prompt.txt'
python scripts/third_concept_cli.py --A 'H2O MEMORY' --B 'TRAJEXA' --corpus 'D:\develop\works\crossrag\crossrag_past_engineering_corpus.json' --encoder hash --limit 5 --seed 0 --emit-prompt --with-math > 'D:\inbox\crossrag-third-concept\t4\h2o_trajexa_prompt_with_math.txt'
```

【事実】全テストは `python -m unittest discover -s tests -v` で実行し、終了コード0、`Ran 26 tests in 2.156s`、`OK` だった。

【事実】実コーパス本文を含むJSON、Markdown比較、評価シートはリポジトリ外の `D:\inbox\crossrag-third-concept\` に置いた。リポジトリ内にはコーパス本文を複写していない。
