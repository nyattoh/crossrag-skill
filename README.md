# CrossRAG

CrossRAGは、入力した概念から既存コーパスのモチーフを複数の数値的な見方で取り出し、文章やアイデアを考えるモデルへ別の出発点を渡す実験的なスキルです。検索を実行するのはローカルのPython、最終的な文章を書くのはスキルを使うホスト側のモデルです。

使う価値は、同じ問いを「似ているもの」「文脈と異なる方向」「概念ペアの向き」などで見直し、発想の材料を増やせることです。これは発想支援の候補検索であり、事実確認、因果関係の発見、創造性や通常のRAGに対する優位性を保証しません。

## アーキテクチャ

![CrossRAG architecture: ordered context concepts and a corpus share one encoder; similarity, projection-residual, oriented-wedge, and reproducible random ranking return JSON motifs to a separate host LLM.](docs/images/architecture.svg)

同じ文脈・コーパス・エンコーダに対して、類似、直交残差、向き付き楔積、再現可能なランダム順位などを使います。図の平面と矢印は幾何学的な関係の模式図であり、高次元の楔積を一意な3次元の法線として扱うものではありません。直交残差は文脈全体が張る空間から計算し、楔積の向きは先頭2概念から計算します。先頭2概念を入れ替えると楔積成分の符号は反転しますが、重みを混ぜた最終順位全体が逆転するとは限りません。

既定はオフラインの文字 n-gram ハッシュ特徴で、意味埋め込みではありません。意味埋め込みは SentenceTransformer / E5 を使う任意の選択肢です。検索の出力は既存コーパスのモチーフと数値スコアを含む JSON で、創作文章や未検証の仮説を書くのは別のホストモデルです。順位は実験的なヒューリスティックであり、創造性や発見の向上を証明せず、モチーフも事実の証拠にはなりません。

図は実装 `93396d8f` に基づきます。[計算式と限界](skill/crossrag/references/methodology.md)も参照してください。

## 入っているもの

インストール単位は [`skill/crossrag`](skill/crossrag) です。このディレクトリを、そのままエージェントのスキル置き場へコピーします。

```text
skill/crossrag/
  LICENSE
  SKILL.md
  requirements.txt
  agents/openai.yaml
  scripts/run.py        # CLI
  scripts/engine.py     # 検索本体
  scripts/third_concept.py
  scripts/third_concept_cli.py  # third-concept, normal-ray, random-orthogonal CLI
  scripts/serendipity_ray.py
  scripts/residual_ray.py
  scripts/compare_third_concept.py
  references/methodology.md
  references/modes.md
  references/example-corpus.json
  tests/                # 回帰テストと合成fixture
```

## 何に使うか

- 小説、詩、比喩、意外性のあるコピーを書くとき
- 分野をまたいだ着想や、変わった研究上のつながりを出すとき
- 同じコーパスで、類似検索と楔積検索の出方を比べるとき

事実の裏取りには使いません。取り出したモチーフは着想であり、証拠でも、そのまま写す文でもありません。順位はヒューリスティックで、通常のRAGより優れていることは示しません。

## まず一度使う

リポジトリのルートで、必要な依存関係を一度だけ入れます。

```powershell
python -m pip install -r skill/crossrag/requirements.txt
Set-Location skill/crossrag
python scripts/run.py --context "犬、カレー、夕暮れ" --method similar
```

この最短コマンドは、同梱の22件の日本語例コーパスとオフラインの文字n-gram hashで、類似度による順位を5件返します。JSONの`selected`に含まれる`line`が文章や発想の材料です。`hash`は意味を理解する埋め込みではなく、例コーパスも小さいため、実際の領域には自分のコーパスを指定してください。候補やスコアは事実の根拠ではありません。

`wedge`は既定の方式ですが、hashと短い語の組み合わせでは全候補のwedge成分が0になる場合があります。そのときは出力の`warnings`を読み、wedgeが順位に寄与していないことを明記してください。異なる方式を比べる場合は同じコーパス・エンコーダを使い、方式名だけで性能差を主張しないでください。

スキル対応ホストでは、たとえば「CrossRAGを使って、犬・カレー・夕暮れから一つの意外な関係を選び、短い物語を書いて。検索された文は写さないで」と依頼します。ホスト側のモデルがスキルを読み、検索を実行してから文章を作ります。調査の着想がほしい場合は、発見モードで候補を3つ、つながり・不確実性・確かめる最小テストとともに示すよう依頼します。

## 準備

Python 3.10 以上と NumPy が必要です。既定のハッシュエンコーダはオフラインで動きます。これは文字 n-gram の特徴ベクトルであり、意味埋め込みではありません。OS管理下のPython環境では仮想環境を使ってください。

```powershell
python -m pip install -r skill/crossrag/requirements.txt
```

macOS/Linuxでは、たとえば次のように仮想環境を使えます。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r skill/crossrag/requirements.txt
```

macOS/Linuxで`python`がPython 3.10以上を指さない場合は、`python3`で仮想環境を作成・有効化してから依存を入れてください。管理されたPython環境がpipによる直接インストールを拒否する場合も、仮想環境を使います。

意味埋め込みを使う場合だけ、別途 `sentence-transformers` とモデルがローカルに必要です。初回はモデルのダウンロードが走ります。`--encoder semantic` を付けない限り、ネットワークも有料 API も使いません。

## 検索の実行

`skill/crossrag` をカレントにして実行します。

```powershell
Set-Location skill/crossrag
python scripts/run.py --context "犬、カレー、夕暮れ、帰宅" --mode creative --method wedge --limit 5
python scripts/run.py --context "記憶、法線、検索" --mode discovery --method orthogonal --limit 5
python scripts/run.py --context "犬、カレー" --mode creative --method similar --encoder hash
python scripts/run.py --context "犬、カレー" --corpus references/example-corpus.json
python scripts/third_concept_cli.py --search-mode random-orthogonal --context "雨" --context "時計" --corpus references/example-corpus.json --seed 7 --ray-count 8 --serendipity 60
```

`--corpus`に相対パスを渡す場合は、`skill/crossrag`からの相対パスになります。リポジトリ外のユーザーコーパスには絶対パスを使ってください。`--encoder semantic`はsentence-transformersとモデルをローカルに用意した場合だけ指定します。

`random-orthogonal`は、複数回指定した`--context`の最初の埋め込みを原点にして、その文脈が張るアフィン部分空間の直交補空間へseed付きGaussian方向を投影し、コーパスを幾何条件で探索する実験モードです。少なくとも2個の`--context`を別々に指定します。`--seed`で方向を再現し、`--ray-count`で試す方向数、`--serendipity`（0〜100、既定60）で探索半径と光線コサイン条件を調整します。値に対応する法則の調子は、0–33が`grounded`（控えめな飛躍）、34–66が`playful`（奇妙な法則）、67–100が`absurd`（徹底的にデタラメな法則）です。境界は整数値で示した範囲に含めます。既定値は半径`0.25 × (0.5 + s)`、コサイン閾値`0.85 − 0.35 × s`（`s = serendipity / 100`）です。出力には再現可能な幾何クラスタとして`selected_clusters`と`combined_clusters`を含みます。候補行と出典情報をそのまま返し、意味的な関連性や品質は判定しません。光線の前向き距離`t`、原点からの半径距離`radial`、光線からの距離`ray_distance`を区別し、原点より後ろの候補は`behind_origin`として示します。有効なrankと1件以上の非除外候補行がある状態で光線チューブにhitがない場合は、最近傍の実在行をsingleton fallbackとして必ず返し、fallbackと明示してチューブhitには数えません。コーパスが空、候補がすべて除外、または有効な直交補空間を定められないrankの場合は、候補を捏造せず利用不能状態を返します。

既定のcosine閾値はserendipityから写像しますが、`--ray-cluster-cosine`を指定するとその値が明示的に優先されます。JSONの`params.mapped_cosine_threshold`は写像値、`params.cosine_threshold`は実際に使った閾値、`params.cosine_threshold_source`は採用元（`serendipity_mapping`または`explicit_override`）です。例えば`--ray-cluster-cosine 0.5`を付けた場合、写像値を併記したうえで有効値と採用元を記録します。

法則文を付けない場合、JSONは`awaiting`状態を示し、ホストモデル向けプロンプトに「`この世界が成立する奇妙な法則を一つ作れ`」という指示を含めます。ホストモデルはその指示から法則を作り、同じ引数に`--law`を加えて検索を再実行します。スクリプト自身はLLM/APIを呼びません。READMEの作例は説明用にホストモデルが生成した架空の法則であり、コーパスから検索された文や実行結果ではありません。

awaiting時の`host_prompt`は単なる文字列ではなくJSONオブジェクトです。`contexts`、`selected_clusters`、`tone`、`tone_instruction`、`guardrails`を構造化フィールドとして持ち、同じ内容をまとめた`prompt`文字列も含みます。`required_directive`にもホストへの必須指示を記録します。

架空のホスト生成法則の例（検索結果でも実行結果でもありません）: 「雨の音が聞こえるものだけが、過去を記憶できる。」再実行時には、この文を同じ検索引数へ`--law`として渡します。

```powershell
python scripts/third_concept_cli.py --search-mode random-orthogonal --context "雨" --context "時計" --corpus references/example-corpus.json --seed 7 --ray-count 8 --serendipity 60 --law "雨の音が聞こえるものだけが、過去を記憶できる。"
```

入力不足や不正な値はエラーとして扱います。文脈埋め込みから直交補空間を作れない場合は検索不能の状態を示し、候補がない場合と光線チューブに候補が入らない場合は区別します。有効なrankと非空の非除外候補があるのにチューブhitがない場合は、最近傍の実在行をsingleton fallbackとして必ず返し、fallbackと明示します。コーパスが空、候補がすべて除外、または有効な直交補空間を定められないrankの場合は、利用不能状態を返します。

To inspect corpus passages near rays estimated from residual clusters outside the affine plane through A, B and a source article C, use the experimental normal-ray CLI:

```powershell
python scripts/third_concept_cli.py --A "概念A" --B "概念B" --C "出典記事" --corpus references/example-corpus.json --encoder hash --search-mode ray --ray-radius 0.25 --ray-start-distance 0.25 --ray-cluster-cosine 0.7 --ray-min-cluster-size 2 --limit 5
```

既定の`residual-cluster`方式では、A・B・出典記事Cの埋め込みが定めるアフィン平面から、コーパスの残差を決定的なリーダー方式でクラスタ化し、各クラスタの平均残差方向に光線を定めます。Cは必須です。A・B・Cと全コーパス候補は、単一の`encoder.encode_passage`経路で同じ座標空間に埋め込みます。A・B・Cは入力テキスト、候補は整形済みコーパス行テキスト（ペア行なら例:`aとb。line`）を使い、この方式では`encode_query`を使いません。role prefixを使うエンコーダでは、既知の`query: `または`passage: `ロールprefixがテキスト先頭にあればmarkerとして取り除き、`encode_passage`が全てに同じpassage prefixを付けます。prefix-freeエンコーダではprefixを追加しません。ヒット結果にはコーパスの元のテキストを保持します。出力の`params.encoding_path`は`"encode_passage"`です。意味や関連性の判定、類似度による除外はありません。`--exclude`に一致する行だけを明示的に除外し、平面内に近い残差はクラスタ化できませんが、幾何条件を満たせば検索結果には現れます。A/B/Cに語彙的・意味的に近い行やC自身の行も対象です。光線近傍に条件を満たす実在行がなければ空になります。旧PCA×外積方式は`--ray-method pca-cross`で選べます。距離・クラスタ閾値はエンコーダ間で較正されていません。どちらの方向も意味的事実ではなく、有用性は評価されていません。

標準出力はUTF-8のJSONです。各候補に`cosine`、`orthogonal`、`wedge`と、選んだ手法の順位が入ります。`--mode`はホスト側モデルへ渡す創作・発見の指示、`--method`は候補を並べる数値方式を切り替えます。

| オプション | 意味 |
| --- | --- |
| `--context` | カンマ区切りの概念。2つ以上。楔積では先頭2語の順序が向きを決める |
| `--mode` | `creative`（創作）または `discovery`（発見）。既定は `creative` |
| `--method` | `similar` / `orthogonal` / `wedge` / `random`。既定は `wedge` |
| `--encoder` | `hash`（既定）または `semantic` |
| `--corpus` | `["概念1","概念2","着想"]` または `{"a","b","line"}` の JSON 配列 |
| `--limit` | 返す件数。既定は 5 |

自前のコーパスは、空でない文字列 3 つを 1 件とします。

## 実験例：料理の記事から、思いがけない方向へ（2026-10-21）

**試したかったこと:** 食材の実食記事から、料理の話を自然に広げるのではなく、元の話の腰を折るような遠いモチーフを拾えるか。元記事は非公開とし、ここでは「普段は脇役の食材が中心になり、火の通り方が違ってもおいしいと感じたが、その理由は分からない」という概要だけを扱います。商品名、記事名、元URL、全文は記載しません。

新しい `normal-ray` の `residual-cluster` 方式で、記事 C と二つの概念 A・B からアフィン平面 `C + span(A-C, B-C)` を作り、既存コーパスの候補を平面からの残差方向でクラスタ化しました。使用したのは **384次元の文字 n-gram hash** と、コード内の **22件の標準コーパス（`engine.DEFAULT_CORPUS`）** です。これは意味埋め込みではありません。なお、`references/example-corpus.json` は別の4件の例示データで、今回の試行には使っていません。

| 入力 | 標準設定（半径0.25、クラスタcosine 0.7） | 緩和設定（半径1.1、クラスタcosine 0.4） |
| --- | --- | --- |
| 元記事に基づく非公開入力 | `no_residual_clusters`／光線0本 | `hits_found`／光線6本 |
| 下記の公開可能な匿名概要入力 | `no_residual_clusters`／光線0本 | `hits_found`／光線4本 |

両方とも `--ray-min-cluster-size 2`（既定値）、`--limit 10` で実行しました。**6本という結果の厳密な再現には非公開の入力文が必要**です。文章を匿名化すると埋め込み座標が変わるため、結果も4本に変わりました。

### 匿名化した入力で試す

リポジトリのルートから `skill/crossrag` に移動し、次を実行します（PythonとNumPyが必要）。

```powershell
Set-Location skill/crossrag
$A = "料理の脇役だった食材が主役になる"
$B = "加熱の程度が違ってもおいしい"
$C = "ある料理の試食記事。これまで添え物だった食材が料理の中心になり、軽く火を通したときも、柔らかくなるまで加熱したときも魅力がある。なぜおいしいか説明できず、予想外の美味しさに驚いた。"

python scripts/third_concept_cli.py --A $A --B $B --C $C --search-mode ray --encoder hash --limit 10

python scripts/third_concept_cli.py --A $A --B $B --C $C --search-mode ray --encoder hash --ray-cluster-cosine 0.4 --ray-radius 1.1 --limit 10
```

匿名入力の緩和設定では、例えば次の**既存コーパス由来のペア**が同じ光線の検索結果に現れました（生成された文章ではありません）。

- **時計の針が葉をつける／影が時刻を進める**
- **夕暮れが帰る人を待つ／終電が言えなかった約束を運ぶ**
- **焼きたてのパンが秘密をふくらませる／閉館後の図書室で夢が貸し出される**

元記事を使った非公開入力の緩和設定では、このほか **雨粒と自転車**、**傘と猫の地図** などを含む6本の光線ができました。ここからホスト側モデルが、たとえば「昨日に取り残された家には、今日の鍵が使えない」という着想を書けますが、**これは検索結果そのものではなく、その後の創作**です。

### 分かったこと・分からないこと

- 条件次第で、出発記事から遠く感じられるモチーフを**既存データから抽出**できました。
- ただし緩和した半径は広く、クラスタやヒットが増えたことは「発見性能の向上」や「外積による意味的な発見」の証拠ではありません。このモードは高次元の `A×B` 法線を計算するのではなく、**コーパス残差から光線方向を推定**します。
- 結果は小規模な創作用コーパス、文字特徴、入力文、閾値、候補の並び順に依存します。意味的な意外性、有用性、再現性、通常の検索やランダム選択に対する優位性は未評価です。

次の検証課題は、同一条件での `similar`／`orthogonal`／`random` などとの比較、意味埋め込み・大規模コーパスへの切替、人間による「話の腰を折られた度合い」の評価です。

## 検索方式

同じエンコーダ、同じコーパス、同じ生成条件で比べてください。条件が違うと比較になりません。

- **similar**: 文脈ベクトルの平均と、候補文のコサイン類似度
- **orthogonal**: 文脈が張る空間への射影の残差。コサインの帯域と一貫性の重みを混ぜたヒューリスティック
- **wedge**: 向き付き双ベクトル `a∧b` と `u∧v` のコサイン。先頭2語の順序を入れ替えると符号が反転する。言語的な方向そのものではない
- **random**: 再現可能なハッシュ順位。対照用
- **random-orthogonal**: seed付きランダム方向を文脈部分空間の直交補空間へ投影する実験的な幾何探索。意味検索や品質評価ではない

式、限界、評価の置き方は [methodology.md](skill/crossrag/references/methodology.md)、創作と発見の出力の型は [modes.md](skill/crossrag/references/modes.md) にあります。

## 各ホストへの入れ方

配布・実行単位は`skill/crossrag`フォルダー全体です。`SKILL.md`だけを移さず、`scripts/`、`references/`、`requirements.txt`、`LICENSE`との相対位置を保ってください。Codex・Gemini CLI・Grok Build・Piのユーザー共通パスは`~/.agents/skills/crossrag`です。Claude Codeのユーザーパスは`~/.claude/skills/crossrag`で、`.agents/skills`は探索先に含まれません。Codex向けの`agents/openai.yaml`はCodex UI用の任意メタデータです。

リポジトリのルートからPowerShellで配置する例:

```powershell
$shared = Join-Path $HOME ".agents\skills\crossrag"
$claude = Join-Path $HOME ".claude\skills\crossrag"
New-Item -ItemType Directory -Force -Path $shared, $claude | Out-Null
robocopy .\skill\crossrag $shared /E /XD .ruff_cache __pycache__ .pytest_cache
if ($LASTEXITCODE -ge 8) { throw "CrossRAG shared install failed: $LASTEXITCODE" }
robocopy .\skill\crossrag $claude /E /XD .ruff_cache __pycache__ .pytest_cache
if ($LASTEXITCODE -ge 8) { throw "CrossRAG Claude install failed: $LASTEXITCODE" }
```

macOS/Linuxでの例:

```bash
mkdir -p "$HOME/.agents/skills/crossrag" "$HOME/.claude/skills/crossrag"
rsync -a --exclude='.ruff_cache/' --exclude='__pycache__/' --exclude='.pytest_cache/' skill/crossrag/ "$HOME/.agents/skills/crossrag/"
rsync -a --exclude='.ruff_cache/' --exclude='__pycache__/' --exclude='.pytest_cache/' skill/crossrag/ "$HOME/.claude/skills/crossrag/"
```

| ホスト | ユーザー領域の配置先 | 認識確認・明示起動 |
| --- | --- | --- |
| Codex | `~/.agents/skills/crossrag/` | `/skills`で一覧、`$crossrag`で明示起動。表示されなければ再起動 |
| Claude Code | `~/.claude/skills/crossrag/` | `/skills`で一覧、`/crossrag`で明示起動。新しいskill directoryをセッション中に追加したら`/reload-skills` |
| Gemini CLI | `~/.agents/skills/crossrag/`または`~/.gemini/skills/crossrag/` | `gemini skills list`または`/skills list`で確認し、説明に合う依頼を自然文で送る。個別skillのslash起動は公式CLI資料に記載なし。`/skills reload`で再読込 |
| Grok Build | `~/.agents/skills/crossrag/`または`~/.grok/skills/crossrag/` | `/skills`で一覧、`/crossrag`で明示起動。表示されなければ再起動 |
| Pi | `~/.agents/skills/crossrag/`または`~/.pi/agent/skills/crossrag/` | `/skill:crossrag`で明示起動、`/reload`で再読込 |

上記は各ホストの公式探索先とコマンドに基づく案内です。プロジェクト領域の場所や信頼設定はホストごとに異なります。配布・設定の説明は、各ホストで実際にスキルがロードされ検索が動作したことを保証しません。実機確認の範囲は、完了報告で分けて示します。

同じホストで複数の探索先に同名のスキルを置くと、重複または優先順位の警告が出る場合があります。共有パスとホスト固有パスのどちらか一方を使ってください。Gemini CLIでは`gemini skills link <path>`も選べますが、共有パスへのコピーと同時には使わないでください。

スクリプトを実行できないときは検索を実行したと装わず、必要なセットアップを伝えてください。言語だけで下書きする場合も、数値検索による候補を使っていないことを明示します。

## 出典

検索本体はCrossRAG Lab v0.1を基にした研究プロトタイプです。背景として[Geometric Deep Learning](https://arxiv.org/abs/1603.09320)と[Sentence-Transformersの類似度](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)を参照しています。

各ホストの探索場所と起動方法は変わるため、利用前に[Codex](https://developers.openai.com/codex/skills)、[Claude Code](https://code.claude.com/docs/en/skills)、[Gemini CLI](https://geminicli.com/docs/cli/using-agent-skills/)、[Grok Build](https://docs.x.ai/build/features/skills-plugins-marketplaces)、[Pi](https://pi.dev/docs/latest/skills)の公式資料も確認してください。

## ライセンス

MIT License で公開しています。詳細は [LICENSE](LICENSE) を参照してください。

NumPy、Sentence-Transformers、利用する埋め込みモデルなどの外部依存物には、それぞれのライセンスが適用されます。
