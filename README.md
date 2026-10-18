# CrossRAG

CrossRAGは、入力した概念から既存コーパスのモチーフを複数の数値的な見方で取り出し、文章やアイデアを考えるモデルへ別の出発点を渡す実験的なスキルです。検索を実行するのはローカルのPython、最終的な文章を書くのはスキルを使うホスト側のモデルです。

使う価値は、同じ問いを「似ているもの」「文脈と異なる方向」「概念ペアの向き」などで見直し、発想の材料を増やせることです。これは発想支援の候補検索であり、事実確認、因果関係の発見、創造性や通常のRAGに対する優位性を保証しません。

## アーキテクチャ

![CrossRAG architecture: ordered context concepts and a corpus share one encoder; similarity, projection-residual, oriented-wedge, and reproducible random ranking return JSON motifs to a separate host LLM.](docs/images/architecture.svg)

同じ文脈・コーパス・エンコーダに対して、類似、直交残差、向き付き楔積、再現可能なランダム順位の4手法を比較する構成です。図の平面と矢印は幾何学的な関係の模式図であり、高次元の楔積を一意な3次元の法線として扱うものではありません。直交残差は文脈全体が張る空間から計算し、楔積の向きは先頭2概念から計算します。先頭2概念を入れ替えると楔積成分の符号は反転しますが、重みを混ぜた最終順位全体が逆転するとは限りません。

既定はオフラインの文字 n-gram ハッシュ特徴で、意味埋め込みではありません。意味埋め込みは SentenceTransformer / E5 を使う任意の選択肢です。検索の出力は既存コーパスのモチーフと数値スコアを含む JSON で、創作文章や未検証の仮説を書くのは別のホストモデルです。順位は実験的なヒューリスティックであり、創造性や発見の向上を証明せず、モチーフも事実の証拠にはなりません。

図は実装 `93396d8f` に基づきます。[計算式と限界](skill/crossrag/references/methodology.md)も参照してください。

## 入っているもの

インストール単位は [`skill/crossrag`](skill/crossrag) です。このディレクトリを、そのままエージェントのスキル置き場へコピーします。

```text
skill/crossrag/
  SKILL.md
  requirements.txt
  agents/openai.yaml
  scripts/run.py        # CLI
  scripts/engine.py     # 検索本体
  references/methodology.md
  references/modes.md
  references/example-corpus.json
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

Python 3 と NumPy が必要です。既定のハッシュエンコーダはオフラインで動きます。これは文字 n-gram の特徴ベクトルであり、意味埋め込みではありません。

```powershell
python -m pip install -r skill/crossrag/requirements.txt
```

意味埋め込みを使う場合だけ、別途 `sentence-transformers` とモデルがローカルに必要です。初回はモデルのダウンロードが走ります。`--encoder semantic` を付けない限り、ネットワークも有料 API も使いません。

## 検索の実行

`skill/crossrag` をカレントにして実行します。

```powershell
Set-Location skill/crossrag
python scripts/run.py --context "犬、カレー、夕暮れ、帰宅" --mode creative --method wedge --limit 5
python scripts/run.py --context "記憶、法線、検索" --mode discovery --method orthogonal --limit 5
python scripts/run.py --context "犬、カレー" --mode creative --method similar --encoder semantic
python scripts/run.py --context "犬、カレー" --corpus references/example-corpus.json
```

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

## 4つの手法

同じエンコーダ、同じコーパス、同じ生成条件で比べてください。条件が違うと比較になりません。

- **similar**: 文脈ベクトルの平均と、候補文のコサイン類似度
- **orthogonal**: 文脈が張る空間への射影の残差。コサインの帯域と一貫性の重みを混ぜたヒューリスティック
- **wedge**: 向き付き双ベクトル `a∧b` と `u∧v` のコサイン。先頭2語の順序を入れ替えると符号が反転する。言語的な方向そのものではない
- **random**: 再現可能なハッシュ順位。対照用

式、限界、評価の置き方は [methodology.md](skill/crossrag/references/methodology.md)、創作と発見の出力の型は [modes.md](skill/crossrag/references/modes.md) にあります。

## 各ホストへの入れ方

配布・実行単位は`skill/crossrag`フォルダー全体です。`SKILL.md`だけを移さず、`scripts/`、`references/`、`requirements.txt`との相対位置を保ってください。ユーザー領域では、Codex・Gemini CLI・Grok Build・Piが共通して認識する`~/.agents/skills/crossrag`へ置けます。Claude Codeは`~/.claude/skills/crossrag`にも同じフォルダーが必要です。

リポジトリのルートからPowerShellで配置する例:

```powershell
$shared = Join-Path $HOME ".agents\skills\crossrag"
$claude = Join-Path $HOME ".claude\skills\crossrag"
New-Item -ItemType Directory -Force -Path $shared, $claude | Out-Null
Copy-Item -Path .\skill\crossrag\* -Destination $shared -Recurse -Force
Copy-Item -Path .\skill\crossrag\* -Destination $claude -Recurse -Force
```

macOS/Linuxでの例:

```bash
mkdir -p "$HOME/.agents/skills/crossrag" "$HOME/.claude/skills/crossrag"
cp -R skill/crossrag/. "$HOME/.agents/skills/crossrag/"
cp -R skill/crossrag/. "$HOME/.claude/skills/crossrag/"
```

| ホスト | ユーザー領域の配置先 | 認識確認・明示起動 |
| --- | --- | --- |
| Codex | `~/.agents/skills/crossrag/` | `/skills`で一覧、`$crossrag`で明示指定 |
| Claude Code | `~/.claude/skills/crossrag/` | `/crossrag`で明示起動 |
| Gemini CLI | `~/.agents/skills/crossrag/` または `~/.gemini/skills/crossrag/` | `/skills list`、必要なら`/skills reload` |
| Grok Build | `~/.agents/skills/crossrag/` または `~/.grok/skills/crossrag/` | `/skills`で一覧、`/crossrag`で明示起動 |
| Pi | `~/.agents/skills/crossrag/` または `~/.pi/agent/skills/crossrag/` | `/skill:crossrag`で明示起動 |

スキルを追加した後に一覧へ出ない場合は、ホストの再読込み・再起動方法を使ってください。プロジェクト単位に置く場合はホストごとに探索先が異なるため、各公式資料を確認してください。上表は配置形式の案内であり、実機でのスキル認識・実行を検証したという意味ではありません。

スクリプトを実行できないときは検索を実行したと装わず、必要なセットアップを伝えてください。言語だけで下書きする場合も、数値検索による候補を使っていないことを明示します。

## 出典

検索本体はCrossRAG Lab v0.1を基にした研究プロトタイプです。背景として[Geometric Deep Learning](https://arxiv.org/abs/1603.09320)と[Sentence-Transformersの類似度](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)を参照しています。

各ホストの探索場所と起動方法は変わることがあるため、利用前に公式資料も確認してください: [Codex](https://developers.openai.com/codex/skills)、[Claude Code](https://code.claude.com/docs/en/skills)、[Gemini CLI](https://geminicli.com/docs/cli/using-agent-skills/)、[Grok Build](https://docs.x.ai/build/features/skills-plugins-marketplaces)、[Pi](https://pi.dev/docs/latest/skills)。

## ライセンス

MIT License で公開しています。詳細は [LICENSE](LICENSE) を参照してください。

NumPy、Sentence-Transformers、利用する埋め込みモデルなどの外部依存物には、それぞれのライセンスが適用されます。
