# CrossRAG

創作と発見のための、実験的な幾何学検索スキルです。既存の概念コーパスに対して、類似・直交残差・向き付き楔積（外積）・ランダムの4通りで候補を取り出し、同じ条件で比べられます。

検索はローカルの Python が数値で行います。文章を書くのは、結果を受け取った側のモデルです。楔積そのものが文章を生成するわけではなく、創造性が向上したという証明にもなりません。

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

事実の裏取りには使いません。取り出したモチーフは着想であり、証拠でも、そのまま写す文でもありません。順位はヒューリスティックで、通常の RAG より優れていることは示しません。

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

標準出力は UTF-8 の JSON です。各候補に `cosine`、`orthogonal`、`wedge` と、選んだ手法の順位が入ります。

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

## エージェントへの入れ方

`skill/crossrag` を、ホストのスキルディレクトリへフォルダごと置きます。相対パス（`scripts/` と `references/`）を保つため、中だけをばらして置かないでください。

スキルは実行できるとき、楔積と類似を同じコーパスで取り、そのモチーフを着想として文章または仮説にします。スクリプトが動かないときは、数学的な検索をしたと書かず、言語だけの下書きだと明示します。

## 出典

検索本体は CrossRAG Lab v0.1 を基にした研究プロトタイプです。背景として [Geometric Deep Learning](https://arxiv.org/abs/1603.09320) と [Sentence-Transformers の類似度](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html) を参照しています。
