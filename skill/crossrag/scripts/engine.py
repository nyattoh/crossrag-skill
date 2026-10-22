"""CrossRAG: experimental creative retrieval using vectors and oriented bivectors.
This is a research prototype, NOT evidence that exterior algebra improves prose.
"""
from __future__ import annotations
import hashlib
import math
import re
from functools import lru_cache
import numpy as np

DEFAULT_CORPUS = [
 ('湯気','占い','湯気の動きから、まだ帰らない人の足取りを占う。'),
 ('犬','留守番','犬は家族が戻るまで、玄関の時間を守っている。'),
 ('台所','記憶','台所には、もういない人の癖が残っている。'),
 ('スプーン','羅針盤','スプーンが小さな羅針盤になり、帰り道を指す。'),
 ('雨','手紙','雨粒は窓に届く、宛先のない手紙だった。'),
 ('影','時計','伸びた影が、誰にも知らせず時刻を進める。'),
 ('電車','約束','終電が運ぶのは乗客だけでなく、言えなかった約束だ。'),
 ('カレー','郷愁','煮込む香りには、帰る場所の記憶が混ざる。'),
 ('猫','地図','猫の歩いた場所だけが、街の裏側の地図になる。'),
 ('窓','海','窓の向こうに、朝だけ現れる見えない海がある。'),
 ('靴','旅','靴は履き主より先に旅の続きを知っている。'),
 ('星','郵便','星がひと晩遅れの便りを届ける。'),
 ('パン','秘密','焼きたてのパンが、誰かの秘密をふくらませる。'),
 ('時計','植物','時計の針が、静かに葉をつけ始めた。'),
 ('橋','名前','古い橋だけが、忘れられた名前を覚えている。'),
 ('雪','足音','雪が足音を集めて、冬の辞書を作る。'),
 ('自転車','風','自転車は風を押して、知らない町へ出た。'),
 ('図書室','夢','閉館後の図書室で、夢だけが貸し出される。'),
 ('傘','月','傘をひらくと、月の裏側が覗いた。'),
 ('魚','夢','魚は眠るたび、水の外の街を夢に見る。'),
 ('鍵','夕暮れ','鍵穴に夕暮れがつまって、扉が開かなかった。'),
 ('夕暮れ','約束','夕暮れは、帰ると言った人だけを待っていた。'),
]

RANKING_METRICS = {
    'similar': 'cosine',
    'orthogonal': 'orthogonal_residual',
    'wedge': 'gram_bivector_cos',
}

CORPUS_SOURCES = {'demo', 'explicit', 'conversation'}


def normalise(a):
    a = np.asarray(a, dtype=np.float64)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else np.zeros_like(a)


def cosine(a, b):
    return float(np.dot(normalise(a), normalise(b)))


class HashEncoder:
    """Offline character-ngram feature vectors; *not semantic embeddings*."""
    name = 'hash-ngram（文字特徴・意味理解ではない）'
    def __init__(self, dim=384):
        self.dim = dim
    @lru_cache(maxsize=8192)
    def _cached(self, text):
        text = re.sub(r'\s+', '', text.strip().lower())
        vec = np.zeros(self.dim, dtype=np.float64)
        for n, weight in ((1, 0.35), (2, 1.0), (3, 0.7)):
            for i in range(max(0, len(text) - n + 1)):
                gram = text[i:i + n]
                digest = hashlib.blake2b(gram.encode('utf-8'), digest_size=8).digest()
                h = int.from_bytes(digest, 'little')
                vec[h % self.dim] += weight * (1 if (h >> 9) & 1 else -1)
        return tuple(normalise(vec))
    def encode(self, texts):
        return np.stack([np.asarray(self._cached(t)) for t in texts])


class SentenceEncoder:
    name = ''
    def __init__(self, model_name='intfloat/multilingual-e5-small'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name + '（意味埋め込み）'
    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)


def gram_bivector_cos(a, b, u, v):
    """Cosine between a wedge b and u wedge v, without constructing d^2 matrices."""
    aa = cosine(a, b)
    uv = cosine(u, v)
    denom = math.sqrt(max(0, 1 - aa * aa) * max(0, 1 - uv * uv))
    if denom < 1e-8:
        return 0.
    return float(np.clip((cosine(a, u) * cosine(b, v) - cosine(a, v) * cosine(b, u)) / denom, -1, 1))


def orthogonal_residual(candidate, anchors):
    """Residual length after projection onto span of the context anchors."""
    anchors = np.asarray(anchors)
    # QR/SVD tolerates linearly dependent anchors
    _, sing, vt = np.linalg.svd(anchors, full_matrices=False)
    rank = int(np.sum(sing > 1e-7))
    if rank == 0:
        return 1.
    basis = vt[:rank]
    return float(np.clip(np.linalg.norm(normalise(candidate) - normalise(candidate) @ basis.T @ basis), 0, 1))


def run(context, corpus=None, mode='orthogonal', limit=5, encoder=None, use_demo_corpus=False, corpus_source=None):
    if corpus is None:
        if not use_demo_corpus:
            raise ValueError('corpus が指定されていません。会話データまたはコーパスを渡すか、デモ実行時のみ use_demo_corpus=True を明示してください。')
        corpus = DEFAULT_CORPUS
        resolved_source = 'demo' if corpus_source is None else corpus_source
    else:
        resolved_source = 'explicit' if corpus_source is None else corpus_source
    if resolved_source not in CORPUS_SOURCES:
        raise ValueError(f'unknown corpus_source: {resolved_source}')
    if len(corpus) < 2:
        raise ValueError('素材は少なくとも2件必要です')
    if mode not in RANKING_METRICS:
        raise ValueError('unknown mode')
    words = [s.strip() for s in re.split(r'[、,，/／\s]+', context) if s.strip()]
    if len(words) < 2:
        raise ValueError('コンテキストを「犬、カレー、夕暮れ」のように2語以上入力してください')
    enc = encoder or HashEncoder()
    ranking_metric = RANKING_METRICS[mode]
    all_text = words + [f'{a}と{b}。{line}' for a, b, line in corpus] + [x for a, b, _ in corpus for x in (a, b)]
    vectors = enc.encode(all_text)
    anchors = vectors[:len(words)]
    candidate_vectors = vectors[len(words):len(words) + len(corpus)]
    pair_vectors = vectors[len(words) + len(corpus):].reshape(len(corpus), 2, -1)
    ctx = normalise(np.mean(anchors, axis=0))
    rows = []
    raw_wedges = []
    for i, ((a, b, line), vec, pair) in enumerate(zip(corpus, candidate_vectors, pair_vectors)):
        sim = cosine(ctx, vec)
        orth = orthogonal_residual(vec, anchors)
        wedge = gram_bivector_cos(anchors[0], anchors[1], pair[0], pair[1])
        raw_wedges.append(wedge)
        if mode == 'similar':
            rank_value = sim
        elif mode == 'orthogonal':
            rank_value = orth
        else:
            rank_value = wedge
        rows.append({
            'index': i,
            'a': a,
            'b': b,
            'line': line,
            'rank_value': round(float(rank_value), 5),
            'ranking_metric': ranking_metric,
            'cosine': round(sim, 5),
            'orthogonal': round(orth, 5),
            'wedge': round(wedge, 5),
        })
    rows.sort(key=lambda x: (-x['rank_value'], x['index']))
    selected = rows[:int(np.clip(limit, 1, 20))]
    warnings = []
    # ponytail: report the degenerate wedge term instead of fixing the encoder; checked on unrounded values over all candidates
    if mode == 'wedge' and all(abs(w) < 1e-12 for w in raw_wedges):
        warnings.append(f'wedge は全候補で0です（エンコーダ: {enc.name}）。wedge の項が順位に効いておらず、順位は入力インデックス順と同じです。')
    return {
        'context': words,
        'mode': mode,
        'ranking_metric': ranking_metric,
        'corpus_source': resolved_source,
        'encoder': enc.name,
        'selected': selected,
        'total': len(corpus),
        'warnings': warnings,
        'note': '順位は指定された単一指標のみで決定。複合スコアは使用しない',
    }


def make_seed_text(result):
    """Explicitly a deterministic collage, not a language model generation."""
    words = result['context']
    selected = result['selected']
    if not selected:
        return ''
    return f'「{"、".join(words)}」という言葉が残っていた。\n{selected[0]["line"]}\n{selected[1]["line"] if len(selected) > 1 else ""}'
