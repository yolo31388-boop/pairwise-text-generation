"""
游戏内自然语言生成与文本生成系统 - 核心模块

功能：
- generate_text(): 基于语言模型的文本生成，支持采样与温度控制
- language_model(): 词表与词概率分布
- ngram(): 一元（unigram）与二元（bigram）N-gram 模型
- text_summarization(): 抽取式与生成式文本摘要
- text_classification(): 朴素贝叶斯文本分类与情感分析
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random
import re
import time


@dataclass
class Config:
    seed: Optional[int] = None
    smoothing: float = 1.0


_TOKEN_RE = re.compile(r"[一-鿿]|[A-Za-z0-9]+|[^\s]")
_SENTENCE_RE = re.compile(r"[^。！？.!?]+[。！？.!?]?")

# 摘要打分用的中英文停用词
_STOPWORDS = frozenset(
    "的 了 和 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 他 她 它 我们 你们 他们 以及 而且 但是 因为 所以 如果 虽然 通过 进行 可以 "
    "the a an and or but of to in on for with is are was were be been it this that as at by"
    .split()
)

# 情感分析词典
_POSITIVE_WORDS = frozenset(
    "好 棒 喜欢 爱 精彩 优秀 漂亮 开心 快乐 满意 推荐 赞 好玩 有趣 完美 惊喜 舒服 流畅 "
    "good great excellent amazing awesome love like best wonderful fantastic nice happy "
    "fun perfect brilliant enjoyable".split()
)
_NEGATIVE_WORDS = frozenset(
    "差 烂 糟 糟糕 讨厌 失望 无聊 难看 难玩 垃圾 坑 卡 崩溃 难 贵 慢 差劲 后悔 恶心 "
    "bad terrible awful boring hate worst poor disappointing broken slow ugly stupid".split()
)


def _tokenize(text: str) -> List[str]:
    """把文本切成词元：英文按单词，中文按单字。"""
    return _TOKEN_RE.findall(text.lower())


def _join_tokens(tokens: List[str]) -> str:
    """把词元拼回文本：中文直接相连，英文单词之间加空格。"""
    parts: List[str] = []
    for token in tokens:
        if parts and re.fullmatch(r"[A-Za-z0-9]+", token) and re.fullmatch(
            r"[A-Za-z0-9]+", parts[-1][-1]
        ):
            parts.append(" ")
        parts.append(token)
    return "".join(parts)


class TextGeneration:
    """游戏内自然语言生成与文本生成系统。"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._rng = random.Random(self.config.get("seed"))
        self._smoothing = float(self.config.get("smoothing", 1.0))
        # 语言模型 / N-gram 状态
        self._vocab: List[str] = []
        self._vocab_set = set()
        self._unigram_counts: Counter = Counter()
        self._bigram_counts: Dict[str, Counter] = defaultdict(Counter)
        self._total_tokens = 0
        # 朴素贝叶斯分类器状态
        self._class_doc_counts: Counter = Counter()
        self._class_token_counts: Dict[str, Counter] = defaultdict(Counter)
        self._class_total_tokens: Counter = Counter()
        self._classifier_vocab = set()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def update(self, dt: float):
        self._state["last_update"] = time.time()
        self._state["dt"] = dt

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 语料拟合（语言模型 / N-gram 共用）
    # ------------------------------------------------------------------
    def fit(self, corpus) -> "TextGeneration":
        """在语料上训练语言模型。corpus 可以是字符串或字符串列表。"""
        if isinstance(corpus, str):
            corpus = [corpus]
        for document in corpus:
            tokens = _tokenize(document)
            if not tokens:
                continue
            self._unigram_counts.update(tokens)
            self._total_tokens += len(tokens)
            padded = ["<s>"] + tokens + ["</s>"]
            for prev, cur in zip(padded, padded[1:]):
                self._bigram_counts[prev][cur] += 1
        self._vocab = sorted(self._unigram_counts)
        self._vocab_set = set(self._vocab)
        return self

    # ------------------------------------------------------------------
    # 语言模型：词表 + 概率分布
    # ------------------------------------------------------------------
    def language_model(self, corpus=None) -> Dict:
        """返回语言模型，包含词表与词的概率分布。"""
        if corpus is not None:
            self.fit(corpus)
        if not self._vocab:
            raise ValueError("语言模型尚未训练，请先调用 fit() 或传入语料")
        distribution = {
            token: count / self._total_tokens
            for token, count in self._unigram_counts.items()
        }
        return {
            "vocab": list(self._vocab),
            "vocab_size": len(self._vocab),
            "distribution": distribution,
            "total_tokens": self._total_tokens,
        }

    # ------------------------------------------------------------------
    # N-gram：一元模型与二元模型
    # ------------------------------------------------------------------
    def ngram(self, corpus=None, n: int = 2) -> Dict:
        """返回 N-gram 模型，n=1 为一元模型，n=2 为二元模型。"""
        if n not in (1, 2):
            raise ValueError("仅支持一元模型(n=1)与二元模型(n=2)")
        if corpus is not None:
            self.fit(corpus)
        if not self._vocab:
            raise ValueError("N-gram 模型尚未训练，请先调用 fit() 或传入语料")
        if n == 1:
            counts = dict(self._unigram_counts)
            total = self._total_tokens
            probabilities = {token: c / total for token, c in counts.items()}
            return {
                "n": 1,
                "counts": counts,
                "probabilities": probabilities,
                "total": total,
            }
        counts = {prev: dict(nexts) for prev, nexts in self._bigram_counts.items()}
        probabilities = {
            prev: {cur: c / sum(nexts.values()) for cur, c in nexts.items()}
            for prev, nexts in self._bigram_counts.items()
        }
        return {"n": 2, "counts": counts, "probabilities": probabilities}

    # ------------------------------------------------------------------
    # 文本生成：采样 + 温度
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_temperature(distribution: Dict[str, float], temperature: float) -> Dict[str, float]:
        """按温度调整概率分布：温度越低越确定，越高越随机。"""
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        adjusted = {t: p ** (1.0 / temperature) for t, p in distribution.items()}
        total = sum(adjusted.values())
        return {t: p / total for t, p in adjusted.items()}

    def _sample(self, distribution: Dict[str, float]) -> str:
        """按概率分布采样一个词元。"""
        threshold = self._rng.random()
        cumulative = 0.0
        last_token = None
        for token, prob in distribution.items():
            cumulative += prob
            last_token = token
            if threshold < cumulative:
                return token
        return last_token

    def _next_distribution(self, context: Optional[str]) -> Dict[str, float]:
        """给定上下文，返回下一个词的概率分布（bigram 回退到 unigram）。"""
        if context and context in self._bigram_counts:
            nexts = self._bigram_counts[context]
            vocab_size = len(self._vocab_set) + 1  # +1 给 </s>
            total = sum(nexts.values()) + self._smoothing * vocab_size
            distribution = {
                token: (count + self._smoothing) / total
                for token, count in nexts.items()
            }
            for token in self._vocab_set:
                distribution.setdefault(token, self._smoothing / total)
            distribution.setdefault("</s>", self._smoothing / total)
            return distribution
        return {
            token: count / self._total_tokens
            for token, count in self._unigram_counts.items()
        }

    def generate_text(
        self,
        prompt: str = "",
        max_length: int = 20,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        """从语言模型采样生成文本，temperature 控制随机性。"""
        if not self._vocab:
            raise ValueError("语言模型尚未训练，请先调用 fit()")
        rng = random.Random(seed) if seed is not None else self._rng
        prompt_tokens = _tokenize(prompt)
        generated: List[str] = []
        context = prompt_tokens[-1] if prompt_tokens else "<s>"
        for _ in range(max(1, max_length)):
            distribution = self._next_distribution(context)
            distribution = self._apply_temperature(distribution, temperature)
            threshold = rng.random()
            cumulative = 0.0
            token = None
            for candidate, prob in distribution.items():
                cumulative += prob
                token = candidate
                if threshold < cumulative:
                    break
            if token == "</s>":
                break
            generated.append(token)
            context = token
        result = _join_tokens(prompt_tokens + generated)
        self._history.append(
            {
                "type": "generate_text",
                "prompt": prompt,
                "temperature": temperature,
                "result": result,
                "time": time.time(),
            }
        )
        return result

    # ------------------------------------------------------------------
    # 文本摘要：抽取式 + 生成式
    # ------------------------------------------------------------------
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]

    def _extractive_summary(self, sentences: List[str], max_sentences: int) -> List[str]:
        """抽取式摘要：按词频给句子打分，选出得分最高的句子（保持原文顺序）。"""
        word_freq: Counter = Counter()
        for sentence in sentences:
            for token in _tokenize(sentence):
                if token not in _STOPWORDS and not re.fullmatch(r"[^\w一-鿿]", token):
                    word_freq[token] += 1
        scored = []
        for index, sentence in enumerate(sentences):
            tokens = [
                t for t in _tokenize(sentence)
                if t not in _STOPWORDS and not re.fullmatch(r"[^\w一-鿿]", t)
            ]
            score = sum(word_freq[t] for t in tokens) / max(1, len(tokens))
            scored.append((index, score))
        top = sorted(scored, key=lambda item: item[1], reverse=True)[:max_sentences]
        chosen = sorted(index for index, _ in top)
        return [sentences[i] for i in chosen]

    def _abstractive_summary(self, sentences: List[str], max_sentences: int) -> str:
        """生成式摘要：用训练好的语言模型围绕高频主题词生成新句子。"""
        word_freq: Counter = Counter()
        for sentence in sentences:
            for token in _tokenize(sentence):
                if token not in _STOPWORDS and re.fullmatch(r"[A-Za-z0-9一-鿿]+", token):
                    word_freq[token] += 1
        if not word_freq:
            return ""
        topic = word_freq.most_common(1)[0][0]
        if not self._vocab:
            self.fit(sentences)
        generated = self.generate_text(prompt=topic, max_length=16, temperature=0.8)
        return generated if generated else sentences[0]

    def text_summarization(
        self,
        text: str,
        max_sentences: int = 1,
        method: str = "extractive",
    ) -> str:
        """生成文本摘要，method 支持 extractive（抽取式）与 abstractive（生成式）。"""
        if method not in ("extractive", "abstractive"):
            raise ValueError("method 必须是 'extractive' 或 'abstractive'")
        sentences = self._split_sentences(text)
        if not sentences:
            return ""
        max_sentences = max(1, min(max_sentences, len(sentences)))
        if method == "extractive":
            summary = "".join(self._extractive_summary(sentences, max_sentences))
        else:
            summary = self._abstractive_summary(sentences, max_sentences)
        self._history.append(
            {"type": "text_summarization", "method": method, "result": summary, "time": time.time()}
        )
        return summary

    # ------------------------------------------------------------------
    # 文本分类：朴素贝叶斯 + 情感分析
    # ------------------------------------------------------------------
    def train_classifier(self, samples) -> "TextGeneration":
        """训练朴素贝叶斯分类器。samples 为 (文本, 类别) 列表。"""
        for text, label in samples:
            tokens = _tokenize(text)
            self._class_doc_counts[label] += 1
            self._class_token_counts[label].update(tokens)
            self._class_total_tokens[label] += len(tokens)
            self._classifier_vocab.update(tokens)
        return self

    def _nb_scores(self, text: str) -> Dict[str, float]:
        if not self._class_doc_counts:
            raise ValueError("分类器尚未训练，请先调用 train_classifier()")
        tokens = _tokenize(text)
        total_docs = sum(self._class_doc_counts.values())
        vocab_size = max(1, len(self._classifier_vocab))
        scores = {}
        for label, doc_count in self._class_doc_counts.items():
            log_prob = math.log(doc_count / total_docs)
            denominator = self._class_total_tokens[label] + vocab_size
            for token in tokens:
                count = self._class_token_counts[label].get(token, 0)
                log_prob += math.log((count + 1) / denominator)
            scores[label] = log_prob
        return scores

    def text_classification(self, text: str, labels: Optional[List[str]] = None) -> str:
        """朴素贝叶斯文本分类，返回最可能的类别。"""
        scores = self._nb_scores(text)
        if labels is not None:
            scores = {k: v for k, v in scores.items() if k in labels}
        if not scores:
            raise ValueError("没有可用的类别")
        return max(scores, key=scores.get)

    def sentiment_analysis(self, text: str) -> str:
        """情感分析：基于情感词典，返回 positive / negative / neutral。"""
        tokens = _tokenize(text)
        pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"
