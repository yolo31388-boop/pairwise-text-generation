"""
游戏内自然语言生成与文本生成系统 - 核心模块

提供：
- generate_text()        基于语言模型的文本生成（采样 + 温度）
- language_model()       语言模型（词表 + 概率分布）
- ngram()                N-gram 模型（一元 / 二元）
- text_summarization()   文本摘要（抽取式 / 生成式）
- text_classification()  文本分类（朴素贝叶斯 / 情感分析）
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math
import random
import re
import time

START_TOKEN = "<s>"
END_TOKEN = "</s>"

_SENTENCE_SPLIT_RE = re.compile(r"[^.!?。！？]+[.!?。！？]?")
_WORD_RE = re.compile(r"[A-Za-z0-9']+|[一-鿿]")

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome", "fantastic",
    "wonderful", "love", "loved", "like", "liked", "best", "fun",
    "happy", "brilliant", "perfect", "beautiful", "enjoy", "enjoyed",
    "win", "winner", "epic", "nice", "cool",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "hated",
    "boring", "broken", "bug", "bugs", "buggy", "crash", "crashes",
    "sad", "angry", "poor", "disappointing", "disappointed", "fail",
    "failed", "lag", "laggy", "ugly", "stupid",
}


def tokenize(text: str) -> List[str]:
    """把文本切成词 token（小写）。"""
    return _WORD_RE.findall(text.lower())


def detokenize(tokens: Sequence[str]) -> str:
    """把 token 序列还原为可读文本。"""
    return " ".join(tokens)


def split_sentences(text: str) -> List[str]:
    """按句子边界切分文本。"""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.findall(text) if s.strip()]


def _apply_temperature(distribution: Dict[str, float], temperature: float) -> Dict[str, float]:
    """按温度缩放概率分布。temperature=0 退化为贪心（argmax）。"""
    if not distribution:
        return {}
    if temperature <= 0:
        best = max(distribution, key=distribution.get)
        return {best: 1.0}
    scaled = {w: p ** (1.0 / temperature) for w, p in distribution.items() if p > 0}
    total = sum(scaled.values())
    if total <= 0:
        uniform = 1.0 / len(distribution)
        return dict.fromkeys(distribution, uniform)
    return {w: p / total for w, p in scaled.items()}


def sample_from_distribution(
    distribution: Dict[str, float],
    temperature: float = 1.0,
    rng: Optional[random.Random] = None,
) -> str:
    """按温度从概率分布中采样一个词。"""
    rng = rng or random
    scaled = _apply_temperature(distribution, temperature)
    words = list(scaled.keys())
    weights = [scaled[w] for w in words]
    return rng.choices(words, weights=weights, k=1)[0]


class NgramModel:
    """N-gram 语言模型，支持一元（n=1）和二元（n=2），带加一平滑。"""

    def __init__(self, n: int = 1, smoothing: float = 1.0):
        if n < 1:
            raise ValueError("n 必须 >= 1")
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.vocabulary: set = set()

    def train(self, corpus: Sequence) -> "NgramModel":
        """在语料上训练。corpus 元素可以是句子字符串或 token 列表。"""
        for item in corpus:
            tokens = tokenize(item) if isinstance(item, str) else list(item)
            if not tokens:
                continue
            padded = [START_TOKEN] * (self.n - 1) + tokens + [END_TOKEN]
            self.vocabulary.update(tokens)
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                self.ngram_counts[ngram] += 1
                self.context_counts[ngram[:-1]] += 1
        return self

    @property
    def vocab_size(self) -> int:
        # 分布空间包含句末符 </s>
        return len(self.vocabulary) + 1

    def _normalize_context(self, context: Sequence[str]) -> Tuple[str, ...]:
        context = tuple(context)[-(self.n - 1):] if self.n > 1 else ()
        if self.n > 1 and len(context) < self.n - 1:
            context = (START_TOKEN,) * (self.n - 1 - len(context)) + context
        return context

    def probability(self, word: str, context: Sequence[str] = ()) -> float:
        """P(word | context)，带加一平滑。"""
        context = self._normalize_context(context)
        count = self.ngram_counts.get(context + (word,), 0)
        context_total = self.context_counts.get(context, 0)
        return (count + self.smoothing) / (context_total + self.smoothing * self.vocab_size)

    def probability_distribution(self, context: Sequence[str] = ()) -> Dict[str, float]:
        """给定上下文，返回下一个词的概率分布（和为 1）。"""
        context = self._normalize_context(context)
        candidates = sorted(self.vocabulary) + [END_TOKEN]
        dist = {w: self.probability(w, context) for w in candidates}
        total = sum(dist.values())
        if total > 0:
            dist = {w: p / total for w, p in dist.items()}
        return dist

    def sample_next(
        self,
        context: Sequence[str] = (),
        temperature: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> str:
        """从下一个词的概率分布中按温度采样。"""
        return sample_from_distribution(
            self.probability_distribution(context), temperature=temperature, rng=rng
        )

    def generate(
        self,
        prompt: str = "",
        max_length: int = 20,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        """从模型生成文本：采样 + 温度控制。"""
        rng = random.Random(seed)
        tokens = tokenize(prompt)
        for _ in range(max_length):
            word = self.sample_next(tokens, temperature=temperature, rng=rng)
            if word == END_TOKEN:
                break
            tokens.append(word)
        return detokenize(tokens)


class NaiveBayesClassifier:
    """多项式朴素贝叶斯文本分类器（拉普拉斯平滑）。"""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.class_counts: Counter = Counter()
        self.word_counts: Dict[str, Counter] = defaultdict(Counter)
        self.vocabulary: set = set()

    def train(self, examples: Sequence[Tuple[str, str]]) -> "NaiveBayesClassifier":
        for text, label in examples:
            self.class_counts[label] += 1
            for token in tokenize(text):
                self.word_counts[label][token] += 1
                self.vocabulary.add(token)
        return self

    @property
    def labels(self) -> List[str]:
        return sorted(self.class_counts)

    def predict(self, text: str) -> str:
        scores = self.scores(text)
        if not scores:
            raise ValueError("分类器尚未训练")
        return max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def scores(self, text: str) -> Dict[str, float]:
        tokens = tokenize(text)
        total_docs = sum(self.class_counts.values())
        if total_docs == 0:
            return {}
        vocab_size = max(len(self.vocabulary), 1)
        result = {}
        for label, class_count in self.class_counts.items():
            log_prob = math.log(class_count / total_docs)
            word_total = sum(self.word_counts[label].values())
            denom = word_total + self.smoothing * vocab_size
            for token in tokens:
                count = self.word_counts[label].get(token, 0)
                log_prob += math.log((count + self.smoothing) / denom)
            result[label] = log_prob
        return result


@dataclass
class Config:
    pass


class TextGeneration:
    """游戏内自然语言生成与文本生成系统。"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._language_model: Optional[NgramModel] = None
        self._classifier: Optional[NaiveBayesClassifier] = None
        corpus = self.config.get("corpus")
        if corpus:
            self.language_model(corpus)
        training_data = self.config.get("training_data")
        if training_data:
            self.train_classifier(training_data)

    def update(self, dt: float):
        self._state["last_update"] = time.time()

    def reset(self):
        self._state = {}
        self._history = []
        self._language_model = None
        self._classifier = None

    # ------------------------------------------------------------------
    # 语言模型
    # ------------------------------------------------------------------
    def language_model(self, corpus: Optional[Sequence] = None, n: int = 2) -> NgramModel:
        """构建（或返回）语言模型：带词表和概率分布。"""
        if corpus is not None:
            self._language_model = NgramModel(n=n).train(corpus)
        if self._language_model is None:
            raise ValueError("尚未提供语料，无法构建语言模型")
        return self._language_model

    # ------------------------------------------------------------------
    # N-gram
    # ------------------------------------------------------------------
    def ngram(self, corpus: Sequence, n: int = 1) -> NgramModel:
        """训练 N-gram 模型：n=1 为一元模型，n=2 为二元模型。"""
        model = NgramModel(n=n).train(corpus)
        self._history.append(("ngram", n))
        return model

    # ------------------------------------------------------------------
    # 文本生成
    # ------------------------------------------------------------------
    def generate_text(
        self,
        prompt: str = "",
        max_length: int = 20,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        corpus: Optional[Sequence] = None,
    ) -> str:
        """从语言模型采样生成文本，temperature 控制随机性。"""
        model = self.language_model(corpus) if corpus is not None else self.language_model()
        text = model.generate(prompt=prompt, max_length=max_length,
                              temperature=temperature, seed=seed)
        self._history.append(("generate_text", text))
        return text

    # ------------------------------------------------------------------
    # 文本摘要
    # ------------------------------------------------------------------
    def text_summarization(
        self,
        text: str,
        method: str = "extractive",
        num_sentences: int = 1,
        max_keywords: int = 5,
    ) -> str:
        """生成摘要。method="extractive" 抽取式，method="abstractive" 生成式。"""
        if method == "extractive":
            summary = self._extractive_summary(text, num_sentences)
        elif method == "abstractive":
            summary = self._abstractive_summary(text, max_keywords)
        else:
            raise ValueError(f"未知摘要方法: {method}")
        self._history.append(("summarize", method))
        return summary

    def _extractive_summary(self, text: str, num_sentences: int) -> str:
        """抽取式摘要：按词频给句子打分，选出最重要的句子。"""
        sentences = split_sentences(text)
        if not sentences:
            return ""
        freq = Counter(tokenize(text))
        scored = []
        for index, sentence in enumerate(sentences):
            tokens = tokenize(sentence)
            score = sum(freq[t] for t in tokens) / max(len(tokens), 1)
            scored.append((index, score, sentence))
        top = sorted(scored, key=lambda item: (-item[1], item[0]))[:num_sentences]
        top.sort(key=lambda item: item[0])
        return " ".join(sentence for _, _, sentence in top)

    def _abstractive_summary(self, text: str, max_keywords: int) -> str:
        """生成式摘要：提取关键词并重新组织成新句子。"""
        freq = Counter(tokenize(text))
        if not freq:
            return ""
        keywords = [w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:max_keywords]]
        if len(keywords) == 1:
            body = keywords[0]
        else:
            body = ", ".join(keywords[:-1]) + " and " + keywords[-1]
        return f"In summary, the text is about {body}."

    # ------------------------------------------------------------------
    # 文本分类
    # ------------------------------------------------------------------
    def train_classifier(self, examples: Sequence[Tuple[str, str]]) -> NaiveBayesClassifier:
        """用 (文本, 标签) 样本训练朴素贝叶斯分类器。"""
        self._classifier = NaiveBayesClassifier().train(examples)
        return self._classifier

    def text_classification(
        self,
        text: str,
        method: str = "naive_bayes",
        training_data: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        """文本分类。method="naive_bayes" 朴素贝叶斯，method="sentiment" 情感分析。"""
        if method == "naive_bayes":
            classifier = self.train_classifier(training_data) if training_data is not None else self._classifier
            if classifier is None:
                raise ValueError("朴素贝叶斯分类器尚未训练")
            result = classifier.predict(text)
        elif method == "sentiment":
            result = self._sentiment_analysis(text)
        else:
            raise ValueError(f"未知分类方法: {method}")
        self._history.append(("classify", method, result))
        return result

    def _sentiment_analysis(self, text: str) -> str:
        """基于情感词典的情感分析，返回 positive / negative / neutral。"""
        tokens = tokenize(text)
        pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"
