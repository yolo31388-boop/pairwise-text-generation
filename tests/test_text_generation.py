"""
游戏内自然语言生成与文本生成系统 - 验收测试
运行方式：python -m pytest tests/test_text_generation.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.text_generation import TextGeneration, NgramModel, NaiveBayesClassifier, END_TOKEN


class TestTextGeneration:
    def setup_method(self):
        self.corpus = [
            "the dragon attacks the hero",
            "the hero blocks the dragon",
            "the dragon flees from the hero",
            "the hero defeats the dragon",
        ]
        self.training_data = [
            ("great game fun love it", "positive"),
            ("amazing epic awesome win", "positive"),
            ("terrible broken game hate it", "negative"),
            ("bad boring laggy worst game", "negative"),
        ]
        self.system = TextGeneration({"corpus": self.corpus, "training_data": self.training_data})

    # ---- 文本生成：采样 + 温度 ----
    def test_case_01(self):
        text = self.system.generate_text(seed=42)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_case_02(self):
        a = self.system.generate_text(prompt="the", temperature=1.0, seed=7)
        b = self.system.generate_text(prompt="the", temperature=1.0, seed=7)
        c = self.system.generate_text(prompt="the", temperature=1.0, seed=99)
        assert a == b
        assert isinstance(c, str)

    def test_case_03(self):
        greedy1 = self.system.generate_text(prompt="the", temperature=0.0, seed=1)
        greedy2 = self.system.generate_text(prompt="the", temperature=0.0, seed=2)
        assert greedy1 == greedy2
        model = self.system.language_model()
        assert model.sample_next(("the",), temperature=0.0) == \
            max(model.probability_distribution(("the",)).items(), key=lambda kv: kv[1])[0]

    # ---- 语言模型：词表 + 概率分布 ----
    def test_case_04(self):
        model = self.system.language_model()
        assert {"dragon", "hero", "attacks"} <= model.vocabulary
        assert model.vocab_size == len(model.vocabulary) + 1

    def test_case_05(self):
        model = self.system.language_model()
        dist = model.probability_distribution(("the",))
        assert set(dist) == set(model.vocabulary) | {END_TOKEN}
        assert all(0.0 < p <= 1.0 for p in dist.values())
        assert abs(sum(dist.values()) - 1.0) < 1e-9

    def test_case_06(self):
        model = self.system.language_model()
        dist = model.probability_distribution(("the",))
        assert "dragon" in dist
        assert model.probability("dragon", ("the",)) > 0

    # ---- N-gram：一元 + 二元 ----
    def test_case_07(self):
        uni = self.system.ngram(self.corpus, n=1)
        assert isinstance(uni, NgramModel) and uni.n == 1
        assert uni.ngram_counts[("dragon",)] == 4
        assert abs(uni.probability("dragon") - (4 + 1) /
                   (sum(uni.context_counts.values()) + uni.vocab_size)) < 1e-9

    def test_case_08(self):
        bi = self.system.ngram(self.corpus, n=2)
        assert bi.n == 2
        assert bi.ngram_counts[("the", "dragon")] == 4
        assert bi.ngram_counts[("the", "hero")] == 4

    def test_case_09(self):
        bi = self.system.ngram(self.corpus, n=2)
        dist = bi.probability_distribution(("the",))
        assert abs(sum(dist.values()) - 1.0) < 1e-9
        assert dist["dragon"] > dist["flees"]

    # ---- 文本摘要：抽取式 + 生成式 ----
    def test_case_10(self):
        text = ("The dragon attacks the village. The hero defeats the dragon. "
                "Everyone celebrates the brave hero.")
        summary = self.system.text_summarization(text, method="extractive", num_sentences=1)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        assert summary.rstrip(".") + "." in text or summary in sentences
        assert len(summary) <= len(text)

    def test_case_11(self):
        text = ("The dragon attacks the village with fire. The brave hero defeats "
                "the dragon and saves everyone. The village celebrates the hero.")
        summary = self.system.text_summarization(text, method="abstractive")
        assert isinstance(summary, str) and summary
        assert 0 < len(summary) < len(text)

    # ---- 文本分类：朴素贝叶斯 + 情感分析 ----
    def test_case_12(self):
        assert self.system.text_classification(
            "amazing fun game love it", method="naive_bayes") == "positive"
        assert self.system.text_classification(
            "terrible broken game hate it", method="naive_bayes") == "negative"

    def test_case_13(self):
        assert self.system.text_classification(
            "this game is great and amazing, I love it", method="sentiment") == "positive"

    def test_case_14(self):
        assert self.system.text_classification(
            "the game is terrible and broken, worst experience", method="sentiment") == "negative"

    # ---- 端到端流水线 ----
    def test_case_15(self):
        system = TextGeneration()
        model = system.language_model(self.corpus)
        generated = system.generate_text(prompt="the", max_length=5, temperature=0.0, seed=3)
        assert isinstance(generated, str) and generated.startswith("the")
        for word in generated.split()[1:]:
            assert word in model.vocabulary
        summary = system.text_summarization(
            "The dragon attacks. The hero wins the fight.", method="extractive")
        assert summary
        system.train_classifier(self.training_data)
        assert system.text_classification("fun awesome game", method="naive_bayes") == "positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
