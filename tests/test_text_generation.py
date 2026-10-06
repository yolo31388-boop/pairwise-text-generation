"""
游戏内自然语言生成与文本生成系统 - 验收测试
运行方式：python -m pytest tests/test_text_generation.py -q
共 15 个测试用例
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.text_generation import TextGeneration

try:
    import pytest
except ImportError:  # 环境无 pytest 时也可直接运行本文件
    pytest = None


CORPUS = [
    "the dragon attacks the castle",
    "the hero attacks the dragon",
    "the dragon destroys the castle",
    "the hero saves the castle",
    "the dragon roars",
]

ARTICLE = (
    "游戏新版本上线了新的副本。"
    "新的副本带来了全新的玩法。"
    "玩家对新副本的评价非常好。"
    "开发团队表示会持续优化副本体验。"
)


def _raises(exc_type, func, *args, **kwargs):
    """不依赖 pytest 的异常断言辅助函数。"""
    try:
        func(*args, **kwargs)
    except exc_type:
        return True
    return False


class TestTextGeneration:
    def setup_method(self):
        self.system = TextGeneration(config={"seed": 42})

    # ---------- generate_text：采样与温度 ----------
    def test_case_01(self):
        # fit 后能生成文本，且生成内容来自词表
        self.system.fit(CORPUS)
        text = self.system.generate_text(max_length=5)
        assert isinstance(text, str)
        assert len(text) > 0
        vocab = set(self.system.language_model()["vocab"])
        for token in text.split():
            assert token in vocab

    def test_case_02(self):
        # 相同 seed 的采样结果可复现，不同 seed 都能正常采样
        self.system.fit(CORPUS)
        first = self.system.generate_text(max_length=8, seed=7)
        second = self.system.generate_text(max_length=8, seed=7)
        third = self.system.generate_text(max_length=8, seed=123)
        assert first == second
        assert isinstance(third, str) and len(third) > 0

    def test_case_03(self):
        # 温度控制随机性：低温恒定选中最高概率词，高温允许不同采样
        self.system.fit(CORPUS)
        low_temperature = {
            self.system.generate_text(prompt="the", max_length=1, temperature=0.01, seed=s)
            for s in range(10)
        }
        assert low_temperature == {"the dragon"}
        high_temperature = {
            self.system.generate_text(prompt="the", max_length=1, temperature=10.0, seed=s)
            for s in range(40)
        }
        assert len(high_temperature) > 1

    # ---------- language_model：词表与概率分布 ----------
    def test_case_04(self):
        model = self.system.language_model(CORPUS)
        assert "dragon" in model["vocab"]
        assert model["vocab_size"] == len(model["vocab"])
        assert model["vocab_size"] > 0
        assert abs(sum(model["distribution"].values()) - 1.0) < 1e-9

    def test_case_05(self):
        model = self.system.language_model(CORPUS)
        # the 出现 9 次，总词元 23 个
        assert model["total_tokens"] == 23
        assert abs(model["distribution"]["the"] - 9 / 23) < 1e-9
        assert abs(model["distribution"]["dragon"] - 4 / 23) < 1e-9

    # ---------- ngram：一元模型与二元模型 ----------
    def test_case_06(self):
        uni = self.system.ngram(CORPUS, n=1)
        assert uni["n"] == 1
        assert uni["counts"]["the"] == 9
        assert abs(sum(uni["probabilities"].values()) - 1.0) < 1e-9
        assert abs(uni["probabilities"]["hero"] - 2 / 23) < 1e-9

    def test_case_07(self):
        bi = self.system.ngram(CORPUS, n=2)
        assert bi["n"] == 2
        assert bi["counts"]["the"]["dragon"] == 4
        for context, probs in bi["probabilities"].items():
            assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert abs(bi["probabilities"]["the"]["dragon"] - 4 / 9) < 1e-9

    def test_case_08(self):
        # 未训练时抛错；n 只支持 1 和 2
        assert _raises(ValueError, self.system.ngram)
        self.system.fit(CORPUS)
        assert _raises(ValueError, self.system.ngram, None, 3)

    # ---------- text_summarization：抽取式与生成式 ----------
    def test_case_09(self):
        # 抽取式摘要：从原文抽取句子，且是原文子串
        summary = self.system.text_summarization(ARTICLE, max_sentences=1, method="extractive")
        assert isinstance(summary, str) and len(summary) > 0
        assert summary in ARTICLE

    def test_case_10(self):
        # 抽取式摘要：max_sentences 控制抽取的句子数量
        summary = self.system.text_summarization(ARTICLE, max_sentences=2, method="extractive")
        assert summary in ARTICLE
        assert summary.count("。") == 2

    def test_case_11(self):
        # 生成式摘要：围绕高频主题词生成新句子
        summary = self.system.text_summarization(ARTICLE, method="abstractive")
        assert isinstance(summary, str) and len(summary) > 0
        assert "副" in summary or "本" in summary

    # ---------- text_classification：朴素贝叶斯与情感分析 ----------
    def test_case_12(self):
        # 朴素贝叶斯：训练后能正确分类新文本
        samples = [
            ("这款游戏画面精美玩法有趣", "游戏"),
            ("新游戏副本非常刺激", "游戏"),
            ("今天股票大涨基金飘红", "财经"),
            ("央行发布最新货币政策", "财经"),
        ]
        self.system.train_classifier(samples)
        assert self.system.text_classification("游戏副本真好玩") == "游戏"
        assert self.system.text_classification("股票基金行情") == "财经"

    def test_case_13(self):
        # 朴素贝叶斯：未训练时抛错，训练后可限定候选类别
        assert _raises(ValueError, self.system.text_classification, "任意文本")
        self.system.train_classifier(
            [("猫咪可爱狗狗忠诚", "动物"), ("代码编译运行调试", "编程")]
        )
        assert self.system.text_classification("狗狗和猫咪", labels=["动物", "编程"]) == "动物"

    def test_case_14(self):
        # 情感分析：正面 / 负面 / 中性
        assert self.system.sentiment_analysis("这个游戏太好玩了，非常精彩，推荐") == "positive"
        assert self.system.sentiment_analysis("糟糕透顶，无聊又失望") == "negative"
        assert self.system.sentiment_analysis("今天星期三") == "neutral"

    def test_case_15(self):
        # 情感分析支持英文文本
        assert self.system.sentiment_analysis("this game is great and amazing") == "positive"
        assert self.system.sentiment_analysis("this game is bad and boring") == "negative"


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__, "-v"])
    else:
        # 无 pytest 环境下的简易运行器
        case = TestTextGeneration()
        passed = 0
        for name in sorted(n for n in dir(case) if n.startswith("test_case_")):
            case.setup_method()
            getattr(case, name)()
            passed += 1
            print("PASSED", name)
        print("%d passed" % passed)
