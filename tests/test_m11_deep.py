"""M11 用例：深度评分/过滤/关键词（测试文档 docs/06 §2；LLM 全部 mock 不走网络）。"""

from __future__ import annotations

import asyncio

import pytest

from daily_picks import deep as deep_mod
from daily_picks.deep import deep_analyze, deep_filter, format_keywords
from daily_picks.models import Article, ScoredArticle


def make_article(source: str = "rss", title: str = "AI 编程工具实战",
                 summary: str | None = "用 AI 写代码的十个技巧") -> Article:
    return Article(source=source, source_key="k1", title=title,
                   url="https://example.com/k1", summary=summary)


def make_scored(article_id: int = 1, title: str = "AI 编程工具实战",
                score: float = 10.0) -> ScoredArticle:
    return ScoredArticle(article=make_article(title=title), score=score, article_id=article_id)


class FakeSeqLLM:
    """按调用顺序返回预置 JSON 文本；记录 user 消息与并发度（T-DEEP-08 用）。

    2026-09-04：回复序号在进入时用 lock 顺序分配（并发 sleep 后分配会竞态错位——
    deep_filter 加共享 httpx client 后暴露，导致回复与 candidate 错位）。
    """

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.users: list[str] = []
        self._lock = asyncio.Lock()

    async def chat(self, system: str, user: str, json_mode: bool = True) -> str:
        async with self._lock:
            idx = self.calls
            self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            self.users.append(user)
            return self.replies[min(idx, len(self.replies) - 1)]
        finally:
            self.active -= 1


VALID_JSON = ('{"deep_score": 78, "keywords": ["AI", "编程", "代码"],'
              ' "reason": "文中用具体数据对比了三种方案，缓存实测尤其有参考价值。"}')


class TestDeepAnalyze:
    # T-DEEP-01 深度评分有效
    async def test_valid_result(self):
        llm = FakeSeqLLM([VALID_JSON])
        result = await deep_analyze(make_article(), llm, {"AI": 2.0})
        assert result.deep_score == 78
        assert result.ok is True
        assert result.keywords == ["AI", "编程", "代码"]
        assert "AI" in llm.users[0] and result.reason  # 兴趣关键词注入 user 消息 + 输出完整

    # T-DEEP-02 评分越界回退
    async def test_score_out_of_range(self):
        llm = FakeSeqLLM(['{"deep_score": 150, "keywords": ["AI", "编程", "代码"], "reason": "r"}'])
        result = await deep_analyze(make_article(), llm, {})
        assert result.ok is False

    async def test_score_non_numeric(self):
        llm = FakeSeqLLM(['{"deep_score": "78", "keywords": ["AI", "编程", "代码"], "reason": "r"}'])
        result = await deep_analyze(make_article(), llm, {})
        assert result.ok is False

    # T-DEEP-03 关键词不足
    async def test_keywords_too_few_keeps_original(self):
        llm = FakeSeqLLM(['{"deep_score": 60, "keywords": ["代码"], "reason": "r"}'])
        result = await deep_analyze(make_article(), llm, {})
        assert result.ok is False
        assert result.keywords == ["代码"]  # 数量不足（<3）但原文命中，保留原值

    # T-DEEP-04 理由含禁用词
    async def test_banned_reason_word(self):
        llm = FakeSeqLLM(['{"deep_score": 80, "keywords": ["AI", "编程", "代码"], "reason": "本文深入浅出"}'])
        result = await deep_analyze(make_article(), llm, {})
        assert result.ok is False

    async def test_empty_reason_falls_back(self):
        llm = FakeSeqLLM(['{"deep_score": 80, "keywords": ["Rust", "异步", "实战"], "reason": ""}'])
        result = await deep_analyze(make_article(title="Rust 异步实战"), llm, {})
        assert result.ok is True  # 回退文案，非失败
        assert "Rust 异步实战" in result.reason

    async def test_invalid_json_returns_not_ok(self):
        llm = FakeSeqLLM(["不是JSON"])
        result = await deep_analyze(make_article(), llm, {})
        assert result.ok is False


def _replies(scores: list[int], extra: str = "") -> list[str]:
    out = []
    for s in scores:
        # R12: brief 原 % 格式化触发 UP031（brief 自要求 ruff 零告警），改 f-string 输出等价
        out.append(f'{{"deep_score": {s}, "keywords": ["AI", "编程", "代码"], "reason": "文中引用具体数据论证观点{extra}"}}')
    return out


class TestDeepFilter:
    @pytest.fixture(autouse=True)
    def _no_network_fetch(self, monkeypatch):
        """屏蔽 deep_filter 的正文抓取真实网络请求（2026-09-04 起 deep_filter 自带 httpx client）。"""

        async def _no_body(*args, **kwargs):
            return ""

        monkeypatch.setattr(deep_mod, "fetch_article_text", _no_body)

    # T-DEEP-05 批量过滤保高分
    async def test_keeps_high_scores(self):
        llm = FakeSeqLLM(_replies([78, 55, 30]))
        candidates = [make_scored(article_id=i, score=50.0 - i) for i in (1, 2, 3)]
        # R12（计划缺陷最小修复）：brief 原 threshold=60 会触发降阈值重试（1 < DEEP_MIN_COUNT=5 → 50 放行 55）致断言 [1] 失败；
        # 改 70 后重试仍触发（60 ≥ 0 且 1 < 5）但 55/30 依旧不过，断言与意图不变（保高分）
        filtered, results = await deep_filter(candidates, llm, threshold=70)
        assert [sa.article_id for sa in filtered] == [1]
        assert len(results) == 3          # 全量 DeepResult
        assert [r.article_id for r in results] == [1, 2, 3]  # 回填 article_id

    # T-DEEP-06 fail-open 保留失败篇（超时 mock）
    async def test_fail_open_keeps_failed(self, monkeypatch):
        monkeypatch.setattr(deep_mod, "DEEP_TIMEOUT_S", 0.01)  # 缩短超时避免测试拖慢（freezegun 冻结会挂死，勿加 frozen_now）

        class SlowLLM:
            async def chat(self, system, user, json_mode=True):
                await asyncio.sleep(0.1)  # 超过 0.01s → wait_for 超时
                return VALID_JSON

        filtered, results = await deep_filter([make_scored(article_id=1)], SlowLLM(), threshold=60)
        assert [sa.article_id for sa in filtered] == [1]  # 超时篇保留
        assert results[0].ok is False

    # T-DEEP-07 过滤后不足降阈值重试
    async def test_low_yield_lowers_threshold_once(self):
        llm = FakeSeqLLM(_replies([55, 52, 50]))
        candidates = [make_scored(article_id=i) for i in (1, 2, 3)]
        filtered, _ = await deep_filter(candidates, llm, threshold=60)
        assert [sa.article_id for sa in filtered] == [1, 2, 3]  # 阈值降为 50 后全部保留
        assert llm.calls == 3  # 不重复调用 LLM（复用首轮结果）

    # T-DEEP-08 并发上限 ≤3
    async def test_concurrency_capped_at_three(self):
        llm = FakeSeqLLM(_replies([70] * 5))
        candidates = [make_scored(article_id=i) for i in range(1, 6)]
        await deep_filter(candidates, llm, threshold=60)
        assert llm.max_active <= 3

    async def test_empty_candidates(self):
        llm = FakeSeqLLM([])
        filtered, results = await deep_filter([], llm, threshold=60)
        assert filtered == [] and results == []

    async def test_weights_passed_to_analyze(self, monkeypatch):
        seen: list[dict[str, float]] = []
        original = deep_mod.deep_analyze

        async def spy(article, llm, weights, client=None):
            seen.append(weights)
            return await original(article, llm, weights, client)

        monkeypatch.setattr(deep_mod, "deep_analyze", spy)
        await deep_filter([make_scored(article_id=1)], FakeSeqLLM([VALID_JSON]),
                          threshold=60, weights={"AI": 2.0})
        assert seen == [{"AI": 2.0}]


class TestFormatKeywords:
    # T-DEEP-09 format_keywords
    def test_join_and_truncate(self):
        assert format_keywords(["A", "B", "C", "D", "E", "F"]) == "A、B、C、D、E"
        assert format_keywords(["A", "B", "C"]) == "A、B、C"
        assert format_keywords([]) == ""
    # T-DEEP-10（模板兜底摘要）渲染断言在 tests/test_digest_v3.py::TestBuildDigestV3::test_missing_deep_falls_back_to_summary


# ---- v3 run_once 集成（对齐 tests/test_e2e.py 的 respx mock 写法；cli.py 不在覆盖率范围，验证行为）----

import json as json_mod
from pathlib import Path

import httpx

from daily_picks.cli import run_once

RSS_URL = "https://sspai.com/feed"
RSS_URL2 = "https://www.ruanyifeng.com/blog/atom.xml"
BILI_URL = "https://api.bilibili.com/x/web-interface/popular"
ZHIHU_URL = "https://api.zhihu.com/topstory/hot-lists/total"
JUEJIN_URL = "https://api.juejin.cn/recommend_api/v1/article/recommend_all_feed"
HN_URL = "https://hn.algolia.com/api/v1/search"
INFOQ_URL = "https://www.infoq.cn/feed"
LLM_URL = "https://api.deepseek.com/chat/completions"

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def mock_sources(mock_http) -> None:
    mock_http.get(RSS_URL).mock(return_value=httpx.Response(200, content=load("rss_sample.xml")))
    mock_http.get(RSS_URL2).mock(return_value=httpx.Response(200, content=load("rss_sample.xml")))
    mock_http.get(BILI_URL).mock(
        return_value=httpx.Response(200, json=json_mod.loads(load("bilibili_sample.json"))))
    mock_http.get(ZHIHU_URL).mock(
        return_value=httpx.Response(200, json=json_mod.loads(load("zhihu_sample.json"))))
    mock_http.post(JUEJIN_URL).mock(
        return_value=httpx.Response(200, json=json_mod.loads(load("juejin_sample.json"))))
    mock_http.get(HN_URL).mock(
        return_value=httpx.Response(200, json=json_mod.loads(load("hnews_sample.json"))))
    mock_http.get(INFOQ_URL).mock(return_value=httpx.Response(200, content=load("infoq_sample.xml")))


def llm_reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


DEEP_JSON = ('{"deep_score": 70, "keywords": ["AI", "编程", "代码"],'
             ' "reason": "文章用具体数据对比了三种方案的落地成本，缓存实测尤其有参考价值。"}')
RANK_JSON = ('{"picks": [{"article_id": 1, "rank": 1, "reason": "AI主题深度"},'
             ' {"article_id": 2, "rank": 2, "reason": "工具链实测"},'
             ' {"article_id": 3, "rank": 3, "reason": "架构演进"}]}')


class TestRunOnceV3:
    @pytest.fixture(autouse=True)
    def _no_network_fetch(self, monkeypatch):
        """屏蔽 deep_filter 的正文抓取真实网络请求（2026-09-04 起 deep_filter 自带 httpx client）。"""

        async def _no_body(*args, **kwargs):
            return ""

        monkeypatch.setattr(deep_mod, "fetch_article_text", _no_body)

    async def test_v3_deep_path_outputs_v3_digest(self, sample_config, tmp_path, mock_http,
                                                  frozen_now, monkeypatch):
        cfg = sample_config
        cfg.push.dry_run_file = str(tmp_path / "logs" / "last_digest.md")
        cfg.profile.enabled = True
        cfg.profile.top_n = 3
        cfg.profile.deep_threshold = 60
        cfg.profile.deep_candidates = 40
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        mock_sources(mock_http)
        # 8 篇新文章 → 8 次 deep chat + 1 次 rank chat（respx 按序出队）
        mock_http.post(LLM_URL).mock(side_effect=[llm_reply(DEEP_JSON)] * 8 + [llm_reply(RANK_JSON)])

        assert await run_once(cfg, dry_run=True) == 0
        text = Path(cfg.push.dry_run_file).read_text(encoding="utf-8")
        assert text.startswith("📚 今日深度精选（3条）")
        assert "关键词：AI、编程、代码" in text
        assert "推荐理由：文章用具体数据" in text
        assert text.count("关键词：") == 3

    async def test_v3_no_key_skips_deep(self, sample_config, tmp_path, mock_http,
                                        frozen_now, monkeypatch):
        cfg = sample_config
        cfg.push.dry_run_file = str(tmp_path / "logs" / "last_digest.md")
        cfg.profile.enabled = True
        cfg.profile.top_n = 3
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)  # 确保无 key（本地 .env 可能注入真实 key）
        mock_sources(mock_http)
        # 无 DEEPSEEK_API_KEY：deep 跳过，等同 v2 降级；LLM 端点不应被调用
        assert await run_once(cfg, dry_run=True) == 0
        text = Path(cfg.push.dry_run_file).read_text(encoding="utf-8")
        assert "📚 今日深度精选" in text  # 模板仍是 v3（profile.enabled）
        assert "摘要：" in text or "今日无精选内容" in text  # deep 缺位 → 摘要兜底


class TestKeywordValidation:
    """可验证校验（T-DEEP-11，2026-09-04 修复：LLM 编造关键词防护）。"""

    def test_fabricated_all_removed(self):
        # 关键词全部不在原文 → 清空（推送模板走摘要兜底）
        result = deep_mod._validate_keywords(["上下文窗口", "外部记忆"], "标题：AI 编程\n摘要：写代码")
        assert result == []

    def test_fabricated_partially_filtered(self):
        result = deep_mod._validate_keywords(["AI", "不存在的词", "代码"], "标题：AI 编程\n摘要：写代码")
        assert result == ["AI", "代码"]  # 只保留原文命中项

    def test_empty_and_non_string_ignored(self):
        result = deep_mod._validate_keywords(["", None, "AI"], "标题：AI 编程")
        assert result == ["AI"]


class TestFetchBody:
    """正文抓取与清洗（docs/04 §6.2，2026-09-04 修复）。"""

    def test_strip_html(self):
        raw = "<html><head><style>.x{}</style></head><body><p>正文内容</p><script>bad()</script><nav>菜单</nav>结尾</body></html>"
        text = deep_mod._strip_html(raw)
        assert "正文内容" in text and "结尾" in text
        assert "bad" not in text and "菜单" not in text and "{" not in text

    def test_strip_html_entities(self):
        assert deep_mod._strip_html("<p>a &amp; b</p>") == "a & b"

    async def test_fetch_html_success_and_truncate(self):
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text="<html><body><p>" + "字" * 5000 + "</p></body></html>", headers={"content-type": "text/html"}))
        async with httpx.AsyncClient(transport=transport) as client:
            text = await deep_mod.fetch_article_text("https://example.com/x", client, max_chars=100)
            assert len(text) == 100

    async def test_fetch_non_html_returns_empty(self):
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text="{}", headers={"content-type": "application/json"}))
        async with httpx.AsyncClient(transport=transport) as client:
            assert await deep_mod.fetch_article_text("https://example.com/x", client) == ""

    async def test_fetch_error_returns_empty(self):
        transport = httpx.MockTransport(lambda req: httpx.Response(500))
        async with httpx.AsyncClient(transport=transport) as client:
            assert await deep_mod.fetch_article_text("https://example.com/x", client) == ""

    async def test_short_summary_triggers_body_fetch_and_body_used(self, monkeypatch):
        """summary < 200 字且传 client → 抓正文，DeepResult.body_used=True。"""
        calls = {"n": 0}

        async def fake_fetch(url, client, **kw):
            calls["n"] += 1
            return "正文包含 AI 编程 代码 详细讲解"

        monkeypatch.setattr(deep_mod, "fetch_article_text", fake_fetch)
        llm = FakeSeqLLM([VALID_JSON])
        article = make_article(summary="短摘要")  # < 200 字
        transport = httpx.MockTransport(lambda req: httpx.Response(200))
        async with httpx.AsyncClient(transport=transport) as client:
            result = await deep_analyze(article, llm, {"AI": 2.0}, client)
        assert calls["n"] == 1
        assert result.body_used is True
        assert "正文包含" in llm.users[0]  # 正文进入 user prompt

    async def test_long_summary_skips_fetch(self, monkeypatch):
        calls = {"n": 0}

        async def fake_fetch(url, client, **kw):
            calls["n"] += 1
            return "x"

        monkeypatch.setattr(deep_mod, "fetch_article_text", fake_fetch)
        llm = FakeSeqLLM([VALID_JSON])
        article = make_article(summary="长" * 300)  # ≥ 200 字不抓
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as client:
            result = await deep_analyze(article, llm, {"AI": 2.0}, client)
        assert calls["n"] == 0
        assert result.body_used is False
