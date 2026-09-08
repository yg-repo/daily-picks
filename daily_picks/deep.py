"""深度评分 + 关键词提取（docs/04 §6.2 / docs/05 §2.1，M11 核心）。"""

from __future__ import annotations

import asyncio
import html as html_lib
import json
import logging
import re
from dataclasses import dataclass

import httpx

from daily_picks.llm import LLMClient, LLMError
from daily_picks.models import Article, ScoredArticle

logger = logging.getLogger("daily_picks.deep")

DEEP_SCORE_MIN: float = 0.0    # LLM 深度评分（0-100）输出下限校验
KEYWORDS_MIN: int = 3
KEYWORDS_MAX: int = 5
DEEP_TIMEOUT_S: float = 60.0   # 单篇分析超时秒数（docs/05 §2.1；8/31 实测 30s 超时率高，调至 60）
DEEP_MIN_COUNT: int = 5        # 过滤后不足该数触发降阈值重试（对齐 profile.top_n 默认值，docs/04 §6.2 修订）

# 正文抓取（2026-09-04 修复：此前只喂 title+短摘要，LLM 基于标题联想编造关键词/理由。
# summary < FETCH_BODY_MIN_SUMMARY 且 url 可抓时，抓正文辅助分析——docs/04 §6.2）
FETCH_BODY_MIN_SUMMARY: int = 200   # summary 短于此字数触发正文抓取
FETCH_BODY_MAX_CHARS: int = 3000    # 正文送入 LLM 的最大字符数
FETCH_BODY_TIMEOUT_S: float = 10.0

# 待剥离的 HTML 区块（正文清洗）
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0"
_STRIP_BLOCK_RE = re.compile(
    r"<(script|style|nav|footer|header|aside|form|noscript)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# 推荐理由禁用词（docs/04 §7，写死在代码侧校验）
BANNED_REASONS = ("深入浅出", "受益匪浅", "干货满满", "值得一读", "不容错过")

# 深度分析 system prompt（docs/05 §5.1 模板，原样使用）
# 2026-09-02 修订：新增可读性维度（面向普通技术爱好者，硬核前置知识文扣分）与
# 反纯资讯约束（无观点无分析的产品发布/资讯报道 deep_score < 40）。
DEEP_SYSTEM_PROMPT = (
    "你是资深内容编辑，评估文章思考深度。评分标准：观点原创性(35%)、论证扎实度(25%)、"
    "信息密度(20%)、启发价值(10%)、可读性(10%)。目标读者是普通技术爱好者而非底层专家："
    "需要大量领域前置知识才能读懂的文章，可读性给低分。纯资讯/产品发布报道（只有事实罗列，"
    "无观点、无分析、无洞察）deep_score 直接给 40 以下。"
    "推荐理由必须引用文章的具体观点/数据/场景，禁止使用任何套话。"
)


@dataclass
class DeepResult:
    article_id: int
    deep_score: int             # 0-100
    keywords: list[str]         # 3-5 个
    reason: str                 # 具体推荐理由（引用文章细节，禁止空泛）
    ok: bool                    # LLM 输出有效？
    body_used: bool = False     # 分析是否基于抓取的正文（True）或仅 title+summary（False）


def _strip_html(raw: str) -> str:
    """HTML → 纯文本：去 script/style/nav 等区块，去标签，解实体，压缩空白。"""
    text = _STRIP_BLOCK_RE.sub(" ", raw)
    text = _TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    return _WS_RE.sub(" ", text).strip()


async def fetch_article_text(url: str, client: httpx.AsyncClient,
                             max_chars: int = FETCH_BODY_MAX_CHARS) -> str:
    """抓取 url 正文并清洗为纯文本，截断 max_chars。失败/非 HTML/超时 → ""（fail-open）。"""
    try:
        resp = await client.get(url, headers={"User-Agent": _UA}, timeout=FETCH_BODY_TIMEOUT_S)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype.lower():
            return ""
        text = _strip_html(resp.text)
        return text[:max_chars]
    except Exception as e:  # noqa: BLE001
        logger.debug("正文抓取失败 url=%s: %s", url, e)
        return ""


def _validate_keywords(keywords: list[str], text: str) -> list[str]:
    """剔除未在给定文本（title+summary+正文）中出现的关键词（LLM 编造防护）。
    全部未命中 → []（推送模板走摘要兜底，docs/06 T-DEEP-11）。"""
    lowered = text.lower()
    valid = [k for k in keywords if k and k.lower() in lowered]
    return valid[:KEYWORDS_MAX]


async def deep_analyze(article: Article, llm: LLMClient,
                       weights: dict[str, float],
                       client: httpx.AsyncClient | None = None) -> DeepResult:
    """LLM 深度分析：输入 title+summary+url（summary 过短且可抓时附正文）+用户兴趣。

    2026-09-04 修复（docs/04 §6.2）：summary < FETCH_BODY_MIN_SUMMARY 时用 client 抓正文，
    防止 LLM 仅凭标题联想编造；关键词经 _validate_keywords 做代码侧可验证校验。
    解析失败/输出非法 → ok=False（不抛异常，fail-open 依据，docs/05 §2.1）。
    article_id 由调用方（deep_filter）回填——本函数签名无 id 参数。
    """
    summary = article.summary or ""
    body = ""
    body_used = False
    if client is not None and len(summary) < FETCH_BODY_MIN_SUMMARY and article.url:
        body = await fetch_article_text(article.url, client)
        body_used = bool(body)
    full_text = f"{article.title}\n{summary}\n{body}"
    keywords_text = "、".join(sorted(weights, key=weights.get, reverse=True)) or "（暂无）"
    user = (
        f"标题：{article.title}\n摘要：{summary}\n"
        + (f"正文（节选）：{body}\n" if body else "")
        + f"URL：{article.url}\n用户兴趣关键词：{keywords_text}\n"
        + '输出 JSON：{"deep_score": <0-100整数>, "keywords": [3-5个名词短语，必须直接来自标题/摘要/正文原文], '
        + '"reason": "2-3句，只引用给定文本中真实存在的内容，禁止编造或联想未提供的信息，'
        + "禁止'深入浅出/受益匪浅/干货满满/值得一读/不容错过'\"}"
    )
    try:
        text = await llm.chat(DEEP_SYSTEM_PROMPT, user, json_mode=True)
        data = json.loads(text or "{}")
    except (LLMError, json.JSONDecodeError, TypeError) as e:
        logger.warning("deep_analyze 失败（fail-open）: %s", e)
        return DeepResult(article_id=0, deep_score=0, keywords=[], reason="",
                          ok=False, body_used=body_used)
    if not isinstance(data, dict):
        return DeepResult(article_id=0, deep_score=0, keywords=[], reason="",
                          ok=False, body_used=body_used)

    ok = True
    score_raw = data.get("deep_score")
    if not (isinstance(score_raw, int) and not isinstance(score_raw, bool)
            and DEEP_SCORE_MIN <= score_raw <= 100):
        ok = False
    keywords_raw = data.get("keywords")
    if not isinstance(keywords_raw, list) or len(keywords_raw) < KEYWORDS_MIN:
        ok = False
    keywords = [str(k) for k in keywords_raw[:KEYWORDS_MAX]] if isinstance(keywords_raw, list) else []
    # 可验证校验：剔除原文中不存在的关键词（LLM 编造防护，docs/04 §6.2）
    keywords = _validate_keywords(keywords, full_text)

    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = f"文章围绕 {article.title} 展开"   # docs/05 §2.1 回退文案
    elif any(word in reason for word in BANNED_REASONS):
        ok = False

    return DeepResult(article_id=0, deep_score=score_raw if isinstance(score_raw, int) else 0,
                      keywords=keywords, reason=reason, ok=ok, body_used=body_used)


async def deep_filter(candidates: list[ScoredArticle], llm: LLMClient,
                      threshold: int,
                      weights: dict[str, float] | None = None) -> tuple[list[ScoredArticle], list[DeepResult]]:
    """批量 deep_analyze（并发 ≤3，单篇 DEEP_TIMEOUT_S 超时），保留 deep_score >= threshold 的候选。

    ok=False（LLM 失败/超时/输出非法）的文章**保留**（fail-open，不因 deep 故障丢候选）；
    过滤后不足 DEEP_MIN_COUNT 条且 threshold-10 >= 0 时，按首轮结果降阈值重过滤一次
    （不重复调用 LLM，docs/04 §10 降级表）。返回 (过滤后候选, 全量 DeepResult 列表)。
    """
    if not candidates:
        return [], []
    weights = weights or {}
    sem = asyncio.Semaphore(3)

    async def _analyze_one(sa: ScoredArticle, client: httpx.AsyncClient) -> DeepResult:
        async with sem:
            try:
                result = await asyncio.wait_for(
                    deep_analyze(sa.article, llm, weights, client), timeout=DEEP_TIMEOUT_S)
            except TimeoutError:
                logger.warning("deep 分析超时 article_id=%s（fail-open 保留）", sa.article_id)
                return DeepResult(article_id=sa.article_id, deep_score=0,
                                  keywords=[], reason="", ok=False)
            result.article_id = sa.article_id if sa.article_id is not None else 0
            return result

    # 共享 client：正文抓取复用连接（2026-09-04 修复，docs/04 §6.2）
    async with httpx.AsyncClient(timeout=FETCH_BODY_TIMEOUT_S) as client:
        results = await asyncio.gather(*(_analyze_one(sa, client) for sa in candidates))

    def _keep(r: DeepResult, th: int) -> bool:
        return (not r.ok) or r.deep_score >= th

    filtered = [sa for sa, r in zip(candidates, results) if _keep(r, threshold)]
    if len(filtered) < DEEP_MIN_COUNT and threshold - 10 >= 0:
        logger.info("deep 过滤后不足 %d 条，降阈值 %d 重试一次", DEEP_MIN_COUNT, threshold - 10)
        filtered = [sa for sa, r in zip(candidates, results) if _keep(r, threshold - 10)]
    return filtered, results


def format_keywords(keywords: list[str]) -> str:
    """'k1、k2、k3' 顿号拼接，超 5 个截断（docs/04 §6.2）。"""
    return "、".join(keywords[:KEYWORDS_MAX])
