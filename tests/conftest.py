"""pytest 全局 fixture（测试文档 §2）。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import respx
from freezegun import freeze_time

from daily_picks.config import RootConfig, load_config, write_default_config
from daily_picks.storage import Storage


@pytest.fixture
def tmp_db(tmp_path: Path) -> Storage:
    """临时目录下的空库（test.db），已 init_schema。"""
    storage = Storage(tmp_path / "test.db")
    storage.init_schema()
    return storage


@pytest.fixture
def sample_config(tmp_path: Path) -> RootConfig:
    """全默认 RootConfig（write_default_config 生成后 load，storage.db_path 指向临时目录）。"""
    path = tmp_path / "config.yaml"
    write_default_config(path)
    cfg = load_config(str(path))
    cfg.storage.db_path = str(tmp_path / "data" / "test.db")
    return cfg


@pytest.fixture
def mock_http():
    """respx mock 上下文（assert_all_mocked=True），供各用例注册路由。

    assert_all_called=False：允许"注册但故意不调用"的路由（如 T-E2E-11 断言
    tracking 关闭时零追踪请求）；对齐 test_llm/test_publisher/test_sources 的写法。
    """
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as m:
        yield m


@pytest.fixture
def frozen_now():
    """冻结本地时间为 2026-08-27 08:00（UTC+8），并返回该时刻 naive datetime。

    real_asyncio=True：只冻结墙钟，不冻结事件循环的单调时钟——否则 freezegun 冻结
    time.monotonic 会导致 asyncio.sleep/wait_for 的定时器永不触发，async 用例挂死
    （如 tenacity 重试退避等待）。
    """
    with freeze_time("2026-08-27 08:00:00", tz_offset=8, real_asyncio=True):
        yield datetime(2026, 8, 27, 8, 0, 0)

@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """统一测试环境（2026-09-18 CI 修复）：

    ① 固定时区 Asia/Shanghai——run_date 由 datetime.now().astimezone(tz) 生成，
       CI runner 为 UTC 时日期会漂移一天（如冻结 08-27 变成 08-28），导致日期断言失败；
    ② 清除代理环境变量——httpx 默认 trust_env=True，本机 all_proxy(socks5) 与
       CI 无代理的差异会让 respx 拦截路径不一致（未拦截时真实请求打到企业微信，
       返回 errcode 93000 造成"本地过、CI 挂"）。
    """
    import time as _time
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    _time.tzset()
    for k in ("all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY",
              "https_proxy", "HTTPS_PROXY", "no_proxy", "NO_PROXY"):
        monkeypatch.delenv(k, raising=False)
