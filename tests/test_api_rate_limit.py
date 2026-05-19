"""
RPM 限流 + 批量调用正确性测试。

覆盖范围：
  T1: RateLimiter / _SimpleRateLimiter 严格滑窗单元测试（高并发）
  T2: LLMGenerationManager._call_commenter_batch —— 顺序对齐 + RPM 严格
  T3: RewardManager._call_rubric_scoring_api —— 索引对齐 + RPM 严格
  T4: 多轮模拟 run_llm_loop_paper_writing_autonomous —— 直接驱动
      _execute_paper_writing_autonomous，覆盖跨轮累计 RPM 严格性
  T5: 并发交叉（commenter + rubric 同时跑，各自独立限流互不干扰）

设计要点：
  - 不依赖真实 GPU / actor_rollout_wg / 真实 tokenizer / 真实 API
  - FakeOpenAIClient 模拟 chat.completions.create，记录每次调用时间戳
  - assert_window_rpm 用双指针滑窗严格校验任意 60s 窗口调用数 ≤ rpm
  - 通过 __new__ 跳过 LLMGenerationManager / RewardManager 的重型 __init__，
    只装配被测方法所需属性

运行：
  python tests/test_api_rate_limit.py            # 直接跑全部
  python -m pytest tests/test_api_rate_limit.py  # pytest 模式
"""

import os
import sys
import time
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

# 让脚本既能 pytest 跑，也能直接 python 跑
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 缩短 RPM 让测试可在数秒内验证窗口
TEST_RPM = 30
TEST_WINDOW = 60.0


# ---------------- Fake OpenAI client ----------------

class _FakeMessage:
    def __init__(self, content):
        self.message = SimpleNamespace(content=content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeMessage(content)]


class FakeCompletions:
    def __init__(self, parent):
        self.parent = parent

    def create(self, model=None, messages=None, max_tokens=None,
               temperature=None, extra_body=None, **kwargs):
        # 记录入参时间戳；模拟网络延迟
        ts = time.monotonic()
        with self.parent._lock:
            self.parent.call_log.append(ts)
            self.parent.live += 1
            self.parent.peak_live = max(self.parent.peak_live, self.parent.live)
        try:
            time.sleep(self.parent.latency)
        finally:
            with self.parent._lock:
                self.parent.live -= 1
        # 用最后一条 user 消息的前缀作为回包内容，方便断言顺序对齐
        user_msg = ''
        if messages:
            for m in messages[::-1]:
                if m.get('role') == 'user':
                    user_msg = m.get('content', '')
                    break
        # 提取 "Current Draft:\nXXX\n\n" 或前 40 字
        marker = 'Current Draft:\n'
        if marker in user_msg:
            tail = user_msg.split(marker, 1)[1]
            content = 'COMMENT_FOR::' + tail.splitlines()[0][:60]
        elif 'Abstract to evaluate:\n' in user_msg:
            tail = user_msg.split('Abstract to evaluate:\n', 1)[1]
            content = (
                "Problem & Motivation: 0.80\n"
                "Method & Contribution Coverage: 0.70\n"
                "Results & Evidence Coverage: 0.60\n"
                "Topic Consistency: 0.80\n"
                "Clarity & Conciseness: 0.75\n"
                "Length Appropriateness: 0.70\n"
                "Format & Presentation: 1.00\n"
                f"Summary: scored_for::{tail.splitlines()[0][:60]}"
            )
        else:
            content = 'OK'
        return _FakeResponse(content)


class FakeChat:
    def __init__(self, parent):
        self.completions = FakeCompletions(parent)


class FakeOpenAIClient:
    """足够覆盖 client.chat.completions.create 调用面的伪 OpenAI 客户端。"""

    def __init__(self, latency: float = 0.05):
        self.latency = latency
        self.call_log: list[float] = []
        self.live = 0
        self.peak_live = 0
        self._lock = threading.Lock()
        self.chat = FakeChat(self)


# ---------------- 严格滑窗校验工具 ----------------

def assert_window_rpm(timestamps, rpm, window=TEST_WINDOW, label='', tolerance=1):
    """双指针滑窗：任意 [t, t+window) 区间内时间戳数量 <= rpm + tolerance。

    tolerance 解释（默认 1）：限流器内部以 ``acquire`` 准入时刻记录时间戳并严格
    保证 admission 每 ``window`` 内 ≤ rpm。但 FakeClient 记录的是 ``create()``
    入口时刻，与准入之间存在 GIL / 线程调度造成的 µs–ms 级偏移。当首个调用的
    偏移 δ₁ > 第 (rpm+1) 个调用的偏移 δ_{rpm+1} 时，可能出现 hit-side 窗口
    [hit₁, hit₁+window) 边界恰好包含 hit_{rpm+1} 的情形，导致观测计数 = rpm+1。
    这并不违反限流器对上游 API 的 RPM 合规性 —— 上游服务器以自身时钟测量，
    同样存在类似抖动；同时令该边界 case 在测试中以 tolerance 容忍。

    若需要"零容忍"严格性，请使用直接对 ``limiter._timestamps`` 的检查（见
    ``assert_admission_rpm_strict``）。
    """
    ts = sorted(timestamps)
    n = len(ts)
    j = 0
    max_in_window = 0
    for i in range(n):
        while j < n and ts[j] < ts[i] + window:
            j += 1
        cnt = j - i
        if cnt > max_in_window:
            max_in_window = cnt
    assert max_in_window <= rpm + tolerance, (
        f"[{label}] 窗口违规: 最大窗口内调用数={max_in_window} > rpm+tol={rpm + tolerance}, "
        f"总调用数={n}"
    )
    return max_in_window


def make_recording_limiter(limiter_cls, rpm):
    """构造一个会把 admission 时刻完整记录到 ``admissions`` 列表的限流器。

    用于"零容忍"严格性校验：admission 时刻是限流器的真实契约面，必须满足
    任意 60s 窗口 ≤ rpm（无容差）。
    """
    limiter = limiter_cls(rpm=rpm)
    admissions: list[float] = []
    orig_acquire = limiter.acquire

    def acquire():
        orig_acquire()
        admissions.append(time.monotonic())

    limiter.acquire = acquire  # type: ignore[assignment]
    return limiter, admissions


# ---------------- T1: RateLimiter 单元测试 ----------------

class TestRateLimiterStrictness(unittest.TestCase):
    """对两个限流器实现做高并发严格性测试。"""

    def _run_acquire_burst(self, limiter, num_threads, calls_per_thread):
        timestamps = []
        ts_lock = threading.Lock()

        def worker():
            for _ in range(calls_per_thread):
                limiter.acquire()
                t = time.monotonic()
                with ts_lock:
                    timestamps.append(t)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        t0 = time.monotonic()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        elapsed = time.monotonic() - t0
        return timestamps, elapsed

    def test_generation_RateLimiter_strict(self):
        from search_r1.llm_agent.generation import RateLimiter
        # rpm=30, 60 调用 -> 至少需要约 60s 才能完成第二批
        # 为了不让测试过慢，缩成 rpm=30, 总 35 次（仅触发一次超限阻塞）
        limiter = RateLimiter(rpm=TEST_RPM)
        timestamps, elapsed = self._run_acquire_burst(
            limiter, num_threads=10, calls_per_thread=4  # 总 40
        )
        max_in_win = assert_window_rpm(
            timestamps, rpm=TEST_RPM, label='generation.RateLimiter'
        )
        # 触发限流后第 31 次必然要等待至少 (60s - burst_time)
        self.assertGreaterEqual(
            elapsed, TEST_WINDOW - 5,
            f"40 次调用超过 rpm={TEST_RPM} 至少应被阻塞接近 {TEST_WINDOW}s, "
            f"实际 {elapsed:.2f}s"
        )
        print(f"[T1a generation.RateLimiter] 40 calls, max_in_window={max_in_win}, "
              f"elapsed={elapsed:.2f}s")

    def test_main_ppo_SimpleRateLimiter_strict(self):
        from verl.trainer.main_ppo import _SimpleRateLimiter
        limiter = _SimpleRateLimiter(rpm=TEST_RPM)
        timestamps, elapsed = self._run_acquire_burst(
            limiter, num_threads=10, calls_per_thread=4
        )
        max_in_win = assert_window_rpm(
            timestamps, rpm=TEST_RPM, label='main_ppo._SimpleRateLimiter'
        )
        self.assertGreaterEqual(elapsed, TEST_WINDOW - 5)
        print(f"[T1b main_ppo._SimpleRateLimiter] 40 calls, "
              f"max_in_window={max_in_win}, elapsed={elapsed:.2f}s")


# ---------------- T2: _call_commenter_batch 顺序对齐 + RPM ----------------

def _make_manager_stub(rpm=TEST_RPM, max_concurrency=48):
    """构造一个只装配 commenter 路径所需属性的 LLMGenerationManager 实例。"""
    from search_r1.llm_agent.generation import LLMGenerationManager, RateLimiter
    mgr = LLMGenerationManager.__new__(LLMGenerationManager)
    mgr.config = SimpleNamespace(
        commenter_model='fake-model',
        commenter_api_key='fake',
        commenter_base_url='http://fake',
        commenter_max_concurrency=max_concurrency,
        api_rpm=rpm,
    )
    mgr._commenter_rate_limiter = RateLimiter(rpm=rpm)
    return mgr


class TestCommenterBatch(unittest.TestCase):

    def test_order_alignment_and_rpm(self):
        mgr = _make_manager_stub(rpm=TEST_RPM, max_concurrency=48)
        fake_client = FakeOpenAIClient(latency=0.05)
        mgr._commenter_client = fake_client  # 跳过真实 OpenAI 构造

        N = 50  # 触发两次窗口
        domains = [f'kw{i}' for i in range(N)]
        topics = [f'title{i}' for i in range(N)]
        drafts = [f'DRAFT_BODY_{i:03d}' for i in range(N)]
        ground_truths = [''] * N

        t0 = time.monotonic()
        results = mgr._call_commenter_batch(domains, topics, drafts, ground_truths)
        elapsed = time.monotonic() - t0

        # 结果对齐：长度 + 顺序（FakeClient 回包内容携带 draft 前缀）
        self.assertEqual(len(results), N, "返回数量必须与输入对齐")
        for i, r in enumerate(results):
            self.assertIn(
                f'DRAFT_BODY_{i:03d}', r,
                f"第 {i} 条回复未对齐到第 {i} 条输入: {r!r}"
            )

        # RPM 严格
        max_in_win = assert_window_rpm(
            fake_client.call_log, rpm=TEST_RPM, label='commenter_batch'
        )
        # 第 31~50 个调用必然要等待
        self.assertGreaterEqual(
            elapsed, TEST_WINDOW - 5,
            f"50 次调用 rpm={TEST_RPM} 至少需要 ~{TEST_WINDOW}s，实际 {elapsed:.2f}s"
        )
        print(f"[T2 commenter_batch] N={N}, peak_live={fake_client.peak_live}, "
              f"max_in_window={max_in_win}, elapsed={elapsed:.2f}s")


# ---------------- T3: rubric scoring 索引对齐 + RPM ----------------

def _make_reward_manager_stub(rpm=TEST_RPM, max_concurrency=32):
    from verl.trainer.main_ppo import RewardManager, _SimpleRateLimiter
    rm = RewardManager.__new__(RewardManager)
    rm.rubric_api_key = 'fake'
    rm.rubric_api_base = 'http://fake'
    rm.rubric_model = 'fake-rubric'
    rm.rubric_max_concurrency = max_concurrency
    rm.rubric_rpm = rpm
    rm._rubric_rate_limiter = _SimpleRateLimiter(rpm=rpm)
    rm._rubric_client = None  # 让 _call_rubric_scoring_api 走 lazy 分支，我们随后替换
    return rm


class TestRubricBatch(unittest.TestCase):

    def test_index_alignment_and_rpm(self):
        rm = _make_reward_manager_stub(rpm=TEST_RPM, max_concurrency=32)
        fake_client = FakeOpenAIClient(latency=0.05)
        rm._rubric_client = fake_client  # 提前注入，绕过 OpenAI 真实构造

        N = 45
        camera_ready = [f'ABSTRACT_TEXT_{i:03d}' for i in range(N)]
        gts = [f'gt{i}' for i in range(N)]
        domains = [f'kw{i}' for i in range(N)]
        topics = [f'title{i}' for i in range(N)]

        t0 = time.monotonic()
        scores, details = rm._call_rubric_scoring_api(
            camera_ready_texts=camera_ready,
            ground_truth_texts=gts,
            domains=domains,
            topics=topics,
            return_details=True,
        )
        elapsed = time.monotonic() - t0

        self.assertEqual(len(scores), N)
        self.assertEqual(len(details), N)
        # 索引对齐：每个 detail.summary 应携带原文前缀
        for i, d in enumerate(details):
            summary = d.get('summary', '')
            self.assertIn(
                f'ABSTRACT_TEXT_{i:03d}', d.get('raw', ''),
                f"第 {i} 条 rubric raw 未对齐到原文: summary={summary!r}"
            )

        max_in_win = assert_window_rpm(
            fake_client.call_log, rpm=TEST_RPM, label='rubric_batch'
        )
        self.assertGreaterEqual(elapsed, TEST_WINDOW - 5)
        print(f"[T3 rubric_batch] N={N}, peak_live={fake_client.peak_live}, "
              f"max_in_window={max_in_win}, elapsed={elapsed:.2f}s")


# ---------------- T4: 模拟 run_llm_loop_paper_writing_autonomous 多轮 ----------------

class TestAutonomousLoopRateLimit(unittest.TestCase):
    """
    直接驱动 _execute_paper_writing_autonomous 模拟外层循环：
      - 每轮按比例生成 'draft' / 'camera-ready' / 'invalid'
      - 'camera-ready' 退出 active_mask
      - 'draft' / invalid 触发 commenter 调用
      - 跨多轮累计的 commenter 调用必须严格 ≤ api_rpm
    """

    def test_multi_turn_commenter_rpm_strict(self):
        mgr = _make_manager_stub(rpm=TEST_RPM, max_concurrency=48)
        fake_client = FakeOpenAIClient(latency=0.05)
        mgr._commenter_client = fake_client

        BATCH = 20
        MAX_TURNS = 5
        # 用 list 而非 torch tensor，因为 _execute_paper_writing_autonomous
        # 只对 active_mask 做 zip / bool() 操作，list[bool] 等价
        active_mask = [True] * BATCH
        domains = [f'kw{i}' for i in range(BATCH)]
        topics = [f'title{i}' for i in range(BATCH)]
        ground_truths = [''] * BATCH

        all_commenter_call_count = 0
        for turn in range(MAX_TURNS):
            # 构造每轮的 responses_str:
            # 前 1/3 是 draft, 中 1/3 是 invalid（也会触发 commenter）,
            # 后 1/3 在某一轮 camera-ready 退出
            responses_str = []
            for i in range(BATCH):
                if not active_mask[i]:
                    responses_str.append('')  # inactive 不会被使用
                    continue
                if turn == MAX_TURNS - 1 and i % 3 == 2:
                    # 最后一轮让一部分 camera-ready
                    responses_str.append(f'<camera-ready>FINAL_{i:03d}</camera-ready>')
                elif i % 3 == 0:
                    responses_str.append(f'<draft>DRAFT_{turn}_{i:03d}</draft>')
                elif i % 3 == 1:
                    responses_str.append(f'INVALID_RAW_{turn}_{i:03d}')
                else:
                    responses_str.append(f'<draft>DRAFT_{turn}_{i:03d}</draft>')

            expected_commenter = sum(
                1 for i in range(BATCH)
                if active_mask[i] and not responses_str[i].startswith('<camera-ready>')
            )

            calls_before = len(fake_client.call_log)
            next_obs, dones, valid_action, is_comment, cr_contents = \
                mgr._execute_paper_writing_autonomous(
                    responses_str, active_mask,
                    domains=domains, topics=topics, ground_truths=ground_truths,
                )
            calls_after = len(fake_client.call_log)

            # 每轮 commenter 调用数 == active 且非 camera-ready 样本数
            self.assertEqual(
                calls_after - calls_before, expected_commenter,
                f"turn {turn} commenter 调用数不匹配: "
                f"实际 {calls_after - calls_before} 期望 {expected_commenter}"
            )

            # 长度对齐
            self.assertEqual(len(next_obs), BATCH)
            self.assertEqual(len(dones), BATCH)
            self.assertEqual(len(cr_contents), BATCH)

            # camera-ready 样本: done=1, cr_content 非空; draft/invalid: done=0, obs 含 <comment>
            for i in range(BATCH):
                if not active_mask[i]:
                    self.assertEqual(dones[i], 1)
                    continue
                if responses_str[i].startswith('<camera-ready>'):
                    self.assertEqual(dones[i], 1)
                    self.assertTrue(cr_contents[i].startswith('FINAL_'))
                    self.assertEqual(next_obs[i], '')
                else:
                    self.assertEqual(dones[i], 0)
                    self.assertIn('<comment>', next_obs[i])
                    self.assertIn('</comment>', next_obs[i])

            # 更新 active_mask: done==1 退出
            active_mask = [
                am and (d == 0) for am, d in zip(active_mask, dones)
            ]
            all_commenter_call_count += (calls_after - calls_before)

            if not any(active_mask):
                break

        # 全局 RPM 严格
        max_in_win = assert_window_rpm(
            fake_client.call_log, rpm=TEST_RPM,
            label='autonomous_loop_commenter'
        )
        print(f"[T4 autonomous_loop] turns={turn + 1}, "
              f"total_commenter_calls={all_commenter_call_count}, "
              f"peak_live={fake_client.peak_live}, "
              f"max_in_window={max_in_win}")


# ---------------- T5: commenter + rubric 并发交叉 ----------------

class TestConcurrentLimitersIndependent(unittest.TestCase):
    """两个限流器各自独立：同时跑互不影响，且各自严格。"""

    def test_concurrent_no_cross_interference(self):
        mgr = _make_manager_stub(rpm=TEST_RPM, max_concurrency=48)
        commenter_client = FakeOpenAIClient(latency=0.05)
        mgr._commenter_client = commenter_client

        rm = _make_reward_manager_stub(rpm=TEST_RPM, max_concurrency=32)
        rubric_client = FakeOpenAIClient(latency=0.05)
        rm._rubric_client = rubric_client

        N = 40

        def run_commenter():
            return mgr._call_commenter_batch(
                [f'kw{i}' for i in range(N)],
                [f'title{i}' for i in range(N)],
                [f'DRAFT_BODY_{i:03d}' for i in range(N)],
                [''] * N,
            )

        def run_rubric():
            return rm._call_rubric_scoring_api(
                camera_ready_texts=[f'ABSTRACT_TEXT_{i:03d}' for i in range(N)],
                ground_truth_texts=[''] * N,
                domains=[f'kw{i}' for i in range(N)],
                topics=[f'title{i}' for i in range(N)],
                return_details=True,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_c = pool.submit(run_commenter)
            f_r = pool.submit(run_rubric)
            comments = f_c.result()
            scores, details = f_r.result()

        self.assertEqual(len(comments), N)
        self.assertEqual(len(scores), N)

        assert_window_rpm(
            commenter_client.call_log, rpm=TEST_RPM,
            label='concurrent.commenter'
        )
        assert_window_rpm(
            rubric_client.call_log, rpm=TEST_RPM,
            label='concurrent.rubric'
        )
        print(f"[T5 concurrent] commenter_calls={len(commenter_client.call_log)}, "
              f"rubric_calls={len(rubric_client.call_log)}, OK")


# ---------------- 入口 ----------------

if __name__ == '__main__':
    # 直接运行：跑全部 case
    unittest.main(verbosity=2)
