"""
MIRROR 记忆评估台 —— 扁平全量注入 vs 图涌现注入,头对头。

回答用户的核心问题:"这套复杂图框架,在 MIRROR 真实规模下,真的比简单记忆记录好吗?
好在哪?从什么规模开始才划算?"

指标(固定 LLM 质量,用确定性 summarizer,只测图基座):
  - 注入字符成本: 每轮进 system prompt 的字符数(扁平线性增长,图有界)
  - 命中率: 一组探针任务的答案是否出现在注入的上下文里
  - 压缩比: flat_chars / graph_chars

语料: 多主题事件流,含重复模式(应折叠成 concept)、截止、一次性噪声。
运行: python eval_memory.py
"""

from __future__ import annotations

import os
import re
import tempfile

import memory

# ── 语料: 主题 → 该主题的若干事件变体(重复出现 → 折叠候选) ──
TOPICS = {
    "poetry": [  # 强重复(strength≈4 → 会结晶 intent)
        "user uses poetry for python package management",
        "user runs poetry to manage dependencies",
        "project uses poetry never pip for installs",
        "remember poetry is the package manager here",
    ],
    "neovim": [  # 中重复(strength≈3)
        "user prefers neovim as editor",
        "user edits code in neovim",
        "neovim is the users editor of choice",
    ],
    "timezone": [  # 弱重复(strength≈2, 低于 INTENT_THRESHOLD, 留在 episodic)
        "user is in UTC+8 timezone",
        "user local time is UTC plus 8",
    ],
    "snakecase": [  # 中重复
        "user prefers snake_case for variable names",
        "code style uses snake_case not camelCase",
        "naming convention is snake_case",
    ],
}

DEADLINES = [
    "release v2 by Friday this week",
    "ship the API migration before next Monday",
]

NOISE = [
    "looked up python GIL article",
    "checked the weather forecast",
    "read a blog post about rust ownership",
    "browsed hacker news briefly",
    "searched for a regex cheat sheet",
]

# 探针: (描述, 答案子串, 期望命中) —— 答案必须在注入上下文里出现
PROBES = [
    ("package manager pref", "poetry"),
    ("editor pref", "neovim"),
    ("timezone", "UTC"),
    ("naming style", "snake_case"),
    ("release deadline", "Friday"),
]


def _gen_events(n: int, rng) -> list[str]:
    """生成 ~n 条事件: 各主题重复若干次 + 截止 + 噪声。"""
    evs: list[str] = []
    # 每个主题完整加入(保证有重复模式)
    for variants in TOPICS.values():
        evs.extend(variants)
    evs.extend(DEADLINES)
    # 用噪声 + 主题循环填充到 n
    pool = NOISE + [v for vs in TOPICS.values() for v in vs] + DEADLINES
    while len(evs) < n:
        evs.append(rng.choice(pool))
    return evs[:n]


def _det_summarizer(prompt: str) -> str:
    """确定性 summarizer: 按主题关键词给规范 concept 语句。
    固定 LLM 质量为'完美抽象',从而只评估图基座(聚类/压缩/分级)本身。"""
    out = []
    blocks = prompt.split("[")[1:]
    for i, block in enumerate(blocks, 1):
        bl = block.lower()
        if "poetry" in bl:
            out.append(f"{i}. User uses poetry for Python package management")
        elif "neovim" in bl:
            out.append(f"{i}. User prefers neovim as editor")
        elif "snake_case" in bl or "snake" in bl:
            out.append(f"{i}. User prefers snake_case naming")
        elif "utc" in bl or "timezone" in bl:
            out.append(f"{i}. User is in UTC+8 timezone")
        elif "friday" in bl or "release" in bl or "monday" in bl:
            out.append(f"{i}. User has upcoming release deadlines")
        else:
            out.append(f"{i}. Miscellaneous user activity")
    return "\n".join(out)


def _flat_inject(events: list[str]) -> str:
    """扁平基线: 全量事件拼进上下文(旧 remember/forget 系统的行为)。"""
    return "MEMORY:\n" + "\n".join(f"- {e}" for e in events)


def _graph_inject(events: list[str], tmp_dir: str) -> str:
    """图路径: remember 全部 → consolidate(确定性) → select_context。"""
    # 重置到临时图
    memory._G = None
    memory.MEMORY_DIR = tmp_dir
    memory.GRAPH_FILE = os.path.join(tmp_dir, "graph.json")
    memory.LEGACY_FILE = os.path.join(tmp_dir, "memory.json")
    if os.path.exists(memory.GRAPH_FILE):
        os.remove(memory.GRAPH_FILE)
    for e in events:
        memory.remember(e)
    memory.consolidate(_det_summarizer)
    return memory.context_block()


def _hit_rate(text: str) -> tuple[int, int, int]:
    """返回 (命中数, 总探针, 漏掉的探针数)。"""
    low = text.lower()
    hit = sum(1 for _, ans in PROBES if ans.lower() in low)
    return hit, len(PROBES), len(PROBES) - hit


def main() -> None:
    import random
    rng = random.Random(42)
    sizes = [12, 24, 50, 100, 200]

    print("MIRROR memory eval — flat (inject-all) vs graph (emergent)\n")
    print(f"{'N':>5} | {'flat chars':>10} | {'flat hit':>9} | "
          f"{'graph chars':>11} | {'graph hit':>10} | {'compress':>8} | {'graph misses'}")
    print("-" * 95)

    for n in sizes:
        events = _gen_events(n, rng)
        tmp = tempfile.mkdtemp()

        flat = _flat_inject(events)
        graph = _graph_inject(events, tmp)

        fh, _, fm = _hit_rate(flat)
        gh, _, gm = _hit_rate(graph)
        compress = len(flat) / max(1, len(graph))

        print(f"{n:>5} | {len(flat):>10} | {fh}/{len(PROBES):<6} | "
              f"{len(graph):>11} | {gh}/{len(PROBES):<7} | "
              f"{compress:>7.2f}x | {gm}")

    print("\nReading guide:")
    print("- flat chars grows ~linearly with N; graph chars is bounded (capped bands).")
    print("- flat hit is always full (everything injected). graph hit drops when weak/old")
    print("  facts fall outside the capped immediate/working bands — that's the real cost.")
    print("- compress = flat/graph: the crossover where graph's savings justify its complexity.")


if __name__ == "__main__":
    main()
