"""
MIRROR 主动性评估 —— 证明记忆/自进化不再被动等待 agent 手动触发。

核心断言:
  1. 空记忆 → _maybe_auto_consolidate 是 no-op(成本闸门:不花 LLM)
  2. 攒够挣扎信号(≥2)→ 自动折叠 → reflect 涌现 capability gap
  3. 缺口在 context_block 里以「命令式」呈现(指向 self_evolve)
  4. 折叠自消耗信号(pending_signals → 0),天然自限,不每轮重复
  5. 低信号(<2)→ 不触发折叠(闸门有效)

对照: 改动前缺口永不涌现(consolidate_memory 从不被自动调),
       agent 看不到 ⚡ → 永远不会主动 self_evolve。这就是被动根因。
"""

import os
import tempfile

import memory
from agent import Agent
from tools import ToolRegistry

PASS, FAIL = [], []


def _check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'XX'}] {name}{(' — ' + detail) if detail else ''}")


def _stub(prompt, system="x"):
    """LLM 替身: 只对 reflect 风格 prompt 返回一条 BOUNDARY。"""
    low = prompt.lower()
    if "struggl" in low or "boundary" in low or "genuine" in low:
        return "BOUNDARY: parse pdf documents to text | because: repeated pdf_to_text failures"
    return ""


def main() -> int:
    tmp = tempfile.mkdtemp()
    memory._G = None
    memory.MEMORY_DIR = tmp
    memory.GRAPH_FILE = os.path.join(tmp, "graph.json")
    memory.LEGACY_FILE = os.path.join(tmp, "memory.json")

    ag = Agent(api_key="dummy", tools=ToolRegistry())
    ag._llm_summarize = _stub

    print("[1] 成本闸门: 空记忆不触发折叠")
    before = memory.graph_stats()
    ag._maybe_auto_consolidate()
    _check("empty memory → no-op", memory.graph_stats() == before,
           "graph unchanged (no LLM spent)")

    print("\n[2] 攒够信号 → 自动折叠 → 涌现缺口")
    for _ in range(3):
        memory.record_signal("failure", "pdf_to_text failed: no pdf tool available")
    _check("3 signals queued", memory.pending_signals() == 3)
    ag._maybe_auto_consolidate()
    gaps = memory.capability_gaps()
    _check("auto-fold surfaced a gap", bool(gaps),
           (gaps[0][1][:50] if gaps else "(none)"))

    print("\n[3] 缺口在 context 里命令式呈现")
    ctx = memory.context_block()
    _check("context directs self_evolve", "self_evolve()" in ctx and "THIS turn" in ctx)

    print("\n[4] 折叠自消耗信号(自限)")
    _check("signals consumed after fold", memory.pending_signals() == 0,
           "reflected →不会再重复触发")

    print("\n[5] 低信号不触发(<2)")
    memory._G = None
    memory.MEMORY_DIR = tempfile.mkdtemp()
    memory.GRAPH_FILE = os.path.join(memory.MEMORY_DIR, "graph.json")
    memory.LEGACY_FILE = os.path.join(memory.MEMORY_DIR, "memory.json")
    memory.record_signal("failure", "one-off blip")
    _check("1 signal present", memory.pending_signals() == 1)
    folded = ag._maybe_auto_consolidate()
    _check("below threshold → no gap created", not memory.capability_gaps(),
           "cost gate held (<2 signals, <8 events)")

    print("\n" + "=" * 60)
    print(f"PASS: {len(PASS)}   FAIL: {len(FAIL)}")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        return 1
    print("\nPASS: memory is ACTIVE — struggle auto-folds into capability gaps,")
    print("surfaced imperatively, without the agent manually calling consolidate_memory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
