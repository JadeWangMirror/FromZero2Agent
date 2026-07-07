"""
MIRROR 自进化闭环评估 —— 证明核心论点成立(可证伪):

  记忆浮现能力边界  →  self_evolve 评估造工具  →  create_tool 落地
                     →  图自动闭环(缺口标 done + covered_by)

这是用户整套记忆系统要解决的事("什么时候该造工具")的端到端验证。
之前各部件单测过,但完整闭环没测过。
"""

import os
import tempfile

import memory
from filetools import create_file_tools
from toolforge import ToolForge
from tools import ToolRegistry


def det_reflect(prompt: str) -> str:
    """确定性反思器:看到 pdf 失败信号 → 产出 pdf 边界。"""
    if "pdf" in prompt.lower():
        return "BOUNDARY: parse pdf documents to text | because: repeated pdf failures"
    return ""


def main() -> int:
    # ── 记忆指向临时图 ──
    tmp_m = tempfile.mkdtemp()
    memory._G = None
    memory.MEMORY_DIR = tmp_m
    memory.GRAPH_FILE = os.path.join(tmp_m, "graph.json")
    memory.LEGACY_FILE = os.path.join(tmp_m, "memory.json")

    # ── 1) 种挣扎信号 → reflect → 浮现 PDF 边界 ──
    for _ in range(3):
        memory.record_signal("failure", "pdf_to_text failed: cannot parse pdf documents")
    rrep = memory.reflect(det_reflect)
    print("reflect:", rrep)
    gaps_before = memory.capability_gaps()
    print("gaps before:", [s for _, s in gaps_before])
    assert any("pdf" in s.lower() for _, s in gaps_before), \
        "FAIL: PDF boundary should surface from failure signals"

    # ── 2) self_evolve 无参 → 应把 PDF gap 纳入评估 ──
    base = tempfile.mkdtemp()
    reg = ToolRegistry()
    for t in create_file_tools(base_dir=base):
        reg.register(t)
    tf = ToolForge(reg, base_dir=base)
    se = tf.self_evolve()
    print("\nself_evolve (excerpt):\n", se[:380], "...")
    assert "pdf" in se.lower(), "FAIL: self_evolve should assess the surfaced pdf gap"

    # ── 3) agent 造工具 —— create_tool 内部自动 satisfy_gap 闭环 ──
    code = "def execute(path):\n    return f'parsed {path} (mock)'\n"
    created = tf.create_tool(
        name="pdf_to_text",
        description="Parse pdf documents to text. Use when you need to read pdf content.",
        parameters={"path": {"type": "string"}},
        code=code,
        test_code="assert 'parsed' in execute('x.pdf')",
    )
    print("\ncreate_tool:", created)
    assert ("satisfied" in created.lower()) or ("gap" in created.lower()), \
        "FAIL: create_tool should report closing the memory gap"

    # ── 4) 验证闭环 ──
    gaps_after = memory.capability_gaps()
    print("gaps after:", [s for _, s in gaps_after])
    assert not any("pdf" in s.lower() for _, s in gaps_after), \
        "FAIL: PDF gap should be closed after create_tool"
    assert reg.get("pdf_to_text") is not None, "FAIL: tool should be registered"

    # ── 5) 图里那个 intent 被标 done + covered_by ──
    g = memory._graph()
    pdf_intent = [d for _, d in g.nodes(data=True)
                  if d.get("type") == "intent"
                  and "pdf" in d["data"].get("statement", "").lower()]
    assert pdf_intent, "FAIL: pdf intent should exist in graph"
    assert pdf_intent[0]["data"].get("status") == "done", \
        "FAIL: pdf intent should be marked done"
    assert pdf_intent[0]["data"].get("covered_by") == "pdf_to_text", \
        "FAIL: pdf intent should record covered_by the new tool"

    tf.delete_tool("pdf_to_text")
    print("\nPASS: full self-evolution loop verified —")
    print("  struggle → boundary surfaced → self_evolve assessed → create_tool built")
    print("  → gap auto-closed (done + covered_by) → graph reflects the capability acquired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
