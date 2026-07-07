"""
MIRROR 边界检测评估 —— 反思 pass 是否真能浮现"能力边界",且不误报合法重复。

这是用户的核心诉求:"意图浮现要能解决'什么时候该造工具 / 哪里是我的能力边界'"。

验证两件事:
  1. 召回: 失败/绕路信号 → 反思后浮现为 capability_gap(真正的边界被看见)
  2. 精度(架构保证): 合法的普通重复(如读 5 个文件)永不被当成边界 —— 因为
     reflect 只看 signal 事件,普通 event 根本不进它的视野。

运行: python eval_boundary.py
"""

import os
import tempfile

import memory


def _setup(tmp):
    memory._G = None
    memory.MEMORY_DIR = tmp
    memory.GRAPH_FILE = os.path.join(tmp, "graph.json")
    memory.LEGACY_FILE = os.path.join(tmp, "memory.json")
    if os.path.exists(memory.GRAPH_FILE):
        os.remove(memory.GRAPH_FILE)


def main() -> None:
    tmp = tempfile.mkdtemp()
    _setup(tmp)

    # ── 真边界 A: PDF 解析失败 x3(失败是最强的边界信号)──
    for _ in range(3):
        memory.record_signal("failure",
                             "pdf_to_text failed: no tool to parse pdf documents")
    # ── 真边界 B: markdown→slides 手搓绕路 x2 ──
    memory.record_signal("workaround",
                         "manual python: convert markdown to pptx slides (28 lines)")
    memory.record_signal("workaround",
                         "manual python: build slide deck from md (31 lines)")
    # ── 合法重复(非信号): 读多个文件 —— 绝不该被当边界 ──
    for fn in ["a.py", "b.py", "c.py", "d.py", "e.py"]:
        memory.remember(f"read file {fn} to understand structure")

    leaked = {"v": False}

    def summarizer(prompt: str) -> str:
        """确定性反思器:识别 pdf/slides 边界。同时检测是否误喂了合法重复。"""
        low = prompt.lower()
        if ("read file" in low) or ("read_file" in low):
            leaked["v"] = True       # 普通事件不该出现在 reflect 的输入里
        out = []
        if "pdf" in low:
            out.append("BOUNDARY: parse PDF documents to text | because: 3 pdf failures")
        if ("slides" in low) or ("pptx" in low) or ("markdown" in low):
            out.append("BOUNDARY: convert markdown to slides | because: manual python workarounds")
        return "\n".join(out)

    report = memory.reflect(summarizer)
    gaps = memory.capability_gaps()
    print("REFLECT:", report)
    print("surfaced boundaries:", [s for _, s in gaps])
    print("legitimate read-file repetition leaked into reflect?", leaked["v"])
    print()

    # ── 断言 ──
    assert any("pdf" in s.lower() for _, s in gaps), "FAIL: PDF boundary should surface"
    assert any(("slides" in s.lower()) or ("markdown" in s.lower())
               for _, s in gaps), "FAIL: slides boundary should surface"
    assert not leaked["v"], "FAIL: legitimate read_file repetition leaked into reflect"
    # 反思过的信号被标记,不会重复评估
    g = memory._graph()
    unreflected = sum(1 for _, d in g.nodes(data=True)
                      if d.get("data", {}).get("signal") and not d.get("reflected"))
    assert unreflected == 0, "FAIL: reflected signals should be marked"

    print("PASS:")
    print("  - real boundaries (PDF parse, markdown→slides) surfaced from struggle signals;")
    print("  - legitimate repetition (reading 5 files) correctly IGNORED (precision by architecture: ")
    print("    reflect only sees signal events, never plain ones);")
    print("  - reflected signals marked to avoid re-assessment.")
    print()
    print(memory.context_block())


if __name__ == "__main__":
    main()
