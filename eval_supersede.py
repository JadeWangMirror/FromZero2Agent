"""
MIRROR 矛盾/取代评估 —— 记忆能否正确更新过时信息。

扁平记忆的硬伤:用户从 poetry 切到 uv,它会同时留下"用 poetry"和"用 uv",
自相矛盾。图应识别取代关系,把旧的标 superseded 不再浮现。
"""

import os
import tempfile

import memory


def det(prompt: str) -> str:
    """确定性 LLM 替身:区分 consolidate 与 resolve_supersede 两类 prompt。"""
    low = prompt.lower()
    if "supersede" in low and "replaces" in low:
        return ("SUPERSEDE: user switched to uv for python packages replaces "
                "user uses poetry for python packages")
    out = []
    for i, block in enumerate(prompt.split("[")[1:], 1):
        bl = block.lower()
        if "poetry" in bl:
            out.append(f"{i}. User uses poetry for python packages")
        elif "uv" in bl:
            out.append(f"{i}. User switched to uv for python packages")
        else:
            out.append(f"{i}. misc")
    return "\n".join(out)


def main() -> int:
    tmp = tempfile.mkdtemp()
    memory._G = None
    memory.MEMORY_DIR = tmp
    memory.GRAPH_FILE = os.path.join(tmp, "graph.json")
    memory.LEGACY_FILE = os.path.join(tmp, "memory.json")

    # 1) 先建立 poetry concept
    for c in ["user uses poetry for python packages",
              "user runs poetry for python deps",
              "project uses poetry for python"]:
        memory.remember(c)
    memory.consolidate(det)
    poetry = [n for n, d in memory._graph().nodes(data=True)
              if d.get("type") == "concept" and "poetry" in d["data"].get("statement", "").lower()]
    assert poetry, "FAIL: poetry concept should exist"
    print("poetry concept established:", poetry[0])

    # 2) 后来切到 uv
    for c in ["user switched to uv for python packages",
              "user now uses uv for python deps",
              "project migrated to uv for python"]:
        memory.remember(c)
    memory.consolidate(det)
    uv = [n for n, d in memory._graph().nodes(data=True)
          if d.get("type") == "concept" and "uv" in d["data"].get("statement", "").lower()]
    assert uv, "FAIL: uv concept should exist"
    print("uv concept established:", uv[0])

    # 3) resolve_supersede
    rep = memory.resolve_supersede(det)
    print("resolve_supersede:", rep)

    # 4) 断言: poetry 被标 superseded, uv 仍活跃
    g = memory._graph()
    pstatus = g.nodes[poetry[0]]["data"].get("status")
    ustatus = g.nodes[uv[0]]["data"].get("status")
    print(f"poetry status={pstatus}, uv status={ustatus}")
    assert pstatus == "superseded", "FAIL: outdated poetry concept should be superseded"
    assert ustatus != "superseded", "FAIL: uv (the replacement) should stay active"

    # 5) context_block 不该再浮现 poetry, 应浮现 uv
    ctx = memory.context_block().lower()
    assert "poetry" not in ctx, "FAIL: superseded poetry should not surface in context"
    assert "uv" in ctx, "FAIL: active uv should surface"
    print("\ncontext_block (working band):")
    print(ctx)

    print("\nPASS: supersede works — outdated concept marked superseded & hidden; "
          "replacement surfaces. Memory updates instead of contradicting itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
