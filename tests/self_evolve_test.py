"""
自进化系统确定性测试 — 不调用 LLM，纯逻辑验证。

运行: python tests/self_evolve_test.py
覆盖: 查重 / 决策(BUILD vs SKIP) / 创建+回滚 / 审查 / 改进 /
      使用统计 / 热更新 / 组合复用 / 持久化重载。
"""

from __future__ import annotations

import os
import sys
import shutil

# 让脚本能 import 项目根目录的模块
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from filetools import create_file_tools          # noqa: E402
from tools import ToolRegistry, create_default_registry  # noqa: E402
from toolforge import ToolForge                  # noqa: E402
from webtools import create_web_tools            # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}  {detail}")


def build_forge() -> tuple[ToolForge, ToolRegistry]:
    reg = create_default_registry()
    for t in create_file_tools():
        reg.register(t)
    for t in create_web_tools():
        reg.register(t)
    forge = ToolForge(reg)
    reg._stats_sink = forge.record_usage
    return forge, reg


# ── 各测试用例 ────────────────────────────────────────────

def test_find_similar():
    print("\n[test] find_similar_tools")
    forge, _ = build_forge()
    r = forge.find_similar_tools("read a file")
    check("read_file ranked #1 for 'read a file'", "read_file" in r.split("\n")[1])
    check("reports gap for novel capability",
          "gap" in forge.find_similar_tools("compute orbital trajectory").lower())


def test_propose_skip_duplicate():
    print("\n[test] propose_tool SKIP (duplicate)")
    forge, _ = build_forge()
    out = forge.propose_tool("read the content of a file", reuse_signal="recurring")
    check("verdict SKIP", "VERDICT: SKIP" in out, out)
    check("recommends read_file", "read_file" in out, out)


def test_propose_skip_once():
    print("\n[test] propose_tool SKIP (one-off)")
    forge, _ = build_forge()
    out = forge.propose_tool("double the number 5", reuse_signal="once")
    check("verdict SKIP for one-off", "VERDICT: SKIP" in out, out)
    check("points to run_python", "run_python" in out, out)


def test_propose_build():
    print("\n[test] propose_tool BUILD (real gap)")
    forge, _ = build_forge()
    out = forge.propose_tool("render a 3d mesh from obj file", reuse_signal="recurring")
    check("verdict BUILD for novel recurring capability", "VERDICT: BUILD" in out, out)
    check("gives suggested name", "suggested_name" in out, out)


def test_create_rollback_on_test_fail():
    print("\n[test] create_tool rolls back on failing test")
    forge, reg = build_forge()
    out = forge.create_tool(
        "rb_bad", "bad tool", {"x": {"type": "integer"}},
        "def execute(x=0):\n    return x + 100",  # 故意写错
        "assert execute(1) == 999",  # 必然失败的测试
    )
    check("reports test failure", "test FAILED" in out or "FAILED" in out, out)
    check("tool NOT registered", reg.get("rb_bad") is None)
    check("tool dir removed", not forge._exists("rb_bad"))


def test_create_and_use():
    print("\n[test] create_tool + use")
    forge, reg = build_forge()
    out = forge.create_tool(
        "cu_wordcount", "Count words in a string.", {"text": {"type": "string"}},
        "def execute(text=\"\"):\n    return str(len(str(text).split()))",
        "assert execute(\"a b c\") == \"3\"",
    )
    check("created", "created" in out.lower(), out)
    check("usable via registry", reg.execute("cu_wordcount", {"text": "one two three four"}) == "4")
    # 审查应发现缺 try/except
    rev = forge.review_tool("cu_wordcount")
    check("review flags missing error handling", "error handling" in rev and "WARN" in rev, rev)
    forge.delete_tool("cu_wordcount")


def test_review_and_improve():
    print("\n[test] review_tool + improve_tool")
    forge, reg = build_forge()
    forge.create_tool(
        "ri_tool", "Demo tool.", {"n": {"type": "integer"}},
        "def execute(n=0):\n    return str(n * 2)",
    )  # 无测试、无 try、无校验
    rev = forge.review_tool("ri_tool")
    check("review runs", "Review of 'ri_tool'" in rev, rev)
    check("flags missing test", "test coverage" in rev and "WARN" in rev, rev)
    imp = forge.improve_tool("ri_tool", focus="add error handling + test")
    check("improve gives fix direction", "FOCUS" in imp and "update_tool" in imp, imp)
    # 修正后复审：错误处理 + 测试应 PASS
    forge.update_tool(
        "ri_tool",
        code="def execute(n=0):\n    try:\n        return str(int(n) * 2)\n    except Exception as e:\n        return f\"Error: {e}\"",
        test_code="assert execute(3) == \"6\"",
    )
    rev2 = forge.review_tool("ri_tool")
    check("error handling now PASS", "error handling: PASS" in rev2, rev2)
    check("test coverage now PASS", "test coverage: PASS" in rev2, rev2)
    forge.delete_tool("ri_tool")


def test_usage_tracking():
    print("\n[test] usage tracking + stats")
    forge, reg = build_forge()
    for _ in range(7):
        reg.execute("calculator", {"expression": "1+1"})
    for _ in range(3):
        reg.execute("get_current_time", {})
    stats = forge.usage_stats()
    check("calculator recorded 7 calls", "calculator" in stats and "calls=7" in stats, stats)
    check("get_current_time recorded 3 calls", "calls=3" in stats, stats)
    # 强制落盘 + 重载
    forge._save_stats()
    forge2, _ = build_forge()  # 重新构造会 _load_stats
    stats2 = forge2.usage_stats()
    check("stats persisted across restart", "calls=7" in stats2, stats2)
    # unused 视图
    unused = forge.usage_stats(detail="unused")
    check("unused view lists 0-call tools", "unused" in unused.lower() or "low-use" in unused.lower(), unused)
    # 清理
    if os.path.isfile(forge._stats_path):
        os.remove(forge._stats_path)


def test_hot_reload():
    print("\n[test] update_tool hot-reload (no stale code)")
    forge, reg = build_forge()
    forge.create_tool(
        "hr_ver", "versioned.", {"x": {"type": "integer"}},
        "def execute(x=0):\n    return \"v1\"",
        "assert execute() == \"v1\"",
    )
    check("initial v1", reg.execute("hr_ver", {}) == "v1")
    forge.update_tool("hr_ver", code="def execute(x=0):\n    return \"v2\"")
    check("after update returns v2", reg.execute("hr_ver", {}) == "v2", "stale code!")
    forge.delete_tool("hr_ver")


def test_composition():
    print("\n[test] tool composition via use()")
    forge, reg = build_forge()
    # word_count 利用 use('read_file') 读文件再数词
    forge.create_tool(
        "cmp_wcfile", "Count words in a file via composition.",
        {"path": {"type": "string"}},
        "def execute(path):\n"
        "    content = use('read_file', path=path)\n"
        "    return str(len(content.split()))",
        "assert True",  # 轻量测试，use 在测试沙箱内不可用故只校验语法
    )
    # 实际执行：写一个临时文件再数
    reg.execute("write_file", {"path": "_cmp_test.txt", "content": "hello world from spark"})
    n = reg.execute("cmp_wcfile", {"path": "_cmp_test.txt"})
    check("composition reads file + counts words", n == "4", n)
    reg.execute("delete_file", {"path": "_cmp_test.txt"})
    forge.delete_tool("cmp_wcfile")


def test_self_evolve_flow():
    print("\n[test] self_evolve orchestration")
    forge, _ = build_forge()
    out = forge.self_evolve("I keep converting json to csv by hand")
    check("self_evolve has verdict section", "VERDICT" in out, out)
    check("self_evolve has gap analysis", "CAPABILITY GAP" in out, out)


def main() -> int:
    print("=" * 60)
    print("SPARK self-evolution test suite")
    print("=" * 60)
    # 清理任何残留测试工具
    forge_clean, _ = build_forge()
    for name in ["rb_bad", "cu_wordcount", "ri_tool", "hr_ver", "cmp_wcfile", "tc_test", "char_count"]:
        if forge_clean._exists(name):
            forge_clean.delete_tool(name)

    for fn in [
        test_find_similar,
        test_propose_skip_duplicate,
        test_propose_skip_once,
        test_propose_build,
        test_create_rollback_on_test_fail,
        test_create_and_use,
        test_review_and_improve,
        test_usage_tracking,
        test_hot_reload,
        test_composition,
        test_self_evolve_flow,
    ]:
        try:
            fn()
        except Exception as e:
            global _failed
            _failed += 1
            print(f"  [FAIL] {fn.__name__} raised: {type(e).__name__}: {e}")

    # 最终清理
    for name in ["rb_bad", "cu_wordcount", "ri_tool", "hr_ver", "cmp_wcfile"]:
        if forge_clean._exists(name):
            forge_clean.delete_tool(name)
    if os.path.isfile(os.path.join(forge_clean.custom_dir, "_usage.json")):
        os.remove(os.path.join(forge_clean.custom_dir, "_usage.json"))

    print("\n" + "=" * 60)
    print(f"Result: {_passed} passed, {_failed} failed")
    print("=" * 60)
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
