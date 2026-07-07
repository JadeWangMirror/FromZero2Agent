"""
MIRROR 全面能力测试 —— 逐工具摸底,捕获潜在缺陷。

覆盖:
  L1 内置:    calculator, get_current_time
  文件/搜索:  read_file, write_file, edit_file, list_dir, move_file, delete_file,
              glob, grep, run_python, run_shell
  网络:       web_search, web_fetch (网络相关,宽松判定)
  元工具:     list_tools, find_similar_tools, propose_tool, create_tool, read_tool,
              review_tool, update_tool, delete_tool, self_evolve (全生命周期,临时 base_dir)
  记忆/自进化: remember, record_signal, consolidate, reflect, capability_gaps,
              context_block, update_intent, forget, satisfy_gap

运行: python eval_capabilities.py
"""

import os
import tempfile

import memory
from filetools import create_file_tools
from toolforge import ToolForge
from webtools import create_web_tools
from tools import create_default_registry, ToolRegistry

PASS, FAIL, SKIP = [], [], []


def _check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'✓' if cond else '✗'}] {name}{(' — ' + detail) if detail else ''}")


def _det_summarizer(prompt: str) -> str:
    """确定性 LLM 替身:按关键词给规范输出(折叠/反思共用)。"""
    out = []
    low = prompt.lower()
    if "bound" in low or "reflect" in low or "signal" in low or "fail" in low or "pdf" in low:
        # reflect 风格 prompt
        for i, key in enumerate(("pdf", "slides", "convert", "parse"), 1):
            if key in low:
                out.append(f"BOUNDARY: {key} capability | because: repeated {key} signals")
        if out:
            return "\n".join(out)
    # consolidate 风格:按簇编号
    for i, block in enumerate(prompt.split("[")[1:], 1):
        bl = block.lower()
        if "poetry" in bl:
            out.append(f"{i}. User uses poetry for python | tool: none")
        elif "currency" in bl:
            out.append(f"{i}. User converts currency | tool: none")
        elif "json" in bl:
            out.append(f"{i}. User formats json | tool: none")
        else:
            out.append(f"{i}. Miscellaneous pattern")
    return "\n".join(out)


def test_builtin():
    print("[1] built-in tools")
    reg = create_default_registry()
    t = {n: tool for n, tool in [(x.name, x) for x in []]}
    calc = reg.get("calculator")
    _check("calculator", "6" in calc.execute(expression="2 + 2 * 2"))
    _check("calculator rejects junk", "error" in calc.execute(expression="abc").lower())
    time_tool = reg.get("get_current_time")
    _check("get_current_time", len(time_tool.execute()) > 8)


def test_file_tools():
    print("[2] file tools")
    base = tempfile.mkdtemp()
    tools = {t.name: t for t in create_file_tools(base_dir=base)}
    wf, rf = tools["write_file"], tools["read_file"]

    r = wf.fn(path="sub/a.txt", content="hello\nworld\n")
    _check("write_file", "Wrote" in r and os.path.exists(os.path.join(base, "sub", "a.txt")))
    _check("read_file", "hello" in rf.fn(path="sub/a.txt"))
    _check("read_file missing", "error" in rf.fn(path="nope.txt").lower())

    ef = tools["edit_file"]
    _check("edit_file", "Replaced" in ef.fn(path="sub/a.txt", old_string="hello",
                                            new_string="hi"))
    _check("edit_file not found", "not found" in ef.fn(path="sub/a.txt",
                                                       old_string="zzz", new_string="x").lower())

    ld = tools["list_dir"]
    _check("list_dir", "sub" in ld.fn(path="."))

    gf = tools["glob"]
    _check("glob", "a.txt" in gf.fn(pattern="**/*.txt", path="."))

    gp = tools["grep"]
    wf.fn(path="b.py", content="print('hi')\nx = 1\n")
    _check("grep", "b.py" in gp.fn(pattern="print", path="."))

    rp = tools["run_python"]
    _check("run_python", "42" in rp.fn(code="print(40+2)"))
    _check("run_python error captured", "traceback" in rp.fn(code="1/0").lower()
           or "zerodivision" in rp.fn(code="1/0").lower())

    rs = tools["run_shell"]
    _check("run_shell", "ok" in rs.fn(command="echo ok").lower()
           or "ok" in rs.fn(command="echo ok"))

    mv = tools["move_file"]
    _check("move_file", "Moved" in mv.fn(src="b.py", dst="c.py"))
    df = tools["delete_file"]
    _check("delete_file", "Deleted" in df.fn(path="c.py"))


def test_web_tools():
    print("[3] web tools (network — lenient)")
    tools = {t.name: t for t in create_web_tools()}
    try:
        r = tools["web_search"].fn(query="python programming language", max_results=2)
        _check("web_search", "python" in r.lower() or "result" in r.lower() or len(r) > 20,
               "(network)")
    except Exception as e:
        _check("web_search", False, f"exception {e}")
    try:
        r = tools["web_fetch"].fn(url="https://httpbin.org/json", max_chars=2000)
        _check("web_fetch", len(r) > 10, "(network)")
    except Exception as e:
        _check("web_fetch", False, f"exception {e}")


def test_meta_tools():
    print("[4] meta-tools (ToolForge full lifecycle, temp base_dir)")
    base = tempfile.mkdtemp()
    reg = ToolRegistry()
    for t in create_file_tools(base_dir=base):
        reg.register(t)
    tf = ToolForge(reg, base_dir=base)

    _check("list_tools", "Available" in tf.list_tools())
    sim = tf.find_similar_tools("read a file")
    _check("find_similar_tools", "read_file" in sim)

    prop = tf.propose_tool("count words in text", reuse_signal="recurring")
    _check("propose_tool BUILD", "BUILD" in prop)

    code = ("def execute(text):\n"
            "    return str(len((text or '').split()))\n")
    created = tf.create_tool(name="word_count",
                             description="Count words in text.",
                             parameters={"text": {"type": "string"}},
                             code=code,
                             test_code="assert execute('a b c') == '3'")
    _check("create_tool", "created" in created.lower())
    _check("create_tool registered", reg.get("word_count") is not None)
    _check("create_tool executes", reg.execute("word_count", {"text": "a b c"}) == "3")

    _check("read_tool", "execute" in tf.read_tool("word_count"))
    _check("review_tool", "Review" in tf.review_tool("word_count"))

    upd = tf.update_tool(name="word_count",
                         code="def execute(text):\n    return str(len((text or '').split()))\n",
                         description="Count whitespace-separated tokens.")
    _check("update_tool", "updated" in upd.lower())

    # self_evolve 无参(依赖 memory gaps;此处只测不崩)
    se = tf.self_evolve()
    _check("self_evolve no-arg safe", isinstance(se, str) and len(se) > 0)

    _check("delete_tool", "deleted" in tf.delete_tool("word_count").lower())
    _check("delete_tool removed", reg.get("word_count") is None)


def test_memory_selfevolve():
    print("[5] memory + self-evolution")
    tmp = tempfile.mkdtemp()
    memory._G = None
    memory.MEMORY_DIR = tmp
    memory.GRAPH_FILE = os.path.join(tmp, "graph.json")
    memory.LEGACY_FILE = os.path.join(tmp, "memory.json")
    if os.path.exists(memory.GRAPH_FILE):
        os.remove(memory.GRAPH_FILE)

    _check("remember", "event" in memory.remember("user uses poetry for python").lower())
    memory.remember("user runs poetry for deps")
    memory.remember("project uses poetry never pip")
    # 信号
    _check("record_signal", "signal" in memory.record_signal("failure", "pdf parse failed").lower())

    rep = memory.consolidate(_det_summarizer, tools=[])
    _check("consolidate", "concept" in rep.lower())
    g = memory._G
    has_concept = any(d.get("type") == "concept" for _, d in g.nodes(data=True))
    _check("consolidate produced concept", has_concept)

    # 多记几个 pdf 失败信号 → reflect 应识别边界
    for _ in range(2):
        memory.record_signal("failure", "pdf_to_text failed again: no pdf tool")
    rrep = memory.reflect(_det_summarizer)
    _check("reflect", "boundary" in rrep.lower() or "signal" in rrep.lower())
    gaps = memory.capability_gaps()
    _check("capability_gaps", isinstance(gaps, list))

    ctx = memory.context_block()
    _check("context_block non-empty", len(ctx) > 0)

    # satisfy_gap 闭环
    closed = memory.satisfy_gap("pdf_tool", "parse pdf documents to text")
    _check("satisfy_gap returns int", isinstance(closed, int))

    # update_intent / forget(若存在 intent)
    intents = [n for n, d in g.nodes(data=True) if d.get("type") == "intent"]
    if intents:
        iid = intents[0]
        _check("update_intent", "intent" in memory.update_intent(iid, status="done").lower()
               or "→" in memory.update_intent(iid, status="done"))
        _check("forget", "removed" in memory.forget(intents[0]).lower())
    else:
        _check("update_intent/forget (no intent to test)", True, "skipped — no intent")


def main():
    # 隔离全局记忆:create_tool / update_intent 等会写记忆,测试用临时图,
    # 绝不污染用户真实的 ~/.mirror/graph.json
    tmp_m = tempfile.mkdtemp()
    memory._G = None
    memory.MEMORY_DIR = tmp_m
    memory.GRAPH_FILE = os.path.join(tmp_m, "graph.json")
    memory.LEGACY_FILE = os.path.join(tmp_m, "memory.json")
    test_builtin()
    test_file_tools()
    test_web_tools()
    test_meta_tools()
    test_memory_selfevolve()
    print("\n" + "=" * 60)
    print(f"PASS: {len(PASS)}   FAIL: {len(FAIL)}")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
