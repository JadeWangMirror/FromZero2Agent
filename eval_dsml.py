"""
MIRROR DSML 工具调用兜底 —— 证明模型以 DSML 文本吐工具调用时,工具仍能被触发。

DeepSeek 流式端点有时不返回原生 Anthropic tool_use 块,而是把工具调用以 DSML
文本塞进 text。MIRROR 原先不认 → 工具永不执行 → 会话卡死断开。
本评估用用户实际遇到的原文验证解析正确,且原生路径不受影响。
"""
import llm

PASS, FAIL = [], []
def check(n, c, d=""):
    (PASS if c else FAIL).append(n)
    print(f"  [{'OK' if c else 'XX'}] {n}{(' — '+d) if d else ''}")

# 用户实际遇到的原文(逐字)
USER_DSML = (
    '< | DSML | tool_calls>\n'
    '< | DSML | invoke name="read_file">\n'
    '< | DSML | parameter name="path" string="true">D:\Desktop\test\parser.py</ | DSML | parameter>\n'
    '</ | DSML | invoke>\n'
    '</ | DSML | tool_calls>'
)

print("[1] 用户原文 → 正确解析成 read_file(path=...)")
calls = llm._parse_dsml_tool_calls(USER_DSML)
check("解析出 1 个调用", len(calls) == 1)
check("工具名 read_file", calls and calls[0]["name"] == "read_file")
check("path 参数原样(反斜杠不丢)",
      calls and calls[0]["input"].get("path") == r"D:\Desktop	est\parser.py")

print("\n[2] 多工具 + 数值/字符串类型还原")
multi = (
    '< | DSML | tool_calls>\n'
    '< | DSML | invoke name="screenshot_url">\n'
    '< | DSML | parameter name="url" string="true">https://example.com</ | DSML | parameter>\n'
    '< | DSML | parameter name="width" string="false">1280</ | DSML | parameter>\n'
    '</ | DSML | invoke>\n'
    '< | DSML | invoke name="calculator">\n'
    '< | DSML | parameter name="expression" string="true">2+2</ | DSML | parameter>\n'
    '</ | DSML | invoke>\n'
    '</ | DSML | tool_calls>'
)
c2 = llm._parse_dsml_tool_calls(multi)
check("解析出 2 个调用", len(c2) == 2)
check("width 还原为 int 1280", c2[0]["input"].get("width") == 1280
      and isinstance(c2[0]["input"].get("width"), int))
check("第二个是 calculator", c2[1]["name"] == "calculator")

print("\n[3] _finalize_blocks: DSML text → tool_use 块")
out = llm._finalize_blocks([{"type": "text", "text": USER_DSML}])
check("DSML text 被转成 tool_use",
      any(b.get("type") == "tool_use" and b.get("name") == "read_file" for b in out))
check("DSML 原文不再残留",
      not any("DSML" in b.get("text", "") for b in out if b.get("type") == "text"))

print("\n[4] 混合:自然语言前缀 + DSML → 保留前缀")
mixed = 'Let me check that file.\n' + USER_DSML
out2 = llm._finalize_blocks([{"type": "text", "text": mixed}])
check("前缀自然语言保留",
      any("Let me check" in b.get("text", "") for b in out2 if b.get("type") == "text"))
check("同时产出 tool_use",
      any(b.get("type") == "tool_use" for b in out2))

print("\n[5] 原生路径不受影响")
native = [{"type": "text", "text": "hi"},
          {"type": "tool_use", "id": "t1", "name": "x", "input": {}}]
out3 = llm._finalize_blocks(native)
check("有原生 tool_use → 不触发 DSML 解析", out3 == native)
plain = llm._finalize_blocks([{"type": "text", "text": "just chatting"}])
check("普通文本 → 原样", plain == [{"type": "text", "text": "just chatting"}])

print("\n" + "=" * 60)
print(f"PASS: {len(PASS)}   FAIL: {len(FAIL)}")
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    raise SystemExit(1)
print("\nPASS: 模型以 DSML 文本吐工具调用时,MIRROR 仍能正确触发工具;")
print("      原生 tool_use 路径完全不受影响。")
