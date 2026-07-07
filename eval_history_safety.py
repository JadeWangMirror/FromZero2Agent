"""
MIRROR 历史异常安全 —— 异常中断的对话也必须落盘且可恢复。

证明:
  [unit] _finalize_interrupted 三种结尾都补成合法形态
  [integ] run() 中途抛 400 → 异常照常抛出,但上下文已落盘:
          - 用户的指令保留
          - 已完成轮的工具结果保留
          - 历史以 assistant 收尾(轮替合法)
          - 下次请求无孤儿 tool_use(不再触发 400)
"""
import os, tempfile
import memory
from agent import Agent
from tools import ToolRegistry, Tool

PASS, FAIL = [], []
def check(n, c, d=""):
    (PASS if c else FAIL).append(n)
    print(f"  [{'OK' if c else 'XX'}] {n}{(' — '+d) if d else ''}")

class FB:
    def __init__(self, **kw):
        for k,v in kw.items(): setattr(self,k,v)
def text_block(t): return FB(type="text", text=t)
def tool_use_block(name, inp, id): return FB(type="tool_use", id=id, name=name, input=inp)
def resp(*b): return FB(content=list(b))

class HTTP400(Exception): pass

class FakeLLM:
    def __init__(self, script):
        self.script=list(script)
        self.usage={"input":0,"output":0,"calls":0,"cache_read":0,"cache_creation":0}
        self.last={"input":0,"output":0}; self.context_window=128000; self.model="x"
    def send_stream(self, **kw):
        item=self.script.pop(0)
        if isinstance(item, Exception): raise item
        return item

def has_tool_result_for(msgs, tid):
    return any(m.get("role")=="user" and isinstance(m.get("content"),list)
               and any(b.get("tool_use_id")==tid for b in m["content"]) for m in msgs)

def main():
    tmp=tempfile.mkdtemp()
    memory._G=None; memory.MEMORY_DIR=tmp
    memory.GRAPH_FILE=os.path.join(tmp,"g.json"); memory.LEGACY_FILE=os.path.join(tmp,"m.json")

    print("[unit] _finalize_interrupted")
    # 1) orphan tool_use
    m=[{"role":"user","content":"hi"},
       {"role":"assistant","content":[{"type":"tool_use","id":"tu9","name":"x","input":{}}]}]
    Agent._finalize_interrupted(m)
    check("orphan tool_use → 补 tool_result", has_tool_result_for(m,"tu9"))
    check("末尾补 assistant 保轮替", m[-1]["role"]=="assistant")
    # 2) trailing user-text
    m=[{"role":"assistant","content":[{"type":"text","text":"hi"}]},
       {"role":"user","content":"unanswered"}]
    Agent._finalize_interrupted(m)
    check("trailing user-text → 补 assistant", m[-1]["role"]=="assistant")
    # 3) clean assistant text → 不动
    m=[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"text","text":"yo"}]}]
    before=list(m); Agent._finalize_interrupted(m)
    check("clean assistant → 不动", m==before)

    print("\n[integ] run() 中途 400 → 异常抛出但上下文落盘")
    reg=ToolRegistry()
    reg.register(Tool("echo","echo back",{"x":{"type":"string"}},
                      lambda x="":"E:"+str(x), required=["x"]))
    ag=Agent(api_key="dummy", tools=reg)
    ag.llm=FakeLLM([ resp(text_block("calling echo"), tool_use_block("echo",{"x":"1"},"tu1")),
                    HTTP400("tool_use must be followed by tool_result") ])
    raised=False
    try:
        ag.run("please echo 1")
    except Exception:
        raised=True
    check("异常确实抛出(不吞)", raised)
    hist=ag._history
    check("上下文已落盘(非空)", len(hist)>0)
    check("用户指令被保留",
          any(m.get("role")=="user" and "echo 1" in str(m.get("content")) for m in hist))
    check("已完成轮的工具结果保留", has_tool_result_for(hist,"tu1"))
    check("历史以 assistant 收尾", hist[-1]["role"]=="assistant")

    # 关键:模拟下次 run 拼装 → sanitize 后无孤儿 tool_use
    nxt=Agent._sanitize_messages(list(hist)+[{"role":"user","content":"next"}])
    orphan=False
    for i,m in enumerate(nxt):
        if m.get("role")=="assistant" and isinstance(m.get("content"),list):
            for b in m["content"]:
                if b.get("type")=="tool_use":
                    nu = nxt[i+1] if i+1<len(nxt) else {}
                    if not (nu.get("role")=="user" and isinstance(nu.get("content"),list)
                            and any(r.get("tool_use_id")==b.get("id") for r in nu["content"])):
                        orphan=True
    check("下次请求无孤儿 tool_use(不再 400)", not orphan)

    print("\n"+"="*60)
    print(f"PASS: {len(PASS)}   FAIL: {len(FAIL)}")
    if FAIL: print("FAILED:", ", ".join(FAIL)); return 1
    print("\nPASS: 异常中断的对话也被持久化 + 补成合法形态,")
    print("      下一轮既不失忆、也不因孤儿 tool_use 触发 400。")
    return 0

if __name__=="__main__": raise SystemExit(main())
