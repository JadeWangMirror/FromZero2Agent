"""
MIRROR memory — CogniFold 式概念图(类脑皮层折叠)。

不做检索。记忆是一张类型化图,每轮注入上下文的内容由**拓扑**决定,而不是查询:
event 折叠成 concept,concept 结晶成 intent;按 PageRank + 近因 + 紧急度
读出 immediate / working / background 三档——目标从拓扑密度里自然涌现。

设计参考 CogniFold(https://github.com/OpenNorve/CogniFold)的有效部分:
  节点: event(海马/episodic) · concept(皮层/neocortical) · intent(前额/prefrontal) · time
  边:   GROUNDS · CAUSES · TRIGGERS · REINFORCES · PART_OF ·
        DERIVED_FROM · DEADLINE_FOR · RELATED_TO   (类型化、带权)
  折叠(涌现发生处)—— 四个结构性债务,以图重写解决:
    accumulation  事件不断落盘
    completion    重复事件聚类 → LLM 抽象成 concept;强 concept → intent 结晶
    compression   近义 concept 合并(REINFORCES + 吸收源)
    decay         边权按龄衰减,低权节点剪枝(event 可褪,皮层知识保留)

刻意省略(用户明确要求"避免检索式记忆"):
  ✗ recall(query) / bm25 / 向量检索 —— 没有"查询"原语
  ✗ 读取靠 select_context() 读图当前状态,不靠 query 匹配

NetworkX 后端(与 CogniFold 同)。存储 ~/.mirror/graph.json,旧 memory.json 自动迁移。
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime

import networkx as nx

MEMORY_DIR = os.path.expanduser("~/.mirror")
GRAPH_FILE = os.path.join(MEMORY_DIR, "graph.json")
LEGACY_FILE = os.path.join(MEMORY_DIR, "memory.json")

# ── 读窗预算(每轮注入的条数,有界) ──
IMM_EVENTS = 4
IMM_INTENTS = 5
WORK_CONCEPTS = 6

# ── 折叠阈值 ──
SHARED_TOKENS = 2             # 共享 ≥ 此数内容词 → 同类(主判定,短句友好)
STRICT_JACCARD = 0.30         # 或: 单词重叠但 Jaccard 占主导 → 同类
MIN_CLUSTER = 2               # 至少几条相似事件才折叠成 concept
INTENT_THRESHOLD = 3          # concept 支撑度 ≥ 此值 → 结晶 intent
DEDUP_JACCARD = 0.45          # concept 间相似到此程度 → 合并(compression)
COVERAGE_JACCARD = 0.12       # concept 与某工具描述 Jaccard ≥ 此值 → 视为被该工具覆盖
DECAY_HALF_LIFE_H = 240.0     # 边权半衰期(10 天)
PRUNE_WEIGHT = 0.05           # 边权低于此值 → 剪掉

# ── 边类型 ──
GROUNDS = "GROUNDS"
CAUSES = "CAUSES"
TRIGGERS = "TRIGGERS"
REINFORCES = "REINFORCES"
PART_OF = "PART_OF"
DERIVED_FROM = "DERIVED_FROM"
DEADLINE_FOR = "DEADLINE_FOR"
RELATED_TO = "RELATED_TO"

# ── 时序抽取(前瞻记忆: 截止/计划时间) ──
_DEADLINE_RE = re.compile(
    r"\b(by|before|due|deadline|ship|release|launch|schedule|plan)\b"
    r"|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b(tomorrow|tonight|eod|eow|weekend)\b",
    re.IGNORECASE,
)

_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with",
    "that", "this", "it", "is", "are", "be", "by", "from", "as", "at",
    "into", "your", "you", "i", "we", "they", "my", "me", "was", "were",
    "do", "does", "did", "but", "not", "so", "if", "user",
}
_TOK = re.compile(r"[a-z0-9一-鿿]+")

# ── prompt-injection 扫描(写入前把关) ──
_INJ = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+(dan|developer|root|admin)\b", re.I),
    re.compile(r"</?\s*(system|prompt|instruction)\s*>", re.I),
    re.compile(r"\<\|im_(start|end)\|>"),
]
_INVISIBLE = re.compile(r"[​-‏ - ⁠﻿]")


def _scan(content: str) -> str | None:
    if _INVISIBLE.search(content):
        return "contains invisible Unicode characters"
    for pat in _INJ:
        if pat.search(content):
            return f"prompt-injection pattern: {pat.pattern[:40]}"
    return None


# ── 基础工具 ─────────────────────────────────────────────

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOK.findall((text or "").lower())
            if len(t) > 1 and t not in _STOP}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _recency_iso(ts: str, half_life_h: float = DECAY_HALF_LIFE_H) -> float:
    """[0,1] 近因分:1/(1+age/half)。缺 ts 给低分(不主导)。"""
    if not ts:
        return 0.2
    try:
        t = datetime.fromisoformat(ts)
    except Exception:
        return 0.2
    age_h = max(0.05, (datetime.now() - t).total_seconds() / 3600)
    return 1.0 / (1.0 + age_h / half_life_h)


# ── 图持久化(JSON: nodes + edges) ──────────────────────

def _load_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    if not os.path.exists(GRAPH_FILE):
        return g
    try:
        with open(GRAPH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return g
    for n in data.get("nodes", []):
        g.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for e in data.get("edges", []):
        g.add_edge(e["src"], e["dst"], type=e.get("type", RELATED_TO),
                   weight=e.get("weight", 1.0))
    return g


def _save_graph(g: nx.DiGraph) -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    data = {
        "nodes": [{"id": n, **d} for n, d in g.nodes(data=True)],
        "edges": [
            {"src": u, "dst": v, "type": d.get("type", RELATED_TO),
             "weight": d.get("weight", 1.0)}
            for u, v, d in g.edges(data=True)
        ],
    }
    with open(GRAPH_FILE, "w", encoding="utf-8", newline="") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _next_id(g: nx.DiGraph, prefix: str) -> str:
    mx = 0
    for n in g.nodes:
        if isinstance(n, str) and n.startswith(prefix):
            try:
                mx = max(mx, int(n[len(prefix):]))
            except ValueError:
                pass
    return f"{prefix}{mx + 1:03d}"


# ── 单例(惰性加载 + 旧格式迁移) ────────────────────────

_G: nx.DiGraph | None = None


def _graph() -> nx.DiGraph:
    global _G
    if _G is None:
        _G = _load_graph()
        _migrate_legacy(_G)
    return _G


def _migrate_legacy(g: nx.DiGraph) -> None:
    """旧 memory.json → event 节点。迁移后改名 .migrated 避免重复。"""
    if not os.path.exists(LEGACY_FILE):
        return
    if os.path.exists(LEGACY_FILE + ".migrated"):
        return
    try:
        with open(LEGACY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        data = {}
    moved = 0
    for m in data.get("memories", []):
        eid = _next_id(g, "e-")
        g.add_node(eid, type="event",
                   data={"content": m.get("content", ""), "tags": m.get("tags", [])},
                   created=m.get("created_at", _now()),
                   last_touched=m.get("created_at", _now()),
                   consolidated=False)
        moved += 1
    if moved:
        _save_graph(g)
    try:
        os.replace(LEGACY_FILE, LEGACY_FILE + ".migrated")
    except OSError:
        pass


# ── 写入原语(唯一的"写":落 event) ─────────────────────

def remember(content: str, tags: str = "") -> str:
    """提交一条原始事件到海马层。真正的理解发生在 consolidate 折叠之后。"""
    content = (content or "").strip()
    if not content:
        return "[!] remember: 'content' is required."
    bad = _scan(content)
    if bad:
        return f"[!] rejected ({bad}) — not saved."
    g = _graph()
    eid = _next_id(g, "e-")
    now = _now()
    g.add_node(eid, type="event",
               data={"content": content,
                     "tags": [t.strip() for t in (tags or "").split(",") if t.strip()]},
               created=now, last_touched=now, consolidated=False)
    _save_graph(g)
    return (f"[✓] event {eid} committed to hippocampal layer. "
            f"Run consolidate_memory to fold it into the cortex.")


def forget(node_id: str) -> str:
    """按 id 删除任意层节点(event/concept/intent)。"""
    g = _graph()
    if node_id not in g:
        return f"[!] forget: node '{node_id}' not found in graph."
    g.remove_node(node_id)
    _save_graph(g)
    return f"[✓] removed node {node_id} (and its edges)."


def update_intent(intent_id: str, status: str = "pending",
                  urgency: int | None = None) -> str:
    """更新 intent 状态(pending/in_progress/done/skipped)与紧急度。"""
    g = _graph()
    if intent_id not in g:
        return f"[!] update_intent: '{intent_id}' not found."
    d = g.nodes[intent_id]
    if d.get("type") != "intent":
        return f"[!] '{intent_id}' is not an intent (type={d.get('type')})."
    d["data"]["status"] = status or d["data"].get("status", "pending")
    if urgency is not None:
        try:
            d["data"]["urgency"] = max(1, min(5, int(urgency)))
        except (TypeError, ValueError):
            pass
    d["last_touched"] = _now()
    _save_graph(g)
    return (f"[✓] intent {intent_id} → status={d['data']['status']}, "
            f"urgency={d['data'].get('urgency', 1)}")


# ── 折叠循环(涌现发生处) ───────────────────────────────

def _related(a: set, b: set) -> bool:
    """同类判定: 共享 ≥2 内容词, 或单词重叠但 Jaccard 占主导(该词很显著)。
    用双重标准避免 'prefers' 这类半通用词桥接无关话题(无嵌入的关键词兜底)。"""
    if not a or not b:
        return False
    if len(a & b) >= SHARED_TOKENS:
        return True
    return _jaccard(a, b) >= STRICT_JACCARD


def _cluster(nodes: list[str], toks: dict) -> list[list[str]]:
    """贪心聚类(Jaccard 或共享词)。"""
    assigned: set[str] = set()
    clusters: list[list[str]] = []
    for n in nodes:
        if n in assigned:
            continue
        tn = toks.get(n, set())
        cl = [n]
        assigned.add(n)
        for m in nodes:
            if m in assigned:
                continue
            if _related(tn, toks.get(m, set())):
                cl.append(m)
                assigned.add(m)
        clusters.append(cl)
    return clusters


def _concept_strength(g: nx.DiGraph, concept: str) -> int:
    """concept 被多少 event/概念支撑 = 入边中 DERIVED_FROM 条数。"""
    d = g.nodes[concept].get("data", {})
    if d.get("strength"):
        return d["strength"]
    return sum(1 for _, _, e in g.in_edges(concept, data=True)
               if e.get("type") == DERIVED_FROM)


def _has_intent_for(g: nx.DiGraph, concept: str) -> bool:
    for n, d in g.nodes(data=True):
        if d.get("type") == "intent" and g.has_edge(n, concept):
            return True
    return False


def _extract_temporal(g: nx.DiGraph, events: list[tuple]) -> int:
    """前瞻记忆: 给截止/计划事件挂 time 节点 + DEADLINE_FOR 边。

    这是图挣钱的差异点之一 —— 扁平表表达不了"这件事有截止时间"。
    让时序事实在读取时不被 recency 淹没,并填充非 DERIVED_FROM 的真实结构。
    返回新建 time 节点数。
    """
    new_t = 0
    due_nodes: dict[str, str] = {
        d["data"].get("label"): n for n, d in g.nodes(data=True)
        if d.get("type") == "time"
    }
    for n, d in events:
        content = d.get("data", {}).get("content", "")
        m = _DEADLINE_RE.search(content)
        if not m:
            continue
        label = m.group(0).lower().strip()
        d["data"]["due"] = label
        if label not in due_nodes:
            tid = _next_id(g, "t-")
            g.add_node(tid, type="time", data={"label": label},
                       created=_now(), last_touched=_now())
            due_nodes[label] = tid
            new_t += 1
        # event → time (该事件 deadline_for 此时序锚点)
        if not g.has_edge(n, due_nodes[label]):
            g.add_edge(n, due_nodes[label], type=DEADLINE_FOR, weight=2.0)
    return new_t


def _decay_and_prune(g: nx.DiGraph) -> int:
    """边权按龄衰减;极低权边剪掉;孤立且老的已折叠 event 剪掉(皮层知识保留)。"""
    pruned = 0
    for u, v, e in list(g.edges(data=True)):
        age = min(_recency_iso(g.nodes[u].get("last_touched")),
                  _recency_iso(g.nodes[v].get("last_touched")))
        e["weight"] = e.get("weight", 1.0) * age
        if e["weight"] < PRUNE_WEIGHT:
            g.remove_edge(u, v)
    for n, d in list(g.nodes(data=True)):
        if d.get("type") == "event" and d.get("consolidated"):
            if g.degree(n) == 0 and _recency_iso(d.get("last_touched")) < 0.1:
                g.remove_node(n)
                pruned += 1
    return pruned


def consolidate(summarize_fn, tools=None) -> str:
    """折叠循环: accumulation → completion → compression → decay。

    summarize_fn(prompt)->str 由 Agent 注入 LLM(批量抽象 concept 语句)。
    tools=[(name,desc),...] 当前工具清单 —— 用于覆盖判定:未被任何工具覆盖的
    强 concept 结晶成 kind=capability_gap 的 intent,作为自进化的需求信号。
    返回人类可读报告。涌现发生在此:concept 与 intent 不被查询,而是被算出。
    """
    g = _graph()
    tool_toks = [(name, _tokens(name.replace("_", " ") + " " + (desc or "")))
                 for name, desc in (tools or [])]
    events = [(n, d) for n, d in g.nodes(data=True)
              if d.get("type") == "event" and not d.get("consolidated")]

    # ── completion: 聚类 + LLM 批量抽象 ──
    toks = {n: _tokens(d.get("data", {}).get("content", "")) for n, d in events}
    clusters = _cluster([n for n, _ in events], toks)
    foldable = [c for c in clusters if len(c) >= MIN_CLUSTER]

    statements: dict[int, str] = {}
    covered: dict[int, str] = {}        # cluster idx → 覆盖它的工具名(LLM 判定)
    if foldable:
        blocks = []
        for i, cl in enumerate(foldable, 1):
            sample = "\n".join(f"  - {g.nodes[n]['data'].get('content', '')[:140]}"
                               for n in cl[:10])
            blocks.append(f"[{i}]\n{sample}")
        # 有工具清单时,让 LLM 在同一次折叠调用里顺带判定覆盖(搭便车,零额外消耗),
        # 替代脆的 Jaccard 阈值 —— 语义判断交给 LLM。
        tool_inv = ""
        fmt = "one line each, no preamble."
        if tool_toks:
            tool_inv = "\n\nExisting tools:\n" + "\n".join(
                f"- {name}: {(desc or '').splitlines()[0][:80]}"
                for name, desc in (tools or []) if name) or ""
            fmt = ("one line each, format: 'N. <concept statement> | tool: <toolname>' "
                   "where <toolname> is the EXISTING tool that already covers this concept, "
                   "or 'none' if uncovered. no preamble.")
        prompt = (
            "Below are numbered clusters of recurring observations from past sessions. "
            "For EACH cluster, write ONE concise concept statement: a general fact, "
            "preference, or pattern they all exemplify (<=1 sentence). Output a numbered "
            "list matching the cluster numbers, " + fmt + "\n\n" + "\n".join(blocks)
            + tool_inv
        )
        try:
            raw = summarize_fn(prompt) or ""
        except Exception:
            raw = ""
        for line in raw.splitlines():
            m = re.match(r"\s*\(?\s*(\d+)[\).:\-]\s*(.+?)(?:\s*\|\s*tool:\s*(.+))?\s*$", line)
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= len(foldable):
                    statements[idx] = m.group(2).strip()
                    t = (m.group(3) or "").strip()
                    if tool_toks and t and t.lower() not in ("none", "n/a", "-", ""):
                        covered[idx] = t

    new_c = merged_c = 0
    for i, cl in enumerate(foldable, 1):
        stmt = statements.get(i)
        if not stmt:
            continue
        st = _tokens(stmt)
        # compression: 与已有 concept 去重合并
        target = None
        for cn, cd in g.nodes(data=True):
            if cd.get("type") == "concept" and _jaccard(
                    st, _tokens(cd["data"].get("statement", ""))) >= DEDUP_JACCARD:
                target = cn
                break
        cov = covered.get(i)
        if target is not None:
            for src in cl:
                if g.has_edge(src, target):
                    g[src][target]["weight"] = g[src][target].get("weight", 1.0) + 1.0
                else:
                    g.add_edge(src, target, type=DERIVED_FROM, weight=1.0)
            g.nodes[target]["data"]["strength"] = \
                g.nodes[target]["data"].get("strength", 0) + len(cl)
            if cov:
                g.nodes[target]["data"]["covered_by"] = cov
            g.nodes[target]["last_touched"] = _now()
            merged_c += 1
        else:
            cid = _next_id(g, "c-")
            cdata = {"statement": stmt, "strength": len(cl)}
            if cov:
                cdata["covered_by"] = cov
            g.add_node(cid, type="concept", data=cdata,
                       created=_now(), last_touched=_now(), consolidated=True)
            for src in cl:
                g.add_edge(src, cid, type=DERIVED_FROM, weight=1.0)
            new_c += 1
        for src in cl:
            g.nodes[src]["consolidated"] = True

    # ── completion: intent 结晶(涌现) + 能力缺口判定 ──
    new_i = gaps = 0
    for n, d in list(g.nodes(data=True)):
        if d.get("type") != "concept":
            continue
        strength = _concept_strength(g, n)
        if strength >= INTENT_THRESHOLD and not _has_intent_for(g, n):
            stmt = d["data"]["statement"]
            covered_by = d["data"].get("covered_by")   # LLM 在折叠时判定的覆盖
            # 有工具清单时:被覆盖=served, 未覆盖=capability_gap; 无清单=task
            kind = "served" if covered_by else ("capability_gap" if tool_toks else "task")
            iid = _next_id(g, "i-")
            idata = {"statement": stmt, "status": "pending",
                     "urgency": min(5, 2 + strength // 2), "kind": kind}
            if covered_by:
                idata["covered_by"] = covered_by
            g.add_node(iid, type="intent", data=idata,
                       created=_now(), last_touched=_now())
            g.add_edge(iid, n, type=DERIVED_FROM, weight=1.0)
            new_i += 1
            if kind == "capability_gap":
                gaps += 1

    # ── 时序抽取(前瞻记忆: 截止/计划) ──
    new_t = _extract_temporal(g, events)

    pruned = _decay_and_prune(g)
    _save_graph(g)

    parts = [f"Folded {len(events)} event(s) → {new_c} new concept(s)"]
    if merged_c:
        parts.append(f"{merged_c} reinforced")
    if new_i:
        parts.append(f"{new_i} intent(s) crystallized")
    if gaps:
        parts.append(f"{gaps} capability gap(s)")
    if new_t:
        parts.append(f"{new_t} time anchor(s)")
    if pruned:
        parts.append(f"{pruned} decayed node(s) pruned")
    return ", ".join(parts) + "."


# ── 记忆↔自进化 接口 ────────────────────────────────────

def capability_gaps() -> list[tuple[str, str]]:
    """当前未被工具覆盖、未完成的能力缺口 [(intent_id, statement)]。
    自进化的需求信号 —— 这些是重复出现但无现成工具的操作。"""
    g = _graph()
    return [(n, d["data"].get("statement", ""))
            for n, d in g.nodes(data=True)
            if d.get("type") == "intent"
            and d["data"].get("kind") == "capability_gap"
            and d["data"].get("status") != "done"]


def satisfy_gap(tool_name: str, desc: str = "") -> int:
    """新工具上线 → 把它覆盖的 capability_gap intent 标记 done + covered_by。

    供需闭环:create_tool 后调用,让图反映"这个缺口已被新工具补上"。
    返回闭环的缺口数。
    """
    g = _graph()
    ttoks = _tokens(tool_name.replace("_", " ") + " " + (desc or ""))
    if not ttoks:
        return 0
    closed = 0
    for n, d in g.nodes(data=True):
        if (d.get("type") == "intent"
                and d["data"].get("kind") == "capability_gap"
                and d["data"].get("status") != "done"
                and _jaccard(_tokens(d["data"].get("statement", "")), ttoks) >= COVERAGE_JACCARD):
            d["data"]["status"] = "done"
            d["data"]["covered_by"] = tool_name
            d["last_touched"] = _now()
            closed += 1
    if closed:
        _save_graph(g)
    return closed


# ── 涌现式读取(无 query,读图当前状态) ─────────────────

def _pagerank(g: nx.DiGraph) -> dict:
    if g.number_of_nodes() == 0:
        return {}
    try:
        return nx.pagerank(g, weight="weight")
    except Exception:
        return {n: 1.0 / g.number_of_nodes() for n in g.nodes}


def select_context(g: nx.DiGraph) -> str:
    """读出三档上下文。没有 query —— 浮上来什么由拓扑(PageRank)+近因+紧急度决定。"""
    if g.number_of_nodes() == 0:
        return ""

    pr = _pagerank(g)

    def rec(d):
        return _recency_iso(d.get("last_touched") or d.get("created"))

    # immediate: 活跃 intent(紧急度 × 拓扑 × 近因)
    intents = sorted(
        ((n, d) for n, d in g.nodes(data=True)
         if d.get("type") == "intent" and d["data"].get("status") != "done"),
        key=lambda nd: nd[1]["data"].get("urgency", 1) * (0.4 + pr.get(nd[0], 0)) * rec(nd[1]),
        reverse=True,
    )[:IMM_INTENTS]

    # immediate: 最近 event(海马近因主导)
    events = sorted(
        ((n, d) for n, d in g.nodes(data=True) if d.get("type") == "event"),
        key=lambda nd: nd[1].get("created", ""),
        reverse=True,
    )[:IMM_EVENTS]
    # 前瞻: 截止/计划事件,不论新旧都浮上来(前瞻记忆,不被 recency 淹没)
    recent_ids = {n for n, _ in events}
    due = [(n, d) for n, d in g.nodes(data=True)
           if d.get("type") == "event"
           and d["data"].get("due")
           and n not in recent_ids][:4]

    # working: 强 concept(PageRank × 近因 × log strength)
    concepts = sorted(
        ((n, d) for n, d in g.nodes(data=True) if d.get("type") == "concept"),
        key=lambda nd: (0.4 + pr.get(nd[0], 0)) * rec(nd[1])
                       * math.log(1 + nd[1]["data"].get("strength", 1)),
        reverse=True,
    )[:WORK_CONCEPTS]

    nE = sum(1 for _, d in g.nodes(data=True) if d.get("type") == "event")
    nC = sum(1 for _, d in g.nodes(data=True) if d.get("type") == "concept")
    nI = sum(1 for _, d in g.nodes(data=True) if d.get("type") == "intent")

    lines = ["", "═ MEMORY (emergent — graph topology, not retrieval) " + "═" * 8]
    gap_n = sum(1 for _, d in intents if d["data"].get("kind") == "capability_gap")
    if intents:
        hdr = f"▸ EMERGENT INTENTS ({nI} crystallized"
        if gap_n:
            hdr += f"; {gap_n} capability gap(s) ⚡"
        hdr += "):"
        lines.append(hdr)
        for _, d in intents:
            kind = d["data"].get("kind", "task")
            covered = d["data"].get("covered_by")
            if kind == "capability_gap":
                tag = "  ⚡ no tool covers → self_evolve/propose_tool"
            elif covered:
                tag = f"  (covered by {covered})"
            else:
                tag = ""
            lines.append(f"   • [{d['data'].get('status', '?')}] "
                         f"{d['data'].get('statement', '')}  "
                         f"(urg={d['data'].get('urgency', 1)}){tag}")
    if events:
        lines.append("▸ IMMEDIATE (recent events):")
        for _, d in events:
            lines.append(f"   - {d['data'].get('content', '')[:160]}")
    if due:
        lines.append("▸ DUE / PENDING (prospective — deadlines, recency-proof):")
        for _, d in due:
            lines.append(f"   ⏰ {d['data'].get('content', '')[:160]}")
    if concepts:
        lines.append("▸ WORKING (reinforced concepts, PageRank-ranked):")
        for _, d in concepts:
            lines.append(f"   · {d['data'].get('statement', '')}  "
                         f"(×{d['data'].get('strength', 1)})")
    lines.append(f"▸ graph: {nE} events · {nC} concepts · {nI} intents  "
                 f"(consolidate_memory to fold)")
    lines.append("═" * 60)
    return "\n".join(lines)


def context_block() -> str:
    """供 system prompt 注入的涌现式记忆块。失败返回空串(不影响主流程)。"""
    try:
        return select_context(_graph())
    except Exception:
        return ""


def graph_stats() -> str:
    """图规模概览(给 /memory 命令或工具用)。"""
    g = _graph()
    if g.number_of_nodes() == 0:
        return "Memory graph is empty. Use remember(...) to commit events."
    by_type = {}
    for _, d in g.nodes(data=True):
        t = d.get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1
    pending = sum(1 for _, d in g.nodes(data=True)
                  if d.get("type") == "event" and not d.get("consolidated"))
    return (f"nodes={g.number_of_nodes()} edges={g.number_of_edges()} "
            f"by_type={by_type} unconsolidated_events={pending}")
