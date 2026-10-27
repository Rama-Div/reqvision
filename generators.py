"""ReqVision - UML (Graphviz), test cases and code skeleton generators."""
from __future__ import annotations

import base64
import glob
import keyword
import os
import re
import shutil
import subprocess
from typing import Optional

from analysis import obj_head, singular, snake


# --------------------------------------------------------------------------
# Graphviz helpers
# --------------------------------------------------------------------------
def ensure_graphviz_on_path() -> None:
    """On Windows the installer often does not add Graphviz to PATH - find it."""
    if shutil.which("dot"):
        return
    bases = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
             os.environ.get("LOCALAPPDATA")]
    for base in filter(None, bases):
        for pattern in ("Graphviz*", os.path.join("Programs", "Graphviz*")):
            for root in glob.glob(os.path.join(base, pattern)):
                bin_dir = os.path.join(root, "bin")
                if os.path.exists(os.path.join(bin_dir, "dot.exe")) or os.path.exists(os.path.join(bin_dir, "dot")):
                    os.environ["PATH"] += os.pathsep + bin_dir
                    return


ensure_graphviz_on_path()


def _run_dot(dot: str, timeout: int):
    return subprocess.run(["dot", "-Tpng", "-Gdpi=110"], input=dot.encode("utf-8"),
                          capture_output=True, timeout=timeout)


def render(dot: str) -> tuple[str, Optional[str]]:
    """DOT source -> (data-URI PNG, error). Uses the `dot` binary directly.

    The very first call on Windows can take a long time (Graphviz builds its font cache),
    so a timeout is retried once with a longer limit.
    """
    try:
        try:
            res = _run_dot(dot, 60)
        except subprocess.TimeoutExpired:
            res = _run_dot(dot, 180)
    except FileNotFoundError:
        return "", "Graphviz (dot) غير موجود في PATH. ثبّته ثم أعد تشغيل الخادم."
    except subprocess.TimeoutExpired:
        return "", "انتهت مهلة توليد المخطط."
    if res.returncode != 0 or not res.stdout:
        return "", res.stderr.decode("utf-8", "replace")[:300] or "dot failed"
    return "data:image/png;base64," + base64.b64encode(res.stdout).decode(), None


def warmup() -> None:
    """Run once at server start so the first user request is not slow."""
    try:
        _run_dot('digraph G { a -> b [label="x"]; }', 180)
    except Exception:
        pass


def q(s: str) -> str:
    """Quote a plain string (backslashes are neutralised)."""
    return '"' + str(s).replace("\\", "/").replace('"', "'").replace("\n", " ") + '"'


def qraw(s: str) -> str:
    """Quote a string that already contains DOT escapes such as \\n or \\l."""
    return '"' + s.replace('"', "'") + '"'


def safe(s: str) -> str:
    return re.sub(r"[{}|<>\"\\]", "", str(s))


def camel(s: str) -> str:
    out = "".join(w[:1].upper() + w[1:] for w in re.findall(r"[A-Za-z0-9]+", s))
    if not out:
        return "Item"
    if out[0].isdigit() or keyword.iskeyword(out):
        out = "C" + out
    return out


def functional(reqs: list[dict]) -> list[dict]:
    return [r for r in reqs if r["type"] == "Functional" and not r["negated"]]


def empty_dot(msg: str) -> str:
    return f'digraph G {{ node [shape=note, fontname="Arial"]; n [label={q(msg)}]; }}'


def label_of(r: dict) -> str:
    return f"{r['verb'].capitalize()} {r['object']}".strip()


# --------------------------------------------------------------------------
# UML diagrams
# --------------------------------------------------------------------------
def use_case_dot(reqs: list[dict]) -> str:
    fr = functional(reqs)
    if not fr:
        return empty_dot("No functional requirements found")
    L = ["digraph UseCase {", "rankdir=LR;", 'node [fontname="Arial", fontsize=10];']
    actors: list[str] = []
    for r in fr:
        if r["actor"] != "System" and r["actor"] not in actors:
            actors.append(r["actor"])
    for a in actors:
        L.append(f'{q("A_" + a)} [label={q(a)}, shape=box, style="rounded,filled", fillcolor="#FFE0B2"];')
    L.append('subgraph cluster_sys { label="System Boundary"; style=dashed; color="#6558D3"; fontname="Arial";')
    for r in fr:
        extra = f"\\n[{safe(r['condition'])}]" if r["condition"] else ""
        auto = r["actor"] == "System"
        name = label_of(r) + (" (auto)" if auto else "")
        fill = "#F3E5F5" if auto else "#E3F2FD"
        L.append(f'{q(r["id"])} [label={qraw(safe(name) + extra)}, shape=ellipse, '
                 f'style=filled, fillcolor="{fill}"];')
    L.append("}")
    for r in fr:
        if r["actor"] != "System":
            L.append(f'{q("A_" + r["actor"])} -> {q(r["id"])} [arrowhead=none];')
    L.append("}")
    return "\n".join(L)


def class_dot(reqs: list[dict]) -> str:
    fr = functional(reqs)
    if not fr:
        return empty_dot("No functional requirements found")
    classes: dict[str, list[str]] = {}
    entities: set[str] = set()
    assoc: list[tuple[str, str, str]] = []
    for r in fr:
        cname = camel(r["actor"])
        classes.setdefault(cname, []).append(f"+ {r['method']}()")
        if r["object"]:
            e = camel(singular(r["object"].split()[-1]) if len(r["object"].split()) == 1 else r["object"])
            entities.add(e)
            assoc.append((cname, e, r["verb"]))
    L = ["digraph Class {", "rankdir=BT;", 'node [shape=record, style=filled, fillcolor="#F5F5F5", fontname="Arial", fontsize=10];',
         'edge [fontname="Arial", fontsize=9];']
    for name, methods in classes.items():
        body = "".join(safe(m) + "\\l" for m in dict.fromkeys(methods))
        L.append(f"{q(name)} [label={qraw('{' + name + '||' + body + '}')}];")
    for e in sorted(entities):
        if e in classes:
            continue
        L.append(f'{q(e)} [label={qraw("{" + e + "|+ id: int\\l+ name: str\\l|}")}, fillcolor="#E8F5E9"];')
    for a, e, v in dict.fromkeys(assoc):
        if a != e:
            L.append(f'{q(a)} -> {q(e)} [arrowhead=vee, label={q(v)}];')
    L.append("}")
    return "\n".join(L)


def sequence_dot(reqs: list[dict]) -> str:
    fr = functional(reqs)
    if not fr:
        return empty_dot("No functional requirements found")
    actors: list[str] = []
    for r in fr:
        if r["actor"] != "System" and r["actor"] not in actors:
            actors.append(r["actor"])
    order = actors + ["System"]
    n = len(fr)
    L = ["digraph Seq {", "rankdir=TB;", "nodesep=1.4; ranksep=0.45; splines=line;",
         'node [fontname="Arial", fontsize=10]; edge [fontname="Arial", fontsize=9];']
    L.append("{ rank=same;")
    for k, p in enumerate(order):
        L.append(f'h{k} [label={q(p)}, shape=box, style="rounded,filled", fillcolor="#EDE7F6", group=g{k}];')
    for k in range(len(order) - 1):
        L.append(f"h{k} -> h{k + 1} [style=invis];")
    L.append("}")
    for i in range(n):
        L.append("{ rank=same;")
        for k in range(len(order)):
            L.append(f'p{k}_{i} [shape=point, width=0.02, label="", group=g{k}];')
        for k in range(len(order) - 1):
            L.append(f"p{k}_{i} -> p{k + 1}_{i} [style=invis];")
        L.append("}")
    for k in range(len(order)):
        L.append(f"h{k} -> p{k}_0 [style=dashed, arrowhead=none, color=gray];")
        for i in range(n - 1):
            L.append(f"p{k}_{i} -> p{k}_{i + 1} [style=dashed, arrowhead=none, color=gray];")
    sys_k = len(order) - 1
    for i, r in enumerate(fr):
        label = f"{i + 1}. {safe(label_of(r)).lower()}"
        if r["condition"]:
            label += f"\\n[{safe(r['condition'])}]"
        src = order.index(r["actor"]) if r["actor"] in order else sys_k
        if src == sys_k:
            L.append(f"p{sys_k}_{i} -> p{sys_k}_{i} [label={qraw(label)}, arrowhead=vee];")
        else:
            L.append(f"p{src}_{i} -> p{sys_k}_{i} [label={qraw(label)}, arrowhead=vee, constraint=false];")
    L.append("}")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Test cases
# --------------------------------------------------------------------------
def example_pair(obj: str, verb: str = "") -> tuple[str, str]:
    o = obj.lower()
    if verb in ("login", "signin", "logon"):
        return "registered username + correct password", "registered username + wrong password"
    if verb in ("signup", "register") and not obj:
        return "new unique email + strong password", "email already registered"
    if "email" in o:
        return "user@example.com", "user@@example"
    if "password" in o:
        return "Str0ng!Pass2026", "empty string or '123'"
    if any(k in o for k in ("file", "text", "srs", "document", "report")):
        return "sample_srs.txt (valid, 2 KB)", "empty file or unsupported format"
    if "date" in o:
        return "2026-10-25", "2026-13-45"
    if any(k in o for k in ("id", "number", "quantity", "amount", "age")):
        return "42", "-1 or 'abc'"
    if obj:
        return f"valid {obj}", f"empty or malformed {obj}"
    return "valid request", "malformed request"


def make_tests(reqs: list[dict]) -> list[dict]:
    tests: list[dict] = []

    def add(r, ttype, desc, steps, inp, expected):
        tests.append({"test_id": f"TC-{len(tests) + 1:03d}", "requirement_id": r["id"],
                      "type": ttype, "description": desc, "steps": steps,
                      "example_input": inp, "expected_result": expected})

    for r in reqs:
        obj = r["object"]
        good, bad = example_pair(obj, r['verb'])
        what = f"{r['verb']} {obj}".strip()
        c = r["constraint"]
        if r["negated"]:
            add(r, "Negative", f"{r['actor']} tries a forbidden action: {what}",
                [f"Sign in as {r['actor']}", f"Attempt to {what}", "Observe the response"],
                good, f"System refuses to {what} and shows an explanatory message.")
            continue
        if r["type"] == "Non-Functional":
            if c:
                raw, unit = c.get("raw_value", c["value"]), c.get("raw_unit", c["unit"])
                if c["unit"] == "%":
                    ok, ko = f"{raw:g}%", f"{max(raw - 1, 0):g}%"
                else:
                    ok = f"{round(raw * (0.9 if c['op'] == '<=' else 1.1), 3):g} {unit}"
                    ko = f"{round(raw * (1.1 if c['op'] == '<=' else 0.9), 3):g} {unit}"
                add(r, "Boundary", f"Measure within the limit ({c['display']}): {r['text']}",
                    ["Prepare a realistic workload", "Measure the metric", "Compare with the limit"],
                    f"measured value = {ok}", "Requirement is met (pass).")
                add(r, "Boundary", f"Measure beyond the limit ({c['display']}): {r['text']}",
                    ["Prepare a heavy workload", "Measure the metric", "Compare with the limit"],
                    f"measured value = {ko}", "Requirement is violated (fail) and is reported.")
            else:
                add(r, "Positive", f"Verify quality attribute: {r['text']}",
                    ["Define a measurable threshold with the stakeholder", "Measure", "Compare"],
                    "n/a", "A measurable acceptance criterion exists and is met.")
            continue
        steps = [f"Sign in as {r['actor']}" if r["actor"] != "System" else "Trigger the system process"]
        if r["condition"]:
            steps.append(f"Make sure: {r['condition']}")
        steps += [f"{what.capitalize()} with the example input", "Observe the response"]
        add(r, "Positive", f"{r['actor']} performs: {what}", steps, good,
            f"System completes '{r['verb']}' successfully and the state is updated.")
        add(r, "Negative", f"{r['actor']} attempts '{what}' with invalid data",
            [steps[0], f"Submit invalid input for '{what}'", "Observe the response"], bad,
            "System rejects the input and shows a clear error message; nothing is changed.")
        if r["condition"]:
            add(r, "Negative", f"Condition not met ({r['condition']}) for: {what}",
                [steps[0], f"Make sure the condition is FALSE: {r['condition']}", f"Try to {what}"],
                good, "System does not perform the action and explains why.")
    return tests


# --------------------------------------------------------------------------
# Code skeleton
# --------------------------------------------------------------------------
def make_code(reqs: list[dict]) -> str:
    fr = functional(reqs)
    constraints = [r for r in reqs if r not in fr]
    out = ['"""Auto-generated skeleton by ReqVision - fill in the method bodies."""',
           "from dataclasses import dataclass", ""]
    entities: dict[str, None] = {}
    for r in fr:
        if r["object"]:
            words = r["object"].split()
            entities[camel(singular(words[-1]) if len(words) == 1 else r["object"])] = None
    actor_names = {camel(r["actor"]) for r in fr}
    for e in entities:
        if e not in actor_names:
            out += ["", "@dataclass", f"class {e}:", "    id: int = 0", "    name: str = \"\"", ""]
    by_actor: dict[str, list[dict]] = {}
    for r in fr:
        by_actor.setdefault(r["actor"], []).append(r)
    for actor, items in by_actor.items():
        out += ["", f"class {camel(actor)}:", f'    """Actor: {safe(actor)}"""', ""]
        for r in items:
            param = ""
            if r["object"] and r["object"].split():
                words = r["object"].split()
                ent = camel(singular(words[-1]) if len(words) == 1 else r["object"])
                param = f", {snake(ent)}: {ent}" if ent not in actor_names else ""
            doc = safe(r["text"])[:100].replace("'", "")
            out.append(f"    def {r['method']}(self{param}) -> None:")
            out.append(f'        """{r["id"]}: {doc}"""')
            if r["condition"]:
                out.append(f"        # precondition: {safe(r['condition'])}")
            out += ["        raise NotImplementedError", ""]
    if constraints:
        out += ["", "# ---- Constraints / non-functional requirements (not methods) ----"]
        for r in constraints:
            tag = "FORBIDDEN" if r["negated"] else (r["subtype"] or "NFR")
            out.append(f"# {r['id']} [{tag}] {safe(r['text'])[:110]}")
    return "\n".join(out).strip() + "\n"