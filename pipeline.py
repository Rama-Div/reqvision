"""ReqVision - text -> full analysis result (no web framework needed)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from analysis import (MAX_REQUIREMENTS, MAX_TEXT_CHARS, analyze_consistency,
                      extract_requirements, finalize)
from generators import (class_dot, make_code, make_tests, render, sequence_dot,
                        use_case_dot)
from segmenter import split_story


def analyze_text(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("SRS text cannot be empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"النص طويل جداً (الحد {MAX_TEXT_CHARS} حرف).")

    # Story-style input (no full stops / bullets): one requirement per sentence.
    text = split_story(text)

    reqs, unparsed, engine, note = extract_requirements(text)
    if not reqs:
        raise ValueError(note or "لم أتمكن من استخراج أي متطلب. جرّب جملاً مثل: The user can upload a file.")
    finalize(reqs)
    findings = analyze_consistency(reqs)
    sources = {
        "use_case_diagram": use_case_dot(reqs),
        "class_diagram": class_dot(reqs),
        "sequence_diagram": sequence_dot(reqs),
    }
    result: dict = {
        "status": "success", "engine": engine, "engine_note": note,
        "requirements": reqs, "unparsed": unparsed, "findings": findings,
        "test_cases": make_tests(reqs), "generated_code": make_code(reqs),
        "diagram_errors": {},
    }
    with ThreadPoolExecutor(max_workers=3) as pool:
        rendered = dict(zip(sources, pool.map(render, sources.values())))
    for key, (img, err) in rendered.items():
        result[key] = img
        if err:
            result["diagram_errors"][key] = err
    sev = {"high": 0, "medium": 0, "low": 0}
    for f in findings:
        sev[f["severity"]] += 1
    result["summary"] = {
        "requirements": len(reqs),
        "actors": len({r["actor"] for r in reqs}),
        "functional": sum(r["type"] == "Functional" for r in reqs),
        "non_functional": sum(r["type"] == "Non-Functional" for r in reqs),
        "tests": len(result["test_cases"]),
        "issues": sev,
        "truncated": len(reqs) >= MAX_REQUIREMENTS,
    }
    return result
