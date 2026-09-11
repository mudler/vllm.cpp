#!/usr/bin/env python3
"""Prepare a row-owned secondary source copy with one retained input leaf.

The pristine archive remains unchanged. This overlay qualifies only this row's
bounded synthetic model and never advances a global oracle pin.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PINS = {
    "stock": ("10bf611e533d81f739128304991c5e133c6aebd8",
              "bc1c497c4da0951a4f98de50e397bcb632b2e40d4f24b8c58203c964d4d6183b"),
    "fork": ("36fe8e1cc7f2b3b8c92fdda0ab07600141921786",
             "e7d4915cc66bc6187f42ad016f9b1288f8e1702e15da4e63d9862cf8afcb8267"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): digest(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and ".git" not in path.relative_to(root).parts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", choices=PINS)
    parser.add_argument("pristine", type=Path)
    parser.add_argument("overlay", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    pin, graph_sha = PINS[args.oracle]
    relative = "src/llama-graph.cpp"
    original = args.pristine / relative
    if digest(original) != graph_sha:
        raise RuntimeError("pristine graph source differs from the recorded pin")
    if args.overlay.exists() or args.evidence.exists():
        raise RuntimeError("overlay and evidence destinations must be new")
    before = inventory(args.pristine)
    old = original.read_text()
    anchor = "llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {"
    if old.count(anchor) != 1:
        raise RuntimeError("hybrid builder anchor is not unique")
    start = old.index(anchor)
    end = old.index("\n}", start)
    function = old[start:end]
    line = "    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());\n"
    if function.count(line) != 1:
        raise RuntimeError("recurrent copy construction is not unique")
    replacement = function.replace(line, line + "    ggml_build_forward_expand(gf, inp_rs->s_copy);\n")
    new = old[:start] + replacement + old[end:]
    shutil.copytree(args.pristine, args.overlay, symlinks=True,
                    ignore=shutil.ignore_patterns(".git"))
    (args.overlay / relative).write_text(new)
    after = inventory(args.overlay)
    changed = [path for path in sorted(set(before) | set(after))
               if before.get(path) != after.get(path)]
    if changed != [relative] or inventory(args.pristine) != before:
        raise RuntimeError("overlay changed an unrelated file or the pristine source")
    args.evidence.mkdir(parents=True)
    patch = args.evidence / "retain-hybrid-copy.patch"
    patch.write_text("".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                               fromfile="a/" + relative, tofile="b/" + relative)))
    report = {"oracle": args.oracle, "pristine_pin": pin, "scope": "bounded synthetic model only",
              "pristine": str(args.pristine.resolve()), "overlay": str(args.overlay.resolve()),
              "pristine_files": before, "overlay_files": after, "changed_files": changed,
              "patch_sha256": digest(patch), "pristine_restoration_verified": True}
    (args.evidence / "source-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pin": pin, "changed": changed, "patch_sha256": digest(patch)}))


if __name__ == "__main__":
    main()
