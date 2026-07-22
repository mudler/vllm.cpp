#!/usr/bin/env python3
"""Regenerate src/vt/vulkan/vulkan_spirv.h from src/vt/vulkan/shaders/*.comp.

BACKEND-VULKAN, W0. This is the vllm.cpp equivalent of llama.cpp's
`ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` (@ 237ad9b96),
which likewise shells out to a GLSL compiler and emits the SPIR-V as C arrays
into a generated header (`ggml-vulkan-shaders.hpp`).

ONE DELIBERATE DIFFERENCE, and it is the whole reason this is a script and not a
build step: llama.cpp runs its generator AT BUILD TIME and therefore REQUIRES
`glslc` on every build machine. Neither of our boxes has one — `dgx.casa`
(aarch64, the GB10 gate box) and the dev box both ship the Vulkan LOADER but no
`glslc`/`glslangValidator`/`libshaderc`, and neither grants sudo to install one
(measured 2026-07-22). Runtime GLSL compilation would need libshaderc linked in,
which we also do not have and which would be a compiled third-party dependency
(.agents/discipline.md forbids those; third_party/ is single-header only).

So the SPIR-V is COMMITTED, as a generated header, and the build needs NO shader
toolchain at all — it is hermetic on every box including CI. The cost is that
this script must be re-run, and its output committed, whenever a `.comp` changes;
`tests/vt/test_vulkan_backend.cpp` asserts the header is non-empty and every
kernel loads, and the header records the compiler that produced it.

Usage:
    scripts/gen-vulkan-spirv.py [--compiler PATH] [--check]

--check re-generates into memory and fails if the committed header is stale,
which is what a CI job would run if a shader toolchain is ever available there.

Note on the SDK version: .agents/specs/backend-fanout-metal-vulkan-xpu.md
§ Risks/decisions 4 flags that Ubuntu's packaged `glslc` is shaderc 2023.8, too
old for llama.cpp's coopmat2 feature probe. W0 uses no cooperative-matrix path,
but the header below records the exact compiler version regardless so a later
work row (V3) can tell whether the committed SPIR-V predates a compiler that
understands coopmat2.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SHADER_DIR = REPO / "src" / "vt" / "vulkan" / "shaders"
OUT_HEADER = REPO / "src" / "vt" / "vulkan" / "vulkan_spirv.h"

# Vulkan 1.1 is the floor the backend requires (src/vt/vulkan/vulkan_context.cpp).
TARGET_ENV = "vulkan1.1"

CANDIDATE_COMPILERS = ("glslang", "glslangValidator", "glslc")


def find_compiler(explicit: str | None) -> pathlib.Path:
    if explicit:
        p = pathlib.Path(explicit)
        if not p.exists():
            sys.exit(f"compiler not found: {explicit}")
        return p
    for name in CANDIDATE_COMPILERS:
        found = shutil.which(name)
        if found:
            return pathlib.Path(found)
    sys.exit(
        "no GLSL->SPIR-V compiler found (looked for: "
        + ", ".join(CANDIDATE_COMPILERS)
        + "). Install a current Vulkan SDK / glslang, or pass --compiler."
    )


def compiler_version(cc: pathlib.Path) -> str:
    out = subprocess.run([str(cc), "--version"], capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip().splitlines()
    return text[0].strip() if text else str(cc)


def compile_one(cc: pathlib.Path, src: pathlib.Path) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        spv = pathlib.Path(td) / (src.stem + ".spv")
        if cc.name == "glslc":
            cmd = [str(cc), f"--target-env={TARGET_ENV}", "-O", "-fshader-stage=compute",
                   "-I", str(SHADER_DIR), "-o", str(spv), str(src)]
        else:
            # -g0 strips debug names (OpName/OpSource), which is most of the
            # committed size. `-Os` is deliberately NOT passed: measured on these
            # shaders it INFLATES the blob because
            # its inlining pass expands the dtype-erased load/store helpers (the
            # exact numbers are 130,208 bytes with -Os vs 109,436 without), and
            # the driver's own optimizer does that work at pipeline creation
            # anyway.
            cmd = [str(cc), "-V", "--target-env", TARGET_ENV, "-g0",
                   f"-I{SHADER_DIR}", "-o", str(spv), str(src)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not spv.exists():
            sys.exit(f"{src.name}: shader compilation failed\n{res.stdout}\n{res.stderr}")
        data = spv.read_bytes()
    if len(data) % 4 != 0:
        sys.exit(f"{src.name}: SPIR-V length {len(data)} is not a multiple of 4")
    if data[:4] not in (b"\x03\x02\x23\x07", b"\x07\x23\x02\x03"):
        sys.exit(f"{src.name}: output does not start with the SPIR-V magic number")
    return data


def render(blobs: dict[str, bytes], version: str) -> str:
    lines: list[str] = []
    add = lines.append
    add("// GENERATED FILE — DO NOT EDIT BY HAND.")
    add("// Regenerate with: scripts/gen-vulkan-spirv.py")
    add("//")
    add("// SPIR-V for the Vulkan backend's compute kernels (BACKEND-VULKAN, W0),")
    add("// compiled from the GLSL in src/vt/vulkan/shaders/*.comp.")
    add("//")
    add("// WHY THE SPIR-V IS COMMITTED rather than compiled by the build, as")
    add("// llama.cpp's vulkan-shaders-gen does at build time: neither of our boxes")
    add("// has a GLSL compiler (no glslc/glslangValidator/libshaderc, no sudo), and")
    add("// libshaderc would be a compiled third-party dependency, which")
    add("// .agents/discipline.md forbids. Committing the SPIR-V makes the build")
    add("// hermetic everywhere — including CI and the aarch64 gate box — at the cost")
    add("// of re-running the generator when a .comp changes. See the generator's")
    add("// docstring for the full rationale.")
    add("//")
    add(f"// Produced by: {version}")
    add(f"// Target environment: {TARGET_ENV}")
    add("#ifndef VT_VULKAN_VULKAN_SPIRV_H_")
    add("#define VT_VULKAN_VULKAN_SPIRV_H_")
    add("")
    add("#include <cstddef>")
    add("#include <cstdint>")
    add("")
    add("namespace vt::vulkan {")
    add("")
    for name in sorted(blobs):
        words = [int.from_bytes(blobs[name][i:i + 4], "little")
                 for i in range(0, len(blobs[name]), 4)]
        add(f"inline constexpr uint32_t kSpv_{name}[] = {{")
        for i in range(0, len(words), 8):
            chunk = ", ".join(f"0x{w:08x}u" for w in words[i:i + 8])
            add(f"    {chunk},")
        add("};")
        add("")
    add("// Name -> SPIR-V module. The NAME is the shader's file stem and is also the")
    add("// key the pipeline cache uses (src/vt/vulkan/vulkan_context.cpp).")
    add("struct SpirvModule {")
    add("  const char* name;")
    add("  const uint32_t* words;")
    add("  size_t word_count;")
    add("};")
    add("")
    add("inline constexpr SpirvModule kSpirvModules[] = {")
    for name in sorted(blobs):
        add(f'    {{"{name}", kSpv_{name}, sizeof(kSpv_{name}) / sizeof(uint32_t)}},')
    add("};")
    add("")
    add("}  // namespace vt::vulkan")
    add("")
    add("#endif  // VT_VULKAN_VULKAN_SPIRV_H_")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler", default=None)
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed header is stale instead of rewriting it")
    args = ap.parse_args()

    cc = find_compiler(args.compiler)
    version = compiler_version(cc)
    sources = sorted(SHADER_DIR.glob("*.comp"))
    if not sources:
        sys.exit(f"no shaders found in {SHADER_DIR}")

    blobs = {src.stem: compile_one(cc, src) for src in sources}
    text = render(blobs, version)

    if args.check:
        current = OUT_HEADER.read_text() if OUT_HEADER.exists() else ""
        # The compiler-version line legitimately differs between machines, so
        # compare everything else.
        def strip_version(s: str) -> str:
            return "\n".join(l for l in s.splitlines() if not l.startswith("// Produced by:"))
        if strip_version(current) != strip_version(text):
            sys.exit("vulkan_spirv.h is STALE — re-run scripts/gen-vulkan-spirv.py")
        print("vulkan_spirv.h is up to date")
        return

    OUT_HEADER.write_text(text)
    total = sum(len(b) for b in blobs.values())
    print(f"wrote {OUT_HEADER.relative_to(REPO)}: {len(blobs)} modules, {total} bytes of SPIR-V")
    print(f"compiler: {version}")


if __name__ == "__main__":
    main()
