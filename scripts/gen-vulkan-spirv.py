#!/usr/bin/env python3
"""Regenerate src/vt/vulkan/vulkan_spirv.h from src/vt/vulkan/shaders/*.comp.

BACKEND-VULKAN, W0. This is the vllm.cpp equivalent of llama.cpp's
`ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` (@ 237ad9b96),
which likewise shells out to a GLSL compiler and emits the SPIR-V as C arrays
into a generated header (`ggml-vulkan-shaders.hpp`).

ONE DELIBERATE DIFFERENCE, and it is the whole reason this is a script and not a
build step: llama.cpp runs its generator AT BUILD TIME and therefore REQUIRES
`glslc` on every build machine. Runtime GLSL compilation would need libshaderc
linked in, which we do not have and which would be a compiled third-party
dependency (.agents/discipline.md forbids those; third_party/ is single-header
only).

WHICH COMPILER TO RUN THIS WITH, corrected 2026-09-02 (BACKEND-VULKAN-EXL3,
#2530). This paragraph used to read "Neither of our boxes has one -- `dgx.casa`
(aarch64, the GB10 gate box) and the dev box both ship the Vulkan LOADER but no
`glslc`/`glslangValidator`/`libshaderc`, and neither grants sudo to install one
(measured 2026-07-22)". That is now stale in BOTH directions, and the correction
matters because the first half reads as permission to use the system compiler:

  * `/usr/bin/glslc` DOES exist on the dev box today -- shaderc 2023.8,
    spirv-tools 2023.6, glslang 14.0.0.
  * It CANNOT regenerate this tree. `vt_matmul_coopmat.comp` requires
    `GL_EXT_bfloat16`, which glslang 14.0.0 does not know, so `--check` with the
    system compiler exits on that shader before reaching any other:
    `error: '#extension' : extension not supported: GL_EXT_bfloat16`.
  * The working route needs NO sudo and CI already uses it: unpack the PINNED
    glslang 16.5.0 RELEASE TARBALL into a temp directory and put it first on
    PATH. That is what .github/workflows/ci.yml's `vulkan-spirv-freshness` job
    does, and it is what reproduces the committed blob byte for byte.

        curl -fsSL -o /tmp/glslang.tar.gz \
          https://github.com/KhronosGroup/glslang/releases/download/16.5.0/glslang-16.5.0-linux-x86_64-release.tar.gz
        mkdir -p /tmp/glslang && tar xzf /tmp/glslang.tar.gz -C /tmp/glslang
        PATH=/tmp/glslang/bin:$PATH scripts/gen-vulkan-spirv.py

VERIFY BEFORE YOU REGENERATE, because this script rewrites a 1.5 MB generated
file and the WRONG compiler rewrites every module rather than only yours. Run
`--check` on the UNMODIFIED tree first; it must print "committed SPIR-V is up to
date". Then your regeneration's diff is additions only, which is how
BACKEND-VULKAN-EXL3 added two modules for +948 lines and -0.

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
OUT_SOURCE = REPO / "src" / "vt" / "vulkan" / "vulkan_spirv.cpp"

# Vulkan 1.1 is the floor the backend requires (src/vt/vulkan/vulkan_context.cpp).
TARGET_ENV = "vulkan1.1"

CANDIDATE_COMPILERS = ("glslang", "glslangValidator", "glslc")

# Shaders that get a second compilation with extra -D defines, producing an
# additional SPIR-V module with a suffix. Used for VT_IDOT: the _dev TQ
# shaders compile once as the scalar fallback and once with -DVT_IDOT for the
# dotPacked4x8EXT path. The device-side dispatch picks the _idot variant when
# VK_KHR_shader_integer_dot_product is enabled (vulkan_context.cpp).
EXTRA_VARIANTS = {
    "vt_matmul_bt_tq1_0_dev": [("-DVT_IDOT", "_idot")],
    "vt_matmul_bt_tq2_dev": [("-DVT_IDOT", "_idot")],
    "vt_matmul_bt_tq1_0_grouped_dev": [("-DVT_IDOT", "_idot")],
    "vt_matmul_bt_tq2_grouped_dev": [("-DVT_IDOT", "_idot")],
    "vt_moe_gate_up_swiglu_grouped_tq1_0": [("-DVT_IDOT", "_idot")],
    "vt_moe_gate_up_swiglu_grouped_tq2": [("-DVT_IDOT", "_idot")],
}


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


def compile_one(cc: pathlib.Path, src: pathlib.Path, extra_defs: list[str] | None = None) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        spv = pathlib.Path(td) / (src.stem + ".spv")
        defs = extra_defs or []
        if cc.name == "glslc":
            cmd = [str(cc), f"--target-env={TARGET_ENV}", "-O", "-fshader-stage=compute",
                   "-I", str(SHADER_DIR), "-o", str(spv), str(src)] + defs
        else:
            # -g0 strips debug names (OpName/OpSource), which is most of the
            # committed size. `-Os` is deliberately NOT passed: measured on these
            # shaders it INFLATES the blob because
            # its inlining pass expands the dtype-erased load/store helpers (the
            # exact numbers are 130,208 bytes with -Os vs 109,436 without), and
            # the driver's own optimizer does that work at pipeline creation
            # anyway.
            cmd = [str(cc), "-V", "--target-env", TARGET_ENV, "-g0",
                   f"-I{SHADER_DIR}", "-o", str(spv), str(src)] + defs
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not spv.exists():
            if src.name == "vt_matmul_coopmat.comp":
                print(f"SKIP {src.name}: shader compilation failed")
                return None
            sys.exit(f"{src.name}: shader compilation failed\n{res.stdout}\n{res.stderr}")
        data = spv.read_bytes()
    if len(data) % 4 != 0:
        sys.exit(f"{src.name}: SPIR-V length {len(data)} is not a multiple of 4")
    if data[:4] not in (b"\x03\x02\x23\x07", b"\x07\x23\x02\x03"):
        sys.exit(f"{src.name}: output does not start with the SPIR-V magic number")
    return data


# SPIR-V decoration constants (SPIR-V core spec: OpDecorate = 71 in §3.32.3
# Annotation Instructions, Decoration SpecId = 1 in §3.20). Parsing the emitted
# module is the only source of truth here: glslang offers no side-channel listing
# of the SpecIds it produced, and a hand-maintained list beside the shaders is
# exactly the kind of duplicate that drifts silently.
SPIRV_OP_DECORATE = 71
SPIRV_DECORATION_SPEC_ID = 1
SPIRV_HEADER_WORDS = 5

# The rest of the annotation/type opcodes the BINDING ACCESS reflection below
# needs (SPIR-V core spec §3.52 Instructions, §3.20 Decoration).
SPIRV_OP_TYPE_POINTER = 32
SPIRV_OP_VARIABLE = 59
SPIRV_OP_MEMBER_DECORATE = 72
SPIRV_DECORATION_NON_WRITABLE = 24
SPIRV_DECORATION_BINDING = 33

# Mirrors kMaxDispatchBindings in src/vt/vulkan/vulkan_context.cpp. The mask
# below is one bit per binding, so a shader with more bindings than this could
# not be described and must fail LOUDLY rather than silently reporting the
# overflow bindings as read-only.
MAX_BINDINGS = 32


def _instructions(blob: bytes):
    """Yield (opcode, [operand words]) for one SPIR-V module."""
    words = [int.from_bytes(blob[i:i + 4], "little") for i in range(0, len(blob), 4)]
    i = SPIRV_HEADER_WORDS
    while i < len(words):
        word_count = words[i] >> 16
        opcode = words[i] & 0xFFFF
        if word_count == 0:
            sys.exit("malformed SPIR-V: zero-length instruction")
        yield opcode, words[i + 1:i + word_count]
        i += word_count


def binding_access(blob: bytes, name: str) -> tuple[int, int]:
    """Return (binding_count, writable_mask) for one SPIR-V module.

    WHY THIS IS PARSED OUT OF THE COMPILED MODULE AND NOT WRITTEN DOWN BY HAND.
    The Vulkan backend can only skip a pipeline barrier between two dispatches if
    it can prove they do not share a buffer in a hazardous way, and that proof
    needs each dispatch's READ set and WRITE set. Nothing on the host side knows
    them: `VulkanContext::Dispatch` receives one flat array of VkBuffers with no
    roles attached, and guessing a role from the binding index would be exactly
    the unverified assumption that turns a missing barrier into silently wrong
    numbers.

    The GLSL already states it, and glslang ENFORCES it: a `readonly buffer`
    block is a compile error to write to, and the compiler records the promise as
    a `NonWritable` decoration in the emitted module. So the committed SPIR-V is
    an authoritative, machine-checked statement of which bindings the shader can
    write, and the freshness check on this generator keeps it from drifting from
    the .comp source.

    A binding is reported WRITABLE unless a NonWritable decoration is positively
    found for it, so an unrecognised encoding degrades to "assume it is written",
    which costs a barrier rather than correctness.
    """
    binding_of: dict[int, int] = {}       # variable id -> binding number
    var_non_writable: set[int] = set()    # variable ids decorated NonWritable
    member_non_writable: set[tuple[int, int]] = set()
    pointee_of: dict[int, int] = {}       # pointer type id -> pointee type id
    ptr_type_of: dict[int, int] = {}      # variable id -> pointer type id
    for opcode, ops in _instructions(blob):
        if opcode == SPIRV_OP_DECORATE and len(ops) >= 3:
            if ops[1] == SPIRV_DECORATION_BINDING:
                binding_of[ops[0]] = ops[2]
        elif opcode == SPIRV_OP_DECORATE and len(ops) == 2:
            if ops[1] == SPIRV_DECORATION_NON_WRITABLE:
                var_non_writable.add(ops[0])
        elif opcode == SPIRV_OP_MEMBER_DECORATE and len(ops) >= 3:
            if ops[2] == SPIRV_DECORATION_NON_WRITABLE:
                member_non_writable.add((ops[0], ops[1]))
        elif opcode == SPIRV_OP_TYPE_POINTER and len(ops) == 3:
            pointee_of[ops[0]] = ops[2]
        elif opcode == SPIRV_OP_VARIABLE and len(ops) >= 3:
            ptr_type_of[ops[1]] = ops[0]

    if not binding_of:
        sys.exit(f"{name}: no descriptor bindings found in the SPIR-V")
    mask = 0
    for var, binding in binding_of.items():
        if binding >= MAX_BINDINGS:
            sys.exit(f"{name}: binding {binding} is at or above the backend's "
                     f"{MAX_BINDINGS}-binding limit")
        block = pointee_of.get(ptr_type_of.get(var, -1), -1)
        read_only = var in var_non_writable or (block, 0) in member_non_writable
        if not read_only:
            mask |= 1 << binding
    count = max(binding_of.values()) + 1
    # The host binds descriptors 0..count-1 densely, so a hole would mean the
    # mask's bit positions no longer line up with the dispatch's buffer array.
    if len(set(binding_of.values())) != count:
        sys.exit(f"{name}: descriptor bindings are not a dense 0..{count - 1} range")
    return count, mask


def spec_ids(blob: bytes) -> list[int]:
    """Return the SpecId values decorated in one SPIR-V module, sorted ascending."""
    words = [int.from_bytes(blob[i:i + 4], "little") for i in range(0, len(blob), 4)]
    ids: set[int] = set()
    i = SPIRV_HEADER_WORDS
    while i < len(words):
        word_count = words[i] >> 16
        opcode = words[i] & 0xFFFF
        if word_count == 0:
            sys.exit("malformed SPIR-V: zero-length instruction")
        # OpDecorate <target-id> <decoration> [<literal>...]
        if opcode == SPIRV_OP_DECORATE and word_count >= 4:
            if words[i + 2] == SPIRV_DECORATION_SPEC_ID:
                ids.add(words[i + 3])
        i += word_count
    return sorted(ids)


BANNER = [
    "// GENERATED FILE - DO NOT EDIT BY HAND.",
    "// Regenerate with: scripts/gen-vulkan-spirv.py",
    "//",
    "// SPIR-V for the Vulkan backend's compute kernels (BACKEND-VULKAN),",
    "// compiled from the GLSL in src/vt/vulkan/shaders/*.comp.",
    "//",
    "// WHY THE SPIR-V IS COMMITTED rather than compiled by the build, as",
    "// llama.cpp's vulkan-shaders-gen does at build time: the build then needs NO",
    "// shader toolchain on any machine, which is hermetic everywhere including CI",
    "// and the aarch64 gate box. libshaderc would also be a compiled third-party",
    "// dependency, which .agents/discipline.md forbids. The cost is an obligation",
    "// to re-run this generator when a .comp changes, which the",
    "// vulkan-spirv-freshness CI job enforces.",
    "//",
    "// WHY THE WORDS LIVE IN THE .cpp AND NOT THE HEADER (VK-A1): at the target",
    "// shader surface the blobs are megabytes of array initializer, and anything in",
    "// the header is re-parsed by every including TU. The header carries the struct",
    "// and extern declarations; vulkan_spirv.cpp carries the data, so adding shaders",
    "// costs one TU's compile time rather than all of them.",
    "//",
]


def render_header(blobs: dict[str, bytes], version: str) -> str:
    lines = list(BANNER)
    add = lines.append
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
    add("// Name -> SPIR-V module. The NAME is the shader's file stem and is also the")
    add("// key the pipeline cache uses (src/vt/vulkan/vulkan_context.cpp).")
    add("//")
    add("// spec_ids lists the SPECIALIZATION CONSTANT IDs the module declares, parsed")
    add("// from its OpDecorate SpecId instructions and sorted ascending. The host")
    add("// passes specialization values BY ID, and Vulkan SILENTLY IGNORES a map entry")
    add("// whose ID the module does not declare - so without this table a host/shader")
    add("// drift produces WRONG NUMBERS instead of a clean failure.")
    add("//")
    add("// binding_count is the number of descriptor bindings the module declares, and")
    add("// writable_mask has bit i set iff binding i is NOT decorated NonWritable, i.e.")
    add("// iff the shader is permitted to WRITE that buffer. Both are parsed out of the")
    add("// compiled module rather than written down beside it.")
    add("//")
    add("// This is the READ/WRITE SET the dispatch path needs to decide whether two")
    add("// dispatches are genuinely independent. Dispatch() is handed one flat array of")
    add("// VkBuffers with no roles attached, and inferring a role from the binding index")
    add("// would be an unverified assumption whose failure mode is a MISSING BARRIER and")
    add("// therefore silently wrong numbers. `readonly` in the GLSL is enforced by")
    add("// glslang (writing such a block is a compile error) and recorded as NonWritable")
    add("// in the SPIR-V, so this mask is a machine-checked fact about the shader.")
    add("//")
    add("// A bit is SET unless read-only is positively proven, so anything the")
    add("// reflection does not understand degrades to an extra barrier, never a")
    add("// missing one.")
    add("struct SpirvModule {")
    add("  const char* name;")
    add("  const uint32_t* words;")
    add("  size_t word_count;")
    add("  const uint32_t* spec_ids;")
    add("  size_t spec_id_count;")
    add("  uint32_t binding_count;")
    add("  uint32_t writable_mask;")
    add("};")
    add("")
    add("// DEFINED IN vulkan_spirv.cpp. The array is `extern` and therefore of unknown")
    add("// bound here, so kSpirvModuleCount is the ONLY way to size it - use it rather")
    add("// than sizeof/sizeof.")
    add("extern const SpirvModule kSpirvModules[];")
    add("extern const size_t kSpirvModuleCount;")
    add("")
    add("}  // namespace vt::vulkan")
    add("")
    add("#endif  // VT_VULKAN_VULKAN_SPIRV_H_")
    return "\n".join(lines) + "\n"


def render_source(blobs: dict[str, bytes], version: str) -> str:
    lines = list(BANNER)
    add = lines.append
    add(f"// Produced by: {version}")
    add(f"// Target environment: {TARGET_ENV}")
    add("")
    add('#include "vulkan_spirv.h"')
    add("")
    add("namespace vt::vulkan {")
    add("namespace {")
    add("")
    for name in sorted(blobs):
        words = [int.from_bytes(blobs[name][i:i + 4], "little")
                 for i in range(0, len(blobs[name]), 4)]
        add(f"constexpr uint32_t kSpv_{name}[] = {{")
        for i in range(0, len(words), 8):
            chunk = ", ".join(f"0x{w:08x}u" for w in words[i:i + 8])
            add(f"    {chunk},")
        add("};")
        add("")
    for name in sorted(blobs):
        ids = spec_ids(blobs[name])
        if ids:
            add(f"constexpr uint32_t kSpecIds_{name}[] = {{")
            add("    " + ", ".join(f"{i}u" for i in ids) + ",")
            add("};")
            add("")
    add("}  // namespace")
    add("")
    add("const SpirvModule kSpirvModules[] = {")
    for name in sorted(blobs):
        ids = spec_ids(blobs[name])
        idp = f"kSpecIds_{name}" if ids else "nullptr"
        count, mask = binding_access(blobs[name], name)
        add(f'    {{"{name}", kSpv_{name}, sizeof(kSpv_{name}) / sizeof(uint32_t), '
            f'{idp}, {len(ids)}, {count}u, 0x{mask:08x}u}},')
    add("};")
    add("const size_t kSpirvModuleCount = sizeof(kSpirvModules) / sizeof(kSpirvModules[0]);")
    add("")
    add("}  // namespace vt::vulkan")
    return "\n".join(lines) + "\n"


def strip_version(s: str) -> str:
    """Drop the compiler-version line, which legitimately differs per machine."""
    return "\n".join(l for l in s.splitlines() if not l.startswith("// Produced by:"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler", default=None)
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed artifacts are stale instead of rewriting them")
    args = ap.parse_args()

    cc = find_compiler(args.compiler)
    version = compiler_version(cc)
    sources = sorted(SHADER_DIR.glob("*.comp"))
    if not sources:
        sys.exit(f"no shaders found in {SHADER_DIR}")

    blobs = {}
    for src in sources:
        b = compile_one(cc, src)
        if b is not None:
            blobs[src.stem] = b
        # Compile extra variants (e.g. -DVT_IDOT) for shaders that declare them.
        for define, suffix in EXTRA_VARIANTS.get(src.stem, []):
            b = compile_one(cc, src, extra_defs=[define])
            if b is not None:
                blobs[src.stem + suffix] = b
    outputs = ((OUT_HEADER, render_header(blobs, version)),
               (OUT_SOURCE, render_source(blobs, version)))

    if args.check:
        stale = [path.relative_to(REPO) for path, text in outputs
                 if strip_version(path.read_text() if path.exists() else "")
                 != strip_version(text)]
        if stale:
            sys.exit("STALE, re-run scripts/gen-vulkan-spirv.py: "
                     + ", ".join(str(p) for p in stale))
        print("committed SPIR-V is up to date")
        return

    for path, text in outputs:
        path.write_text(text)
    total = sum(len(b) for b in blobs.values())
    print(f"wrote {OUT_HEADER.relative_to(REPO)} + {OUT_SOURCE.relative_to(REPO)}: "
          f"{len(blobs)} modules, {total} bytes of SPIR-V")
    print(f"compiler: {version}")


if __name__ == "__main__":
    main()
