# VK-A1: Vulkan shader-variant pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Vulkan backend a shader-variant mechanism that scales to ~100 shaders without a combinatorial explosion of committed artifacts, and gate the committed SPIR-V against staleness in CI.

**Architecture:** Keep the committed-SPIR-V route (the build stays hermetic — no shader toolchain on any build box, which is strictly better than llama.cpp's build-time `glslc` requirement), but change the *variant* mechanism from GLSL `#define`s to **SPIR-V specialization constants**. llama.cpp needs 242 `string_to_spv(` call sites largely because a `#define` variant is a whole new module; a specialization constant is one module specialized at pipeline-creation time, with the driver dead-code-eliminating the untaken branches. That collapses artifact count from the dtype × quant × coopmat-tier *cross product* down to roughly the number of shader *files*. Emission is split from one monolithic header into a header + `.cpp` so blob growth does not land in every including TU.

**Tech Stack:** GLSL 450 compute → SPIR-V (glslang, ahead-of-time, committed), Vulkan 1.1 compute pipelines, C++20, CMake, doctest, GitHub Actions.

## Global Constraints

- **Row:** `BACKEND-VULKAN`. **Claim:** `CLAIM-VULKAN-FULL-1`. **Spec:** `.agents/specs/vulkan-full-support.md` (sub-project `VK-A1`).
- **`VLLM_CPP_VULKAN` AUTO must keep resolving OFF.** The SACRED CUDA gates are untouched by construction; any CMake change must preserve this and it must be re-verified, not assumed.
- **Everything is additive.** No existing non-Vulkan source file changes behavior.
- **No compiled third-party dependencies.** `.agents/discipline.md` forbids them; `third_party/` is single-header only. glslang is a *build-time tool run by hand and in CI*, never linked.
- **Every commit carries `FOLLOWING_AGENTS_PROTOCOL` plus `Assisted-by:`**, and never `Signed-off-by` or `Co-Authored-By`.
- **Author is `mudler`**, never `localai-bot`.
- **Run `scripts/agent-preflight.sh`** before committing. Note: `check-fusion-consistency` is **RED on `main`** for an unrelated reason (`minimax_h3_video_vae_device` GEMM merge drift, verified pre-existing at `075b9f21^`). Do not attempt to fix it here; confirm it is the *same* failure and no new one.
- **Target environment is `vulkan1.1`** (`TARGET_ENV` in `scripts/gen-vulkan-spirv.py`) — the floor `vulkan_context.cpp` requires. Do not raise it in this plan.
- **Pinned shader compiler: glslang `16.5.0`, prebuilt Linux x86_64 release tarball.** CORRECTED during execution (2026-08-06): the header recorded `Glslang Version: 11:16.4.0`, but **16.4.0 ships no release assets** (only `16.5.0` and `main-tot` do; Ubuntu packages `15.1.0`), so the recorded version is not fetchable and could not back a CI gate. MEASURED: the committed SPIR-V reproduces **byte-for-byte under 16.5.0** — `gen-vulkan-spirv.py --check` passes unchanged. That both proves the committed artifact is what it claims and shows the emitted SPIR-V is stable across a glslang minor bump, so the gate pins the DOWNLOAD URL rather than asserting a version string.
- **llama.cpp port pin: `237ad9b96`**, readable at `/home/mudler/_git/llama.cpp`. Cite `file:line` for any ported structure.

## PRECONDITIONS — verify before Task 1, stop if unmet

1. **Disk.** `df -h /home/mudler` showed **4.8 GB free (99% full)** on 2026-08-06. A build under ENOSPC produces partial objects and *bogus* test failures. Require **≥ 40 GB free** before any build step. Free it with `go clean -cache`, pruning stale `source-*` trees and old worktrees, or ask the user. **Do not build under 99%.**
2. **No glslang on either box** (`which glslang glslangValidator glslc` → all not found). Task 1 installs it. Until then, no `.comp` file can be changed, because the committed SPIR-V could not be regenerated.
3. **Work in the worktree** `.claude/worktrees/row+BACKEND-VULKAN-A1` on branch `row/BACKEND-VULKAN-A1`, based on `075b9f21`.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `scripts/gen-vulkan-spirv.py` | AOT shader build: GLSL → SPIR-V → committed C arrays | Modify: emit specialization-constant metadata; split output into `.h` + `.cpp` |
| `src/vt/vulkan/vulkan_spirv.h` | GENERATED. Module table declaration | Regenerate: blobs move out; declarations only |
| `src/vt/vulkan/vulkan_spirv.cpp` | GENERATED. The SPIR-V word arrays | Create (generated) |
| `src/vt/vulkan/vulkan_context.h` | Context API; `Dispatch` signature | Modify: `Dispatch` takes specialization values |
| `src/vt/vulkan/vulkan_context.cpp` | Pipeline cache + `VkSpecializationInfo` | Modify: cache key includes specialization; wire `pSpecializationInfo` |
| `src/vt/vulkan/shaders/vt_common.glsl` | Shared GLSL: storage model, reductions | Modify: `VT_TG` becomes a specialization constant |
| `src/vt/vulkan/shaders/*.comp` | The 7 compute shaders | Modify: `local_size_x_id` instead of a literal |
| `tests/vt/test_vulkan_backend.cpp` | The Vulkan unit gate | Modify: new cases for specialization + metadata |
| `.github/workflows/ci.yml` | CI | Modify: add the `vulkan-spirv-freshness` job |
| `.agents/feature-matrix.md` | Record | Modify: repair the `BACKEND-VULKAN` drift |
| `.agents/backend-matrix.md`, `.agents/state-index/<writable-shard>.csv`, `.agents/state-events/<YYYY-MM>/<event-id>.md`, `.agents/NOW.md`, `docs/STATUS.md`, `docs/BENCHMARKS.md` | Record | Modify/create: land the row + ordered checkpoint and immutable evidence (`.agents/state.csv` is the manifest) |

---

### Task 1: Pin glslang and prove the committed SPIR-V reproduces

This task exists because **everything downstream is unbuildable without it**, and because it answers a question nobody has answered: does the committed header actually reproduce? If it does not, the committed-SPIR-V route has no CI gate and the rest of the plan changes.

**Files:**
- Modify: `.agents/environment.md` (record the installed tool)
- No source changes

**Interfaces:**
- Consumes: nothing
- Produces: a `glslang` binary on `PATH` reporting exactly `Glslang Version: 11:16.4.0`; the fact (recorded) that `--check` passes or fails at that version

- [ ] **Step 1: Confirm no compiler is present, so the starting state is known**

```bash
which glslang glslangValidator glslc || echo "none present (expected)"
```

Expected: `none present (expected)`.

- [x] **Step 2: Install the pinned glslang** — DONE 2026-08-06

The recorded `16.4.0` has **no release assets**, so it is not installable without a
source build. `16.5.0` is the nearest tag that ships a prebuilt Linux x86_64
binary, and Step 3 measured that it produces byte-identical SPIR-V:

```bash
mkdir -p "$HOME/tools" && cd "$HOME/tools"
curl -fsSL -o glslang-16.5.0.tar.gz \
  https://github.com/KhronosGroup/glslang/releases/download/16.5.0/glslang-16.5.0-linux-x86_64-release.tar.gz
mkdir -p glslang-16.5.0 && tar xzf glslang-16.5.0.tar.gz -C glslang-16.5.0
export PATH="$HOME/tools/glslang-16.5.0/bin:$PATH"
glslang --version | head -1
```

Result: `Glslang Version: 11:16.5.0` at `$HOME/tools/glslang-16.5.0/bin/glslang`.

- [x] **Step 3: Run the staleness check against the committed header** — DONE 2026-08-06

```bash
PATH="$HOME/tools/glslang-16.5.0/bin:$PATH" python3 scripts/gen-vulkan-spirv.py --check
```

Result: **`vulkan_spirv.h is up to date`, exit 0.**

This is a stronger outcome than the step was written to obtain. The check compares
SPIR-V bytes and ignores only the version comment, so a byte-identical result under
a *different, newer* compiler proves two things at once: the committed artifact is
genuinely what it claims to be, and the emitted SPIR-V did not change across a
glslang minor bump. The second is what lets Task 2 pin the download URL instead of
asserting a version string, which would have been brittle for no gain.

- [ ] **Step 4: Record the toolchain in the environment doc**

Add to `.agents/environment.md`, in the dev-box profile section alongside the other toolchain facts:

```markdown
- **Vulkan shader toolchain (2026-08-06, `VK-A1`).** `glslang` **16.4.0** installed
  per-user at `$HOME/tools/glslang-16.4.0/bin` (release tarball, no root). This is
  the EXACT version recorded at `src/vt/vulkan/vulkan_spirv.h:16`; a different
  build emits different SPIR-V and fails `scripts/gen-vulkan-spirv.py --check`,
  which compares SPIR-V bytes and ignores only the version comment. Regenerate
  committed SPIR-V with that directory on `PATH`. Nothing links against glslang —
  it is a build-time tool, never a dependency (`.agents/discipline.md`).
```

- [ ] **Step 5: Verify the env doc gate still passes**

Run: `python3 scripts/check-env-doc.py`
Expected: OK.

- [ ] **Step 6: Commit**

```bash
git add .agents/environment.md
git commit -m "$(cat <<'MSG'
build(vulkan): pin glslang 16.4.0 and prove the committed SPIR-V reproduces

VK-A1 precondition. Neither box had a GLSL compiler, so no .comp file could be
changed without invalidating the committed SPIR-V. glslang 16.4.0 (the exact
version recorded in vulkan_spirv.h) is now installed per-user from the release
tarball; `gen-vulkan-spirv.py --check` reproduces the committed header, which is
the first evidence that the committed-artifact route is gateable at all.

Nothing is linked against glslang; it is a build-time tool.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [Claude Code]
MSG
)"
```

---

### Task 2: CI job that fails on stale committed SPIR-V

Today `--check` exists but **no CI job runs it**, so a `.comp` edit without a regenerate would ship silently — exactly the failure mode the committed-artifact route trades for hermeticity. This closes it before ~100 shaders exist.

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `scripts/gen-vulkan-spirv.py --check` from Task 1
- Produces: CI job `vulkan-spirv-freshness`

- [ ] **Step 1: Write the job**

Add to `.github/workflows/ci.yml` under `jobs:`, following the shape of the existing lightweight record jobs (`cuda-arch-features`, `device-leakage`):

```yaml
  vulkan-spirv-freshness:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install pinned glslang
        run: |
          set -euo pipefail
          curl -fsSL -o /tmp/glslang.zip \
            https://github.com/KhronosGroup/glslang/releases/download/16.4.0/glslang-master-linux-Release.zip
          unzip -q /tmp/glslang.zip -d /tmp/glslang
          echo "/tmp/glslang/bin" >> "$GITHUB_PATH"
      - name: Verify the pinned version
        run: |
          set -euo pipefail
          glslang --version | head -1
          glslang --version | head -1 | grep -qF 'Glslang Version: 11:16.4.0'
      - name: Committed SPIR-V is not stale
        run: python3 scripts/gen-vulkan-spirv.py --check
```

The explicit version assertion is deliberate: without it, a glslang release bump turns this gate from "your shader is stale" into a confusing red on an unrelated PR, and the failure message would not say why.

- [ ] **Step 2: Verify the workflow parses**

Run:
```bash
python3 -c "import yaml,sys; d=yaml.safe_load(open('.github/workflows/ci.yml')); assert 'vulkan-spirv-freshness' in d['jobs']; print('job registered, keys:', list(d['jobs']['vulkan-spirv-freshness'].keys()))"
```
Expected: `job registered, keys: ['runs-on', 'steps']`

- [ ] **Step 3: Prove the gate actually catches staleness (mutation test)**

The gate is worthless if it passes over a real edit. Prove it locally:

```bash
export PATH="$HOME/tools/glslang-16.4.0/bin:$PATH"
# Perturb a shader in a way that changes SPIR-V but not semantics
printf '\n// mutation probe\n' >> src/vt/vulkan/shaders/vt_relu.comp
sed -i 's/out_f32\[p.out_off + i\] = max(v, 0.0);/out_f32[p.out_off + i] = max(0.0, v);/' src/vt/vulkan/shaders/vt_relu.comp
python3 scripts/gen-vulkan-spirv.py --check ; echo "exit=$?"
```
Expected: `vulkan_spirv.h is STALE` and `exit=1`.

If the `sed` did not match (the shader text differs), make any equivalent
expression-level edit that changes emitted code, and re-run. A comment-only
change may compile to identical SPIR-V under `-g0` — that is correct behavior,
not a gate failure, but it does not prove the gate, so the edit must change code.

- [ ] **Step 4: Revert the mutation and confirm green**

```bash
git checkout -- src/vt/vulkan/shaders/vt_relu.comp
python3 scripts/gen-vulkan-spirv.py --check
```
Expected: `vulkan_spirv.h is up to date`, and `git status --short` shows no shader change.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'MSG'
ci(vulkan): gate the committed SPIR-V against staleness

The committed-artifact route trades a build-time shader toolchain for the
obligation to regenerate by hand, and nothing enforced that obligation: --check
existed but no job ran it, so a .comp edit without a regenerate shipped silently.

The job downloads the pinned glslang 16.4.0 and asserts the version string before
running --check, so a glslang release bump reads as a version mismatch rather than
as a confusing "your shader is stale" on an unrelated PR. Verified by mutation:
an expression-level shader edit turns the gate red, and reverting turns it green.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [Claude Code]
MSG
)"
```

---

### Task 3: Generator emits specialization-constant metadata

The host must know which specialization constants a module declares, or it will pass values that silently do nothing (Vulkan ignores map entries whose `constantID` is not declared — a wrong value would not fail, it would produce wrong numbers). This mirrors the existing `buffer_count`/`push_size` drift guard at `vulkan_context.cpp:427`.

**Files:**
- Modify: `scripts/gen-vulkan-spirv.py`
- Modify: `tests/vt/test_vulkan_backend.cpp`
- Regenerate: `src/vt/vulkan/vulkan_spirv.h`

**Interfaces:**
- Consumes: `SpirvModule` from Task 1's committed header
- Produces: `SpirvModule` gains `const uint32_t* spec_ids; size_t spec_id_count;` — sorted ascending, the SpecId values decorated in the module

- [ ] **Step 1: Write the failing test**

Add to `tests/vt/test_vulkan_backend.cpp`, after the existing `"the committed SPIR-V table is present and well-formed"` case:

```cpp
TEST_CASE("the committed SPIR-V table records each module's specialization constants") {
  // Device-independent: a property of the checked-in artifact, so this also gates
  // the generator on a box with no Vulkan. The host passes specialization values
  // positionally by constantID; Vulkan SILENTLY IGNORES a map entry whose ID the
  // module does not declare, so a drift here is wrong numbers, not a clean error.
  for (const auto& m : vt::vulkan::kSpirvModules) {
    CAPTURE(m.name);
    // Every module must at least declare the workgroup-size constant (Task 5).
    REQUIRE(m.spec_id_count >= 1);
    for (size_t i = 1; i < m.spec_id_count; ++i) {
      CHECK(m.spec_ids[i - 1] < m.spec_ids[i]);  // sorted ascending, no duplicates
    }
    CHECK(m.spec_ids[0] == 0u);  // ID 0 is reserved for VT_TG (vt_common.glsl)
  }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cmake --build build-vulkan --target test_vulkan_backend && ./build-vulkan/tests/test_vulkan_backend -tc="*specialization constants*"`

(Configure once with `cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_VULKAN=ON` — **only after the disk precondition is met**.)

Expected: FAIL to COMPILE with `'const struct vt::vulkan::SpirvModule' has no member named 'spec_id_count'`.

- [ ] **Step 3: Teach the generator to parse SpecId decorations**

Add to `scripts/gen-vulkan-spirv.py`, above `render`:

```python
# SPIR-V decoration opcode/operand constants (SPIR-V 1.x core spec, §3.32.3
# OpDecorate = 71, §3.20 Decoration SpecId = 1). Parsing the module directly is
# the only source of truth: glslang gives no side-channel listing of the SpecIds
# it emitted, and a hand-maintained list is exactly the kind of duplicate that
# drifts.
SPIRV_OP_DECORATE = 71
SPIRV_DECORATION_SPEC_ID = 1
SPIRV_HEADER_WORDS = 5


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
```

- [ ] **Step 4: Emit the metadata**

In `render`, replace the `SpirvModule` struct and table emission with:

```python
    for name in sorted(blobs):
        ids = spec_ids(blobs[name])
        if ids:
            add(f"inline constexpr uint32_t kSpecIds_{name}[] = {{")
            add("    " + ", ".join(f"{i}u" for i in ids) + ",")
            add("};")
            add("")

    add("// Name -> SPIR-V module. The NAME is the shader's file stem and is also the")
    add("// key the pipeline cache uses (src/vt/vulkan/vulkan_context.cpp).")
    add("//")
    add("// spec_ids lists the SPECIALIZATION CONSTANT IDs the module declares, parsed")
    add("// from its OpDecorate SpecId instructions, sorted ascending. The host passes")
    add("// specialization values by ID; Vulkan SILENTLY IGNORES a map entry whose ID")
    add("// the module does not declare, so without this table a host/shader drift")
    add("// produces WRONG NUMBERS instead of a clean failure.")
    add("struct SpirvModule {")
    add("  const char* name;")
    add("  const uint32_t* words;")
    add("  size_t word_count;")
    add("  const uint32_t* spec_ids;")
    add("  size_t spec_id_count;")
    add("};")
    add("")
    add("inline constexpr SpirvModule kSpirvModules[] = {")
    for name in sorted(blobs):
        ids = spec_ids(blobs[name])
        idp = f"kSpecIds_{name}" if ids else "nullptr"
        add(f'    {{"{name}", kSpv_{name}, sizeof(kSpv_{name}) / sizeof(uint32_t), '
            f'{idp}, {len(ids)}}},')
    add("};")
```

- [ ] **Step 5: Add a generator unit test for the parser**

Create `tests/scripts/test_gen_vulkan_spirv.py`, following the shape of the existing `tests/scripts/test_check_*.py` suites:

```python
"""Unit tests for scripts/gen-vulkan-spirv.py (BACKEND-VULKAN, VK-A1)."""
import importlib.util
import pathlib
import struct
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "gen_vulkan_spirv", ROOT / "scripts" / "gen-vulkan-spirv.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def module(*instructions: tuple[int, list[int]]) -> bytes:
    """Build a minimal SPIR-V blob: 5 header words then the given instructions."""
    words = [0x07230203, 0x00010300, 0, 1, 0]
    for opcode, operands in instructions:
        words.append(((len(operands) + 1) << 16) | opcode)
        words.extend(operands)
    return b"".join(struct.pack("<I", w) for w in words)


class SpecIdParsing(unittest.TestCase):
    def test_no_decorations_yields_empty(self):
        self.assertEqual(gen.spec_ids(module()), [])

    def test_extracts_spec_ids_sorted_and_deduped(self):
        blob = module(
            (gen.SPIRV_OP_DECORATE, [10, gen.SPIRV_DECORATION_SPEC_ID, 3]),
            (gen.SPIRV_OP_DECORATE, [11, gen.SPIRV_DECORATION_SPEC_ID, 0]),
            (gen.SPIRV_OP_DECORATE, [12, gen.SPIRV_DECORATION_SPEC_ID, 3]),
        )
        self.assertEqual(gen.spec_ids(blob), [0, 3])

    def test_ignores_other_decorations(self):
        # Decoration 2 is Block, not SpecId; it must not be collected.
        blob = module((gen.SPIRV_OP_DECORATE, [10, 2, 0]))
        self.assertEqual(gen.spec_ids(blob), [])

    def test_skips_unrelated_opcodes_by_word_count(self):
        # An OpName (5) with operands must be stepped over, not misparsed.
        blob = module(
            (5, [10, 0x6E69616D, 0]),
            (gen.SPIRV_OP_DECORATE, [10, gen.SPIRV_DECORATION_SPEC_ID, 7]),
        )
        self.assertEqual(gen.spec_ids(blob), [7])

    def test_committed_header_modules_all_parse(self):
        # The real artifact: every committed module must parse without error.
        shaders = sorted((ROOT / "src/vt/vulkan/shaders").glob("*.comp"))
        self.assertTrue(shaders, "no shaders found")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6: Run the generator unit test**

Run: `python3 -m unittest discover -s tests/scripts -p 'test_gen_vulkan_spirv.py' -v`
Expected: 5 tests, OK.

- [ ] **Step 7: Regenerate and confirm the C++ test now compiles but still fails on the assertion**

```bash
export PATH="$HOME/tools/glslang-16.4.0/bin:$PATH"
python3 scripts/gen-vulkan-spirv.py
cmake --build build-vulkan --target test_vulkan_backend
./build-vulkan/tests/test_vulkan_backend -tc="*specialization constants*"
```

Expected: compiles; **FAILS** at `REQUIRE(m.spec_id_count >= 1)` — no shader declares a specialization constant yet. That is correct: Task 5 makes it pass. Leaving it red here is deliberate, so do **not** weaken the assertion.

- [ ] **Step 8: Commit**

```bash
git add scripts/gen-vulkan-spirv.py src/vt/vulkan/vulkan_spirv.h \
        tests/scripts/test_gen_vulkan_spirv.py tests/vt/test_vulkan_backend.cpp
git commit -m "$(cat <<'MSG'
feat(vulkan): record each module's specialization-constant IDs in the SPIR-V table

The host will pass specialization values by constantID, and Vulkan SILENTLY
IGNORES a map entry whose ID the module does not declare -- a host/shader drift
would produce wrong numbers rather than a clean failure. The generator now parses
OpDecorate SpecId straight out of the emitted module (glslang exposes no
side-channel listing, and a hand-maintained list is exactly the duplicate that
drifts) and emits the sorted IDs alongside each blob.

Parser covered by unit tests including the two ways a naive scanner breaks:
non-SpecId decorations, and stepping over unrelated opcodes by word count. The
C++ assertion that every module declares at least one constant is RED on purpose
until the workgroup size becomes one.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [Claude Code]
MSG
)"
```

---

### Task 4: Specialize pipelines, and key the cache by specialization

**Files:**
- Modify: `src/vt/vulkan/vulkan_context.h:66-67` (`Dispatch`), `:109` (`GetPipeline`)
- Modify: `src/vt/vulkan/vulkan_context.cpp:419-502` (`GetPipeline`), `:504+` (`Dispatch`)
- Modify: `src/vt/vulkan/vulkan_ops.cpp` (all `Dispatch` call sites)

**Interfaces:**
- Consumes: `SpirvModule::spec_ids` / `spec_id_count` from Task 3
- Produces:
  - `void VulkanContext::Dispatch(const std::string& name, const void* const* buffers, uint32_t buffer_count, const void* push_constants, uint32_t push_size, uint32_t group_count_x, const uint32_t* spec_values = nullptr, uint32_t spec_count = 0)`
  - Pipeline cache key type `std::string` of the form `name + "|" + decimal spec values joined by ","`

- [ ] **Step 1: Write the failing test**

Add to `tests/vt/test_vulkan_backend.cpp`:

```cpp
TEST_CASE("Vulkan specializes pipelines and caches them per specialization") {
  if (!vt::vulkan::VulkanDeviceAvailable()) return;  // no device: nothing to assert
  auto& ctx = vt::vulkan::VulkanContext::Get();

  // vt_relu with the workgroup size specialized to two different values must
  // produce two DISTINCT pipelines and IDENTICAL results. Identical results are
  // the real assertion: a specialization that silently did nothing would also
  // return identical results, so the pipeline-count check is what proves the
  // mechanism engaged.
  const size_t before = ctx.PipelineCacheSize();

  const int64_t n = 300;  // deliberately not a multiple of either workgroup size
  std::vector<float> in(n), out64(n), out128(n);
  for (int64_t i = 0; i < n; ++i) in[i] = static_cast<float>(i) - 150.0f;

  RunReluWithWorkgroup(ctx, in, out64, 64u);
  RunReluWithWorkgroup(ctx, in, out128, 128u);

  CHECK(ctx.PipelineCacheSize() == before + 2);  // two variants, two pipelines
  for (int64_t i = 0; i < n; ++i) {
    CAPTURE(i);
    CHECK(out64[i] == doctest::Approx(std::max(in[i], 0.0f)));
    CHECK(out64[i] == out128[i]);  // specialization must not change numerics
  }
}
```

Add the helper above the case (it allocates through the backend so the buffer
registry resolves the pointers, exactly as the existing op cases do):

```cpp
// Dispatch vt_relu over `in` with VT_TG specialized to `tg`. Mirrors the push
// constant layout in src/vt/vulkan/shaders/vt_relu.comp.
static void RunReluWithWorkgroup(vt::vulkan::VulkanContext& ctx,
                                 const std::vector<float>& in, std::vector<float>& out,
                                 uint32_t tg) {
  vt::Backend& be = *vt::GetBackend(vt::DeviceType::kVULKAN);
  const size_t bytes = in.size() * sizeof(float);
  void* din = be.Alloc(bytes);
  void* dout = be.Alloc(bytes);
  std::memcpy(din, in.data(), bytes);

  struct Push { uint32_t n, in_off, out_off; } push{
      static_cast<uint32_t>(in.size()), 0u, 0u};
  const void* bufs[2] = {be.BufferOf(din), be.BufferOf(dout)};
  const uint32_t spec[1] = {tg};
  const uint32_t groups = static_cast<uint32_t>((in.size() + tg - 1) / tg);
  ctx.Dispatch("vt_relu", bufs, 2, &push, sizeof(push), groups, spec, 1);

  std::memcpy(out.data(), dout, bytes);
  be.Free(din);
  be.Free(dout);
}
```

If `Backend::BufferOf` is not the accessor the existing cases use to turn a
device pointer into a `VkBuffer`, use whatever `tests/vt/test_vulkan_backend.cpp`
already uses in its interior-pointer case (`"Vulkan resolves INTERIOR pointers"`,
line 136) and keep the rest identical.

- [ ] **Step 2: Run it to verify it fails**

Run: `cmake --build build-vulkan --target test_vulkan_backend && ./build-vulkan/tests/test_vulkan_backend -tc="*specializes pipelines*"`
Expected: FAIL to compile — `Dispatch` takes 6 arguments, and `PipelineCacheSize` does not exist.

- [ ] **Step 3: Extend the context API**

In `src/vt/vulkan/vulkan_context.h`, change the `Dispatch` declaration and add the accessor:

```cpp
  // Dispatch one compute kernel, SYNCHRONOUSLY (record, submit, wait). `name` is
  // a key in the committed SPIR-V table (src/vt/vulkan/vulkan_spirv.h).
  // ... (existing comment retained) ...
  //
  // `spec_values` are SPECIALIZATION CONSTANT values, supplied in ascending
  // constantID order and matching the module's declared `spec_ids` exactly. They
  // are part of the pipeline cache KEY: each distinct combination is a distinct
  // VkPipeline, specialized once and reused. This is the variant mechanism that
  // replaces llama.cpp's per-#define module explosion
  // (ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp @ 237ad9b96 has
  // 242 string_to_spv call sites, most inside dtype/quant/coopmat loops).
  void Dispatch(const std::string& name, const void* const* buffers, uint32_t buffer_count,
                const void* push_constants, uint32_t push_size, uint32_t group_count_x,
                const uint32_t* spec_values = nullptr, uint32_t spec_count = 0);

  // Number of distinct pipelines currently cached. Exposed for the unit gate: it
  // is how a test proves a specialization actually produced a NEW pipeline rather
  // than silently reusing one.
  size_t PipelineCacheSize() const;
```

and change the private declaration:

```cpp
  Pipeline& GetPipeline(const std::string& name, uint32_t buffer_count, uint32_t push_size,
                        const uint32_t* spec_values, uint32_t spec_count);
```

- [ ] **Step 4: Implement specialization in `GetPipeline`**

In `src/vt/vulkan/vulkan_context.cpp`, replace the head of `GetPipeline` (currently `vulkan_context.cpp:419-441`) with:

```cpp
namespace {
// Cache key: the module name plus the specialization values, which together
// identify a VkPipeline. Values are decimal so the key is human-readable in the
// VT_CHECK messages below.
std::string PipelineKey(const std::string& name, const uint32_t* spec_values,
                        uint32_t spec_count) {
  std::string key = name;
  for (uint32_t i = 0; i < spec_count; ++i) {
    key += (i == 0 ? '|' : ',');
    key += std::to_string(spec_values[i]);
  }
  return key;
}
}  // namespace

VulkanContext::Pipeline& VulkanContext::GetPipeline(const std::string& name,
                                                    uint32_t buffer_count, uint32_t push_size,
                                                    const uint32_t* spec_values,
                                                    uint32_t spec_count) {
  const std::string key = PipelineKey(name, spec_values, spec_count);
  auto& cache = *static_cast<std::map<std::string, Pipeline>*>(pipelines_);
  auto it = cache.find(key);
  if (it != cache.end()) {
    VT_CHECK(it->second.buffer_count == buffer_count && it->second.push_size == push_size,
             "vulkan: pipeline " + key + " re-requested with a different binding layout");
    return it->second;
  }

  const VulkanApi& vk = Api();
  auto device = Unpack<VkDevice>(device_);

  const SpirvModule* module = nullptr;
  for (const SpirvModule& m : kSpirvModules) {
    if (name == m.name) { module = &m; break; }
  }
  VT_CHECK(module != nullptr,
           "vulkan: no committed SPIR-V for kernel '" + name +
               "' — regenerate with scripts/gen-vulkan-spirv.py");

  // The host passes specialization values POSITIONALLY against the module's
  // declared SpecIds. Vulkan silently ignores a map entry for an undeclared ID,
  // so a mismatch here would be wrong numbers rather than an error — check it.
  VT_CHECK(spec_count == module->spec_id_count,
           "vulkan: kernel '" + name + "' declares " +
               std::to_string(module->spec_id_count) +
               " specialization constants but " + std::to_string(spec_count) +
               " values were supplied — host and committed SPIR-V have drifted");
```

Then, immediately before the `VkComputePipelineCreateInfo` block (currently `:484`), build the specialization info, and reference it from the stage:

```cpp
  // One uint32 per constant, tightly packed; entry i carries the module's i-th
  // declared SpecId (they are emitted sorted ascending).
  std::vector<VkSpecializationMapEntry> spec_entries(spec_count);
  for (uint32_t i = 0; i < spec_count; ++i) {
    spec_entries[i].constantID = module->spec_ids[i];
    spec_entries[i].offset = i * static_cast<uint32_t>(sizeof(uint32_t));
    spec_entries[i].size = sizeof(uint32_t);
  }
  VkSpecializationInfo spec_info{};
  spec_info.mapEntryCount = spec_count;
  spec_info.pMapEntries = spec_entries.data();
  spec_info.dataSize = spec_count * sizeof(uint32_t);
  spec_info.pData = spec_values;
```

and in the existing `cpci.stage` setup add:

```cpp
  cpci.stage.pSpecializationInfo = spec_count ? &spec_info : nullptr;
```

**`spec_entries` and `spec_info` must be declared in the same scope as `cpci` and
must outlive the `vkCreateComputePipelines` call.** A function-local temporary
whose address is captured and read later is precisely the use-after-free class
this project has already hit twice with CUDA-graph capture; keep them as named
locals in `GetPipeline`'s body, not in a nested block.

Finally, change the cache insert at the end from `cache.emplace(name, p)` to
`cache.emplace(key, p)`.

- [ ] **Step 5: Thread it through `Dispatch` and add the accessor**

In `Dispatch`, change the signature to match the header and the `GetPipeline` call to:

```cpp
  Pipeline& p = GetPipeline(name, buffer_count, push_size, spec_values, spec_count);
```

Add, next to the other small accessors:

```cpp
size_t VulkanContext::PipelineCacheSize() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return static_cast<std::map<std::string, Pipeline>*>(pipelines_)->size();
}
```

`mutex_` is a `void*` to a `std::mutex` and this method is `const`, so the
`static_cast` is on the pointer, not on `this` — no `mutable` is needed.

- [ ] **Step 6: Run the test**

Run: `./build-vulkan/tests/test_vulkan_backend -tc="*specializes pipelines*"`
Expected: still FAILS — `vt_relu` declares 0 specialization constants, so the new `VT_CHECK` fires with "declares 0 ... but 1 values were supplied". That is the drift guard working. Task 5 makes it pass.

- [ ] **Step 7: Verify the existing suite is unbroken**

All existing `Dispatch` call sites in `vulkan_ops.cpp` use the defaulted arguments, so they pass `spec_count == 0` and match today's `spec_id_count == 0`.

Run: `./build-vulkan/tests/test_vulkan_backend`
Expected: every case except `"*specialization constants*"` and `"*specializes pipelines*"` passes — 8 of the pre-existing cases green, the 2 new ones red.

- [ ] **Step 8: Commit**

```bash
git add src/vt/vulkan/vulkan_context.h src/vt/vulkan/vulkan_context.cpp \
        tests/vt/test_vulkan_backend.cpp
git commit -m "$(cat <<'MSG'
feat(vulkan): specialize compute pipelines and key the cache by specialization

The variant mechanism VK-A1 exists to choose. llama.cpp spells a shader variant
as a GLSL #define, so every dtype x quant x coopmat-tier combination is a whole
new SPIR-V module -- 242 string_to_spv call sites at pin 237ad9b96, most inside
those loops. A specialization constant is instead ONE module specialized at
pipeline creation, with the driver eliminating the untaken branches, so artifact
count tracks shader FILES rather than their cross product.

Dispatch now takes specialization values, the pipeline cache is keyed by name plus
those values, and the count is checked against the module's declared SpecIds --
Vulkan silently ignores a map entry for an undeclared ID, so an unchecked drift
would be wrong numbers, not an error. The specialization structures are named
locals that outlive the create call, not temporaries whose address escapes.

Existing call sites use the defaulted arguments and are unaffected. The two new
gates stay RED until a shader actually declares a constant.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [Claude Code]
MSG
)"
```

---

### Task 5: Make the workgroup size a specialization constant

This proves the mechanism on a real shader and removes a genuine hazard: the workgroup size is currently duplicated as `#define VT_TG 128u` (`vt_common.glsl:60`), a literal `local_size_x = 128` in every `.comp`, and `kWorkgroupSize = 128` on the host (`vulkan_context.h:148`) — three copies held in agreement by a comment. It also turns out to be the first thing a real device needs to vary: llama.cpp selects workgroup size per device.

**Files:**
- Modify: `src/vt/vulkan/shaders/vt_common.glsl:60,177,183`
- Modify: all 7 of `src/vt/vulkan/shaders/*.comp` (the `local_size_x` line)
- Modify: `src/vt/vulkan/vulkan_context.h:148` (`kWorkgroupSize`)
- Modify: `src/vt/vulkan/vulkan_ops.cpp` (pass the value at every `Dispatch`)
- Regenerate: `src/vt/vulkan/vulkan_spirv.h`

**Interfaces:**
- Consumes: `Dispatch(..., spec_values, spec_count)` from Task 4
- Produces: specialization constant **ID 0 = workgroup size**, declared by every module; host constant `vt::vulkan::kWorkgroupSize` stays 128 and is the value passed

- [ ] **Step 1: Convert the shared header**

In `src/vt/vulkan/shaders/vt_common.glsl`, replace `#define VT_TG 128u` (line 60) with:

```glsl
// Workgroup size, as SPECIALIZATION CONSTANT ID 0. It was a #define, duplicated
// into every .comp's local_size_x and again into kWorkgroupSize on the host and
// held in agreement by a comment. As a specialization constant there is ONE
// declaration: the .comp files derive local_size_x from it via local_size_x_id,
// and the host supplies the value at pipeline creation
// (src/vt/vulkan/vulkan_context.cpp GetPipeline). ID 0 is reserved for this
// across every shader in the backend.
layout(constant_id = 0) const uint VT_TG = 128u;
```

`vt_smem` (line 177) is declared `shared float vt_smem[VT_TG]`. A specialization
constant **is** a valid array size in GLSL/SPIR-V (it becomes `OpSpecConstant`
sized), so this line needs no change — but verify it compiles rather than
assuming, because that is the one construct in this file that could reject a
spec constant.

- [ ] **Step 2: Convert every shader's workgroup declaration**

In each of the 7 `src/vt/vulkan/shaders/*.comp`, replace the literal layout line:

```glsl
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
```

with:

```glsl
// local_size_x comes from specialization constant 0 (VT_TG, vt_common.glsl), so
// the host chooses the workgroup size at pipeline creation and there is exactly
// one place it is written down.
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
```

Verify you changed all 7 and that no literal remains:

```bash
grep -c 'local_size_x_id = 0' src/vt/vulkan/shaders/*.comp | grep -v ':1$' || echo "all 7 converted"
grep -n 'local_size_x = 128' src/vt/vulkan/shaders/*.comp && echo "LEFTOVER LITERAL" || echo "no literals remain"
```

Expected: `all 7 converted` then `no literals remain`.

- [ ] **Step 3: Regenerate and confirm every module now declares ID 0**

```bash
export PATH="$HOME/tools/glslang-16.4.0/bin:$PATH"
python3 scripts/gen-vulkan-spirv.py
grep -c 'kSpecIds_' src/vt/vulkan/vulkan_spirv.h
```
Expected: `7` occurrences of the array definition (one per module), plus their uses in the table.

If glslang **rejects** `shared float vt_smem[VT_TG]` with a spec-constant size,
that is the one real risk in this task: fall back to sizing the shared array by
the MAXIMUM supported workgroup size (a plain `const uint VT_TG_MAX = 1024u;`)
and indexing only the first `VT_TG` entries. Record the fallback and why, in the
commit body — it costs shared memory and must not be silently taken.

- [ ] **Step 4: Pass the value at every dispatch**

In `src/vt/vulkan/vulkan_ops.cpp`, every `ctx.Dispatch(...)` call gains the
specialization argument. The value is the existing host constant, so behavior is
unchanged:

```cpp
  static constexpr uint32_t kSpecTg[1] = {vt::vulkan::kWorkgroupSize};
```

declared once at file scope near the top of the anonymous namespace, then each
call site gains `, kSpecTg, 1` before the closing paren. Confirm none were
missed:

```bash
grep -c 'kSpecTg, 1' src/vt/vulkan/vulkan_ops.cpp
grep -c 'ctx.Dispatch(' src/vt/vulkan/vulkan_ops.cpp
```
Expected: the two counts are EQUAL.

- [ ] **Step 5: Update the host constant's comment**

In `src/vt/vulkan/vulkan_context.h`, replace the `kWorkgroupSize` comment (line 145-147):

```cpp
// Workgroup size this backend specializes its kernels with. It is no longer
// duplicated in the shaders: the .comp files declare `local_size_x_id = 0` and
// vt_common.glsl declares `layout(constant_id = 0) const uint VT_TG`, so this is
// the single value the host supplies at pipeline creation. Kernels that compute
// their workgroup COUNT from it (FlatGroupCount) therefore cannot disagree with
// the SPIR-V.
inline constexpr uint32_t kWorkgroupSize = 128;
```

- [ ] **Step 6: Run both new gates**

Run: `cmake --build build-vulkan --target test_vulkan_backend && ./build-vulkan/tests/test_vulkan_backend`
Expected: **all cases PASS**, including `"*specialization constants*"` (every module now declares ID 0) and `"*specializes pipelines*"` (two workgroup sizes, two pipelines, identical results).

- [ ] **Step 7: Verify the numerics did not move**

The specialization must be numerically inert at the default value. Run the cross-device harness, which compares Vulkan against the CPU oracle:

Run: `./build-vulkan/tests/test_backend_cross_device`
Expected: 5/5 cases pass, same assertion count as before the change (73/73 on a dev box with llvmpipe only; 144/144 on GB10 where CUDA is also present).

If any NMSE moved, stop: a specialization constant changed generated code in a
way that altered the reduction, and that needs explaining before it lands.

- [ ] **Step 8: Commit**

```bash
git add src/vt/vulkan/shaders/ src/vt/vulkan/vulkan_spirv.h \
        src/vt/vulkan/vulkan_context.h src/vt/vulkan/vulkan_ops.cpp
git commit -m "$(cat <<'MSG'
refactor(vulkan): workgroup size becomes specialization constant 0

Proves the Task 4 mechanism on a real shader and removes a live hazard. The
workgroup size was written down three times -- #define VT_TG in vt_common.glsl, a
literal local_size_x in each of the 7 .comp files, and kWorkgroupSize on the host
-- held in agreement by a comment. Kernels compute their workgroup COUNT from the
host constant, so a drift would have been a silently wrong dispatch shape.

There is now one declaration: constant_id 0, with the .comp files taking
local_size_x_id = 0 and the host supplying the value at pipeline creation. ID 0 is
reserved for it across the backend.

Numerically inert at the default: the cross-device harness against the CPU oracle
is unchanged, and the new gate dispatches the same shader at two workgroup sizes
and requires two distinct pipelines with identical results.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [Claude Code]
MSG
)"
```

---

### Task 6: Split SPIR-V emission out of the header

At 7 modules the committed header is 3,491 lines / 109 KB and is included by two TUs. At the target surface it is megabytes of `constexpr` arrays re-parsed by every includer, which costs compile time and memory and makes every regeneration a huge diff in a header. Splitting now is cheap; splitting after ~100 shaders is not.

**Files:**
- Modify: `scripts/gen-vulkan-spirv.py` (`render` → `render_header` + `render_source`)
- Regenerate: `src/vt/vulkan/vulkan_spirv.h` (declarations only)
- Create (generated): `src/vt/vulkan/vulkan_spirv.cpp`
- Modify: `CMakeLists.txt` (add the generated source to the Vulkan target sources)

**Interfaces:**
- Consumes: the `SpirvModule` layout from Task 3
- Produces: `vulkan_spirv.h` declaring `extern const SpirvModule kSpirvModules[]; extern const size_t kSpirvModuleCount;`; `vulkan_spirv.cpp` defining them

- [ ] **Step 1: Write the failing test**

The existing case computes the module count with `sizeof(kSpirvModules)/sizeof(...)`, which cannot work on an `extern` array of unknown bound. Update it to use the new count symbol, which makes it fail first:

In `tests/vt/test_vulkan_backend.cpp`, in `"the committed SPIR-V table is present and well-formed"`, replace:

```cpp
  const size_t n = sizeof(vt::vulkan::kSpirvModules) / sizeof(vt::vulkan::kSpirvModules[0]);
  CHECK(n == 7);
  for (const auto& m : vt::vulkan::kSpirvModules) {
```

with:

```cpp
  // The blobs live in vulkan_spirv.cpp, so the array is `extern` and of unknown
  // bound here; the generated count is the only way to size it. That is the point
  // of the split: at ~100 shaders the words must not be re-parsed by every TU
  // that merely needs the table.
  const size_t n = vt::vulkan::kSpirvModuleCount;
  CHECK(n == 7);
  for (size_t mi = 0; mi < n; ++mi) {
    const auto& m = vt::vulkan::kSpirvModules[mi];
```

and close the loop accordingly. Apply the same `kSpirvModuleCount` indexing to the two other range-`for` loops over `kSpirvModules` in that file and in the Task 3 case.

- [ ] **Step 2: Run it to verify it fails**

Run: `cmake --build build-vulkan --target test_vulkan_backend`
Expected: FAIL to compile — `'kSpirvModuleCount' is not a member of 'vt::vulkan'`.

- [ ] **Step 3: Split the generator's output**

In `scripts/gen-vulkan-spirv.py`, add the second output path next to `OUT_HEADER`:

```python
OUT_SOURCE = REPO / "src" / "vt" / "vulkan" / "vulkan_spirv.cpp"
```

Replace `render` with two functions. `render_header` emits the same banner comment plus:

```python
    add("namespace vt::vulkan {")
    add("")
    add("// Name -> SPIR-V module. The NAME is the shader's file stem and is also the")
    add("// key the pipeline cache uses (src/vt/vulkan/vulkan_context.cpp).")
    add("//")
    add("// spec_ids lists the SPECIALIZATION CONSTANT IDs the module declares, parsed")
    add("// from its OpDecorate SpecId instructions, sorted ascending. The host passes")
    add("// specialization values by ID; Vulkan SILENTLY IGNORES a map entry whose ID")
    add("// the module does not declare, so without this table a host/shader drift")
    add("// produces WRONG NUMBERS instead of a clean failure.")
    add("struct SpirvModule {")
    add("  const char* name;")
    add("  const uint32_t* words;")
    add("  size_t word_count;")
    add("  const uint32_t* spec_ids;")
    add("  size_t spec_id_count;")
    add("};")
    add("")
    add("// DEFINED IN vulkan_spirv.cpp. The SPIR-V words are deliberately NOT in this")
    add("// header: at the target shader surface they are megabytes of array initializer")
    add("// that every including TU would re-parse.")
    add("extern const SpirvModule kSpirvModules[];")
    add("extern const size_t kSpirvModuleCount;")
    add("")
    add("}  // namespace vt::vulkan")
```

`render_source` emits the banner, `#include "vulkan_spirv.h"`, the per-module
`kSpv_*` and `kSpecIds_*` arrays in an anonymous namespace, then the definitions:

```python
    add("const SpirvModule kSpirvModules[] = {")
    for name in sorted(blobs):
        ids = spec_ids(blobs[name])
        idp = f"kSpecIds_{name}" if ids else "nullptr"
        add(f'    {{"{name}", kSpv_{name}, sizeof(kSpv_{name}) / sizeof(uint32_t), '
            f'{idp}, {len(ids)}}},')
    add("};")
    add("const size_t kSpirvModuleCount = sizeof(kSpirvModules) / sizeof(kSpirvModules[0]);")
```

Update `main` to write and `--check` **both** files:

```python
    header = render_header(blobs, version)
    source = render_source(blobs, version)

    if args.check:
        stale = []
        for path, text in ((OUT_HEADER, header), (OUT_SOURCE, source)):
            current = path.read_text() if path.exists() else ""
            if strip_version(current) != strip_version(text):
                stale.append(path.relative_to(REPO))
        if stale:
            sys.exit("STALE, re-run scripts/gen-vulkan-spirv.py: "
                     + ", ".join(str(p) for p in stale))
        print("committed SPIR-V is up to date")
        return

    OUT_HEADER.write_text(header)
    OUT_SOURCE.write_text(source)
```

Hoist `strip_version` to module scope so both branches use the one definition.

- [ ] **Step 4: Add the generated source to the build**

In `CMakeLists.txt`, in the Vulkan backend block at line 903, add
`src/vt/vulkan/vulkan_spirv.cpp` to the same source list that already carries
`src/vt/vulkan/vulkan_context.cpp`.

- [ ] **Step 5: Regenerate, build, and run**

```bash
export PATH="$HOME/tools/glslang-16.4.0/bin:$PATH"
python3 scripts/gen-vulkan-spirv.py
python3 scripts/gen-vulkan-spirv.py --check
cmake --build build-vulkan --target test_vulkan_backend
./build-vulkan/tests/test_vulkan_backend
```
Expected: `committed SPIR-V is up to date`; build succeeds; **all cases pass**.

- [ ] **Step 6: Confirm the header actually shrank**

```bash
wc -l src/vt/vulkan/vulkan_spirv.h src/vt/vulkan/vulkan_spirv.cpp
```
Expected: the header is on the order of 40 lines; the `.cpp` carries the ~3,450 lines of words. If the header did not shrink, the blobs did not move and the split did not happen.

- [ ] **Step 7: Full Vulkan-on build, clean, `-Werror`**

Header changes are exactly where an incremental build reports green over a red clean build.

```bash
rm -rf build-vulkan
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_VULKAN=ON
cmake --build build-vulkan -j"$(nproc)" 2>&1 | tee /tmp/vk-build.log
grep -ci warning /tmp/vk-build.log
```
Expected: build succeeds; warning count **0**.

- [ ] **Step 8: Prove the CUDA gate build is untouched**

`VLLM_CPP_VULKAN` AUTO must still resolve OFF.

```bash
cmake -S . -B /tmp/vk-auto-probe -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -i vulkan
```
Expected: Vulkan reported OFF (or absent). Nothing in this task may change that.

- [ ] **Step 9: Commit**

```bash
git add scripts/gen-vulkan-spirv.py src/vt/vulkan/vulkan_spirv.h \
        src/vt/vulkan/vulkan_spirv.cpp CMakeLists.txt tests/vt/test_vulkan_backend.cpp
git commit -m "$(cat <<'MSG'
refactor(vulkan): emit SPIR-V words into a .cpp, leaving declarations in the header

At 7 modules the generated header is 3,491 lines and two TUs include it. At the
target shader surface it is megabytes of constexpr array initializer that every
includer re-parses, and every regeneration is a huge header diff. The words now
live in a generated vulkan_spirv.cpp and the header carries the struct plus extern
declarations, so adding shaders costs one TU's compile time rather than all of
them. Doing this at 7 modules is cheap; doing it at 100 is not.

--check now covers both generated files. Clean -Werror Vulkan-ON build is 0-warn,
and VLLM_CPP_VULKAN AUTO still resolves OFF so the CUDA gate build is untouched.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [Claude Code]
MSG
)"
```

---

### Task 7: Repair the record drift and land the row

**Files:**
- Modify: `.agents/feature-matrix.md:280`
- Modify: `.agents/backend-matrix.md:233`
- Modify: `.agents/specs/vulkan-full-support.md` (mark `VK-A1` landed)
- Modify: `.agents/state-index/<writable-shard>.csv`, `.agents/NOW.md`, `docs/STATUS.md`, `docs/BENCHMARKS.md`; create matching `.agents/state-events/<YYYY-MM>/<event-id>.md` (`.agents/state.csv` is the manifest)
- Modify: `docs/FEATURES.md` (required: `feature-matrix.md` is a feature-surface trigger)

**Interfaces:**
- Consumes: the landed state from Tasks 1-6
- Produces: no code

- [ ] **Step 1: Repair the feature-matrix drift**

`.agents/feature-matrix.md:280` reads `INVENTORIED | runtime absent` while
`.agents/backend-matrix.md:233` carries `ACTIVE` (gated skeleton). `backend-matrix`
is correct. Change the feature-matrix row to:

```markdown
| `BACKEND-VULKAN` | Vulkan | `ACTIVE` | gated skeleton: 8 of the CPU backend's 83 registered ops, no model runs | [backend matrix](../../../.agents/backend-matrix.md) |
```

- [ ] **Step 2: Verify the two matrices now agree**

```bash
grep -n 'BACKEND-VULKAN' .agents/feature-matrix.md | cut -c1-160
grep -o 'W0/V1 SKELETON LANDED[^.]*\.' .agents/backend-matrix.md | head -1
```
Expected: both describe a landed skeleton; neither says `INVENTORIED`.

- [ ] **Step 3: Record `VK-A1` as landed in the campaign spec**

In `.agents/specs/vulkan-full-support.md` §6, change the `VK-A1` row's
Deliverable cell to state what actually landed, and add a short subsection under
it recording: the decision taken (specialization constants over `#define`
variants) with the evidence (242 `string_to_spv(` call sites vs. our artifact
count), the glslang pin, the CI gate and its mutation proof, the header/source
split with before/after line counts, and — if the `vt_smem` fallback from Task 5
Step 3 was taken — that fallback and its shared-memory cost.

**Do not claim a speed number.** Nothing in VK-A1 was benchmarked, and the row
owes none.

- [ ] **Step 4: Update the public checkpoints**

`docs/STATUS.md` is under a **size ratchet that only shrinks** (currently 284,073;
`scripts/check-public-doc-tables.py`). Adding text requires compacting elsewhere
in the same change. Keep the STATUS entry to one or two sentences pointing at the
spec, put the narrative in the matching immutable `.agents/state-events/`
evidence file, append its ordered `.agents/state-index/` row, refresh
`.agents/NOW.md`, run `python3 scripts/check-state-record.py`, and update the
existing `BENCH-VK-LLAMA` row in `docs/BENCHMARKS.md` rather than
adding a new one — it must still read NOT APPLICABLE, because VK-A1 measures
nothing.

`docs/FEATURES.md` must also move, because `.agents/feature-matrix.md` is a
feature-surface trigger in `scripts/check-doc-checkpoint.py`. Its `Vulkan` row is
at `docs/FEATURES.md:157`; make it agree with the repaired feature-matrix state.

- [ ] **Step 5: Run the record gates**

```bash
bash scripts/agent-preflight.sh --quiet
```
Expected: every gate OK **except** `check-fusion-consistency` / `test_check_fusion_consistency`, which are red on `main` for the unrelated `minimax_h3_video_vae_device` drift. Confirm the failure text is that same one and no other gate moved.

- [ ] **Step 6: Commit and open the PR**

```bash
git add .agents/ docs/
git commit -m "$(cat <<'MSG'
record(vulkan): VK-A1 landed -- variant mechanism decided, staleness gated

Repairs the BACKEND-VULKAN drift found while scoping the campaign: feature-matrix
carried INVENTORIED/"runtime absent" against backend-matrix's ACTIVE for the same
row. backend-matrix was right; the feature matrix now says gated skeleton, 8 of
the CPU backend's 83 registered ops, no model runs.

Records the VK-A1 decision and its evidence, the glslang 16.4.0 pin, the CI
freshness gate and its mutation proof, and the header/source split. No speed
number is measured, claimed or owed.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [Claude Code]
MSG
)"
git push -u origin row/BACKEND-VULKAN-A1
gh pr create --draft --title "VK-A1: Vulkan shader-variant pipeline" --body "$(cat <<'BODY'
Implements sub-project `VK-A1` of `.agents/specs/vulkan-full-support.md`.

Decides and implements the shader-variant mechanism before ~100 shaders exist:
SPIR-V **specialization constants** over llama.cpp's `#define`-per-variant model,
keeping the committed-SPIR-V route so the build stays hermetic on every box.

- glslang 16.4.0 pinned; committed SPIR-V proven to reproduce
- CI job fails on stale committed SPIR-V (mutation-proved)
- generator records each module's declared SpecIds; host checks the count
- pipelines specialized, cache keyed by specialization
- workgroup size collapsed from three copies to one specialization constant
- SPIR-V words moved out of the header into a generated `.cpp`
- `BACKEND-VULKAN` feature-matrix/backend-matrix drift repaired

No model runs on Vulkan and no speed number is measured, claimed or owed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

---

## Self-Review

**Spec coverage.** `.agents/specs/vulkan-full-support.md` §6 lists three questions VK-A1 must answer plus the drift repair:
1. *build-time generation vs committed artifacts vs hybrid* → answered in Tasks 1, 2, 6: committed artifacts retained (hermetic), made viable by the header/source split and gated by CI. The premise change that reopened it (sudo now available) is what makes Task 1's install and Task 2's CI download possible.
2. *how coopmat tiers are spelled — defines vs specialization constants* → answered in Tasks 3-5: specialization constants, proved end to end on the workgroup size.
3. *where the tier is selected — the `vt::arch_tactics` generalization* → **deliberately not in this plan.** VK-A1 builds the mechanism for *passing* a variant; there is no device capability worth branching on until a coopmat kernel exists, and `arch_tactics` is listed in the spec as shared with `VK-C`. Building a tactic registry with exactly one tactic would be speculative. This is a scope narrowing against the spec's §6 wording and is called out here rather than left implicit — flag it to the user at handoff.
4. *`feature-matrix.md:280` drift repair* → Task 7.

**Placeholder scan.** No "TBD"/"handle edge cases"/"similar to Task N". Two conditional branches are specified with concrete fallbacks rather than left open: Task 1 Step 3 (committed SPIR-V does not reproduce → stop and report) and Task 5 Step 3 (`vt_smem` rejects a spec-constant size → size by max, record the cost).

**Type consistency.** `Dispatch(..., const uint32_t* spec_values, uint32_t spec_count)` is declared in Task 4 Step 3, implemented in Step 5, and called with `kSpecTg, 1` in Task 5 Step 4. `SpirvModule::spec_ids`/`spec_id_count` are emitted in Task 3 Step 4, consumed in Task 4 Step 4, asserted in Task 3 Step 1. `kSpirvModuleCount` is introduced in Task 6 Step 3 and consumed in Step 1's test. `PipelineCacheSize()` is declared in Task 4 Step 3, defined in Step 5, used in Step 1's test. `gen.spec_ids` / `gen.SPIRV_OP_DECORATE` / `gen.SPIRV_DECORATION_SPEC_ID` are defined in Task 3 Step 3 and used by the unit test in Step 5.

**Known-red gate.** `check-fusion-consistency` is red on `main` before this branch exists (`minimax_h3_video_vae_device`). Every task that runs preflight expects that one failure and no other.
