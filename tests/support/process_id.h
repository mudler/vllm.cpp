// vllm.cpp original (test harness); no upstream mirror.
//
// ONE portable process id for the whole test tree.
//
// WHY IT EXISTS. Tests name their temporary directories after the running
// process so two concurrent runs cannot collide. `::getpid()` is POSIX, and its
// declaration lives in `<unistd.h>`, which MSVC does not ship at all. So the
// obvious spelling does not fail on Windows — it does not COMPILE, and it takes
// both `windows-msvc-*` lanes down with it. Exactly the shape of issue #603,
// which is why `tests/support/test_env.h` next door exists.
//
// It also fails on a CURRENT POSIX toolchain for the opposite reason. Several
// files called `::getpid()` while including nothing that declares it, and
// compiled anyway because an older libstdc++ pulled `<unistd.h>` in for them.
// gcc 16 does not:
//
//   error: '::getpid' has not been declared; did you mean 'getpt'?
//
// That was fixed once in three files and came back in five more, because each
// new loader test copies the temp-directory helper from the last one. A per-file
// `#ifdef` would be copied just as faithfully and get it wrong again, so the
// portable spelling lands ONCE, here, and new tests include it.
#pragma once

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace vllm_test {

// The current process id, for building a name no other process will pick.
//
// NOT for anything but naming. The two platforms agree that this is a unique
// live-process identifier and agree on nothing else about it, so it is used for
// uniqueness only, never compared against a recorded value or reused after the
// process exits.
inline int ProcessId() {
#if defined(_WIN32)
  return ::_getpid();
#else
  return static_cast<int>(::getpid());
#endif
}

}  // namespace vllm_test
