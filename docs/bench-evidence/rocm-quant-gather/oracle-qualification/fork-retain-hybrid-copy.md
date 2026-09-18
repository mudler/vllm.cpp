# IQ1 fork retain-hybrid-copy overlay

Row: `BACKEND-ROCM-QUANT-GATHER`. Issue: [#3093](https://github.com/mudler/vllm.cpp/issues/3093).

This document records the minimal diff of the IQ1 fork llama.cpp overlay.
The overlay changes one file, `src/llama-graph.cpp`.
It retains `s_copy` as one graph leaf in `build_inp_mem_hybrid`.

```diff
--- a/src/llama-graph.cpp
+++ b/src/llama-graph.cpp
@@ -3466,0 +3467 @@
+    ggml_build_forward_expand(gf, inp_rs->s_copy);
```

The sha256 of the original `fork-retain-hybrid-copy.patch` file was
`2a909c7fe0ef2baafe74f637e11ba0ccc13973eb114ba90f7c75b51e6e8fd241`.

The appliable, manifest-sealed form of this overlay is the archive member
`fork-overlay-evidence/retain-hybrid-copy.patch` inside
[`qualification-captures.tar.gz`](../qualification-captures.tar.gz).
That member carries the change with full context, and the fenced diff above is the
minimal rendering of the same change.
