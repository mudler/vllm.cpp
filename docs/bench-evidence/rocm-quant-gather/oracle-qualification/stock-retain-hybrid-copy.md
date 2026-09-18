# Stock retain-hybrid-copy overlay

Row: `BACKEND-ROCM-QUANT-GATHER`. Issue: [#3093](https://github.com/mudler/vllm.cpp/issues/3093).

This document records the minimal diff of the stock llama.cpp overlay.
The overlay changes one file, `src/llama-graph.cpp`.
It retains `s_copy` as one graph leaf in `build_inp_mem_hybrid`.

```diff
--- a/src/llama-graph.cpp
+++ b/src/llama-graph.cpp
@@ -3484,0 +3485 @@
+    ggml_build_forward_expand(gf, inp_rs->s_copy);
```

The sha256 of the original `stock-retain-hybrid-copy.patch` file was
`5fc4996ffd57bc105c42be6689b55e4786d72fb8481bdd059b740429f407b8e0`.

The appliable, manifest-sealed form of this overlay is the archive member
`stock-overlay-evidence/retain-hybrid-copy.patch` inside
[`qualification-captures.tar.gz`](../qualification-captures.tar.gz).
That member carries the change with full context, and the fenced diff above is the
minimal rendering of the same change.
