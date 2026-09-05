A vLLM@5559679229/gfx1151: 6 prompts, lengths [48, 48, 48, 48, 48, 48]
B llama.cpp b10451/HIP: 6 prompts, lengths [48, 48, 48, 48, 48, 48]
C vllm.cpp ROCm arm: 6 prompts, lengths [48, 48, 48, 48, 48, 48]

### A vLLM  vs  B llama.cpp
| prompt | verdict | first diff | A | B |
|---|---|---:|---:|---:|
| 0 `The capital city of France is` | **TOKEN-EXACT 48/48** | — | — | — |
| 1 `The three primary colors are` | DIVERGE | 35 | 4350 | 5844 |
| 2 `Water boils at a temperature of` | DIVERGE | 4 | 11995 | 29922 |
| 3 `The Pythagorean theorem states that` | **TOKEN-EXACT 48/48** | — | — | — |
| 4 `In 1969, humans first walked on` | DIVERGE | 13 | 6165 | 19820 |
| 5 `A prime number is a natural number` | **TOKEN-EXACT 48/48** | — | — | — |

DIVERGENCES(A vLLM vs B llama.cpp) = 3/6

### C vllm.cpp  vs  B llama.cpp
| prompt | verdict | first diff | A | B |
|---|---|---:|---:|---:|
| 0 `The capital city of France is` | **TOKEN-EXACT 48/48** | — | — | — |
| 1 `The three primary colors are` | DIVERGE | 45 | 303 | 1521 |
| 2 `Water boils at a temperature of` | **TOKEN-EXACT 48/48** | — | — | — |
| 3 `The Pythagorean theorem states that` | DIVERGE | 45 | 25 | 393 |
| 4 `In 1969, humans first walked on` | **TOKEN-EXACT 48/48** | — | — | — |
| 5 `A prime number is a natural number` | DIVERGE | 32 | 16 | 15 |

DIVERGENCES(C vllm.cpp vs B llama.cpp) = 3/6

### C vllm.cpp  vs  A vLLM
| prompt | verdict | first diff | A | B |
|---|---|---:|---:|---:|
| 0 `The capital city of France is` | **TOKEN-EXACT 48/48** | — | — | — |
| 1 `The three primary colors are` | DIVERGE | 35 | 5844 | 4350 |
| 2 `Water boils at a temperature of` | DIVERGE | 4 | 29922 | 11995 |
| 3 `The Pythagorean theorem states that` | DIVERGE | 45 | 25 | 393 |
| 4 `In 1969, humans first walked on` | DIVERGE | 13 | 19820 | 6165 |
| 5 `A prime number is a natural number` | DIVERGE | 32 | 16 | 15 |

DIVERGENCES(C vllm.cpp vs A vLLM) = 5/6

VLLM_vs_LLAMACPP_DIVERGENCES=3/6
VLLMCPP_vs_LLAMACPP_DIVERGENCES=3/6
VLLMCPP_vs_VLLM_DIVERGENCES=5/6
