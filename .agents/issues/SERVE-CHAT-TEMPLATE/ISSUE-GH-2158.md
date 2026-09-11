ID: ISSUE-GH-2158
Title: The GGUF chat-template selection that #2079 wires is gated as a FUNCTION and not as the server path: deleting the `LoadChatTemplateForModel` call site in `server_main.cpp` leaves `test_chat_template` at 37 cases / 147 assertions green, so the #2077 regression could return unseen. MEASURED on the #2079 head merged onto main, not argued. Landed with the gap named because the wiring is five lines at a production entry point and was verified by hand on gfx1100, while the defect makes every GGUF chat request useless. Owed: a case entering through `VllmServerMain` in the `test_serve_residency_config.cpp` re-exec shape. Its obstacle is why this is its own unit of work — the chat-template block sits after the full engine load, so the nonexistent-model-directory trick cannot reach it and the synthetic GGUF stops one step earlier at the missing tokenizer
Row: SERVE-CHAT-TEMPLATE
State: UNKNOWN
Kind: bug
GitHub: 2158
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:839`

### Frozen archive evidence

> | [#2158](https://github.com/mudler/vllm.cpp/issues/2158) | `SERVE-CHAT-TEMPLATE` | The GGUF chat-template selection that #2079 wires is gated as a FUNCTION and not as the server path: deleting the `LoadChatTemplateForModel` call site in `server_main.cpp` leaves `test_chat_template` at 37 cases / 147 assertions green, so the #2077 regression could return unseen. MEASURED on the #2079 head merged onto main, not argued. Landed with the gap named because the wiring is five lines at a production entry point and was verified by hand on gfx1100, while the defect makes every GGUF chat request useless. Owed: a case entering through `VllmServerMain` in the `test_serve_residency_config.cpp` re-exec shape. Its obstacle is why this is its own unit of work — the chat-template block sits after the full engine load, so the nonexistent-model-directory trick cannot reach it and the synthetic GGUF stops one step earlier at the missing tokenizer | bug |

## Resolution

-
