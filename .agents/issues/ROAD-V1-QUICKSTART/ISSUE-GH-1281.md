ID: ISSUE-GH-1281
Title: No page tells a new reader how to run a model. `README.md:160-176` carries a `## Quickstart` whose serve line is `build/examples/server --model /path/to/Qwen3.6-27B`, which asks the reader to build the tree and to already hold a 27B checkpoint, and which names none of the three published lane images at `ghcr.io/mudler/vllm.cpp`. `docs/USAGE.md` is the full reference at more than 4800 lines and is the wrong first document. Adds `docs/QUICKSTART.md`: one `docker run` line needing no build, no GPU and no checkout, the `curl` that answers it, the release-archive equivalent, a model table, and the cache, `HF_TOKEN` and `HF_HUB_OFFLINE` notes. `docs/USAGE.md` stays the reference and wins any disagreement. **The bar is that every row was executed**, image pulled and model fetched and tokens returned, recording the date and host; no row is reasoned about and none is marked as expected to work, which is why the first cut is three rows and not a catalog. The repository identifiers are verified against the live Hub during implementation rather than asserted from memory, because `docs/USAGE.md:1122` already records that its own checkpoint tables name no repository, revision or sha256 for a set of models, so a quickstart row either reuses a pin that exists or creates one. Depends on [#1280](https://github.com/mudler/vllm.cpp/issues/1280), since every line uses the `--model org/repo` form that row adds. The executed runs claim `dgx:gpu0` through `rc run` or `rc hold` and never by `ssh`, which would make the fleet report the device free while the run holds it. A checker asserts that every `docker run` line names a tag the container matrix publishes and that every `--model` line parses under the new grammar; it does not verify the model still exists on the Hub. Related: [#342](https://github.com/mudler/vllm.cpp/issues/342) records public documents drifting from the shipped command-line interface, which this new document inherits and answers with the executed-row bar
Row: ROAD-V1-QUICKSTART
State: UNKNOWN
Kind: feature
GitHub: 1281
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:418`

### Frozen archive evidence

> | [#1281](https://github.com/mudler/vllm.cpp/issues/1281) | `ROAD-V1-QUICKSTART` | No page tells a new reader how to run a model. `README.md:160-176` carries a `## Quickstart` whose serve line is `build/examples/server --model /path/to/Qwen3.6-27B`, which asks the reader to build the tree and to already hold a 27B checkpoint, and which names none of the three published lane images at `ghcr.io/mudler/vllm.cpp`. `docs/USAGE.md` is the full reference at more than 4800 lines and is the wrong first document. Adds `docs/QUICKSTART.md`: one `docker run` line needing no build, no GPU and no checkout, the `curl` that answers it, the release-archive equivalent, a model table, and the cache, `HF_TOKEN` and `HF_HUB_OFFLINE` notes. `docs/USAGE.md` stays the reference and wins any disagreement. **The bar is that every row was executed**, image pulled and model fetched and tokens returned, recording the date and host; no row is reasoned about and none is marked as expected to work, which is why the first cut is three rows and not a catalog. The repository identifiers are verified against the live Hub during implementation rather than asserted from memory, because `docs/USAGE.md:1122` already records that its own checkpoint tables name no repository, revision or sha256 for a set of models, so a quickstart row either reuses a pin that exists or creates one. Depends on [#1280](https://github.com/mudler/vllm.cpp/issues/1280), since every line uses the `--model org/repo` form that row adds. The executed runs claim `dgx:gpu0` through `rc run` or `rc hold` and never by `ssh`, which would make the fleet report the device free while the run holds it. A checker asserts that every `docker run` line names a tag the container matrix publishes and that every `--model` line parses under the new grammar; it does not verify the model still exists on the Hub. Related: [#342](https://github.com/mudler/vllm.cpp/issues/342) records public documents drifting from the shipped command-line interface, which this new document inherits and answers with the executed-row bar | feature |

## Resolution

-
