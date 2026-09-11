ID: ISSUE-GH-384
Title: --served-model-name accepts only one name, but OpenAIServingModels already implements multi-name serving
Row: SERVE-OAI-BASIC
State: OPEN
Kind: UNKNOWN
GitHub: 384
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-11
Updated: 2026-08-11
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## The gap
>
> `OpenAIServingModels` implements multi-name serving completely -- and nothing can reach it.
>
> The class is built on `std::vector<std::string> served_model_names_` and every consumer already honours the whole list:
>
> - `serving_models.h:45` -- `explicit OpenAIServingModels(std::vector<std::string>)`
> - `serving_models.h:50` -- `model_name()` returns `served_model_names_.front()`, "the canonical served id"
> - `serving_models.cpp` -- `is_base_model()` iterates every name; `show_available_models()` emits a `ModelCard` per name, so `/v1/models` would list all of them
> - `check_model()` resolves an absent/empty model to the served one and 404s anything unknown
>
> So a request naming *any* alias would be served, responses would carry the first name, and `/v1/models` would advertise all of them. All of that works today.
>
> The only thing that cannot express it is the CLI. `server_main.cpp:142` declares
>
> ```cpp
> std::string served_model_name;  // default: the model dir name
> ```
>
> parsed at `:281-282` as a single `NextArg`, resolved to a single `std::string` at `:577-581`, and handed to the **convenience single-model ctor** (`serving_models.h:47`) at all three construction sites -- `:636` embeddings, `:684` ASR, `:808` text. The vector ctor is never called from the server.
>
> By your own rule in `AGENTS.md` -- *"A capability that is not reachable through the shared surface is not done"* -- the multi-name port is not done, and the missing piece is one `std::string` that should be a `std::vector<std::string>`.
>
> An observation, offered once and not as a lecture: this is the fourth reachability gap we have run into here, after `marlin-nvfp4` missing `11.0` (#325/#326), the decode-opt kernel being head_dim-256 only (#382), and `fa2` missing `11.0`. The first three are kernel dispatch; this one is the entrypoint layer, which is why it seemed worth naming -- the shape is a capability that is fully built and gated behind something narrower than itself, not a CUDA-specific accident.
>
> ## Divergence from vLLM
>
> vLLM's `--served-model-name` takes multiple values (`nargs='+'`), and its documented contract is exactly what `OpenAIServingModels` already implements: the server responds to any of the provided names, and the `model` field of a response carries the **first** name in the list. Our port mirrored that semantic faithfully one layer down and dropped the multiplicity at the argument layer.
>
> (I am citing vLLM's documented CLI contract, not a `file:line` from your pinned oracle -- I do not have that checkout. Worth confirming against the pin before the flag shape is fixed.)
>
> ## Why it matters in practice
>
> Serving one model under two names on one port is the normal way to migrate clients across a model-version rename without running a second server or a proxy. Concretely, we serve a Mistral checkpoint that some clients address as a versioned id and others by its previous id; today that needs either a second port or a rename flag-day. It is also how an OpenAI-compatible endpoint is usually presented behind a stable alias.
>
> ## Suggested shape
>
> Minimal and confined to the entrypoint:
>
> 1. `Args::served_model_name` -> `std::vector<std::string> served_model_names`.
> 2. `--served-model-name` accepts multiple values, mirroring vLLM. Your existing flags all take exactly one `NextArg`, so **repeating the flag** (`--served-model-name a --served-model-name b`) fits your parser's shape better than `nargs='+'` and needs no lookahead. Comma-separated in a single value is the other option. **This is a convention call and I would rather you pick it than guess** -- vLLM's own form is space-separated, which your parser does not currently do anywhere.
> 3. The `:577-581` default (model dir basename) becomes a one-element vector when the flag is absent -- unchanged behaviour.
> 4. The three construction sites pass the vector; the single-string ctor stays for other callers.
> 5. Everything downstream that wants the canonical id keeps using `models.model_name()`.
>
> Default behaviour is byte-identical when the flag is passed once or not at all.
>
> ## Offer
>
> Happy to implement it once you have picked the flag convention, with a RED-first test -- `is_base_model` / `check_model` / `show_available_models` over a two-name construction, plus an arg-parsing case. Tell me which form you want and whether you would rather have it as one commit or split parse/threading.
>
> ## Scope
>
> Read from the tree at `dbd0d51c`; I have not measured or built anything for this. If any of the three call sites has a reason to stay single-name that I have missed, say so and I will scope it down.
>

## Resolution

-
