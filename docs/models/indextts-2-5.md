# IndexTTS 2.5

IndexTTS 2.5 is not available through `/v1/audio/speech` yet. The server
refuses this family and names the missing production pieces instead of loading
a partial pipeline. Issues [#634](https://github.com/mudler/vllm.cpp/issues/634)
and [#1112](https://github.com/mudler/vllm.cpp/issues/1112) track that work.

The repository contains ports and numerical gates for individual IndexTTS 2.5
stages. Those gates do not make the model servable. The shipped checkpoint also
depends on auxiliary models that it fetches separately, including
`nvidia/bigvgan_v2_22khz_80band_256x`.

Use the upstream `IndexTeam/IndexTTS-2.5` checkpoint only for contributor
verification until the production speech engine supports this family. The
checkpoint conversion commands, stage-golden commands, and manifest evidence
belong to the internal
[`IndexTTS 2.5 specification`](../../.agents/specs/indextts-2-5.md).
