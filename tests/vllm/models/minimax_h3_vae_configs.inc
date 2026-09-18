// GENERATED from the REAL MiniMaxAI/MiniMax-H3 config.json files — DO NOT EDIT BY HAND.
//
// The shipped VAE configs, verbatim. Small enough to embed, and embedding them is
// the point: the parsers are gated against what the checkpoint ACTUALLY says, not
// against a hand-copied summary that could drift.
#pragma once

namespace vllm_test {

inline constexpr const char* kH3AudioVaeConfigJson = R"H3JSON({
  "_class_name": "AutoencoderKLMiniMaxH3Audio",
  "_diffusers_version": "0.36.0.dev0",
  "encoder_dim": 64,
  "encoder_rates": [
    2,
    4,
    4,
    5,
    5
  ],
  "latent_dim": 2048,
  "latent_channels": 32,
  "decoder_dim": 1024,
  "decoder_rates": [
    5,
    5,
    2,
    2,
    2,
    2,
    2
  ],
  "decoder_kernel_sizes": [
    9,
    9,
    4,
    4,
    4,
    4,
    4
  ],
  "num_attention_heads": 8,
  "resblock_kernel_sizes": [
    3,
    7,
    11
  ],
  "resblock_dilation_sizes": [
    [
      1,
      3,
      5
    ],
    [
      1,
      3,
      5
    ],
    [
      1,
      3,
      5
    ]
  ],
  "sampling_rate": 32000,
  "latents_mean": [
    -0.020211687488382354,
    0.3876466479950502,
    -0.04398279799186767,
    -0.28591514936373,
    0.08179686214561671,
    -0.35782641352446604,
    0.040623809960919084,
    -0.01552534501956604,
    -0.223362481667332,
    0.1821006842509091,
    0.2941778783780663,
    -0.07901167601970885,
    -0.056815072777201,
    -0.3699028221860095,
    -0.31616315591624855,
    0.5905951377425391,
    -0.052139568068853864,
    0.013673160263486295,
    -0.03691647864630577,
    0.09732660653298163,
    -0.3394662328788498,
    -0.30685677538541667,
    -0.24504598907458763,
    -0.034698524462007344,
    0.02868032184767538,
    -0.21217779266454084,
    -0.1678263169941987,
    0.3221287889040614,
    -0.1223055851554907,
    0.4356604928128464,
    -0.0502599202236253,
    0.3979258376211797
  ],
  "latents_std": [
    1.6895524230479284,
    2.76263727217653,
    1.7945344281264435,
    1.6801681847309828,
    1.6390226546605453,
    2.7788298348882177,
    1.7659090095747236,
    1.6199757612137327,
    2.6336525640336896,
    1.8539356672817833,
    2.5056497896915633,
    1.811019237886178,
    1.9579657790720237,
    1.6685498243529284,
    1.4922469314453364,
    3.298670198067373,
    1.9491804496832168,
    1.8720003270431442,
    1.8334080103291832,
    1.6488070416529093,
    1.6176957696319716,
    1.9131449234774398,
    1.5695245398428617,
    1.6943659940415912,
    1.8318420762504692,
    1.5540637421583379,
    1.9344930328968526,
    1.599198216109855,
    1.718045989838149,
    1.6307219190837705,
    1.8661226051202384,
    1.5613768203168363
  ]
})H3JSON";

inline constexpr const char* kH3VideoVaeConfigJson = R"H3JSON({
  "_class_name": "AutoencoderKLMiniMaxH3",
  "_diffusers_version": "0.36.0.dev0",
  "in_channels": 3,
  "out_channels": 3,
  "latent_channels": 24,
  "block_out_channels": [
    128,
    256,
    256,
    512,
    512,
    1024
  ],
  "layers_per_block": 2,
  "spatial_downsample_factors": [
    2,
    2,
    2,
    2,
    1,
    1
  ],
  "temporal_downsample_factors": [
    1,
    2,
    2,
    1,
    1,
    1
  ],
  "norm_num_groups": 32,
  "norm_eps": 1e-06,
  "spatial_padding_mode": "reflect",
  "decoder_num_layers": 36,
  "decoder_num_attention_heads": 32,
  "decoder_attention_head_dim": 64,
  "decoder_num_register_tokens": 4,
  "decoder_ffn_mult": 4,
  "decoder_rope_theta": 100.0,
  "decoder_rope_dim_ratio": 0.75,
  "decoder_norm_eps": 1e-05,
  "clip_length": 17,
  "token_drop": 3,
  "latents_mean": [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264
  ],
  "latents_std": [
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523
  ]
})H3JSON";

// The HuggingFace video_vae ships in two files: a wrapper config
// (FL2VA/video_vae/config.json) with vae_clip_length, vae_token_drop,
// latent_channels, and latents_mean/std, and a source config
// (FL2VA/video_vae/source/config.json) with the nested vit_decoder_kwargs
// holding the ViT decoder architecture. A real deployment merges them.
// This fixture is that merged config — the parser must resolve values from
// BOTH the flat wrapper keys and the nested vit_decoder_kwargs.
inline constexpr const char* kH3VideoVaeMergedConfigJson = R"H3JSON({
  "_class_name": "MiniMaxH3VideoVAE",
  "source_class_name": "AutoencoderKLLegacy",
  "vae_clip_length": 17,
  "vae_token_drop": 3,
  "latent_channels": 24,
  "in_channels": 3,
  "out_ch": 3,
  "z_channels": 24,
  "time_down": [1, 2, 2, 1, 1, 1],
  "vae_ratio": 16,
  "vae_ratio_t": 4,
  "vit_decoder_kwargs": {
    "dim_head": 64,
    "heads": 32,
    "num_layers": 36,
    "rope_dim_ratio": 0.75,
    "rope_theta": 100.0
  },
  "latents_mean": [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328906198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264
  ],
  "latents_std": [
    1.2223774194717407,
    1.2767263650894165,
    1.68317747116088865,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.96531379222869875,
    1.05698859691619875,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574483,
    0.44988900423049925,
    0.7197399735450745,
    0.69362932443618775,
    2.961095094680786,
    2.7694199085235595,
    3.0496184825897215,
    2.1088054180145265,
    3.276226282119751,
    3.1627357006073,
    2.28168129920959475,
    2.6127843856811525
  ]
})H3JSON";

}  // namespace vllm_test
