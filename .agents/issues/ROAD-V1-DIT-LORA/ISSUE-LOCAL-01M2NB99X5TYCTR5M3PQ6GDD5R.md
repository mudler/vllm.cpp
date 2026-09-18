ID: ISSUE-LOCAL-01M2NB99X5TYCTR5M3PQ6GDD5R
Title: VAE config parser rejects HuggingFace source config format
Row: ROAD-V1-DIT-LORA
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-16
Updated: 2026-09-16
Closed: 2026-09-16

## Problem

ParseMiniMaxH3VideoVaeDecoderConfig only reads flat decoder_* keys. The real HuggingFace config (FL2VA/video_vae/source/config.json) uses nested vit_decoder_kwargs.* keys and alternative names (z_channels, out_ch, time_down). With defaults heads=0 dim_head=0, all weight shapes are zero and Upload fails with 'host buffer does not match the requested shape'.

## Resolution

Parser now resolves three layers: flat decoder_* keys first, then nested vit_decoder_kwargs.* keys, then HuggingFace alternative names (z_channels, out_ch, time_down, vae_clip_length, vae_token_drop). A merged-config test fixture (kH3VideoVaeMergedConfigJson) with the real HuggingFace nested format was added to minimax_h3_vae_configs.inc and gated by a new test case that checks all 16 parsed values. Both the new and existing flat-format tests pass (16/16 and 23/23 assertions).
