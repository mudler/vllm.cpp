// DeepSeek-V4 Flash Vision W6 parity -- the ORACLE side, compiled INSIDE a
// checkout of llama.cpp release b10766 (9400c8946e4da5e7694f2c26d6d4e50e14b690fa).
//
// Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W6, issue #2411. This
// file is not part of vllm.cpp's build. The W6 job script copies it into
// `tools/mtmd/` of the pinned clone and adds one executable linked to the
// oracle's own static `mtmd` library.
//
// It runs the oracle's OWN functions and nothing re-implemented:
//   clip_init                                     the mmproj loader
//   mtmd_image_preprocessor_deepseek4v::preprocess mtmd-image.cpp:1159
//   clip_image_encode                             clip.cpp, the deepseek4v graph
// It exists because `llama-mtmd-cli` needs the 82 GB language model before it
// will encode an image, and the oracle's `mtmd-debug` feeds pre-normalised
// synthetic pixels with `lead_pad` fixed at 0. Setting `lead_pad` here is the
// same write `mtmd_tokenizer` makes at mtmd.cpp:1461-1470.
//
// Stage captures come from the scheduler eval callback `mtmd-debug` also uses,
// on the two names `clip_graph_deepseek4v::build` gives: `vit_out` and
// `aligner_out`. The block itself is the encoder's returned vector, which is
// the same buffer `MTMD_DEBUG_EMBEDDINGS` dumps.
//
// usage: dsv4v-oracle-dump <mmproj.gguf> <image.rgb> <width> <height>
//                          <lead_pad> <outdir> <tag> <threads>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "mtmd-image.h"

static std::vector<float> g_vit, g_cells;
static int64_t g_vit_rc[2] = {0, 0}, g_cells_rc[2] = {0, 0};

static bool eval_cb(ggml_tensor* t, bool ask, void*) {
  const char* n = ggml_get_name(t);
  // EXACT names: the graph also names the reshaped / permuted views
  // "vit_out (reshaped) ...", and a prefix match kept the last of them.
  const bool vit = std::strcmp(n, "vit_out") == 0;
  const bool al = std::strcmp(n, "aligner_out") == 0;
  if (ask) return vit || al;
  if (!(vit || al)) return true;
  if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
    std::fprintf(stderr, "stage %s: type %d contiguous %d, not captured\n", n,
                 (int)t->type, (int)ggml_is_contiguous(t));
    return true;
  }
  std::vector<float>& v = vit ? g_vit : g_cells;
  int64_t* rc = vit ? g_vit_rc : g_cells_rc;
  v.resize(ggml_nelements(t));
  ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
  rc[1] = t->ne[0];
  rc[0] = ggml_nelements(t) / t->ne[0];
  std::fprintf(stderr, "captured %s ne=[%lld,%lld,%lld]\n", n,
               (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2]);
  return true;
}

static void write_f32(const std::string& path, int64_t rows, int64_t cols,
                      const float* data) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "FATAL: cannot write %s\n", path.c_str());
    std::exit(3);
  }
  const int32_t hdr[2] = {(int32_t)rows, (int32_t)cols};
  std::fwrite(hdr, sizeof(hdr), 1, f);
  std::fwrite(data, sizeof(float), (size_t)(rows * cols), f);
  std::fclose(f);
}

int main(int argc, char** argv) {
  if (argc != 9) {
    std::fprintf(stderr,
                 "usage: %s <mmproj> <rgb> <width> <height> <lead_pad> <outdir> "
                 "<tag> <threads>\n",
                 argv[0]);
    return 2;
  }
  const char* mmproj = argv[1];
  const int width = std::atoi(argv[3]);
  const int height = std::atoi(argv[4]);
  const int lead_pad = std::atoi(argv[5]);
  const std::string outdir = argv[6];
  const std::string tag = argv[7];
  const int threads = std::atoi(argv[8]);

  std::ifstream in(argv[2], std::ios::binary);
  std::vector<uint8_t> rgb((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
  if ((int64_t)rgb.size() != (int64_t)width * height * 3) {
    std::fprintf(stderr, "FATAL: rgb holds %zu bytes\n", rgb.size());
    return 3;
  }

  ggml_backend_load_all();
  clip_context_params p{};
  p.use_gpu = false;
  p.device = nullptr;
  p.flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
  p.image_min_tokens = -1;
  p.image_max_tokens = -1;
  p.warmup = false;
  p.cb_eval = eval_cb;
  p.cb_eval_user_data = nullptr;
  p.no_alloc = false;
  p.progress_callback = nullptr;
  p.progress_callback_user_data = nullptr;
  clip_init_result r = clip_init(mmproj, p);
  if (!r.ctx_v) {
    std::fprintf(stderr, "FATAL: clip_init loaded no vision context\n");
    return 3;
  }
  clip_ctx* ctx = r.ctx_v;
  const clip_hparams* hp = clip_get_hparams(ctx);
  std::printf(
      "oracle hparams: patch=%d n_merge=%d eps=%g rope_theta=%g "
      "min_pixels=%d max_n_token=%d mean=[%g,%g,%g] std=[%g,%g,%g]\n",
      hp->patch_size, hp->n_merge, hp->eps, (double)hp->rope_theta,
      hp->image_min_pixels, hp->dsv4_max_n_token, hp->image_mean[0],
      hp->image_mean[1], hp->image_mean[2], hp->image_std[0], hp->image_std[1],
      hp->image_std[2]);

  clip_image_u8 img;
  img.set_size({width, height}, false);
  img.cpy_buf(rgb);
  mtmd_image_preprocessor_deepseek4v pre(ctx);
  mtmd_image_preproc_out pp = pre.preprocess(img);
  if (pp.entries.size() != 1) {
    std::fprintf(stderr, "FATAL: %zu preprocessed entries\n", pp.entries.size());
    return 3;
  }
  clip_image_f32& e = pp.entries[0];
  e.lead_pad = lead_pad;
  // NOISE-FLOOR ARM. Our processor narrows the normalised pixels to bf16
  // before the tower (deepseek_v4_processor.cpp) and the oracle keeps them f32.
  // DSV4V_ROUND_INPUT_BF16=1 applies the same rounding here, so oracle(f32 in)
  // against oracle(bf16 in) measures how far that perturbation ALONE moves the
  // oracle's own output -- the floor ours-vs-oracle is judged against.
  if (const char* rb = std::getenv("DSV4V_ROUND_INPUT_BF16"); rb && rb[0] == '1') {
    std::vector<float> rounded = e.get_ro_buf();
    for (float& v : rounded) {
      uint32_t u;
      std::memcpy(&u, &v, 4);
      u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
      std::memcpy(&v, &u, 4);
    }
    e.cpy_buf(rounded);
    std::printf("oracle input rounded to bf16 (noise-floor arm)\n");
  }
  std::printf("oracle preprocess: %dx%d (from %dx%d) lead_pad=%d\n", e.nx(),
              e.ny(), width, height, e.lead_pad);

  // The normalised input, rearranged into our patch-row order
  // [(vy*nvx+vx)][(c*P+dy)*P+dx], so the two inputs compare element-wise.
  //
  // READ THE `input` STAGE AS A VALUE CHECK ONLY. This loop imposes OUR claimed
  // patch-row order on the oracle's buffer, so `ours-<tag>-input.f32` against
  // `oracle-<tag>-input.f32` measures the normalisation ARITHMETIC (the mean,
  // the standard deviation and the bf16 narrowing) and ASSERTS the ordering
  // rather than measuring it. If our patch order were wrong, this file would
  // still compare exact, because both sides would have been written in the same
  // wrong order.
  //
  // Nothing is blind as a result: ordering is measured DOWNSTREAM, by the
  // block-level permutation check in `dsv4v_w6_compare.py`, which best-matches
  // every one of our image rows against the oracle's own block and requires the
  // identity. The `input` line simply says less than its name suggests, and the
  // gate's ordering evidence is the permutation line, never this one.
  const int P = hp->patch_size;
  const int nvx = e.nx() / P, nvy = e.ny() / P, cols = 3 * P * P;
  const std::vector<float>& buf = e.get_ro_buf();
  std::vector<float> rows((size_t)nvx * nvy * cols);
  for (int vy = 0; vy < nvy; ++vy)
    for (int vx = 0; vx < nvx; ++vx)
      for (int c = 0; c < 3; ++c)
        for (int dy = 0; dy < P; ++dy)
          for (int dx = 0; dx < P; ++dx)
            rows[((size_t)(vy * nvx + vx)) * cols + (c * P + dy) * P + dx] =
                buf[((size_t)(vy * P + dy) * e.nx() + vx * P + dx) * 3 + c];
  write_f32(outdir + "/oracle-" + tag + "-input.f32", (int64_t)nvx * nvy, cols,
            rows.data());

  const int n_tok = clip_n_output_tokens(ctx, &e);
  const int n_embd = clip_n_mmproj_embd(ctx);
  std::vector<float> emb((size_t)n_tok * n_embd);
  if (!clip_image_encode(ctx, threads, &e, emb)) {
    std::fprintf(stderr, "FATAL: clip_image_encode failed\n");
    return 4;
  }
  write_f32(outdir + "/oracle-" + tag + "-block.f32", n_tok, n_embd, emb.data());
  std::printf("oracle block: %d x %d\n", n_tok, n_embd);
  if (!g_vit.empty())
    write_f32(outdir + "/oracle-" + tag + "-vit.f32", g_vit_rc[0], g_vit_rc[1],
              g_vit.data());
  if (!g_cells.empty())
    write_f32(outdir + "/oracle-" + tag + "-cells.f32", g_cells_rc[0],
              g_cells_rc[1], g_cells.data());
  clip_free(ctx);
  std::printf("ORACLE_DONE tag=%s\n", tag.c_str());
  return 0;
}
