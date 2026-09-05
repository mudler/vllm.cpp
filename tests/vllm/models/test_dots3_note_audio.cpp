// dots3-note W7a (#2703) — THE AUDIO TOWER AND ITS FRONT END, against TWO
// INDEPENDENT double-precision references.
//
// WHAT THIS FILE ESTABLISHES, AND WHAT IT DOES NOT. `.agents/specs/
// dots3-note.md` §6.4 records option B: this row has NO oracle and will not get
// one — 576.89 GB bf16 / 298.67 GB fp8 against 119-122 GiB hosts — so
// correctness is argued by a reference written from the upstream Python that
// shares NO helper with the implementation. That is a CONSISTENCY gate: two
// implementations agree, neither is shown to match vLLM, and no performance
// number is claimable on any axis.
//
// THE ROW'S CONVENTION IS TO PROVE THE INDEPENDENCE BY ENUMERATION. W6a listed
// 70 qualified names in its reference namespace, W6b 105 and W6c 45, all
// `std::`. This file has TWO reference namespaces and reports both counts,
// because one reference covering a DFT, a mel filterbank and a 32-layer tower
// with a four-stage temporal mask is too large for a reviewer to hold, and the
// two halves fail in unrelated ways: the front end's hazards are windowing,
// framing and normalisation; the tower's are ordering, masking and bias
// placement.
//
// AND ONE PLACE THERE **IS** A REAL ORACLE, which almost nothing on this row
// gets. `tests/vllm/multimodal/fixtures/voxtral_audio/voxtral_mel_filters_f32.bin`
// is a COMMITTED [201, 128] float32 matrix that
// `scripts/mm/a3_voxtral_oracle_capture.py:141-147` dumped from
// `mistral_common.audio.mel_filter_bank(201, 128, 0.0, 8000.0, 16000)` — a
// third party's implementation of the same formula `nvidia/audio.py:98-106`
// calls. The shared `MelFilterBankSlaney` seam is asserted against it BIT for
// BIT, which settles HTK-versus-Slaney, the `norm` argument and the
// integer-divided-Nyquist detail outright.
//
// Upstream read in `~/_git/vllm` at `9035151d6`. `dots3_note` does not exist at
// the parity pin `5559679229`, so every anchor names that SHA.
#include "vllm/model_executor/models/dots3_note_audio.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include "dots3_note_tiny_fixture.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dots3_note.h"
#include "vllm/multimodal/dots3_note_processor.h"
#include "dots3_note_resample_golden.h"
#include "vllm/multimodal/audio_processor.h"
#include "vllm/multimodal/audio_resample.h"
#include "vllm/multimodal/mel_filter_bank.h"
#include "vllm/multimodal/processing/processor.h"
#include "vllm/multimodal/qwen3vl_processor.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using dots3_tiny::TinyCheckpoint;
using dots3_tiny::TinySpec;

namespace {

// The value tolerance for the W7c-2 resampler (#2828), and its DERIVATION.
//
// Measured on this build the port reproduces scipy's own float32 output BIT FOR
// BIT on four of the five golden cases and to 8.31e-18 on the fifth, so "twice
// the measured floor" would be 1.7e-17. THAT IS NOT THE BOUND, and gating on it
// would be gating on a coincidence of rounding: the port and scipy agree to
// ~1e-15 in DOUBLE, and everything after that is one narrowing store, whose
// granularity is a `float` ulp. A legitimate platform difference — another
// libm's `sin` by one ulp in the filter taps, or a contracted multiply-add in
// the convolution — moves the double answer by ~1e-16 relative, which usually
// narrows to the same float and can narrow to the adjacent one.
//
// So the bound is TWO FLOAT ULPS at the fixtures' peak of ~0.99, which is
// 1.2e-7. Every defect this gate exists to catch is orders above it: a
// one-sample phase shift moves an output by up to 0.34 on these fixtures, and a
// nearest-sample decimation by 0.053 to 0.457.
inline constexpr double kResampleTol = 1.2e-7;

std::string FixtureDir() { return DOTS3_NOTE_CKPT_FIXTURE_DIR; }
std::string VoxtralFixtureDir() { return VOXTRAL_AUDIO_FIXTURE_DIR; }

TinySpec AudioSpec() {
  TinySpec s;
  s.with_audio = true;
  return s;
}

std::vector<float> ReadF32File(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open fixture: " << path);
  f.seekg(0, std::ios::end);
  const std::streamoff bytes = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<float> out(static_cast<size_t>(bytes) / sizeof(float));
  f.read(reinterpret_cast<char*>(out.data()), bytes);
  return out;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// REFERENCE 1 — THE FRONT END.
//
// Written from `nvidia/audio.py:96-126` @ `9035151d6` and from transformers
// `audio_utils.mel_filter_bank:453` / `hertz_to_mel:285` / `mel_to_hertz:321`,
// in double precision, sharing NO helper with `src/`. Every qualified name
// below is `std::`, and the enumeration case MEASURES that from this file's
// own bytes rather than trusting the list.
// ═══════════════════════════════════════════════════════════════════════════
namespace ref_front {

// Every qualified name used in this namespace, MEASURED by the enumeration case
// below — not transcribed. All 11 are `std::`: this reference calls nothing
// from `vllm::`, nothing from `vt::` and nothing from the file under test.
//
//  1 std::size_t (36)         2 std::vector (19)    3 std::int64_t (4)
//  4 std::log (3)             5 std::cos (2)        6 std::min (2)
//  7 std::exp (1)             8 std::log10 (1)      9 std::max (1)
// 10 std::numeric_limits (1) 11 std::sin (1)
//
// 71 occurrences of 11 distinct names. THIS LIST WAS WRONG WHEN THE FILE WAS
// FIRST WRITTEN, and nothing could see it: it read 22 names asserted against a
// hand-written `kQualifiedNames = 22`, and eleven of them — `std::pow`,
// `std::string`, `std::sqrt`, `std::abs`, `std::int16_t`, `std::to_string`,
// `std::fabs`, `std::floor`, `std::ceil`, `std::round`, `std::llround` — are
// used NOWHERE in this namespace. Two (`std::llround`, `std::int16_t`) appear
// nowhere in this FILE except inside that list. A constant compared with a
// literal is a transcription, and a transcription cannot gate what it
// transcribes; adding a `vllm::` call would not have moved it either.
inline constexpr int kDistinctQualifiedNames = 11;
inline constexpr int kQualifiedNameOccurrences = 71;

constexpr double kPi = 3.14159265358979323846;

// `audio_utils.hertz_to_mel(mel_scale="slaney")` :285-296.
double HzToMel(double hz) {
  if (hz >= 1000.0) return 15.0 + std::log(hz / 1000.0) * (27.0 / std::log(6.4));
  return 3.0 * hz / 200.0;
}
// `mel_to_hertz(mel_scale="slaney")` :321-331.
double MelToHz(double mel) {
  if (mel >= 15.0) return 1000.0 * std::exp((std::log(6.4) / 27.0) * (mel - 15.0));
  return 200.0 * mel / 3.0;
}

// `mel_filter_bank(..., norm="slaney", mel_scale="slaney")` :453, returned as
// [n_freq][n_mels] the way upstream returns it.
std::vector<std::vector<double>> Bank(int n_freq, int n_mels, double f_min,
                                      double f_max, int sr) {
  // :516-519 — centres equally spaced in MEL space, then back to Hz.
  const double m0 = HzToMel(f_min), m1 = HzToMel(f_max);
  std::vector<double> fc(static_cast<std::size_t>(n_mels + 2));
  for (int i = 0; i < n_mels + 2; ++i) {
    const double mel = m0 + (m1 - m0) * static_cast<double>(i) /
                                static_cast<double>(n_mels + 1);
    fc[static_cast<std::size_t>(i)] = MelToHz(mel);
  }
  fc[static_cast<std::size_t>(n_mels + 1)] = MelToHz(m1);
  // :528 — the INTEGER-divided Nyquist.
  std::vector<double> ff(static_cast<std::size_t>(n_freq));
  const double top = static_cast<double>(sr / 2);
  for (int k = 0; k < n_freq; ++k) {
    ff[static_cast<std::size_t>(k)] =
        top * static_cast<double>(k) / static_cast<double>(n_freq - 1);
  }
  std::vector<std::vector<double>> out(
      static_cast<std::size_t>(n_freq),
      std::vector<double>(static_cast<std::size_t>(n_mels), 0.0));
  for (int m = 0; m < n_mels; ++m) {
    const double lo = fc[static_cast<std::size_t>(m)];
    const double mid = fc[static_cast<std::size_t>(m) + 1];
    const double hi = fc[static_cast<std::size_t>(m) + 2];
    const double enorm = 2.0 / (hi - lo);  // :532-535, the slaney area norm
    for (int k = 0; k < n_freq; ++k) {
      const double f = ff[static_cast<std::size_t>(k)];
      const double down = (f - lo) / (mid - lo);
      const double up = (hi - f) / (hi - mid);
      const double tri = std::max(0.0, std::min(down, up));
      out[static_cast<std::size_t>(k)][static_cast<std::size_t>(m)] = tri * enorm;
    }
  }
  return out;
}

// `pad_or_trim` (:84-93) + `log_mel_spectrogram` (:117-126), the whole front
// end, as [n_mels][n_frames].
//
// `torch.stft(center=True)` REFLECT-pads by n_fft/2 on each side; the frame
// count is `1 + (padded - n_fft) / hop` and `stft[..., :-1]` drops the last.
// The window is torch's PERIODIC Hann. The spectrogram is POWER (`.abs()**2`).
// The floor is a GLOBAL max minus 8, not a per-band one.
std::vector<std::vector<double>> LogMel(const std::vector<float>& wav,
                                        int pad_to, int n_fft, int hop,
                                        int n_mels, int sr) {
  std::vector<double> x(static_cast<std::size_t>(pad_to), 0.0);
  const std::int64_t copy =
      std::min<std::int64_t>(static_cast<std::int64_t>(wav.size()), pad_to);
  for (std::int64_t i = 0; i < copy; ++i)
    x[static_cast<std::size_t>(i)] = wav[static_cast<std::size_t>(i)];

  const int p = n_fft / 2;
  const int L = pad_to;
  std::vector<double> padded(static_cast<std::size_t>(L + 2 * p), 0.0);
  for (int i = 0; i < L; ++i)
    padded[static_cast<std::size_t>(p + i)] = x[static_cast<std::size_t>(i)];
  for (int i = 0; i < p; ++i)
    padded[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(p - i)];
  for (int i = 0; i < p; ++i) {
    const int src = L - 2 - i;
    padded[static_cast<std::size_t>(p + L + i)] =
        src >= 0 ? x[static_cast<std::size_t>(src)] : 0.0;
  }
  const int frames = 1 + (L + 2 * p - n_fft) / hop - 1;

  std::vector<double> win(static_cast<std::size_t>(n_fft));
  for (int i = 0; i < n_fft; ++i) {
    win[static_cast<std::size_t>(i)] =
        0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) /
                             static_cast<double>(n_fft));
  }
  const int n_freq = 1 + n_fft / 2;
  const std::vector<std::vector<double>> bank =
      Bank(n_freq, n_mels, 0.0, static_cast<double>(sr) / 2.0, sr);

  std::vector<std::vector<double>> out(
      static_cast<std::size_t>(n_mels),
      std::vector<double>(static_cast<std::size_t>(frames), 0.0));
  double gmax = -std::numeric_limits<double>::infinity();
  for (int f = 0; f < frames; ++f) {
    std::vector<double> mag(static_cast<std::size_t>(n_freq), 0.0);
    for (int k = 0; k < n_freq; ++k) {
      double re = 0.0, im = 0.0;
      for (int j = 0; j < n_fft; ++j) {
        const double v = padded[static_cast<std::size_t>(f * hop + j)] *
                         win[static_cast<std::size_t>(j)];
        const double ang = 2.0 * kPi * static_cast<double>(k) *
                           static_cast<double>(j) / static_cast<double>(n_fft);
        re += v * std::cos(ang);
        im -= v * std::sin(ang);
      }
      mag[static_cast<std::size_t>(k)] = re * re + im * im;
    }
    for (int m = 0; m < n_mels; ++m) {
      double acc = 0.0;
      for (int k = 0; k < n_freq; ++k) {
        acc += bank[static_cast<std::size_t>(k)][static_cast<std::size_t>(m)] *
               mag[static_cast<std::size_t>(k)];
      }
      if (acc < 1e-10) acc = 1e-10;
      const double lg = std::log10(acc);
      out[static_cast<std::size_t>(m)][static_cast<std::size_t>(f)] = lg;
      if (lg > gmax) gmax = lg;
    }
  }
  const double floor_v = gmax - 8.0;
  for (auto& row : out) {
    for (double& v : row) {
      if (v < floor_v) v = floor_v;
      v = (v + 4.0) / 4.0;
    }
  }
  return out;
}

}  // namespace ref_front

// ═══════════════════════════════════════════════════════════════════════════
// REFERENCE 2 — THE TOWER.
//
// Written from `nvidia/audio_encoder.py` and `nvidia/audio.py` @ `9035151d6`,
// in double precision, sharing NO helper with `src/` or with `ref_front`. Every
// qualified name is `std::`, measured the same way.
// ═══════════════════════════════════════════════════════════════════════════
namespace ref_tower {

// MEASURED, as above. All 6 are `std::`.
//
//  1 std::size_t (24)   2 std::vector (14)   3 std::int64_t (10)
//  4 std::sqrt (3)      5 std::erf (1)       6 std::exp (1)
//
// 53 occurrences of 6 distinct names. This list read 19 names before it was
// measured; thirteen of them are not used here. That this reference reaches
// only SIX names is itself the point — a 32-layer tower in double precision
// needs `exp`, `erf` and `sqrt` and nothing else, so anything else appearing
// in the measurement is a helper that leaked in from `src/`.
inline constexpr int kDistinctQualifiedNames = 6;
inline constexpr int kQualifiedNameOccurrences = 53;

using Mat = std::vector<std::vector<double>>;

// `RMSNorm.forward` (:36-39): `x * rsqrt(mean(x^2) + eps)` and THEN `weight *`.
// Deliberately upstream's order, not `vt::RmsNorm`'s — see
// `dots3_note_audio.h`'s note on the one formula difference.
Mat RmsNorm(const Mat& x, const std::vector<double>& w, double eps) {
  Mat out(x.size());
  for (std::size_t r = 0; r < x.size(); ++r) {
    double acc = 0.0;
    for (double v : x[r]) acc += v * v;
    const double inv = 1.0 / std::sqrt(acc / static_cast<double>(x[r].size()) + eps);
    out[r].resize(x[r].size());
    for (std::size_t c = 0; c < x[r].size(); ++c)
      out[r][c] = x[r][c] * inv * w[c];
  }
  return out;
}

// `nn.LayerNorm` — mean-subtracting, BIASED variance, weight AND bias.
Mat LayerNorm(const Mat& x, const std::vector<double>& w,
              const std::vector<double>& b, double eps) {
  Mat out(x.size());
  for (std::size_t r = 0; r < x.size(); ++r) {
    const std::size_t n = x[r].size();
    double mean = 0.0;
    for (double v : x[r]) mean += v;
    mean /= static_cast<double>(n);
    double var = 0.0;
    for (double v : x[r]) var += (v - mean) * (v - mean);
    var /= static_cast<double>(n);
    const double inv = 1.0 / std::sqrt(var + eps);
    out[r].resize(n);
    for (std::size_t c = 0; c < n; ++c)
      out[r][c] = (x[r][c] - mean) * inv * w[c] + b[c];
  }
  return out;
}

// y[r][n] = sum_k x[r][k] * w[n][k] (+ b[n]) — `F.linear`.
Mat Linear(const Mat& x, const Mat& w, const std::vector<double>* b) {
  Mat out(x.size());
  for (std::size_t r = 0; r < x.size(); ++r) {
    out[r].assign(w.size(), 0.0);
    for (std::size_t n = 0; n < w.size(); ++n) {
      double acc = b != nullptr ? (*b)[n] : 0.0;
      for (std::size_t k = 0; k < x[r].size(); ++k) acc += x[r][k] * w[n][k];
      out[r][n] = acc;
    }
  }
  return out;
}

// `nn.functional.gelu` with the default `approximate='none'`: the EXACT erf
// form, not the tanh approximation.
double GeluErf(double v) {
  return 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
}

double Silu(double v) { return v / (1.0 + std::exp(-v)); }

// `swiglu` (:42-44): `x1, x2 = x.chunk(2, -1); silu(x1) * x2`. GATE THEN UP.
Mat SwiGlu(const Mat& x) {
  Mat out(x.size());
  for (std::size_t r = 0; r < x.size(); ++r) {
    const std::size_t half = x[r].size() / 2;
    out[r].resize(half);
    for (std::size_t c = 0; c < half; ++c)
      out[r][c] = Silu(x[r][c]) * x[r][c + half];
  }
  return out;
}

// ONE stride-2 padding-1 3x3 Conv2d over `[C][F][T]`, then GELU. Written as the
// four-deep loop upstream's `nn.Conv2d` performs, NOT as an im2col — so the
// implementation's im2col composition is being CHECKED rather than repeated.
std::vector<Mat> Conv2dGelu(const std::vector<Mat>& x, const std::vector<Mat>& w,
                            const std::vector<double>& b, std::size_t out_ch) {
  const std::size_t in_ch = x.size();
  const std::size_t Fi = x[0].size(), Ti = x[0][0].size();
  const std::size_t Fo = (Fi + 2 - 3) / 2 + 1, To = (Ti + 2 - 3) / 2 + 1;
  std::vector<Mat> out(out_ch, Mat(Fo, std::vector<double>(To, 0.0)));
  for (std::size_t co = 0; co < out_ch; ++co) {
    for (std::size_t fo = 0; fo < Fo; ++fo) {
      for (std::size_t to = 0; to < To; ++to) {
        double acc = b[co];
        for (std::size_t ci = 0; ci < in_ch; ++ci) {
          for (std::size_t kf = 0; kf < 3; ++kf) {
            const std::int64_t fi =
                static_cast<std::int64_t>(2 * fo) - 1 + static_cast<std::int64_t>(kf);
            if (fi < 0 || fi >= static_cast<std::int64_t>(Fi)) continue;
            for (std::size_t kt = 0; kt < 3; ++kt) {
              const std::int64_t ti = static_cast<std::int64_t>(2 * to) - 1 +
                                      static_cast<std::int64_t>(kt);
              if (ti < 0 || ti >= static_cast<std::int64_t>(Ti)) continue;
              acc += x[ci][static_cast<std::size_t>(fi)][static_cast<std::size_t>(ti)] *
                     w[co][ci][kf * 3 + kt];
            }
          }
        }
        out[co][fo][to] = GeluErf(acc);
      }
    }
  }
  return out;
}

// `_temporal_mask` (:528-533): zero every position at or past `valid` on the
// TIME axis.
void MaskTime(std::vector<Mat>* x, std::int64_t valid) {
  for (Mat& ch : *x) {
    for (std::vector<double>& row : ch) {
      for (std::size_t t = 0; t < row.size(); ++t) {
        if (static_cast<std::int64_t>(t) >= valid) row[t] = 0.0;
      }
    }
  }
}

}  // namespace ref_tower

// ═══════════════════════════════════════════════════════════════════════════
// REFERENCE 3 — THE SEGMENTATION GEOMETRY (W7b, #2797).
//
// Written from `nvidia/audio.py:193-234` and `nvidia/audio_encoder.py:570-577`
// @ `9035151d6`, sharing NO helper with `src/`, with `ref_front` or with
// `ref_tower`. Every qualified name is `std::`, measured by the same
// enumeration case — which W7b EXTENDS to a third namespace rather than
// writing a second instrument.
//
// WHY A THIRD REFERENCE AND NOT A LONGER SECOND ONE. What chunking gets wrong
// is GEOMETRY — which samples land in which chunk, how many rows each chunk
// contributes, and where each chunk starts in the concatenation — and every one
// of those defects produces correctly-shaped output that an aggregate norm
// cannot see. This namespace computes that geometry and nothing else; the
// heavy numerics stay in `ref_front::LogMel` and `RefTower`, which W7b does not
// touch and drives per segment at the geometry derived here.
// ═══════════════════════════════════════════════════════════════════════════
namespace ref_chunks {

// MEASURED by the enumeration case below, as the other two are.
//
//  1 std::int64_t (20)   2 std::vector (5)
//
// 25 occurrences of 2 distinct names. That this reference reaches only TWO is
// the point of it: a geometry that needs `sqrt`, `exp` or anything from `vt::`
// is not a geometry any more, so anything else appearing in the measurement is
// a helper that leaked in from `src/`.
inline constexpr int kDistinctQualifiedNames = 2;
inline constexpr int kQualifiedNameOccurrences = 25;

// One `chunk_seconds` segment: where it starts, how long it is, how many rows
// it contributes and where those rows land in the concatenation.
struct Segment {
  std::int64_t start = 0;
  std::int64_t length = 0;
  std::int64_t token_len = 0;
  std::int64_t row_offset = 0;
};

// `encode_waveform`'s slicing loop (`audio.py:194-218`), stepping in SECONDS
// exactly as upstream does, with Python's clamping slice stop.
//
// `token_len` is written as upstream's LITERAL `(length - 1) // stride + 1`
// (`:210-212`) and not as the `ceil` the implementation uses, which is the
// whole point of a reference. The two are the same function for every
// `length >= 1` and differ only at `length == 0`, where C++ truncation toward
// zero would give 1 where Python's floor gives 0 — and the `while` condition
// below never emits a zero-length segment, so this transcription is safe HERE
// and would not be in the implementation, where the count is also asked for a
// waveform.
std::vector<Segment> Segments(std::int64_t num_samples, std::int64_t chunk_seconds,
                              std::int64_t sample_rate, std::int64_t hop_length,
                              std::int64_t conv_temporal_stride,
                              std::int64_t merge_factor) {
  const std::int64_t stride = hop_length * conv_temporal_stride * merge_factor;
  std::vector<Segment> out;
  std::int64_t time_step = 0;
  std::int64_t rows = 0;
  while (time_step * sample_rate < num_samples) {
    Segment s;
    s.start = time_step * sample_rate;
    const std::int64_t stop = (time_step + chunk_seconds) * sample_rate;
    s.length = (stop < num_samples ? stop : num_samples) - s.start;
    s.token_len = (s.length - 1) / stride + 1;
    s.row_offset = rows;
    rows += s.token_len;
    out.push_back(s);
    time_step += chunk_seconds;
  }
  return out;
}

// `compute_audio_token_length` (`audio.py:129-147`) — upstream's OWN statement
// of the total, which it defines and never calls. The PROMPT side instead
// counts one `math.ceil(total / stride)` (`common/processor.py:771`), and the
// two are equal for every waveform exactly when `chunk_samples % stride == 0`.
std::int64_t TotalTokens(const std::vector<Segment>& segments) {
  std::int64_t n = 0;
  for (const Segment& s : segments) n += s.token_len;
  return n;
}

// The four temporal-mask lengths for one segment, from ITS OWN sample count:
// `valid_mel_lens = audio_sample_lens // hop_length` per batch element
// (`audio_encoder.py:570-577`), then `(n + 1) // 2` at each of the three
// stride-2 layers (`:550`, `:555`, `:560`). Taking these from the PADDED length
// instead is one of the four defects a norm cannot see.
std::vector<std::int64_t> MaskStages(std::int64_t length,
                                     std::int64_t hop_length) {
  std::vector<std::int64_t> v(4);
  v[0] = length / hop_length;
  v[1] = (v[0] + 1) / 2;
  v[2] = (v[1] + 1) / 2;
  v[3] = (v[2] + 1) / 2;
  return v;
}

}  // namespace ref_chunks

// ═══════════════════════════════════════════════════════════════════════════
// 1c. THE RESAMPLE REFERENCE (W7c-2, #2828).
//
// A FOURTH reference namespace, under the SAME run-time enumeration instrument
// W7a wrote and W7b extended, because reference code the instrument does not
// read is reference code whose independence nothing measures.
//
// It transcribes `scipy.signal.resample_poly` at its defaults from scipy's own
// source — `_signaltools.py::resample_poly`, `_fir_filter_design.py::firwin`,
// `windows._windows.kaiser` — which is the arm upstream's `resample_audio_scipy`
// calls (`vllm/multimodal/audio.py:232-250` @ `9035151d6`). Spec §4.17.3 writes
// the six steps out.
//
// WHAT THIS REFERENCE DOES AND DOES NOT ADD. It is the SAME algorithm as
// `src/`, written twice, so it catches a transcription slip in one of the two
// and nothing else. The actual oracle on this slice is
// `dots3_note_resample_golden.h`, which is scipy's own output; this namespace
// is the row's convention and the second opinion, not the authority. Spec
// §4.17.7 says so in its own words.
// ═══════════════════════════════════════════════════════════════════════════
namespace ref_resample {

// MEASURED by the enumeration case below, as the other three are.
//
//  1 std::int64_t (39)   2 std::size_t (14)   3 std::vector (8)
//  4 std::sin (1)        5 std::sqrt (1)
//
// 63 occurrences of 5 distinct names. That a windowed-sinc filter design needs
// exactly TWO transcendentals is the shape to check: anything from `vt::` or
// `vllm::` appearing in the measurement is a helper that leaked in from `src/`,
// and the enumeration reads this file's own bytes rather than this list.
inline constexpr int kDistinctQualifiedNames = 5;
inline constexpr int kQualifiedNameOccurrences = 63;

// `scipy.special.i0` as its own power series, `sum_k (x^2/4)^k / (k!)^2`. At
// the kaiser default's beta = 5 the argument never exceeds 5.
double I0(double x) {
  const double q = x * x / 4.0;
  double term = 1.0;
  double total = 1.0;
  for (int k = 1; k < 200; ++k) {
    term *= q / (static_cast<double>(k) * static_cast<double>(k));
    total += term;
    if (term < 1e-18 * total) break;
  }
  return total;
}

// `numpy.sinc`.
double Sinc(double x) {
  if (x == 0.0) return 1.0;
  const double p = 3.14159265358979323846 * x;
  return std::sin(p) / p;
}

std::int64_t Gcd(std::int64_t a, std::int64_t b) {
  while (b != 0) {
    const std::int64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

// The whole of `resample_poly(x, up, down)` at defaults, in double.
std::vector<double> ResamplePoly(const std::vector<double>& x,
                                 std::int64_t orig_sr, std::int64_t target_sr) {
  const std::int64_t g = Gcd(orig_sr, target_sr);
  const std::int64_t up = target_sr / g;
  const std::int64_t down = orig_sr / g;
  if (up == 1 && down == 1) return x;

  // firwin(2 * half_len + 1, 1 / max_rate, window=("kaiser", 5.0)), scaled so
  // the taps sum to `up`.
  const std::int64_t max_rate = up > down ? up : down;
  const double f_c = 1.0 / static_cast<double>(max_rate);
  const std::int64_t half_len = 10 * max_rate;
  const std::int64_t numtaps = 2 * half_len + 1;
  const double alpha = 0.5 * static_cast<double>(numtaps - 1);
  const double i0_beta = I0(5.0);
  std::vector<double> h(static_cast<std::size_t>(numtaps));
  double sum = 0.0;
  for (std::int64_t i = 0; i < numtaps; ++i) {
    const double m = static_cast<double>(i) - alpha;
    const double r = m / alpha;
    const double under = 1.0 - r * r;
    const double win = I0(5.0 * std::sqrt(under > 0.0 ? under : 0.0)) / i0_beta;
    h[static_cast<std::size_t>(i)] = f_c * Sinc(f_c * m) * win;
    sum += h[static_cast<std::size_t>(i)];
  }
  for (double& v : h) v = v * static_cast<double>(up) / sum;

  const std::int64_t n_in = static_cast<std::int64_t>(x.size());
  const std::int64_t prod = n_in * up;
  const std::int64_t n_out = prod / down + ((prod % down) != 0 ? 1 : 0);

  const std::int64_t n_pre_pad = down - (half_len % down);
  const std::int64_t n_pre_remove = (half_len + n_pre_pad) / down;
  std::int64_t n_post_pad = 0;
  // `_output_len(len_h, n_in, up, down) = ((n_in - 1) * up + len_h - 1)
  //  // down + 1`.
  while (((n_in - 1) * up + (static_cast<std::int64_t>(h.size()) + n_pre_pad +
                             n_post_pad) - 1) / down + 1 <
         n_out + n_pre_remove) {
    ++n_post_pad;
  }
  std::vector<double> hh(static_cast<std::size_t>(n_pre_pad) + h.size() +
                             static_cast<std::size_t>(n_post_pad),
                         0.0);
  for (std::size_t i = 0; i < h.size(); ++i)
    hh[static_cast<std::size_t>(n_pre_pad) + i] = h[i];
  const std::int64_t len_h = static_cast<std::int64_t>(hh.size());

  std::vector<double> y(static_cast<std::size_t>(n_out), 0.0);
  for (std::int64_t oi = 0; oi < n_out; ++oi) {
    const std::int64_t t = (oi + n_pre_remove) * down;
    std::int64_t j_lo = (t - len_h + up) / up;
    if (j_lo < 0) j_lo = 0;
    std::int64_t j_hi = t / up;
    if (j_hi > n_in - 1) j_hi = n_in - 1;
    double acc = 0.0;
    for (std::int64_t j = j_lo; j <= j_hi; ++j)
      acc += hh[static_cast<std::size_t>(t - j * up)] *
             x[static_cast<std::size_t>(j)];
    y[static_cast<std::size_t>(oi)] = acc;
  }
  return y;
}

// THE DEFECT THE GOLDEN TOLERANCE MUST BE ABLE TO SEE: no anti-alias filter at
// all, just the nearest input sample. It is a legitimate "resampler" on
// band-limited content and a wrong one on anything else, which is why the
// golden set carries a case with a tone above the output Nyquist.
std::vector<double> NearestSample(const std::vector<double>& x,
                                  std::int64_t orig_sr,
                                  std::int64_t target_sr) {
  const std::int64_t n_in = static_cast<std::int64_t>(x.size());
  const std::int64_t g = Gcd(orig_sr, target_sr);
  const std::int64_t up = target_sr / g;
  const std::int64_t down = orig_sr / g;
  const std::int64_t prod = n_in * up;
  const std::int64_t n_out = prod / down + ((prod % down) != 0 ? 1 : 0);
  std::vector<double> y(static_cast<std::size_t>(n_out), 0.0);
  for (std::int64_t i = 0; i < n_out; ++i) {
    std::int64_t j = (i * down + up / 2) / up;
    if (j > n_in - 1) j = n_in - 1;
    y[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(j)];
  }
  return y;
}

}  // namespace ref_resample

// ═══════════════════════════════════════════════════════════════════════════
// A FIFTH REFERENCE (W8a, #2860): the ONE-PASS prompt-update planner.
//
// A second, independent implementation of `apply_token_matches`
// (`vllm/multimodal/processing/processor.py:944-957` @ `9035151d6`) and the
// planner it calls (`:799-857`, `:906-941`), written from the Python and not
// from `src/vllm/multimodal/processing/processor.cpp`. It shares NO helper with
// the implementation; every qualified name is `std::`, measured by the case
// below, which is why this namespace joins the EXISTING instrument rather than
// getting a second one of its own.
//
// It is INTEGER work, so "double precision" does not apply and this file does
// not claim it. What the reference buys is that two independently written
// planners agree on the whole output id stream and on every span, which is the
// property a transcription slip in either one breaks.
//
// The style is deliberately not the implementation's: this one PRE-SCANS every
// target occurrence in the prompt once, then walks the occurrence lists with a
// cursor, where `src/` re-searches from the previous match's end on each step.
// Two shapes of the same rule, so a shared off-by-one has to be made twice.
// ═══════════════════════════════════════════════════════════════════════════
namespace ref_apply {

// MEASURED by the enumeration case below, as the other four are.
//
//  1 std::size_t (12)   2 std::string (2)   3 std::vector (14)
inline constexpr int kDistinctQualifiedNames = 3;
inline constexpr int kQualifiedNameOccurrences = 28;

struct Rule {
  std::string modality;
  std::vector<int> target;
  std::vector<int> pads;  // one placeholder-run length per item
};

struct Span {
  std::string modality;
  int item = 0;
  int offset = 0;
  int length = 0;
};

struct Applied {
  std::vector<int> ids;
  std::vector<Span> spans;
};

// Every position at which `target` occurs in `ids`, overlaps included. The
// EXCLUSION of overlapping matches is upstream's `start_idx = end_idx` advance
// (`processor.py:652-655`) and it is applied by the cursor walk below, not
// here.
inline std::vector<int> AllOccurrences(const std::vector<int>& ids,
                                       const std::vector<int>& target) {
  std::vector<int> at;
  if (target.empty()) return at;
  const std::size_t n = ids.size();
  const std::size_t m = target.size();
  if (n < m) return at;
  for (std::size_t i = 0; i + m <= n; ++i) {
    std::size_t j = 0;
    while (j < m && ids[i + j] == target[j]) ++j;
    if (j == m) at.push_back(static_cast<int>(i));
  }
  return at;
}

inline Applied Apply(const std::vector<int>& ids,
                     const std::vector<Rule>& rules) {
  const std::size_t r_count = rules.size();
  std::vector<std::vector<int>> occ(r_count);
  for (std::size_t r = 0; r < r_count; ++r)
    occ[r] = AllOccurrences(ids, rules[r].target);

  std::vector<std::size_t> cursor(r_count, 0);  // into occ[r]
  std::vector<std::size_t> taken(r_count, 0);   // items consumed

  Applied out;
  int copied = 0;  // how much of `ids` has been emitted
  for (;;) {
    std::size_t pick = r_count;
    int pick_at = 0;
    for (std::size_t r = 0; r < r_count; ++r) {
      if (taken[r] >= rules[r].pads.size()) continue;
      while (cursor[r] < occ[r].size() && occ[r][cursor[r]] < copied)
        ++cursor[r];
      if (cursor[r] >= occ[r].size()) continue;
      const int here = occ[r][cursor[r]];
      if (pick == r_count || here < pick_at) {
        pick = r;
        pick_at = here;
      }
    }
    if (pick == r_count) break;

    const Rule& rule = rules[pick];
    for (int i = copied; i < pick_at; ++i)
      out.ids.push_back(ids[static_cast<std::size_t>(i)]);

    const int pads = rule.pads[taken[pick]];
    Span span;
    span.modality = rule.modality;
    span.item = static_cast<int>(taken[pick]);
    out.ids.push_back(rule.target[0]);
    span.offset = static_cast<int>(out.ids.size());
    span.length = pads;
    for (int i = 0; i < pads; ++i) out.ids.push_back(rule.target[1]);
    out.ids.push_back(rule.target[2]);
    out.spans.push_back(span);

    copied = pick_at + static_cast<int>(rule.target.size());
    ++taken[pick];
  }
  for (int i = copied; i < static_cast<int>(ids.size()); ++i)
    out.ids.push_back(ids[static_cast<std::size_t>(i)]);
  return out;
}

}  // namespace ref_apply

// ═══════════════════════════════════════════════════════════════════════════
// THE ENUMERATION INSTRUMENT.
//
// Reads THIS source file at `DOTS3_AUDIO_TEST_SOURCE` (the same arrangement
// `MODELOPT_MIXED_FIXTURE_DIR` uses to hand a test a path), strips comments and
// string/char literals, takes the span of one reference namespace, and counts
// every `scope::name`. Comments must be stripped or the enumeration LIST above
// would count itself and the instrument would agree with any list it was given
// — which is the exact failure this replaces.
// ═══════════════════════════════════════════════════════════════════════════
namespace {

struct RefNames {
  int distinct = 0;
  int occurrences = 0;
  std::set<std::string> scopes;
  std::set<std::string> names;
};

std::string Join(const std::set<std::string>& s) {
  std::string out;
  for (const std::string& v : s) {
    if (!out.empty()) out += ",";
    out += v;
  }
  return out;
}

bool IsIdentChar(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_';
}

// Comments and literals out, everything else through unchanged.
std::string StripCommentsAndLiterals(const std::string& code) {
  std::string out;
  const size_t n = code.size();
  for (size_t i = 0; i < n;) {
    const char c = code[i];
    if (c == '/' && i + 1 < n && code[i + 1] == '/') {
      const size_t j = code.find('\n', i);
      i = (j == std::string::npos) ? n : j;
    } else if (c == '/' && i + 1 < n && code[i + 1] == '*') {
      const size_t j = code.find("*/", i + 2);
      i = (j == std::string::npos) ? n : j + 2;
    } else if (c >= '0' && c <= '9' && (i == 0 || !IsIdentChar(code[i - 1]))) {
      // A pp-number, taken WHOLE. C++14 lets a numeric literal carry `'` digit
      // separators (`16'000`), and treating that `'` as a char-literal
      // delimiter makes the scan below run to the NEXT `'` and drop everything
      // in between. That is a hole in THIS instrument and not a cosmetic one:
      // two separators bracketing a `vt::` call would hide the call from the
      // enumeration, and the independence property would read GREEN while being
      // false. The token must START at a digit that does not continue an
      // identifier, so `u8'a'` is still a char literal and is still stripped.
      size_t j = i;
      while (j < n) {
        if (IsIdentChar(code[j]) || code[j] == '.') {
          ++j;
        } else if (code[j] == '\'' && j + 1 < n && IsIdentChar(code[j + 1])) {
          j += 2;
        } else if ((code[j] == '+' || code[j] == '-') && j > i &&
                   (code[j - 1] == 'e' || code[j - 1] == 'E' ||
                    code[j - 1] == 'p' || code[j - 1] == 'P')) {
          ++j;
        } else {
          break;
        }
      }
      out.append(code, i, j - i);
      i = j;
    } else if (c == '"' || c == '\'') {
      size_t j = i + 1;
      while (j < n && code[j] != c) j += (code[j] == '\\') ? 2 : 1;
      i = j + 1;
    } else {
      out += c;
      ++i;
    }
  }
  return out;
}

RefNames QualifiedNamesIn(const std::string& ns) {
  std::ifstream in(DOTS3_AUDIO_TEST_SOURCE, std::ios::binary);
  REQUIRE_MESSAGE(in.good(),
                  "the enumeration instrument could not open its own source at "
                      << DOTS3_AUDIO_TEST_SOURCE);
  const std::string src((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  const std::string open = "namespace " + ns + " {";
  const std::string close = "}  // namespace " + ns;
  const size_t a = src.find(open);
  const size_t b = src.find(close);
  REQUIRE(a != std::string::npos);
  REQUIRE(b != std::string::npos);
  REQUIRE(b > a);
  const std::string body = StripCommentsAndLiterals(src.substr(a, b - a));

  RefNames r;
  for (size_t i = 0; i + 1 < body.size(); ++i) {
    if (body[i] != ':' || body[i + 1] != ':') continue;
    // the scope to the left
    size_t s = i;
    while (s > 0 && IsIdentChar(body[s - 1])) --s;
    if (s == i) continue;
    // the name to the right
    size_t e = i + 2;
    size_t t = e;
    while (t < body.size() && IsIdentChar(body[t])) ++t;
    if (t == e) continue;
    const std::string scope = body.substr(s, i - s);
    const std::string name = body.substr(e, t - e);
    r.scopes.insert(scope);
    r.names.insert(scope + "::" + name);
    ++r.occurrences;
  }
  r.distinct = static_cast<int>(r.names.size());
  return r;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// 1. THE ONE REAL ORACLE ON THIS ROW.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("dots3-note W7a: the SHARED slaney bank reproduces the committed voxtral oracle BIT for BIT") {
  // `nvidia/audio.py:98-106` @ `9035151d6`:
  //   mel_filter_bank(num_frequency_bins=1 + 400//2, num_mel_filters=128,
  //                   min_frequency=0.0, max_frequency=16000/2,
  //                   sampling_rate=16000, norm="slaney", mel_scale="slaney")
  const std::vector<float> ours =
      vllm::multimodal::MelFilterBankSlaney(201, 128, 0.0, 8000.0, 16000);
  const std::vector<float> oracle =
      ReadF32File(VoxtralFixtureDir() + "/voxtral_mel_filters_f32.bin");

  REQUIRE(oracle.size() == 201u * 128u);
  REQUIRE(ours.size() == oracle.size());

  // BIT-FOR-BIT, not a tolerance. Both sides construct in double and round once
  // on the store, so anything short of exact would mean a formula difference
  // rather than a rounding one — and this is the ONE assertion on this row that
  // can tell those apart, because the other side of it is
  // `mistral_common.audio.mel_filter_bank` and not a second transcription of
  // ours.
  std::size_t mismatched = 0;
  double worst = 0.0;
  for (std::size_t i = 0; i < ours.size(); ++i) {
    if (ours[i] != oracle[i]) {
      ++mismatched;
      worst = std::max(worst, static_cast<double>(std::fabs(ours[i] - oracle[i])));
    }
  }
  MESSAGE("mel bank vs voxtral oracle: " << mismatched << " of " << ours.size()
                                         << " values differ, worst |delta| " << worst);
  CHECK(mismatched == 0u);

  // The bank is SPARSE, and saying so keeps a reader from reading "25728 values
  // agree" as 25728 independent facts: at 128 mels over 201 bins most of the
  // low-frequency triangles fall between FFT bins and are entirely zero.
  std::size_t nonzero = 0;
  for (float v : oracle)
    if (v != 0.0f) ++nonzero;
  MESSAGE("the bank is sparse: " << nonzero << " of " << oracle.size()
                                 << " values are nonzero");
  CHECK(nonzero > 0u);

  // The TRANSPOSED accessor is the same numbers in the other order — which is
  // what makes Parakeet byte-identical across the extraction rather than
  // merely close.
  const std::vector<float> t =
      vllm::multimodal::MelFilterBankSlaneyTransposed(201, 128, 0.0, 8000.0, 16000);
  REQUIRE(t.size() == ours.size());
  std::size_t t_bad = 0;
  for (int k = 0; k < 201; ++k)
    for (int m = 0; m < 128; ++m)
      if (t[static_cast<std::size_t>(m) * 201 + k] !=
          ours[static_cast<std::size_t>(k) * 128 + m])
        ++t_bad;
  CHECK(t_bad == 0u);
}

TEST_CASE("dots3-note W7a+W7b+W7c-2: the FOUR references share no helper with src/, by enumeration") {
  // The row's convention (W6a 70, W6b 105, W6c 45 qualified names, all `std::`),
  // now over THREE namespaces since W7b (#2797).
  //
  // THIS CASE RE-READS THIS FILE AND COUNTS. It used to compare two
  // hand-written constants with two literals, which measured nothing: the lists
  // beside those constants named 22 and 19 names when the references actually
  // use 11 and 6, and a reference that GREW a `vllm::` call would not have
  // moved either number. The load-bearing assertion below is the SCOPE SET —
  // every qualified name in each reference span resolves through `std::` — and
  // it is computed from the bytes, so one `vllm::` or `vt::` call reddens it.
  const RefNames front = QualifiedNamesIn("ref_front");
  const RefNames tower = QualifiedNamesIn("ref_tower");
  // W7b (#2797) EXTENDS this instrument to its third namespace rather than
  // writing a second one: `ref_chunks` is new reference code, and reference
  // code the instrument does not read is reference code whose independence
  // nothing measures.
  const RefNames chunks = QualifiedNamesIn("ref_chunks");
  // W7c-2 (#2828) extends it AGAIN, to a fourth namespace, for the same reason
  // W7b did rather than writing a second instrument.
  const RefNames resample = QualifiedNamesIn("ref_resample");
  // W8a (#2860) extends it to a FIFTH, for the same reason again: `ref_apply`
  // is new reference code, and reference code the instrument does not read is
  // reference code whose independence nothing measures.
  const RefNames apply = QualifiedNamesIn("ref_apply");

  // The instrument must be shown to have READ something. A parse that found an
  // empty span would otherwise report "zero non-std:: names" and pass.
  REQUIRE(front.occurrences > 0);
  REQUIRE(tower.occurrences > 0);
  REQUIRE(chunks.occurrences > 0);
  REQUIRE(resample.occurrences > 0);
  REQUIRE(apply.occurrences > 0);

  MESSAGE("ref_front: " << front.distinct << " distinct, " << front.occurrences
                        << " occurrences, scopes=" << Join(front.scopes));
  MESSAGE("ref_tower: " << tower.distinct << " distinct, " << tower.occurrences
                        << " occurrences, scopes=" << Join(tower.scopes));
  MESSAGE("ref_chunks: " << chunks.distinct << " distinct, "
                         << chunks.occurrences
                         << " occurrences, scopes=" << Join(chunks.scopes));
  MESSAGE("ref_resample: " << resample.distinct << " distinct, "
                           << resample.occurrences
                           << " occurrences, scopes=" << Join(resample.scopes));
  MESSAGE("ref_apply: " << apply.distinct << " distinct, " << apply.occurrences
                        << " occurrences, scopes=" << Join(apply.scopes));

  // The independence property itself.
  CHECK(Join(front.scopes) == "std");
  CHECK(Join(tower.scopes) == "std");
  CHECK(Join(chunks.scopes) == "std");
  CHECK(Join(resample.scopes) == "std");
  CHECK(Join(apply.scopes) == "std");

  // And the enumerated lists, now that they are the measurement's own output.
  CHECK(front.distinct == ref_front::kDistinctQualifiedNames);
  CHECK(front.occurrences == ref_front::kQualifiedNameOccurrences);
  CHECK(tower.distinct == ref_tower::kDistinctQualifiedNames);
  CHECK(tower.occurrences == ref_tower::kQualifiedNameOccurrences);
  CHECK(chunks.distinct == ref_chunks::kDistinctQualifiedNames);
  CHECK(chunks.occurrences == ref_chunks::kQualifiedNameOccurrences);
  CHECK(resample.distinct == ref_resample::kDistinctQualifiedNames);
  CHECK(resample.occurrences == ref_resample::kQualifiedNameOccurrences);
  CHECK(apply.distinct == ref_apply::kDistinctQualifiedNames);
  CHECK(apply.occurrences == ref_apply::kQualifiedNameOccurrences);

  // AND THE STRIPPER ITSELF, because the property above is only as true as the
  // scan that measures it. `StripCommentsAndLiterals` used to treat every `'`
  // as a char-literal delimiter, so two C++14 digit separators bracketing a
  // `vt::` call hid that call and this case read GREEN with a live reach
  // inside a reference. Nothing above can see that: the clean file carries no
  // separator, so the counts do not move either way. These four are the only
  // standing gate on the pp-number rule.
  const std::string bracketed =
      "double g = 16'000.0 * vt::Scale(vllm::kOne) / 1'280.0;";
  CHECK(StripCommentsAndLiterals(bracketed).find("vt::Scale") !=
        std::string::npos);
  CHECK(StripCommentsAndLiterals(bracketed).find("vllm::kOne") !=
        std::string::npos);
  // A PREFIXED char literal is still a literal, which is what the shorter
  // "ignore a `'` whose previous character is alphanumeric" rule would break.
  CHECK(StripCommentsAndLiterals("u8'v' L't' 'x'").find('v') ==
        std::string::npos);
  // And a comment is still removed whole, separator or not.
  CHECK(StripCommentsAndLiterals("// vt::Scale 16'000\nint a = 1;")
            .find("vt::") == std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. THE FRONT END against `ref_front`.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("dots3-note W7a: the front end agrees with an INDEPENDENT double reference") {
  const TinySpec spec = AudioSpec();
  const TinyCheckpoint ckpt(FixtureDir(), spec);
  vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                          "tiny");
  REQUIRE(cfg.present);
  CHECK(cfg.sampling_rate == 16000);
  CHECK(cfg.chunk_seconds == spec.a_chunk_seconds);
  CHECK(cfg.n_mels == spec.a_mels);
  CHECK(cfg.token_stride() == 1280);
  CHECK(cfg.chunk_samples() == spec.a_chunk_samples());
  CHECK(cfg.chunk_mel_frames() == spec.a_chunk_mel_frames());
  cfg.audio_start_token_id = dots3_tiny::kAudStartId;
  cfg.audio_token_id = dots3_tiny::kAudPadId;
  cfg.audio_end_token_id = dots3_tiny::kAudEndId;
  const vllm::multimodal::Dots3NoteAudioProcessor proc(cfg);

  const std::vector<float> wav = dots3_tiny::FixtureAudioF32(0);
  const vllm::multimodal::AudioKwargs got = proc.ProcessWaveform(
      wav.data(), static_cast<int64_t>(wav.size()), 16000);

  CHECK(got.n_mels == spec.a_mels);
  CHECK(got.n_frames == spec.a_chunk_mel_frames());
  CHECK(got.num_samples == dots3_tiny::kAudioSamples);
  // `ceil(8000 / 1280)` = 7 (`common/processor.py:771`).
  CHECK(got.num_tokens == dots3_tiny::kAudioTokens);

  const std::vector<std::vector<double>> want = ref_front::LogMel(
      wav, static_cast<int>(spec.a_chunk_samples()), cfg.n_fft, cfg.hop_length,
      static_cast<int>(spec.a_mels), 16000);
  REQUIRE(want.size() == static_cast<std::size_t>(spec.a_mels));
  REQUIRE(want[0].size() == static_cast<std::size_t>(spec.a_chunk_mel_frames()));

  double worst = 0.0;
  for (int64_t m = 0; m < spec.a_mels; ++m) {
    for (int64_t f = 0; f < spec.a_chunk_mel_frames(); ++f) {
      const double a =
          got.input_features[static_cast<std::size_t>(m * got.n_frames + f)];
      const double b = want[static_cast<std::size_t>(m)][static_cast<std::size_t>(f)];
      worst = std::max(worst, std::fabs(a - b));
    }
  }
  // Both sides compute in double and differ only in float summation ORDER
  // inside the DFT and the mel projection; the implementation stores f32.
  MESSAGE("front end vs reference: worst |delta| " << worst);
  CHECK(worst < 1e-5);

  // THE PADDED TAIL IS NOT ZERO, which is the whole reason the tower masks. The
  // clip is half the chunk, so frames 50..99 are silence — and their value is
  // the `-8` global-max floor pushed through `(x + 4) / 4`, a NONZERO constant.
  const double tail =
      got.input_features[static_cast<std::size_t>(0 * got.n_frames + 99)];
  MESSAGE("the padded tail sits at " << tail << ", not at 0");
  CHECK(std::fabs(tail) > 0.1);
  // ...and it is the SAME constant everywhere in the tail, which is what makes
  // it leak as a bias rather than as noise.
  for (int64_t m = 0; m < spec.a_mels; ++m) {
    const double v =
        got.input_features[static_cast<std::size_t>(m * got.n_frames + 99)];
    CHECK(std::fabs(v - tail) < 1e-6);
  }
}

TEST_CASE("dots3-note W7c-2: the front end RESAMPLES a wrong rate, and W7b MOVED the length refusal") {
  const TinySpec spec = AudioSpec();
  const TinyCheckpoint ckpt(FixtureDir(), spec);
  vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                          "tiny");
  cfg.audio_token_id = dots3_tiny::kAudPadId;
  const vllm::multimodal::Dots3NoteAudioProcessor proc(cfg);
  const std::vector<float> wav = dots3_tiny::FixtureAudioF32(0);

  SUBCASE("a rate that is not `audio_config.sampling_rate` is RESAMPLED, not refused") {
    // TRUE-BEFORE / FALSE-AFTER, and this subcase is the ownership test for
    // W7c-2 (#2828). It used to assert that `ProcessWaveform` THREW, that the
    // message named W7c-2 and "RESAMPLING IS NOT PORTED", and that the message
    // named libswresample. None of that is true any more. The RED-before is in
    // spec §4.17.11 verbatim.
    const std::vector<float> at22 = dots3_tiny::FixtureAudioF32AtRate(0, 22050);
    REQUIRE(at22.size() == 11025u);
    const vllm::multimodal::AudioKwargs kw =
        proc.ProcessWaveform(at22.data(), static_cast<int64_t>(at22.size()),
                             22050);

    // (a) The waveform the front end saw is the RESAMPLED one:
    // `ceil(11025 * 320 / 441)` = 8000, which is `kAudioSamples`, so the span
    // is the same 7 tokens the 16 kHz clip expands. An UNRESAMPLED 11025-sample
    // waveform would carry `ceil(11025 / 1280)` = 9 tokens, so this assertion
    // alone separates a pass-through, and it does so without reference to any
    // value the resampler produced.
    CHECK(kw.num_samples == dots3_tiny::kAudioSamples);
    CHECK(kw.num_tokens == dots3_tiny::kAudioTokens);

    // (b) THE RESAMPLE IS THE ONLY DIFFERENCE. Feeding the pre-resampled
    // waveform at the target rate produces the same mel BIT FOR BIT, which is
    // what "the same audio, resampled offline, gives the same answer" means at
    // the front end. The values that resample carries are gated against scipy
    // separately, by the golden case.
    const std::vector<float> pre = vllm::multimodal::ResampleAudioScipy(
        at22.data(), static_cast<int64_t>(at22.size()), 22050, 16000);
    REQUIRE(pre.size() == static_cast<std::size_t>(dots3_tiny::kAudioSamples));
    const vllm::multimodal::AudioKwargs off = proc.ProcessWaveform(
        pre.data(), static_cast<int64_t>(pre.size()), 16000);
    REQUIRE(off.input_features.size() == kw.input_features.size());
    std::size_t moved = 0;
    for (std::size_t i = 0; i < kw.input_features.size(); ++i)
      if (kw.input_features[i] != off.input_features[i]) ++moved;
    MESSAGE("resampled in the front end vs pre-resampled: " << moved << " of "
            << kw.input_features.size() << " mel values differ");
    CHECK(moved == 0u);

    // (c) ...and those values are scipy's, checked against the INDEPENDENT
    // second transcription rather than against the same code.
    std::vector<double> xin(at22.size());
    for (std::size_t i = 0; i < at22.size(); ++i)
      xin[i] = static_cast<double>(at22[i]);
    const std::vector<double> ref =
        ref_resample::ResamplePoly(xin, 22050, 16000);
    REQUIRE(ref.size() == pre.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i)
      worst = std::max(worst,
                       std::fabs(ref[i] - static_cast<double>(pre[i])));
    MESSAGE("22050 -> 16000 vs ref_resample: worst |delta| " << worst);
    CHECK(worst <= kResampleTol);
  }

  SUBCASE("...and the encoder-cache key SEPARATES two rates over one buffer") {
    // W7c-2 CREATED this hazard and closes it in the same change. Before it,
    // every served rate was 16000 and the raw waveform was an unambiguous key.
    // Now a file carrying N samples at 16000 Hz and a file carrying THE
    // IDENTICAL N SAMPLES at 44100 Hz decode to identical float buffers. They
    // must not share an encoder-cache entry, because their features differ.
    const std::vector<float>& x = wav;
    const std::string at_target =
        proc.HashAudio(x.data(), static_cast<int64_t>(x.size()), 16000);
    const std::string at_44100 =
        proc.HashAudio(x.data(), static_cast<int64_t>(x.size()), 44100);
    MESSAGE("same buffer, 16000 vs 44100: " << at_target << " / " << at_44100);
    CHECK(at_target != at_44100);

    // ...and at the target rate the three-argument form is the two-argument
    // one, so no existing key moved when this overload was added.
    CHECK(at_target == proc.HashAudio(x.data(), static_cast<int64_t>(x.size())));

    // The features really do differ, which is what makes the separation a
    // correctness requirement rather than a nicety: 8000 samples read as
    // 44100 Hz resample to `ceil(8000 * 160 / 441)` = 2903.
    const vllm::multimodal::AudioKwargs a =
        proc.ProcessWaveform(x.data(), static_cast<int64_t>(x.size()), 16000);
    const vllm::multimodal::AudioKwargs b =
        proc.ProcessWaveform(x.data(), static_cast<int64_t>(x.size()), 44100);
    CHECK(a.num_samples == 8000);
    CHECK(b.num_samples == 2903);
    CHECK(a.num_tokens != b.num_tokens);
  }

  SUBCASE("...and the served path builds that key from ONE resample, not two") {
    // PR #2842 F2. `RouteDots3NoteAudioWav` drove `ProcessWaveform` and then the
    // three-argument `HashAudio`, and each of them resampled: the 1 Hz request
    // in spec §4.17.10 allocated 1220.7 MB TWICE for a 40 KB upload. The route
    // now hands the buffer over. What has to hold is that handing it over and
    // rebuilding it produce the SAME key, because a caller passing the wrong
    // buffer would be a silent cross-request cache defect and not a crash.
    const std::vector<float>& x = wav;
    const auto n = static_cast<std::int64_t>(x.size());

    std::vector<float> shared;
    const vllm::multimodal::AudioKwargs kw =
        proc.ProcessWaveform(x.data(), n, 44100, &shared);

    // (a) The out-parameter is FILLED, and with the waveform the tower
    // consumed rather than with anything else: same length as `num_samples`,
    // and bit-identical to the seam's own answer for the same conversion.
    CHECK(shared.size() == static_cast<std::size_t>(kw.num_samples));
    REQUIRE(shared.size() == 2903u);
    const std::vector<float> off =
        vllm::multimodal::ResampleAudioScipy(x.data(), n, 44100, 16000);
    REQUIRE(off.size() == shared.size());
    std::size_t moved = 0;
    for (std::size_t i = 0; i < off.size(); ++i)
      if (off[i] != shared[i]) ++moved;
    MESSAGE("handed back vs resampled again: " << moved << " of "
            << off.size() << " samples differ");
    CHECK(moved == 0u);

    // (b) ...so the key is the same whichever way it is built. This is the
    // assertion a wrong buffer at the call site fails.
    CHECK(proc.HashAudio(x.data(), n, 44100, &shared) ==
          proc.HashAudio(x.data(), n, 44100));

    // (c) At the target rate nothing is handed back, because then the caller's
    // own pointer already IS the consumed waveform. The vector is pre-filled so
    // that "left empty" is distinguishable from "never written".
    std::vector<float> none(3, 1.0f);
    proc.ProcessWaveform(x.data(), n, 16000, &none);
    CHECK(none.empty());
  }

  SUBCASE("a clip over `chunk_seconds` no longer names W7b, because W7b LIFTED it") {
    // This subcase used to assert "SEGMENTATION IS NOT PORTED" and W7b. It is
    // kept rather than deleted because it is the RED-BEFORE of #2797 written
    // down: the same call, on the same config, now refuses for a DIFFERENT and
    // narrower reason, and a reader who greps for the old string should land
    // here and find out where it went.
    //
    // The tiny fixture's `chunk_seconds` = 1 is 12.5 token strides, so this
    // geometry is one of the non-divisible ones §4.15.3 still refuses PAST ONE
    // CHUNK: 16001 samples is 16000 + 1, whose per-segment sum is 13 + 1 = 14
    // against a placeholder span of ceil(16001/1280) = 13.
    std::vector<float> too_long(static_cast<std::size_t>(spec.a_chunk_samples() + 1),
                                0.1f);
    std::string msg;
    try {
      proc.ProcessWaveform(too_long.data(),
                           static_cast<int64_t>(too_long.size()), 16000);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("SEGMENTATION IS NOT PORTED") == std::string::npos);
    CHECK(msg.find("#2797") != std::string::npos);
    CHECK(msg.find("not a whole number of 1280") != std::string::npos);
    CHECK(msg.find("14 rows") != std::string::npos);
    CHECK(msg.find("span of 13") != std::string::npos);
    CHECK(msg.find("splices audio features") != std::string::npos);
  }
  SUBCASE("...and on a DIVISIBLE geometry the very same length is SERVED") {
    // The other half of the sentence above, and the reason the refusal is about
    // the config's arithmetic rather than about length. `a_chunk_seconds = 2`
    // is 25 whole token strides, and 32001 samples is 2 chunks there.
    TinySpec long_spec = spec;
    long_spec.a_chunk_seconds = dots3_tiny::kAudioLongChunkSeconds;
    const TinyCheckpoint long_ckpt(FixtureDir(), long_spec);
    vllm::multimodal::Dots3NoteAudioProcessorConfig long_cfg =
        vllm::multimodal::LoadDots3NoteAudioProcessorConfig(
            long_ckpt.config_path(), "tiny");
    long_cfg.audio_token_id = dots3_tiny::kAudPadId;
    const vllm::multimodal::Dots3NoteAudioProcessor long_proc(long_cfg);
    std::vector<float> two_chunks(
        static_cast<std::size_t>(long_spec.a_chunk_samples() + 1), 0.1f);
    const vllm::multimodal::AudioKwargs kw = long_proc.ProcessWaveform(
        two_chunks.data(), static_cast<int64_t>(two_chunks.size()), 16000);
    CHECK(kw.num_chunks == 2);
    CHECK(kw.chunk_num_tokens.size() == 2u);
    CHECK(kw.chunk_num_tokens[0] == 25);
    CHECK(kw.chunk_num_tokens[1] == 1);
    CHECK(kw.num_tokens == 26);
  }
  SUBCASE("...and exactly `chunk_samples` is ACCEPTED, so the bound is not off by one") {
    std::vector<float> exact(static_cast<std::size_t>(spec.a_chunk_samples()), 0.1f);
    const vllm::multimodal::AudioKwargs kw = proc.ProcessWaveform(
        exact.data(), static_cast<int64_t>(exact.size()), 16000);
    // `ceil`, not floor: 16000 / 1280 is 12.5, so a full chunk is THIRTEEN
    // tokens. That is also exactly the stem's output length (100 mel frames ->
    // 50 -> 25 -> 13), which is why a full chunk is the largest span the tower
    // can produce and why `chunk_seconds` is the right place to refuse.
    CHECK(kw.num_tokens ==
          (spec.a_chunk_samples() + 1280 - 1) / 1280);
    CHECK(kw.num_tokens == dots3_tiny::kAudioStemFrames);
  }
}

TEST_CASE("dots3-note W7a: `num_tokens` and the mask length are TWO numbers") {
  // §4.14: halving `samples // 160` three times is NOT `ceil(samples / 1280)`.
  // At 1281 samples the mask says 1 and the span says 2, so the span covers a
  // stem row the mask zeroed. A port that derived one from the other would be
  // wrong here and right on the fixture clip, which is why this case exists.
  const TinySpec spec = AudioSpec();
  const TinyCheckpoint ckpt(FixtureDir(), spec);
  vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                          "tiny");
  cfg.audio_token_id = dots3_tiny::kAudPadId;
  const vllm::multimodal::Dots3NoteAudioProcessor proc(cfg);

  const auto mask_len = [](int64_t n) {
    int64_t v = n / 160;
    for (int i = 0; i < 3; ++i) v = (v + 1) / 2;
    return v;
  };
  CHECK(proc.NumAudioTokens(1280) == 1);
  CHECK(mask_len(1280) == 1);
  CHECK(proc.NumAudioTokens(1281) == 2);
  CHECK(mask_len(1281) == 1);
  MESSAGE("at 1281 samples: span " << proc.NumAudioTokens(1281) << ", mask "
                                   << mask_len(1281));
  // The fixture clip is a case where they AGREE, which is why it cannot see the
  // difference on its own.
  CHECK(proc.NumAudioTokens(dots3_tiny::kAudioSamples) == dots3_tiny::kAudioTokens);
  CHECK(mask_len(dots3_tiny::kAudioSamples) == dots3_tiny::kAudioTokens);
}

TEST_CASE("dots3-note W7a: the three marker ids come from the TOKENIZER, and refuse BY NAME") {
  const TinySpec spec = AudioSpec();
  const TinyCheckpoint ckpt(FixtureDir(), spec);
  vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                          "tiny");
  REQUIRE(cfg.present);
  CHECK(cfg.audio_comp_start == "<|audio_comp_start|>");
  CHECK(cfg.audio_comp_span == "<|audio_comp_pad|>");
  CHECK(cfg.audio_comp_end == "<|audio_comp_end|>");
  // NOT resolved by the config loader: only a tokenizer can answer.
  CHECK(cfg.audio_token_id == -1);

  SUBCASE("a tokenizer that carries all three resolves them BY STRING") {
    vllm::multimodal::Dots3NoteAudioProcessorConfig c = cfg;
    vllm::multimodal::Dots3NoteResolveAudioTokenIds(
        &c, [](const std::string& m) -> int32_t {
          if (m == "<|audio_comp_start|>") return dots3_tiny::kAudStartId;
          if (m == "<|audio_comp_end|>") return dots3_tiny::kAudEndId;
          if (m == "<|audio_comp_pad|>") return dots3_tiny::kAudPadId;
          return -1;
        });
    CHECK(c.audio_start_token_id == dots3_tiny::kAudStartId);
    CHECK(c.audio_end_token_id == dots3_tiny::kAudEndId);
    CHECK(c.audio_token_id == dots3_tiny::kAudPadId);
    // THE ORDER IS start, END, pad — the released checkpoint's own
    // (151718 / 151719 / 151720). A port that guessed "start, start+1,
    // start+2" would put the PAD id where the END id belongs, and this is the
    // assertion that catches it.
    CHECK(c.audio_end_token_id == c.audio_start_token_id + 1);
    CHECK(c.audio_token_id == c.audio_start_token_id + 2);
    CHECK(c.audio_token_id != c.audio_start_token_id + 1);
  }
  SUBCASE("a tokenizer missing ONE of them refuses, naming the marker") {
    vllm::multimodal::Dots3NoteAudioProcessorConfig c = cfg;
    std::string msg;
    try {
      vllm::multimodal::Dots3NoteResolveAudioTokenIds(
          &c, [](const std::string& m) -> int32_t {
            if (m == "<|audio_comp_pad|>") return -1;  // the one that is absent
            return 5;
          });
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("<|audio_comp_pad|>") != std::string::npos);
    CHECK(msg.find("audio_comp_span") != std::string::npos);
    CHECK(msg.find("REFUSING rather than defaulting") != std::string::npos);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. THE TOWER against `ref_tower`.
// ═══════════════════════════════════════════════════════════════════════════
namespace {

// The fixture checkpoint's bf16-rounded values, reshaped for the reference.
// Reading them from `TinyCheckpoint::value_of` rather than from the loaded
// `OwnedTensor` is deliberate: the reference is driven from the SAME BYTES the
// loader read, so the comparison measures the FORWARD and not a second copy of
// the weights.
struct TowerRefWeights {
  std::vector<ref_tower::Mat> conv1, conv2, conv3;  // [out][in][9]
  std::vector<double> conv1_b, conv2_b, conv3_b;
  ref_tower::Mat conv_out;  // [D][C*F]
  struct Layer {
    std::vector<double> attn_norm, final_norm;
    ref_tower::Mat q, k, v, o;
    std::vector<double> qb, vb, ob;
    ref_tower::Mat fc1, fc2;
    std::vector<double> fc1_b, fc2_b;
  };
  std::vector<Layer> layers;
  std::vector<double> final_norm;
  std::vector<double> ln_w, ln_b;
  ref_tower::Mat a1, a2;
  std::vector<double> a1_b, a2_b;
};

ref_tower::Mat Reshape(const std::vector<double>& v, std::size_t rows,
                       std::size_t cols) {
  REQUIRE(v.size() == rows * cols);
  ref_tower::Mat out(rows, std::vector<double>(cols));
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c) out[r][c] = v[r * cols + c];
  return out;
}

std::vector<ref_tower::Mat> Reshape3(const std::vector<double>& v,
                                     std::size_t out_ch, std::size_t in_ch) {
  REQUIRE(v.size() == out_ch * in_ch * 9);
  std::vector<ref_tower::Mat> out(out_ch,
                                  ref_tower::Mat(in_ch, std::vector<double>(9)));
  std::size_t i = 0;
  for (std::size_t co = 0; co < out_ch; ++co)
    for (std::size_t ci = 0; ci < in_ch; ++ci)
      for (std::size_t k = 0; k < 9; ++k) out[co][ci][k] = v[i++];
  return out;
}

TowerRefWeights ReadTowerWeights(const TinyCheckpoint& ckpt, const TinySpec& s) {
  const std::string se = "audio_encoder.dots_encoder.speech_encoder.";
  const std::size_t D = static_cast<std::size_t>(s.a_d_model);
  const std::size_t F = static_cast<std::size_t>(s.a_ffn);
  const std::size_t dhs = static_cast<std::size_t>(s.a_dhs);
  TowerRefWeights w;
  w.conv1 = Reshape3(ckpt.value_of(se + "conv2d1.weight"), dhs, 1);
  w.conv1_b = ckpt.value_of(se + "conv2d1.bias");
  w.conv2 = Reshape3(ckpt.value_of(se + "conv2d2.weight"), dhs, dhs);
  w.conv2_b = ckpt.value_of(se + "conv2d2.bias");
  w.conv3 = Reshape3(ckpt.value_of(se + "conv2d3.weight"), dhs, dhs);
  w.conv3_b = ckpt.value_of(se + "conv2d3.bias");
  w.conv_out = Reshape(ckpt.value_of(se + "conv_out.weight"), D,
                       dhs * static_cast<std::size_t>(s.a_freq_after()));
  for (int64_t l = 0; l < s.a_layers; ++l) {
    const std::string p = se + "layers." + std::to_string(l) + ".";
    TowerRefWeights::Layer lw;
    lw.attn_norm = ckpt.value_of(p + "self_attn_layer_norm.weight");
    lw.final_norm = ckpt.value_of(p + "final_layer_norm.weight");
    lw.q = Reshape(ckpt.value_of(p + "self_attn.q_proj.weight"), D, D);
    lw.qb = ckpt.value_of(p + "self_attn.q_proj.bias");
    lw.k = Reshape(ckpt.value_of(p + "self_attn.k_proj.weight"), D, D);
    lw.v = Reshape(ckpt.value_of(p + "self_attn.v_proj.weight"), D, D);
    lw.vb = ckpt.value_of(p + "self_attn.v_proj.bias");
    lw.o = Reshape(ckpt.value_of(p + "self_attn.out_proj.weight"), D, D);
    lw.ob = ckpt.value_of(p + "self_attn.out_proj.bias");
    lw.fc1 = Reshape(ckpt.value_of(p + "fc1.weight"),
                     static_cast<std::size_t>(s.a_fc1_out()), D);
    lw.fc1_b = ckpt.value_of(p + "fc1.bias");
    lw.fc2 = Reshape(ckpt.value_of(p + "fc2.weight"), D, F);
    lw.fc2_b = ckpt.value_of(p + "fc2.bias");
    w.layers.push_back(std::move(lw));
  }
  w.final_norm = ckpt.value_of(se + "layer_norm.weight");
  const std::string ad = "audio_encoder.audio_adapter.proj.";
  const std::size_t AO = static_cast<std::size_t>(s.a_adapter_out());
  w.ln_w = ckpt.value_of(ad + "0.weight");
  w.ln_b = ckpt.value_of(ad + "0.bias");
  w.a1 = Reshape(ckpt.value_of(ad + "1.weight"), AO, D);
  w.a1_b = ckpt.value_of(ad + "1.bias");
  w.a2 = Reshape(ckpt.value_of(ad + "3.weight"), AO, AO);
  w.a2_b = ckpt.value_of(ad + "3.bias");
  return w;
}

// THE WHOLE TOWER, in double precision, from `nvidia/audio_encoder.py:611-736`
// and `nvidia/audio.py:193-282` @ `9035151d6`. `apply_mask` and `rotate_all`
// are MUTATION HANDLES, not options: a case flips one and asserts the answer
// moves, which is how §4.14.8's mutations C and (the rope half of) the
// partial-RoPE risk are measured rather than argued.
ref_tower::Mat RefTower(const TowerRefWeights& w, const TinySpec& s,
                        const std::vector<std::vector<double>>& mel,
                        int64_t num_samples, int64_t num_tokens,
                        bool apply_mask = true, bool rotate_all = false,
                        bool k_has_bias = false) {
  using ref_tower::Mat;
  const std::size_t dhs = static_cast<std::size_t>(s.a_dhs);
  const std::size_t D = static_cast<std::size_t>(s.a_d_model);
  const std::size_t nh = static_cast<std::size_t>(s.a_heads);
  const std::size_t hd = static_cast<std::size_t>(s.a_head_dim());

  // ── the stem, `_conv2d_stem_one_chunk` (:535-562) ───────────────────────
  std::vector<Mat> x(1, mel);
  std::vector<std::int64_t> valid(4);
  valid[0] = num_samples / 160;
  valid[1] = (valid[0] + 1) / 2;
  valid[2] = (valid[1] + 1) / 2;
  valid[3] = (valid[2] + 1) / 2;
  if (apply_mask) ref_tower::MaskTime(&x, valid[0]);
  x = ref_tower::Conv2dGelu(x, w.conv1, w.conv1_b, dhs);
  if (apply_mask) ref_tower::MaskTime(&x, valid[1]);
  x = ref_tower::Conv2dGelu(x, w.conv2, w.conv2_b, dhs);
  if (apply_mask) ref_tower::MaskTime(&x, valid[2]);
  x = ref_tower::Conv2dGelu(x, w.conv3, w.conv3_b, dhs);
  if (apply_mask) ref_tower::MaskTime(&x, valid[3]);

  // `x.permute(0, 3, 1, 2).reshape(B, T, C * F)` (:579-580) — CHANNEL-major.
  const std::size_t Fo = x[0].size(), To = x[0][0].size();
  Mat rows(To, std::vector<double>(dhs * Fo));
  for (std::size_t t = 0; t < To; ++t)
    for (std::size_t c = 0; c < dhs; ++c)
      for (std::size_t f = 0; f < Fo; ++f)
        rows[t][c * Fo + f] = x[c][f][t];

  // `conv_out` (:581), then the varlen pack keeps the first `num_tokens`
  // (:679-681 with B == 1).
  Mat hidden = ref_tower::Linear(rows, w.conv_out, nullptr);
  hidden.resize(static_cast<std::size_t>(num_tokens));

  // ── the rope cache (:69-79, :126-131) ──────────────────────────────────
  const std::size_t rd =
      rotate_all ? hd : static_cast<std::size_t>(s.a_rotary_dim());
  const std::size_t nf = rd / 2;
  Mat cos_t(hidden.size(), std::vector<double>(nf));
  Mat sin_t(hidden.size(), std::vector<double>(nf));
  for (std::size_t t = 0; t < hidden.size(); ++t) {
    for (std::size_t i = 0; i < nf; ++i) {
      const double inv =
          1.0 / std::pow(s.a_rope_theta,
                         static_cast<double>(2 * i) / static_cast<double>(rd));
      cos_t[t][i] = std::cos(static_cast<double>(t) * inv);
      sin_t[t][i] = std::sin(static_cast<double>(t) * inv);
    }
  }

  for (std::size_t l = 0; l < w.layers.size(); ++l) {
    const TowerRefWeights::Layer& lw = w.layers[l];
    const Mat n1 = ref_tower::RmsNorm(hidden, lw.attn_norm, s.rms_eps);
    Mat q = ref_tower::Linear(n1, lw.q, &lw.qb);
    // NO BIAS ON K (`:221`), unless a mutation case asks for one.
    Mat k = ref_tower::Linear(n1, lw.k, k_has_bias ? &lw.qb : nullptr);
    const Mat v = ref_tower::Linear(n1, lw.v, &lw.vb);

    // `apply_rotary_pos_emb` (:146-181): rotate the LEADING `rd` dims of each
    // head, NeoX half-split, and leave the tail alone.
    const auto rope = [&](Mat& m) {
      for (std::size_t t = 0; t < m.size(); ++t) {
        for (std::size_t h = 0; h < nh; ++h) {
          const std::size_t base = h * hd;
          std::vector<double> rot(rd);
          for (std::size_t i = 0; i < rd; ++i) rot[i] = m[t][base + i];
          for (std::size_t i = 0; i < rd; ++i) {
            const double c = cos_t[t][i % nf];
            const double sn = sin_t[t][i % nf];
            const double other = i < nf ? -rot[i + nf] : rot[i - nf];
            m[t][base + i] = rot[i] * c + other * sn;
          }
        }
      }
    };
    rope(q);
    rope(k);

    // Full non-causal MHA per head, softmax in double.
    Mat attn(hidden.size(), std::vector<double>(D, 0.0));
    const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
    for (std::size_t h = 0; h < nh; ++h) {
      const std::size_t base = h * hd;
      for (std::size_t i = 0; i < q.size(); ++i) {
        std::vector<double> sc(k.size());
        double m = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < k.size(); ++j) {
          double dot = 0.0;
          for (std::size_t c = 0; c < hd; ++c)
            dot += q[i][base + c] * k[j][base + c];
          sc[j] = dot * scale;
          m = std::max(m, sc[j]);
        }
        double sum = 0.0;
        for (double& e : sc) {
          e = std::exp(e - m);
          sum += e;
        }
        for (std::size_t c = 0; c < hd; ++c) {
          double acc = 0.0;
          for (std::size_t j = 0; j < k.size(); ++j)
            acc += sc[j] / sum * v[j][base + c];
          attn[i][base + c] = acc;
        }
      }
    }
    const Mat proj = ref_tower::Linear(attn, lw.o, &lw.ob);
    for (std::size_t r = 0; r < hidden.size(); ++r)
      for (std::size_t c = 0; c < D; ++c) hidden[r][c] += proj[r][c];

    const Mat n2 = ref_tower::RmsNorm(hidden, lw.final_norm, s.rms_eps);
    const Mat f1 = ref_tower::Linear(n2, lw.fc1, &lw.fc1_b);
    const Mat act = ref_tower::SwiGlu(f1);
    const Mat f2 = ref_tower::Linear(act, lw.fc2, &lw.fc2_b);
    for (std::size_t r = 0; r < hidden.size(); ++r)
      for (std::size_t c = 0; c < D; ++c) hidden[r][c] += f2[r][c];
  }

  hidden = ref_tower::RmsNorm(hidden, w.final_norm, s.rms_eps);

  // `AudioAdapter` (`audio.py:240-248`): LayerNorm -> Linear -> GELU-erf ->
  // Linear.
  Mat a = ref_tower::LayerNorm(hidden, w.ln_w, w.ln_b, 1e-5);
  a = ref_tower::Linear(a, w.a1, &w.a1_b);
  for (auto& row : a)
    for (double& e : row) e = ref_tower::GeluErf(e);
  return ref_tower::Linear(a, w.a2, &w.a2_b);
}

// The relative L2 the row's other tower gates report.
double RelL2(const std::vector<float>& got, const ref_tower::Mat& want) {
  double num = 0.0, den = 0.0;
  std::size_t i = 0;
  for (const std::vector<double>& row : want) {
    for (double w : row) {
      const double d = static_cast<double>(got[i++]) - w;
      num += d * d;
      den += w * w;
    }
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Everything a tower case needs: the fixture checkpoint, the loaded weights and
// the mel the front end produced.
struct LoadedTower {
  TinySpec spec;
  TinyCheckpoint ckpt;
  vllm::HfConfig config;
  vllm::Dots3NoteAudioParams params;
  vllm::Dots3NoteAudioWeights weights;
  vllm::multimodal::AudioKwargs mel;
  std::vector<std::vector<double>> mel_ref;   // chunk 0's, as W7a had it
  // W7b (#2797): the clip the front end ran on, the segmentation the processor
  // derived, and ONE unpacked mel per chunk.
  std::vector<float> wav;
  std::vector<vllm::multimodal::Dots3NoteAudioProcessor::AudioChunk> chunks;
  std::vector<std::vector<std::vector<double>>> mel_refs;

  // `long_samples > 0` selects the MULTI-CHUNK clip (W7b, #2797) instead of the
  // 0.5 s one, and the caller is expected to have set `spec.a_chunk_seconds` to
  // a chunk that divides — see `kAudioLongChunkSeconds`.
  explicit LoadedTower(TinySpec s = AudioSpec(), int variant = 0,
                       int64_t long_samples = 0)
      : spec(s),
        ckpt(FixtureDir(), s),
        config(vllm::LoadHfConfig(ckpt.config_path())) {
    params = vllm::ParseDots3NoteAudioParams(config);
    REQUIRE(params.present);
    REQUIRE(vllm::Dots3NoteAudioRefusal(params, "", {}).empty());
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(ckpt.weights_path()));
    weights = vllm::MaterializeDots3NoteAudio(shards, params);
    REQUIRE(weights.present);

    vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
        vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                            "tiny");
    cfg.audio_token_id = dots3_tiny::kAudPadId;
    const vllm::multimodal::Dots3NoteAudioProcessor proc(cfg);
    wav = long_samples > 0
              ? dots3_tiny::FixtureAudioLongF32(variant, long_samples)
              : dots3_tiny::FixtureAudioF32(variant);
    chunks = proc.SegmentWaveform(static_cast<int64_t>(wav.size()));
    mel = proc.ProcessWaveform(wav.data(), static_cast<int64_t>(wav.size()),
                               16000);
    // The stacked mel, unpacked into one `[n_mels][n_frames]` per chunk. At one
    // chunk this is exactly what W7a built, and `mel_ref` below still names it.
    const std::size_t per =
        static_cast<std::size_t>(mel.n_mels) * static_cast<std::size_t>(mel.n_frames);
    for (int64_t c = 0; c < mel.num_chunks; ++c) {
      std::vector<std::vector<double>> one(
          static_cast<std::size_t>(mel.n_mels),
          std::vector<double>(static_cast<std::size_t>(mel.n_frames)));
      for (int64_t m = 0; m < mel.n_mels; ++m)
        for (int64_t f = 0; f < mel.n_frames; ++f)
          one[static_cast<std::size_t>(m)][static_cast<std::size_t>(f)] =
              mel.input_features[static_cast<std::size_t>(c) * per +
                                 static_cast<std::size_t>(m * mel.n_frames + f)];
      mel_refs.push_back(std::move(one));
    }
    REQUIRE(!mel_refs.empty());
    mel_ref = mel_refs[0];
  }

  std::vector<float> Run(vllm::Dots3NoteAudioCapture* cap = nullptr) const {
    return vllm::Dots3NoteAudioForward(mel.input_features, mel.num_samples,
                                       mel.num_tokens, /*hop_length=*/160,
                                       weights, params,
                                       vt::GetBackend(vt::DeviceType::kCPU), cap);
  }

  // THE PRODUCTION ENTRY POINT since W7b (#2797) — what `EncodeAudioDots3Note`
  // calls. At one chunk it is `Run` above with the same arguments.
  std::vector<float> RunChunks(
      std::vector<vllm::Dots3NoteAudioCapture>* caps = nullptr) const {
    return vllm::Dots3NoteAudioForwardChunks(
        mel.input_features, mel.chunk_num_samples, mel.chunk_num_tokens,
        /*hop_length=*/160, weights, params,
        vt::GetBackend(vt::DeviceType::kCPU), caps);
  }
};

}  // namespace

TEST_CASE("dots3-note W7a: the AUDIO tower agrees with an INDEPENDENT double reference") {
  const LoadedTower t;
  vllm::Dots3NoteAudioCapture cap;
  const std::vector<float> got = t.Run(&cap);

  REQUIRE(got.size() == static_cast<std::size_t>(dots3_tiny::kAudioTokens *
                                                 t.spec.a_adapter_out()));
  // The four mask lengths, in stage order — captured because a mask that halved
  // with the wrong rounding would still zero SOMETHING.
  REQUIRE(cap.valid_lens.size() == 4u);
  CHECK(cap.valid_lens[0] == dots3_tiny::kAudioSamples / 160);
  CHECK(cap.valid_lens[1] == 25);
  CHECK(cap.valid_lens[2] == 13);
  CHECK(cap.valid_lens[3] == dots3_tiny::kAudioTokens);
  MESSAGE("mask stages: " << cap.valid_lens[0] << " -> " << cap.valid_lens[1]
                          << " -> " << cap.valid_lens[2] << " -> "
                          << cap.valid_lens[3]
                          << ", against a padded mel of " << t.mel.n_frames
                          << " frames and a stem output of "
                          << dots3_tiny::kAudioStemFrames);

  const TowerRefWeights rw = ReadTowerWeights(t.ckpt, t.spec);
  const ref_tower::Mat want =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens);
  REQUIRE(want.size() == static_cast<std::size_t>(dots3_tiny::kAudioTokens));
  REQUIRE(want[0].size() == static_cast<std::size_t>(t.spec.a_adapter_out()));

  const double rel = RelL2(got, want);
  MESSAGE("tower vs reference rel-L2: " << rel);
  // The bf16 envelope of a 2-block tower plus a 3-layer conv stem, with the
  // deliberate `vt::RmsNorm` rounding difference `dots3_note_audio.h` records.
  CHECK(rel < 5e-2);

  // AND THE ANSWER IS NOT A CONSTANT. A tower replaced by a correctly-shaped
  // constant passes every SHAPE assertion above; it cannot pass this one.
  double lo = got[0], hi = got[0];
  for (float v : got) {
    lo = std::min(lo, static_cast<double>(v));
    hi = std::max(hi, static_cast<double>(v));
  }
  MESSAGE("tower output spans [" << lo << ", " << hi << "]");
  CHECK(hi - lo > 1e-3);
}

TEST_CASE("dots3-note W7a: the TEMPORAL MASK changes the answer, at every stage") {
  // §4.14.3. The mel of a zero-padded tail is the `-8` floor through
  // `(x + 4) / 4`, a NONZERO constant, so an unmasked stem leaks it through the
  // 3x3 receptive fields into the LAST VALID tokens. This case measures the
  // leak rather than arguing it: the same reference is run with the four mask
  // stages deleted, and the two answers are asserted to DIFFER.
  const LoadedTower t;
  const TowerRefWeights rw = ReadTowerWeights(t.ckpt, t.spec);
  const ref_tower::Mat masked =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens,
               /*apply_mask=*/true);
  const ref_tower::Mat unmasked =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens,
               /*apply_mask=*/false);

  double worst = 0.0;
  std::size_t worst_row = 0;
  for (std::size_t r = 0; r < masked.size(); ++r) {
    for (std::size_t c = 0; c < masked[r].size(); ++c) {
      const double d = std::fabs(masked[r][c] - unmasked[r][c]);
      if (d > worst) {
        worst = d;
        worst_row = r;
      }
    }
  }
  MESSAGE("deleting the mask moves the answer by up to " << worst
          << ", worst at token " << worst_row << " of " << masked.size());
  CHECK(worst > 1e-3);

  // AND IT REACHES THE LAST KEPT TOKEN, which is the claim that matters: a leak
  // that only touched tokens the span discards would be harmless.
  double last_row = 0.0;
  const std::size_t last = masked.size() - 1;
  for (std::size_t c = 0; c < masked[last].size(); ++c)
    last_row = std::max(last_row, std::fabs(masked[last][c] - unmasked[last][c]));
  MESSAGE("the leak reaches the LAST kept token by " << last_row);
  CHECK(last_row > 1e-3);

  // The IMPLEMENTATION is the masked one.
  const std::vector<float> got = t.Run();
  CHECK(RelL2(got, masked) < 5e-2);
  CHECK(RelL2(got, unmasked) > RelL2(got, masked));
}

TEST_CASE("dots3-note W7a: PARTIAL rope rotates HALF the head, and the tail is untouched") {
  const LoadedTower t;
  CHECK(t.params.head_dim() == t.spec.a_head_dim());
  CHECK(t.params.rotary_dim() == t.spec.a_rotary_dim());
  CHECK(t.params.rotary_dim() * 2 == t.params.head_dim());

  // The cache is [T, rotary_dim] = [cos(rd/2) | sin(rd/2)], and at t == 0 every
  // angle is zero — so the first row is all ones then all zeros. A cache built
  // over `head_dim` frequencies instead would be twice as wide.
  const std::vector<float> cache =
      vllm::Dots3NoteAudioRopeCache(dots3_tiny::kAudioTokens, t.params);
  REQUIRE(cache.size() == static_cast<std::size_t>(dots3_tiny::kAudioTokens *
                                                   t.params.rotary_dim()));
  const int64_t nf = t.params.rotary_dim() / 2;
  for (int64_t i = 0; i < nf; ++i) {
    CHECK(cache[static_cast<std::size_t>(i)] == doctest::Approx(1.0));
    CHECK(cache[static_cast<std::size_t>(nf + i)] == doctest::Approx(0.0));
  }
  // The DENOMINATOR is `rotary_dim`, not `head_dim`. At rotary_dim 4 the two
  // frequencies are theta^0 = 1 and theta^-0.5; over head_dim 8 the second
  // would be theta^-0.25, which is a different number.
  const double want_inv = 1.0 / std::pow(t.spec.a_rope_theta,
                                         2.0 / static_cast<double>(t.params.rotary_dim()));
  CHECK(cache[static_cast<std::size_t>(1 * t.params.rotary_dim() + 1)] ==
        doctest::Approx(std::cos(want_inv)).epsilon(1e-5));

  // ROTATING THE WHOLE HEAD moves the answer, which is what makes the partial
  // rotation an assertion rather than a description.
  const TowerRefWeights rw = ReadTowerWeights(t.ckpt, t.spec);
  const ref_tower::Mat partial =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens,
               /*apply_mask=*/true, /*rotate_all=*/false);
  const ref_tower::Mat full =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens,
               /*apply_mask=*/true, /*rotate_all=*/true);
  double worst = 0.0;
  for (std::size_t r = 0; r < partial.size(); ++r)
    for (std::size_t c = 0; c < partial[r].size(); ++c)
      worst = std::max(worst, std::fabs(partial[r][c] - full[r][c]));
  MESSAGE("rotating the whole head instead of half moves the answer by "
          << worst);
  CHECK(worst > 1e-3);
  const std::vector<float> got = t.Run();
  CHECK(RelL2(got, partial) < RelL2(got, full));
}

TEST_CASE("dots3-note W7a: `k_proj` has NO bias, and giving it one moves the answer") {
  // `nn.Linear(embed_dim, embed_dim, bias=False)` for k against `bias=bias`
  // (default True) for q, v and out (`audio_encoder.py:221-224`) — Whisper's
  // own convention and the obvious thing to get wrong. The checkpoint agrees:
  // 32 each of q/v/out bias and none for k.
  const LoadedTower t;
  const TowerRefWeights rw = ReadTowerWeights(t.ckpt, t.spec);
  const ref_tower::Mat no_k_bias =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens,
               /*apply_mask=*/true, /*rotate_all=*/false, /*k_has_bias=*/false);
  const ref_tower::Mat with_k_bias =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens,
               /*apply_mask=*/true, /*rotate_all=*/false, /*k_has_bias=*/true);
  double worst = 0.0;
  for (std::size_t r = 0; r < no_k_bias.size(); ++r)
    for (std::size_t c = 0; c < no_k_bias[r].size(); ++c)
      worst = std::max(worst, std::fabs(no_k_bias[r][c] - with_k_bias[r][c]));
  MESSAGE("adding a k bias moves the answer by " << worst);
  CHECK(worst > 1e-3);
  const std::vector<float> got = t.Run();
  CHECK(RelL2(got, no_k_bias) < RelL2(got, with_k_bias));

  // AND THE ENUMERATION AGREES: no `k_proj.bias` is claimed, so a checkpoint
  // shipping one would be UNACCOUNTED rather than silently loaded.
  const std::vector<vllm::Dots3NoteTensor> claimed =
      vllm::EnumerateDots3NoteAudioTensors(t.params);
  std::size_t k_bias = 0, q_bias = 0;
  for (const vllm::Dots3NoteTensor& c : claimed) {
    if (c.name.find("k_proj.bias") != std::string::npos) ++k_bias;
    if (c.name.find("q_proj.bias") != std::string::npos) ++q_bias;
  }
  CHECK(k_bias == 0u);
  CHECK(q_bias == static_cast<std::size_t>(t.spec.a_layers));
}

TEST_CASE("dots3-note W7a: two DIFFERENT waveforms give two different tower outputs") {
  const LoadedTower a(AudioSpec(), /*variant=*/0);
  const LoadedTower b(AudioSpec(), /*variant=*/1);
  const std::vector<float> ga = a.Run();
  const std::vector<float> gb = b.Run();
  REQUIRE(ga.size() == gb.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < ga.size(); ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(ga[i] - gb[i])));
  MESSAGE("two waveforms differ in the tower output by up to " << worst);
  CHECK(worst > 1e-3);
}


// ═══════════════════════════════════════════════════════════════════════════
// 3b. W7b (#2797) — THE CHUNK LOOP, and the four seams a norm cannot see.
//
// Spec §4.15. The geometry is `kAudioLongChunkSeconds` = 2 (32000 samples = 25
// token strides, which is what makes the tower's per-chunk sum equal the prompt
// side's single ceil) over an 80000-sample clip: THREE chunks of 32000, 32000
// and 16000, contributing 25, 25 and 13 rows. Three so a reversal is not a swap
// of two halves; a short last one so the truncation and the temporal mask on a
// partly padded chunk are both exercised.
// ═══════════════════════════════════════════════════════════════════════════
namespace {

TinySpec LongAudioSpec() {
  TinySpec s = AudioSpec();
  s.a_chunk_seconds = dots3_tiny::kAudioLongChunkSeconds;
  return s;
}

// The reference geometry for one config, from `ref_chunks` alone.
std::vector<ref_chunks::Segment> RefSegments(
    const vllm::multimodal::Dots3NoteAudioProcessorConfig& cfg,
    std::int64_t num_samples) {
  return ref_chunks::Segments(num_samples, cfg.chunk_seconds, cfg.sampling_rate,
                              cfg.hop_length, cfg.conv_temporal_stride,
                              cfg.merge_factor);
}

// One tower reference block per chunk, each at ITS OWN valid length and row
// count. `num_samples_override` drives the mask from a different length, which
// is how the "mask from the PADDED length" defect is measured rather than
// argued.
std::vector<ref_tower::Mat> RefChunkBlocks(
    const TowerRefWeights& rw, const LoadedTower& t,
    const std::vector<ref_chunks::Segment>& segs,
    std::int64_t num_samples_override = 0, std::int64_t token_len_override = 0) {
  std::vector<ref_tower::Mat> out;
  for (std::size_t i = 0; i < segs.size(); ++i) {
    const std::int64_t ns =
        num_samples_override > 0 ? num_samples_override : segs[i].length;
    const std::int64_t tl =
        token_len_override > 0 ? token_len_override : segs[i].token_len;
    out.push_back(RefTower(rw, t.spec, t.mel_refs[i], ns, tl));
  }
  return out;
}

// The worst |delta| between a block of the concatenated output and a reference
// block, starting at `row_offset`.
double WorstBlockDelta(const std::vector<float>& got, std::int64_t row_offset,
                       const ref_tower::Mat& want, std::int64_t width) {
  double worst = 0.0;
  for (std::size_t r = 0; r < want.size(); ++r) {
    for (std::size_t c = 0; c < want[r].size(); ++c) {
      const std::size_t i =
          static_cast<std::size_t>((row_offset + static_cast<std::int64_t>(r)) *
                                   width) + c;
      REQUIRE(i < got.size());
      worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[r][c]));
    }
  }
  return worst;
}

// Relative L2 of ONE row of the output against one reference row.
double RowRelL2(const std::vector<float>& got, std::int64_t row,
                const std::vector<double>& want, std::int64_t width) {
  double num = 0.0, den = 0.0;
  for (std::size_t c = 0; c < want.size(); ++c) {
    const double d =
        static_cast<double>(got[static_cast<std::size_t>(row * width) + c]) -
        want[c];
    num += d * d;
    den += want[c] * want[c];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

TEST_CASE("dots3-note W7b: the SEGMENTATION is upstream's slicing loop, and the counts balance") {
  // `SegmentWaveform` is the production seam (`audio.py:196-212` @ `9035151d6`)
  // and this drives it against `ref_chunks`, which transcribes upstream's
  // LITERAL `(length - 1) // stride + 1` rather than the implementation's
  // `ceil`. #2797 records that those are the same function for every length the
  // loop can emit, and this case is where that stops being an argument.
  const TinySpec spec = LongAudioSpec();
  const TinyCheckpoint ckpt(FixtureDir(), spec);
  vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                          "tiny");
  cfg.audio_token_id = dots3_tiny::kAudPadId;
  REQUIRE(cfg.chunk_seconds == dots3_tiny::kAudioLongChunkSeconds);
  REQUIRE(cfg.chunk_samples() == 32000);
  REQUIRE(cfg.token_stride() == 1280);
  // THE CONDITION THE WHOLE BRICK RESTS ON (§4.15.3): every segment but the
  // last is exactly `chunk_samples` long, so the tower's per-segment sum equals
  // the prompt side's one ceil for every waveform exactly when this is 0.
  REQUIRE(cfg.chunk_samples() % cfg.token_stride() == 0);
  const vllm::multimodal::Dots3NoteAudioProcessor proc(cfg);

  // Lengths chosen to walk the boundary: one sample, one stride, one sample
  // under and over a whole chunk, the gated clip, and an exact multiple.
  const std::vector<std::int64_t> lengths = {
      1, 1280, 31999, 32000, 32001, dots3_tiny::kAudioLongSamples, 96000};
  for (std::int64_t n : lengths) {
    const std::vector<ref_chunks::Segment> want = RefSegments(cfg, n);
    const std::vector<vllm::multimodal::Dots3NoteAudioProcessor::AudioChunk>
        got = proc.SegmentWaveform(n);
    INFO("num_samples=", n);
    REQUIRE(got.size() == want.size());
    std::int64_t summed = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      INFO("chunk ", i);
      CHECK(got[i].start == want[i].start);
      CHECK(got[i].length == want[i].length);
      CHECK(got[i].num_tokens == want[i].token_len);
      summed += got[i].num_tokens;
    }
    // The segments TILE the waveform: no sample is dropped and none is fed
    // twice. A slice whose stop or start was off by one chunk would still
    // produce well-formed segments.
    std::int64_t covered = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      CHECK(got[i].start == covered);
      covered += got[i].length;
    }
    CHECK(covered == n);
    // And the two upstream token counts agree: the per-segment sum
    // (`audio.py:129-147`) against the prompt side's one ceil
    // (`processor.py:771`), which is `NumAudioTokens` UNCHANGED.
    CHECK(summed == proc.NumAudioTokens(n));
  }

  // The gated geometry, written out so a reader can check the spec against the
  // fixture without running anything.
  const std::vector<vllm::multimodal::Dots3NoteAudioProcessor::AudioChunk> c =
      proc.SegmentWaveform(dots3_tiny::kAudioLongSamples);
  REQUIRE(c.size() == static_cast<std::size_t>(dots3_tiny::kAudioLongChunks));
  CHECK(c[0].length == 32000);
  CHECK(c[1].length == 32000);
  CHECK(c[2].length == 16000);  // SHORT, and 80000 is not a multiple of 32000
  CHECK(c[0].num_tokens == dots3_tiny::kAudioLongFullChunkTokens);
  CHECK(c[1].num_tokens == dots3_tiny::kAudioLongFullChunkTokens);
  CHECK(c[2].num_tokens == dots3_tiny::kAudioLongLastChunkTokens);
  CHECK(c[2].num_tokens < c[1].num_tokens);
  MESSAGE("80000 samples -> " << c.size() << " chunks of " << c[0].length << ", "
                              << c[1].length << ", " << c[2].length
                              << " samples and " << c[0].num_tokens << ", "
                              << c[1].num_tokens << ", " << c[2].num_tokens
                              << " rows");
  CHECK(proc.NumAudioTokens(dots3_tiny::kAudioLongSamples) ==
        dots3_tiny::kAudioLongTokens);
}

TEST_CASE("dots3-note W7b: the front end stacks ONE padded mel per chunk, from that chunk's OWN samples") {
  // The seam a wrong slice offset breaks. Every chunk's mel is asserted against
  // `ref_front::LogMel` OF THAT SEGMENT — so a loop that fed chunk 1 the same
  // samples as chunk 0, or that mel'd the whole waveform once and sliced the
  // result, fails here rather than in a norm over the tower output.
  const TinySpec spec = LongAudioSpec();
  const TinyCheckpoint ckpt(FixtureDir(), spec);
  vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                          "tiny");
  cfg.audio_token_id = dots3_tiny::kAudPadId;
  const vllm::multimodal::Dots3NoteAudioProcessor proc(cfg);

  const std::vector<float> wav = dots3_tiny::FixtureAudioLongF32(
      0, dots3_tiny::kAudioLongSamples);
  REQUIRE(wav.size() == static_cast<std::size_t>(dots3_tiny::kAudioLongSamples));
  const vllm::multimodal::AudioKwargs got = proc.ProcessWaveform(
      wav.data(), static_cast<std::int64_t>(wav.size()), 16000);

  CHECK(got.num_chunks == dots3_tiny::kAudioLongChunks);
  CHECK(got.n_mels == spec.a_mels);
  CHECK(got.n_frames == spec.a_chunk_mel_frames());
  CHECK(got.num_samples == dots3_tiny::kAudioLongSamples);
  CHECK(got.num_tokens == dots3_tiny::kAudioLongTokens);
  // `torch.stack` (`audio.py:220`): one PADDED mel per chunk, all the same
  // width, which is what makes the stack legal upstream.
  REQUIRE(got.input_features.size() ==
          static_cast<std::size_t>(got.num_chunks * got.n_mels * got.n_frames));
  REQUIRE(got.chunk_num_samples.size() ==
          static_cast<std::size_t>(dots3_tiny::kAudioLongChunks));
  REQUIRE(got.chunk_num_tokens.size() ==
          static_cast<std::size_t>(dots3_tiny::kAudioLongChunks));

  const std::vector<ref_chunks::Segment> segs =
      RefSegments(cfg, static_cast<std::int64_t>(wav.size()));
  REQUIRE(segs.size() == static_cast<std::size_t>(got.num_chunks));
  std::int64_t summed = 0;
  for (std::size_t i = 0; i < segs.size(); ++i) {
    CHECK(got.chunk_num_samples[i] == segs[i].length);
    CHECK(got.chunk_num_tokens[i] == segs[i].token_len);
    summed += got.chunk_num_tokens[i];
  }
  CHECK(summed == got.num_tokens);

  const std::size_t per = static_cast<std::size_t>(got.n_mels * got.n_frames);
  double worst_all = 0.0;
  for (std::size_t i = 0; i < segs.size(); ++i) {
    const std::vector<float> segment(
        wav.begin() + static_cast<std::ptrdiff_t>(segs[i].start),
        wav.begin() + static_cast<std::ptrdiff_t>(segs[i].start + segs[i].length));
    const std::vector<std::vector<double>> want = ref_front::LogMel(
        segment, static_cast<int>(spec.a_chunk_samples()), cfg.n_fft,
        cfg.hop_length, static_cast<int>(spec.a_mels), 16000);
    REQUIRE(want.size() == static_cast<std::size_t>(got.n_mels));
    REQUIRE(want[0].size() == static_cast<std::size_t>(got.n_frames));
    double worst = 0.0;
    for (std::int64_t m = 0; m < got.n_mels; ++m) {
      for (std::int64_t f = 0; f < got.n_frames; ++f) {
        const double a = got.input_features[i * per +
                                            static_cast<std::size_t>(m * got.n_frames + f)];
        const double b =
            want[static_cast<std::size_t>(m)][static_cast<std::size_t>(f)];
        worst = std::max(worst, std::fabs(a - b));
      }
    }
    INFO("chunk ", i);
    MESSAGE("chunk " << i << " mel vs reference: worst |delta| " << worst);
    CHECK(worst < 1e-5);
    worst_all = std::max(worst_all, worst);
  }
  MESSAGE("worst over all " << segs.size() << " chunks: " << worst_all);

  // AND THE CHUNKS ARE DIFFERENT MELS. If they were not, every ordering and
  // slicing assertion below would be measuring an identity. The clip sweeps, so
  // chunk 0 and chunk 1 disagree even though both are full.
  double chunk_delta = 0.0;
  for (std::size_t k = 0; k < per; ++k) {
    chunk_delta = std::max(
        chunk_delta, std::fabs(static_cast<double>(got.input_features[k]) -
                               static_cast<double>(got.input_features[per + k])));
  }
  MESSAGE("chunk 0 and chunk 1 differ in the mel by up to " << chunk_delta);
  CHECK(chunk_delta > 1e-2);
}

TEST_CASE("dots3-note W7b: the tower runs each chunk at ITS OWN mask length and concatenates IN ORDER") {
  // THE SEAM CASE. Four defects produce correctly-shaped output and each gets
  // its own assertion rather than a share of an aggregate norm (§4.15.5).
  const TinySpec spec = LongAudioSpec();
  const LoadedTower t(spec, /*variant=*/0, dots3_tiny::kAudioLongSamples);
  REQUIRE(t.mel.num_chunks == dots3_tiny::kAudioLongChunks);
  const std::int64_t width = t.spec.a_adapter_out();

  std::vector<vllm::Dots3NoteAudioCapture> caps;
  const std::vector<float> got = t.RunChunks(&caps);
  REQUIRE(got.size() ==
          static_cast<std::size_t>(dots3_tiny::kAudioLongTokens * width));

  // 1. THE MASK IS PER CHUNK. The two full chunks mask nothing at the last
  //    stage (25 valid of 25 stem rows); the SHORT one masks 13 of 25. A mask
  //    taken from the padded length would read 25 everywhere.
  REQUIRE(caps.size() == static_cast<std::size_t>(dots3_tiny::kAudioLongChunks));
  const std::vector<ref_chunks::Segment> segs = ref_chunks::Segments(
      dots3_tiny::kAudioLongSamples, dots3_tiny::kAudioLongChunkSeconds, 16000,
      160, 8, 1);
  REQUIRE(segs.size() == caps.size());
  for (std::size_t i = 0; i < caps.size(); ++i) {
    const std::vector<std::int64_t> want =
        ref_chunks::MaskStages(segs[i].length, 160);
    REQUIRE(caps[i].valid_lens.size() == 4u);
    INFO("chunk ", i);
    for (std::size_t k = 0; k < 4; ++k) CHECK(caps[i].valid_lens[k] == want[k]);
    MESSAGE("chunk " << i << " mask stages: " << caps[i].valid_lens[0] << " -> "
                     << caps[i].valid_lens[1] << " -> " << caps[i].valid_lens[2]
                     << " -> " << caps[i].valid_lens[3]);
  }
  CHECK(caps[0].valid_lens[3] == dots3_tiny::kAudioLongFullChunkTokens);
  CHECK(caps[2].valid_lens[3] == dots3_tiny::kAudioLongLastChunkTokens);
  CHECK(caps[2].valid_lens[3] < caps[0].valid_lens[3]);

  // 2. EACH CHUNK'S BLOCK IS THAT CHUNK'S ANSWER, at that chunk's row offset.
  const TowerRefWeights rw = ReadTowerWeights(t.ckpt, t.spec);
  const std::vector<ref_tower::Mat> want = RefChunkBlocks(rw, t, segs);
  REQUIRE(want.size() == segs.size());
  for (std::size_t i = 0; i < segs.size(); ++i) {
    INFO("chunk ", i);
    REQUIRE(want[i].size() == static_cast<std::size_t>(segs[i].token_len));
    REQUIRE(want[i][0].size() == static_cast<std::size_t>(width));
    const double worst =
        WorstBlockDelta(got, segs[i].row_offset, want[i], width);
    MESSAGE("chunk " << i << " (rows " << segs[i].row_offset << ".."
                     << segs[i].row_offset + segs[i].token_len - 1
                     << ") vs its own reference: worst |delta| " << worst);
    CHECK(worst < 5e-2);
  }

  // 3. THE BOUNDARIES ARE WHERE THE ROW COUNTS PUT THEM. An off-by-one in the
  //    per-chunk slice moves every later chunk by one row and leaves the shape
  //    intact when the total is fixed up; this compares the FIRST row of each
  //    chunk's block against that chunk's reference row 0 and against the
  //    PREVIOUS chunk's last reference row, and asserts the first is the near
  //    one.
  for (std::size_t i = 1; i < segs.size(); ++i) {
    const double at_seam =
        RowRelL2(got, segs[i].row_offset, want[i][0], width);
    const double if_shifted = RowRelL2(
        got, segs[i].row_offset, want[i - 1][want[i - 1].size() - 1], width);
    INFO("chunk ", i);
    MESSAGE("seam at row " << segs[i].row_offset << ": rel-L2 " << at_seam
                           << " against this chunk's row 0, " << if_shifted
                           << " against the previous chunk's last row");
    CHECK(at_seam < 5e-2);
    CHECK(if_shifted > at_seam);
  }

  // 4. THE ORDER. Every row of a reversed concatenation is a correct row, so
  //    only a comparison that knows WHICH chunk belongs WHERE can see it.
  std::vector<float> reversed;
  for (std::size_t i = segs.size(); i-- > 0;)
    for (const std::vector<double>& row : want[i])
      for (double v : row) reversed.push_back(static_cast<float>(v));
  REQUIRE(reversed.size() == got.size());
  std::vector<float> ordered;
  for (std::size_t i = 0; i < segs.size(); ++i)
    for (const std::vector<double>& row : want[i])
      for (double v : row) ordered.push_back(static_cast<float>(v));
  double ord = 0.0, rev = 0.0;
  for (std::size_t k = 0; k < got.size(); ++k) {
    ord = std::max(ord, std::fabs(static_cast<double>(got[k] - ordered[k])));
    rev = std::max(rev, std::fabs(static_cast<double>(got[k] - reversed[k])));
  }
  MESSAGE("worst |delta| against the ordered concatenation " << ord
          << ", against the reversed one " << rev);
  CHECK(ord < 5e-2);
  CHECK(rev > 10.0 * ord);

  // 5. THE SHORT CHUNK IS TRUNCATED BACK, and truncating is not the same as
  //    dropping rows off the end. The last chunk's 25 stem rows all enter
  //    attention when the slice is not applied, so its KEPT 13 rows change too:
  //    the reference is run at both token counts and the two are asserted to
  //    disagree on the rows the answer keeps.
  const std::vector<ref_tower::Mat> untruncated = RefChunkBlocks(
      rw, t, segs, /*num_samples_override=*/0,
      /*token_len_override=*/dots3_tiny::kAudioLongFullChunkTokens);
  const std::size_t last = segs.size() - 1;
  REQUIRE(untruncated[last].size() ==
          static_cast<std::size_t>(dots3_tiny::kAudioLongFullChunkTokens));
  double trunc_delta = 0.0;
  for (std::size_t r = 0; r < want[last].size(); ++r)
    for (std::size_t c = 0; c < want[last][r].size(); ++c)
      trunc_delta = std::max(
          trunc_delta, std::fabs(want[last][r][c] - untruncated[last][r][c]));
  MESSAGE("not truncating the short chunk moves its KEPT rows by up to "
          << trunc_delta << ", and would add "
          << (dots3_tiny::kAudioLongFullChunkTokens -
              dots3_tiny::kAudioLongLastChunkTokens)
          << " padding-derived rows");
  CHECK(trunc_delta > 1e-3);
  CHECK(WorstBlockDelta(got, segs[last].row_offset, want[last], width) < 5e-2);

  // 6. THE MASK COMES FROM THE VALID LENGTH, NOT THE PADDED ONE. Same shape,
  //    same row count, different numbers — measured on the short chunk, whose
  //    padded tail is half of it.
  const std::vector<ref_tower::Mat> padded_mask = RefChunkBlocks(
      rw, t, segs, /*num_samples_override=*/32000);
  double mask_delta = 0.0;
  for (std::size_t r = 0; r < want[last].size(); ++r)
    for (std::size_t c = 0; c < want[last][r].size(); ++c)
      mask_delta = std::max(
          mask_delta, std::fabs(want[last][r][c] - padded_mask[last][r][c]));
  MESSAGE("masking the short chunk from the PADDED length moves it by up to "
          << mask_delta);
  CHECK(mask_delta > 1e-3);
  // ...and the implementation is the one masked from the VALID length.
  CHECK(WorstBlockDelta(got, segs[last].row_offset, want[last], width) <
        WorstBlockDelta(got, segs[last].row_offset, padded_mask[last], width));

  // 7. AND THE ANSWER IS NOT A CONSTANT.
  double lo = got[0], hi = got[0];
  for (float v : got) {
    lo = std::min(lo, static_cast<double>(v));
    hi = std::max(hi, static_cast<double>(v));
  }
  MESSAGE("the concatenated output spans [" << lo << ", " << hi << "]");
  CHECK(hi - lo > 1e-3);
}

TEST_CASE("dots3-note W7b: ONE chunk takes W7a's path, byte for byte") {
  // The additivity claim, asserted rather than described: at `num_chunks == 1`
  // the chunked entry point calls the single-chunk tower with the same
  // arguments, so the two answers are IDENTICAL and not merely close. The
  // default fixture geometry is used deliberately — it is the one every W7a
  // case runs on.
  const LoadedTower t;
  REQUIRE(t.mel.num_chunks == 1);
  REQUIRE(t.mel.chunk_num_samples.size() == 1u);
  CHECK(t.mel.chunk_num_samples[0] == t.mel.num_samples);
  CHECK(t.mel.chunk_num_tokens[0] == t.mel.num_tokens);
  const std::vector<float> single = t.Run();
  const std::vector<float> chunked = t.RunChunks();
  REQUIRE(single.size() == chunked.size());
  std::size_t differing = 0;
  for (std::size_t i = 0; i < single.size(); ++i)
    if (single[i] != chunked[i]) ++differing;
  MESSAGE("one chunk through the chunked path: " << differing << " of "
                                                 << single.size()
                                                 << " values differ");
  CHECK(differing == 0u);
}

TEST_CASE("dots3-note W7b: the single-chunk tower REFUSES a whole STACK, which is what makes the chunk loop reachable") {
  // The invariant that makes the reachability mutation visible, gated so it is
  // not a mute switch. Handing `Dots3NoteAudioForward` the stacked mel of a
  // multi-chunk clip — which is exactly what the production call site looked
  // like before `Dots3NoteAudioForwardChunks` existed — produces a
  // CORRECTLY-SHAPED answer with the right row count off a mel of the wrong
  // width, so nothing downstream can tell. Upstream's own
  // `assert mel.shape[1] == self.chunk_mel_frames` (`audio.py:215` @
  // `9035151d6`) is the check that can, and this drives it.
  const TinySpec spec = LongAudioSpec();
  const LoadedTower t(spec, /*variant=*/0, dots3_tiny::kAudioLongSamples);
  REQUIRE(t.mel.num_chunks == dots3_tiny::kAudioLongChunks);
  REQUIRE(t.params.chunk_mel_frames() == spec.a_chunk_mel_frames());
  REQUIRE(static_cast<std::int64_t>(t.mel.input_features.size()) ==
          dots3_tiny::kAudioLongChunks * t.params.num_mel_bins *
              t.params.chunk_mel_frames());

  std::string msg;
  try {
    (void)vllm::Dots3NoteAudioForward(
        t.mel.input_features, t.mel.num_samples, t.mel.num_tokens,
        /*hop_length=*/160, t.weights, t.params,
        vt::GetBackend(vt::DeviceType::kCPU));
  } catch (const std::exception& e) {
    msg = e.what();
  }
  INFO("message: ", msg);
  MESSAGE("the stacked mel is refused with: " << msg);
  CHECK(msg.find("not ONE chunk") != std::string::npos);
  CHECK(msg.find("audio.py:215") != std::string::npos);
  CHECK(msg.find("Dots3NoteAudioForwardChunks") != std::string::npos);
  // ...and ONE chunk of that same stack is accepted, so the check is a width
  // check and not a refusal of the whole path.
  const std::size_t per = static_cast<std::size_t>(t.params.num_mel_bins *
                                                   t.params.chunk_mel_frames());
  const std::vector<float> one(t.mel.input_features.begin(),
                               t.mel.input_features.begin() +
                                   static_cast<std::ptrdiff_t>(per));
  const std::vector<float> rows = vllm::Dots3NoteAudioForward(
      one, t.mel.chunk_num_samples[0], t.mel.chunk_num_tokens[0],
      /*hop_length=*/160, t.weights, t.params,
      vt::GetBackend(vt::DeviceType::kCPU));
  CHECK(rows.size() == static_cast<std::size_t>(t.mel.chunk_num_tokens[0] *
                                                t.spec.a_adapter_out()));
}

TEST_CASE("dots3-note W7b: a geometry whose chunk is not a whole number of strides refuses a LONG clip, and serves a short one") {
  // §4.15.3. The tiny fixture's DEFAULT `chunk_seconds` = 1 is exactly such a
  // geometry — 16000 is 12.5 token strides — so this is not a hypothetical
  // config invented to be refused. Past one chunk upstream's own two token
  // counts disagree there, and upstream never compares them.
  const TinySpec spec = AudioSpec();
  const TinyCheckpoint ckpt(FixtureDir(), spec);
  vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                          "tiny");
  cfg.audio_token_id = dots3_tiny::kAudPadId;
  REQUIRE(cfg.chunk_samples() == 16000);
  REQUIRE(cfg.chunk_samples() % cfg.token_stride() != 0);
  const vllm::multimodal::Dots3NoteAudioProcessor proc(cfg);

  // The premise, measured: the two upstream expressions really do disagree
  // here, so the refusal is not decoration.
  const std::int64_t n = 40000;  // 2.5 chunks
  const std::vector<ref_chunks::Segment> segs = RefSegments(cfg, n);
  REQUIRE(segs.size() == 3u);
  MESSAGE("at chunk_seconds=1: the per-segment sum is "
          << ref_chunks::TotalTokens(segs) << " and ceil(total/stride) is "
          << proc.NumAudioTokens(n));
  CHECK(ref_chunks::TotalTokens(segs) == 33);
  CHECK(proc.NumAudioTokens(n) == 32);
  CHECK(ref_chunks::TotalTokens(segs) != proc.NumAudioTokens(n));

  SUBCASE("a waveform past one chunk refuses BY NAME, with the numbers") {
    const std::vector<float> wav = dots3_tiny::FixtureAudioLongF32(0, n);
    std::string msg;
    try {
      proc.ProcessWaveform(wav.data(), static_cast<std::int64_t>(wav.size()),
                           16000);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    INFO("message: ", msg);
    CHECK(msg.find("3 chunks") != std::string::npos);
    CHECK(msg.find("not a whole number of 1280") != std::string::npos);
    CHECK(msg.find("33 rows") != std::string::npos);
    CHECK(msg.find("span of 32") != std::string::npos);
    CHECK(msg.find("#2797") != std::string::npos);
  }
  SUBCASE("...and a waveform INSIDE one chunk is still served, because the sums agree there") {
    // The reason this is a per-request refusal and not an install-time one: a
    // one-segment sum IS `ceil(n / stride)`, so W7a's whole gate still passes
    // on this config.
    const std::vector<float> wav = dots3_tiny::FixtureAudioF32(0);
    const vllm::multimodal::AudioKwargs kw = proc.ProcessWaveform(
        wav.data(), static_cast<std::int64_t>(wav.size()), 16000);
    CHECK(kw.num_chunks == 1);
    CHECK(kw.num_tokens == dots3_tiny::kAudioTokens);
  }
  SUBCASE("...and the EXACT chunk boundary is served: one chunk, not two") {
    std::vector<float> exact(static_cast<std::size_t>(cfg.chunk_samples()), 0.1f);
    const vllm::multimodal::AudioKwargs kw = proc.ProcessWaveform(
        exact.data(), static_cast<std::int64_t>(exact.size()), 16000);
    CHECK(kw.num_chunks == 1);
    CHECK(kw.num_tokens == dots3_tiny::kAudioStemFrames);
  }
}

TEST_CASE("dots3-note W7b: two DIFFERENT long waveforms give two different concatenations") {
  const TinySpec spec = LongAudioSpec();
  const LoadedTower a(spec, /*variant=*/0, dots3_tiny::kAudioLongSamples);
  const LoadedTower b(spec, /*variant=*/1, dots3_tiny::kAudioLongSamples);
  const std::vector<float> ga = a.RunChunks();
  const std::vector<float> gb = b.RunChunks();
  REQUIRE(ga.size() == gb.size());
  REQUIRE(ga.size() == static_cast<std::size_t>(dots3_tiny::kAudioLongTokens *
                                                a.spec.a_adapter_out()));
  double worst = 0.0;
  for (std::size_t i = 0; i < ga.size(); ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(ga[i] - gb[i])));
  MESSAGE("two long waveforms differ in the concatenation by up to " << worst);
  CHECK(worst > 1e-3);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. THE ENUMERATION, and the RELEASED checkpoint's own 430.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("dots3-note W7a: the tower claims ALL 430 of the RELEASED tower's tensors") {
  // The released `audio_config`, read from the committed fixture. 32 layers x
  // 13 tensors = 416, plus 7 stem, 1 final norm and 6 adapter = 430 — the
  // number `bucket_totals` in `index_full.json` records and W2 asserted.
  const vllm::HfConfig cfg =
      vllm::LoadHfConfig(FixtureDir() + "/config.json");
  const vllm::Dots3NoteAudioParams a = vllm::ParseDots3NoteAudioParams(cfg);
  REQUIRE(a.present);
  CHECK(a.d_model == 1280);
  CHECK(a.num_heads == 20);
  CHECK(a.num_layers == 32);
  CHECK(a.ffn_dim == 5120);
  CHECK(a.num_mel_bins == 128);
  CHECK(a.max_source_positions == 6000);
  CHECK(a.downsample_hidden_size == 480);
  CHECK(a.adapter_in_dim == 1280);
  CHECK(a.adapter_out_dim == 5120);
  CHECK(a.head_dim() == 64);
  CHECK(a.rotary_dim() == 32);
  CHECK(a.freq_after() == 16);
  CHECK(a.conv_out_in_dim() == 7680);
  CHECK(a.fc1_out() == 10240);
  // The three MEASURED-DEAD knobs are READ and then never used.
  CHECK(a.conv_chunksize == 500);
  CHECK(a.conv_bucket_step == 10);
  CHECK(a.conv_bucket_max_elements == 20000);

  const std::vector<vllm::Dots3NoteTensor> claimed =
      vllm::EnumerateDots3NoteAudioTensors(a);
  MESSAGE("the released audio tower claims " << claimed.size() << " tensors");
  CHECK(claimed.size() == 430u);
  for (const vllm::Dots3NoteTensor& t : claimed) {
    CHECK(t.name.rfind("audio_encoder.", 0) == 0);
    CHECK_FALSE(t.consumer.empty());
  }
  // NO `conv_out.bias` and NO `k_proj.bias`: both are `bias=False` upstream and
  // both are absent from the released checkpoint. Claiming either would refuse
  // every real load.
  for (const vllm::Dots3NoteTensor& t : claimed) {
    CHECK(t.name.find("conv_out.bias") == std::string::npos);
    CHECK(t.name.find("k_proj.bias") == std::string::npos);
  }
  // And the released tower is ACCEPTED — the whole point of the brick.
  CHECK(vllm::Dots3NoteAudioRefusal(a, "", {}).empty());
}

// ── W7c-1 (#2813): the CHANNEL reduction, at the decoder ────────────────────
//
// Upstream reduces to mono with a plain mean over the channel axes, in two
// independent places and on both of its decode backends:
// `vllm/multimodal/media/audio.py:207-208` (soundfile arm) and `:168-169`
// (PyAV arm) @ `9035151d6`, reached because `load_audio`'s `mono` default is
// True (`:220`); and again on the parser side, where
// `AudioSpec.target_channels` is 1 and `channel_reduction` is
// `ChannelReduction.MEAN` (`vllm/multimodal/audio.py:69-70`, applied at
// `:150-152`), which dots3-note selects at
// `vllm/models/dots3_note/common/processor.py:523-525`.
//
// WHAT THIS CASE MEASURES THAT THE SERVED SUITE CANNOT. The served case proves
// a stereo request reaches the model and gets the MEAN's answer. This one
// drives channel counts no WAV upload sends — 3, 4, 8 — against a reference
// computed in `long double` from the raw int16, and pins the two exactness
// claims spec 4.16.2 makes about the intermediate type.
TEST_CASE("dots3-note W7c-1: the WAV decoder reduces channels by upstream's MEAN") {
  using vllm::multimodal::DecodeWavPcm16MeanToMono;
  using vllm::multimodal::DecodeWavPcm16Mono;

  // An independent reference: the mean in `long double`, from the raw int16.
  // It shares nothing with `src/` — not the accumulator type, not the divisor
  // order, not the parser.
  const auto ref_mean = [](const std::vector<int16_t>& interleaved,
                           int channels) {
    const size_t frames = interleaved.size() / static_cast<size_t>(channels);
    std::vector<long double> out(frames);
    for (size_t f = 0; f < frames; ++f) {
      long double acc = 0.0L;
      for (int c = 0; c < channels; ++c)
        acc += static_cast<long double>(
                   interleaved[f * static_cast<size_t>(channels) +
                               static_cast<size_t>(c)]) /
               32768.0L;
      out[f] = acc / static_cast<long double>(channels);
    }
    return out;
  };

  // A deterministic multi-channel signal: channel c is a different waveform,
  // never a scaled copy, so a decoder that took ONE channel, or averaged the
  // wrong axis, cannot land on the mean by accident.
  const auto make = [](int channels, size_t frames) {
    std::vector<int16_t> pcm(frames * static_cast<size_t>(channels));
    for (size_t f = 0; f < frames; ++f) {
      for (int c = 0; c < channels; ++c) {
        const double t = static_cast<double>(f) / 16000.0;
        const double v =
            0.5 * std::sin(2.0 * 3.14159265358979323846 *
                           (110.0 * (c + 1) + 37.0 * c) * t) +
            0.2 * std::cos(2.0 * 3.14159265358979323846 * (900.0 + 313.0 * c) * t);
        pcm[f * static_cast<size_t>(channels) + static_cast<size_t>(c)] =
            static_cast<int16_t>(std::lround(v * 30000.0));
      }
    }
    return pcm;
  };

  SUBCASE("the mean matches a long-double reference at 1, 2, 3, 4 and 8 channels") {
    for (const int channels : {1, 2, 3, 4, 8}) {
      const std::vector<int16_t> pcm = make(channels, 517);
      const std::vector<uint8_t> wav =
          dots3_tiny::FixtureWavFromPcm16(pcm, 16000, channels);
      const vllm::multimodal::DecodedAudio got =
          DecodeWavPcm16MeanToMono(wav.data(), wav.size());
      const std::vector<long double> want = ref_mean(pcm, channels);
      REQUIRE(got.sampling_rate == 16000);
      REQUIRE(got.samples.size() == want.size());
      long double worst = 0.0L;
      size_t exact = 0;
      for (size_t i = 0; i < want.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<long double>(
                                              got.samples[i]) - want[i]));
        if (static_cast<long double>(got.samples[i]) == want[i]) ++exact;
      }
      MESSAGE("channels=" << channels << ": worst |got-ref| = "
              << static_cast<double>(worst) << ", exact on " << exact << "/"
              << want.size());
      // One float ulp near 1.0 is 6e-8; this is the mean of values in [-1, 1].
      CHECK(static_cast<double>(worst) < 6e-8);
      // Spec 4.16.2: for a POWER-OF-TWO channel count UP TO 512 the port's
      // answer is the exact mean, with no rounding anywhere, because the
      // divisor is a power of two and |acc| <= 2^(15 + log2(C)) <= 2^24 is
      // still an exact float32. The bound is TIGHT and the next case gates it.
      if (channels == 1 || channels == 2 || channels == 4 || channels == 8) {
        CHECK(exact == want.size());
      }
    }
  }

  SUBCASE("at 1024 channels a float32 accumulator is WRONG, and at 512 it is not") {
    // WHY THIS CASE EXISTS. 4.16.2's intermediate-type argument was DERIVED and
    // measured against a long-double reference, and NOT gated: a fresh review
    // replaced the int32 accumulator plus single double divide with the float32
    // accumulator that section argues against, and both suites stayed green. So
    // nothing in the change could tell the shipped decoder from the one the
    // spec rejects. This case is that discriminator, placed exactly at the
    // bound the spec now states.
    //
    // The channel pattern is closed form and near full scale, so |acc| runs at
    // the `32768 * C` ceiling the derivation reasons about. At C = 512 the
    // exact sum is at most 2^24 and every arm agrees to the bit; at C = 1024 it
    // reaches 2^25, a float32 accumulator's PARTIAL sums stop being exact, and
    // it lands ~126 ulps away while this decoder stays exact.
    const auto pattern = [](int channels, size_t frames) {
      std::vector<int16_t> pcm(frames * static_cast<size_t>(channels));
      for (size_t f = 0; f < frames; ++f)
        for (int c = 0; c < channels; ++c)
          pcm[f * static_cast<size_t>(channels) + static_cast<size_t>(c)] =
              static_cast<int16_t>(32000 +
                                   ((c * 7 + static_cast<int>(f) * 37) % 768));
      return pcm;
    };
    // The rejected arm, written out: accumulate `s / 32768` in float32, divide
    // by the channel count in float32. This is the mutation, kept in the test
    // so the case cannot silently stop discriminating.
    const auto f32_accumulator = [](const std::vector<int16_t>& pcm,
                                    int channels) {
      const size_t frames = pcm.size() / static_cast<size_t>(channels);
      std::vector<float> out(frames);
      for (size_t f = 0; f < frames; ++f) {
        float acc = 0.0f;
        for (int c = 0; c < channels; ++c)
          acc += static_cast<float>(
                     pcm[f * static_cast<size_t>(channels) +
                         static_cast<size_t>(c)]) /
                 32768.0f;
        out[f] = acc / static_cast<float>(channels);
      }
      return out;
    };

    for (const int channels : {512, 1024}) {
      const std::vector<int16_t> pcm = pattern(channels, 4);
      const std::vector<uint8_t> wav =
          dots3_tiny::FixtureWavFromPcm16(pcm, 16000, channels);
      const vllm::multimodal::DecodedAudio got =
          DecodeWavPcm16MeanToMono(wav.data(), wav.size());
      const std::vector<long double> want = ref_mean(pcm, channels);
      const std::vector<float> rejected = f32_accumulator(pcm, channels);
      REQUIRE(got.samples.size() == 4);
      REQUIRE(want.size() == 4);

      size_t exact = 0, differs_from_f32 = 0;
      double worst_f32_gap = 0.0;
      for (size_t i = 0; i < want.size(); ++i) {
        // Bit equality against the long-double reference, not a tolerance:
        // both channel counts are inside "the answer is the exact mean".
        if (static_cast<long double>(got.samples[i]) == want[i]) ++exact;
        if (std::memcmp(&got.samples[i], &rejected[i], sizeof(float)) != 0)
          ++differs_from_f32;
        worst_f32_gap = std::max(
            worst_f32_gap,
            std::fabs(static_cast<double>(rejected[i]) -
                      static_cast<double>(want[i])));
      }
      MESSAGE("channels=" << channels << ": exact on " << exact << "/"
              << want.size() << ", differs from the float32 accumulator on "
              << differs_from_f32 << "/" << want.size()
              << ", that accumulator's worst error = " << worst_f32_gap);

      // This decoder is EXACT at both counts. Under a float32 accumulator it is
      // not, at 1024, which is what makes this the intermediate type's gate.
      CHECK(exact == want.size());
      if (channels == 512) {
        // At the bound the two arms are indistinguishable, so the case is not
        // measuring "float32 is always wrong" -- it is measuring WHERE.
        CHECK(differs_from_f32 == 0);
        CHECK(worst_f32_gap == 0.0);
      } else {
        CHECK(differs_from_f32 == want.size());
        CHECK(worst_f32_gap > 1e-6);
      }
    }
  }

  SUBCASE("a ONE-channel file decodes BIT for BIT the same through both entry points") {
    // This is what lets `DecodeWavPcm16Mono` keep its own loop untouched:
    // `parakeet_transcription.cpp`, `chat_mm.cpp` and `test_voxtral_e2e.cpp`
    // call it, and none of them may move by a bit.
    const std::vector<uint8_t> wav =
        dots3_tiny::FixtureAudioWav(0, 16000, /*channels=*/1);
    const vllm::multimodal::DecodedAudio a =
        DecodeWavPcm16Mono(wav.data(), wav.size());
    const vllm::multimodal::DecodedAudio b =
        DecodeWavPcm16MeanToMono(wav.data(), wav.size());
    REQUIRE(a.samples.size() == b.samples.size());
    REQUIRE(!a.samples.empty());
    size_t differ = 0;
    for (size_t i = 0; i < a.samples.size(); ++i)
      if (std::memcmp(&a.samples[i], &b.samples[i], sizeof(float)) != 0)
        ++differ;
    CHECK(differ == 0);
  }

  SUBCASE("a STEREO file whose channels are EQUAL decodes to that channel, bit for bit") {
    // The cross-check between the two loops at C = 2: the mean branch's double
    // arithmetic lands on exactly what the mono branch's float arithmetic does.
    const std::vector<int16_t> m = dots3_tiny::FixtureAudioPcm16(0);
    std::vector<int16_t> dup(m.size() * 2);
    for (size_t i = 0; i < m.size(); ++i) {
      dup[2 * i] = m[i];
      dup[2 * i + 1] = m[i];
    }
    const std::vector<uint8_t> stereo =
        dots3_tiny::FixtureWavFromPcm16(dup, 16000, 2);
    const std::vector<uint8_t> mono = dots3_tiny::FixtureWavFromPcm16(m);
    const vllm::multimodal::DecodedAudio a =
        DecodeWavPcm16Mono(mono.data(), mono.size());
    const vllm::multimodal::DecodedAudio b =
        DecodeWavPcm16MeanToMono(stereo.data(), stereo.size());
    REQUIRE(a.samples.size() == b.samples.size());
    size_t differ = 0;
    for (size_t i = 0; i < a.samples.size(); ++i)
      if (std::memcmp(&a.samples[i], &b.samples[i], sizeof(float)) != 0)
        ++differ;
    CHECK(differ == 0);
  }

  SUBCASE("the FIXTURE stereo clip means back to variant 0, exactly") {
    // The same construction the served case uses, checked at the decoder so a
    // failure separates "the decoder is wrong" from "the seam does not call
    // it".
    const std::vector<uint8_t> wav = dots3_tiny::FixtureAudioWavStereo();
    const vllm::multimodal::DecodedAudio got =
        DecodeWavPcm16MeanToMono(wav.data(), wav.size());
    const std::vector<int16_t> m = dots3_tiny::FixtureAudioPcm16(0);
    REQUIRE(got.samples.size() == m.size());
    size_t differ = 0;
    for (size_t i = 0; i < m.size(); ++i) {
      const float want = static_cast<float>(m[i]) / 32768.0f;
      if (std::memcmp(&got.samples[i], &want, sizeof(float)) != 0) ++differ;
    }
    CHECK(differ == 0);
  }

  SUBCASE("a TRAILING PARTIAL FRAME is dropped, as libsndfile drops it") {
    // Not a refusal: libsndfile reads whole frames and ignores a short tail, so
    // refusing here would be STRICTER than the oracle. The mono path has always
    // truncated (`data_len / 2`); this keeps the multi-channel path the same.
    // `FixtureWavFromPcm16` writes the canonical 44-byte header, so the `data`
    // size lives at offset 40 and the RIFF size at offset 4.
    std::vector<uint8_t> wav =
        dots3_tiny::FixtureWavFromPcm16(make(2, 10), 16000, 2);
    REQUIRE(wav.size() == 44 + 40);  // 10 stereo frames = 20 int16
    wav.push_back(0x11);             // one ORPHAN int16, half a frame
    wav.push_back(0x22);
    const auto put_u32 = [&wav](size_t at, uint32_t v) {
      for (int i = 0; i < 4; ++i)
        wav[at + static_cast<size_t>(i)] =
            static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
    };
    put_u32(40, 42);       // data bytes: 21 int16 = 10 frames + one orphan
    put_u32(4, 36 + 42);   // RIFF size
    const vllm::multimodal::DecodedAudio got =
        DecodeWavPcm16MeanToMono(wav.data(), wav.size());
    CHECK(got.samples.size() == 10);
  }

  SUBCASE("what is STILL refused, by name") {
    const auto why = [](const std::vector<uint8_t>& w) {
      std::string msg;
      try {
        DecodeWavPcm16MeanToMono(w.data(), w.size());
      } catch (const std::exception& e) {
        msg = e.what();
      }
      return msg;
    };
    // Non-16-bit PCM: W7c-2, alongside the rate arm.
    std::vector<uint8_t> bits24 = dots3_tiny::FixtureAudioWav(0, 16000, 2);
    bits24[34] = 24;
    CHECK(why(bits24).find("16-bit") != std::string::npos);
    // A non-PCM `fmt `.
    std::vector<uint8_t> adpcm = dots3_tiny::FixtureAudioWav(0, 16000, 2);
    adpcm[20] = 2;
    CHECK(why(adpcm).find("not PCM") != std::string::npos);
    // A zero channel count is malformed, not an arm.
    std::vector<uint8_t> zeroch = dots3_tiny::FixtureAudioWav(0, 16000, 2);
    zeroch[22] = 0;
    zeroch[23] = 0;
    CHECK(why(zeroch).find("channel count") != std::string::npos);
    // Not a RIFF/WAVE buffer at all.
    const std::vector<uint8_t> junk(64, 0x41);
    CHECK(why(junk).find("RIFF/WAVE") != std::string::npos);
    // ...and the MONO entry point still refuses a stereo file, so nothing that
    // calls it silently changed meaning.
    std::string mono_msg;
    try {
      const std::vector<uint8_t> st = dots3_tiny::FixtureAudioWavStereo();
      DecodeWavPcm16Mono(st.data(), st.size());
    } catch (const std::exception& e) {
      mono_msg = e.what();
    }
    CHECK(mono_msg.find("not mono") != std::string::npos);
  }
}

TEST_CASE("dots3-note W7a: every unported audio arm refuses BY NAME, with its brick") {
  const vllm::HfConfig base =
      vllm::LoadHfConfig(FixtureDir() + "/config.json");
  const auto refuse = [&](const std::function<void(nlohmann::json&)>& edit) {
    nlohmann::json raw = base.raw;
    edit(raw["audio_config"]);
    vllm::HfConfig c = base;
    c.raw = raw;
    return vllm::Dots3NoteAudioRefusal(vllm::ParseDots3NoteAudioParams(c), "", {});
  };

  SUBCASE("the RELEASED config is accepted — the premise of every case below") {
    CHECK(refuse([](nlohmann::json&) {}).empty());
  }
  SUBCASE("a BLOCKWISE-QUANTIZED checkpoint is W9, and outranks everything") {
    const std::string why = vllm::Dots3NoteAudioRefusal(
        vllm::ParseDots3NoteAudioParams(base), "fp8", {128, 128});
    CHECK(why.find("W9") != std::string::npos);
    CHECK(why.find("BLOCKWISE") != std::string::npos);
    // ...and it says the released tower is NOT one, so a reader does not
    // conclude the checkpoint they have is refused.
    CHECK(why.find("430") != std::string::npos);
  }
  SUBCASE("`encoder_type` other than 'dots' — upstream refuses it too") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["encoder_type"] = "whisper"; });
    CHECK(why.find("encoder_type") != std::string::npos);
    CHECK(why.find("audio.py:255-256") != std::string::npos);
  }
  SUBCASE("a non-swiglu activation is a DIFFERENT state dict, not a swap") {
    const std::string why = refuse([](nlohmann::json& a) {
      a["whisper_config"]["activation_function"] = "gelu";
    });
    CHECK(why.find("swiglu") != std::string::npos);
    CHECK(why.find("state dict") != std::string::npos);
  }
  SUBCASE("`use_conv2d_stem` false selects the conv1d stem") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["use_conv2d_stem"] = false; });
    CHECK(why.find("use_conv2d_stem") != std::string::npos);
    CHECK(why.find("1280") != std::string::npos);  // the stride it changes
  }
  SUBCASE("`use_latent_input` selects the latent stem, and OUTRANKS the conv1d arm") {
    // Both flags at once, because `use_latent_input` alone is REFUSED AT PARSE:
    // `use_conv2d_stem and use_latent_input` is mutually exclusive upstream
    // (`audio_encoder.py:450-453`) and this port mirrors that VT_CHECK. The
    // refusal order is upstream's constructor order — latent first, then the
    // stem selection — so a latent config is told about the latent stem rather
    // than about the conv1d one it also implies.
    const std::string why = refuse(
        [](nlohmann::json& a) { a["whisper_config"]["use_latent_input"] = true;
                                a["use_conv2d_stem"] = false; });
    CHECK(why.find("use_latent_input") != std::string::npos);
    CHECK(why.find("LATENT stem") != std::string::npos);
  }
  SUBCASE("...and `use_latent_input` WITH the conv2d stem refuses at PARSE") {
    nlohmann::json raw = base.raw;
    raw["audio_config"]["whisper_config"]["use_latent_input"] = true;
    vllm::HfConfig c = base;
    c.raw = raw;
    std::string msg;
    try {
      vllm::ParseDots3NoteAudioParams(c);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("mutually exclusive") != std::string::npos);
  }
  SUBCASE("`use_causal` changes THREE things at once") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["use_causal"] = true; });
    CHECK(why.find("use_causal") != std::string::npos);
    CHECK(why.find("THREE") != std::string::npos);
  }
  SUBCASE("`use_rms_norm` false is a different state dict") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["use_rms_norm"] = false; });
    CHECK(why.find("use_rms_norm") != std::string::npos);
    // A LayerNorm ships a BIAS none of the 65 norms in the released checkpoint
    // carries, which is what makes this a state-dict change and not a formula
    // one.
    CHECK(why.find("BIAS") != std::string::npos);
    CHECK(why.find("65 norms") != std::string::npos);
  }
  SUBCASE("`use_rope` false needs a positional table nothing ships") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["use_rope"] = false; });
    CHECK(why.find("use_rope") != std::string::npos);
    CHECK(why.find("embed_positions") != std::string::npos);
  }
  SUBCASE("`merge_factor != 1` changes the adapter's INPUT width") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["merge_factor"] = 2; });
    CHECK(why.find("merge_factor") != std::string::npos);
    CHECK(why.find("adapter") != std::string::npos);
  }
  SUBCASE("an adapter that does not land in the TEXT hidden space refuses") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["whisper_adapter_out_dim"] = 4096; });
    CHECK(why.find("whisper_adapter_out_dim") != std::string::npos);
    CHECK(why.find("EncodeMmDots3NoteForCausalLM") != std::string::npos);
  }
  SUBCASE("a checkpoint with NO audio_config refuses, naming the absence") {
    nlohmann::json raw = base.raw;
    raw.erase("audio_config");
    vllm::HfConfig c = base;
    c.raw = raw;
    const vllm::Dots3NoteAudioParams a = vllm::ParseDots3NoteAudioParams(c);
    CHECK_FALSE(a.present);
    const std::string why = vllm::Dots3NoteAudioRefusal(a, "", {});
    CHECK(why.find("no `audio_config`") != std::string::npos);
    CHECK(why.find("multimodal.py:119-126") != std::string::npos);
    // And the enumeration claims NOTHING, so the 430 stay in the deferral
    // bucket rather than becoming missing tensors.
    CHECK(vllm::EnumerateDots3NoteAudioTensors(a).empty());
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. THE RESAMPLER (W7c-2, #2828).
// ═══════════════════════════════════════════════════════════════════════════

// THE ORACLE HERE IS scipy, AND IT IS NOT UPSTREAM'S DEFAULT. Upstream's
// `AudioResampler` defaults to `"pyav"`, which is libswresample, which is not
// bit-identical to ITSELF across CPU dispatch on one binary and one input
// (ffmpeg 6.1.1: 24691 of 32000 samples differ, worst 9.686e-08). A bit-exact
// gate against it is impossible in principle. What is gated is
// `resample_audio_scipy` — another arm of upstream's own switch
// (`vllm/multimodal/audio.py:232-250`, dispatch at `:305-316` @ `9035151d6`),
// which vLLM ships in production for phi4mm (`phi4mm.py:580`).
//
// So this is a CONSISTENCY gate against a stated algorithm and NOT parity with
// upstream's default. Spec §4.17.7 says what that does and does not establish.
//
// A TOLERANCE ALONE WOULD GATE NOTHING, because a resampler that returns its
// input, or returns zeros, passes one. FIVE assertions per case, each naming
// the defect it excludes, and the fifth golden case carries a tone ABOVE the
// output Nyquist because that is the only content that separates a real
// anti-alias filter from picking samples.
TEST_CASE("dots3-note W7c-2: the resampler reproduces scipy's `resample_poly`") {
  REQUIRE(dots3_resample_golden::kNumCases == 5);
  double worst_all = 0.0;
  double worst_ref_all = 0.0;
  for (int c = 0; c < dots3_resample_golden::kNumCases; ++c) {
    const dots3_resample_golden::Case& k = dots3_resample_golden::kCases[c];
    // NOT `CAPTURE(k.name)`: doctest stringifies a `const char*` through its
    // bool conversion, so the case name would read as "true".
    INFO("case: ", std::string(k.name));

    const std::vector<float> got = vllm::multimodal::ResampleAudioScipy(
        k.input, k.n_in, k.orig_sr, k.target_sr);

    // (1) LENGTH: `ceil(n_in * target / orig)`, which is upstream's
    // `n_out = n_in * up; n_out = n_out // down + bool(n_out % down)`. This is
    // what a resampler that RETURNED ITS INPUT cannot pass: at 8000 -> 16000 it
    // would be 80 long and this is 160.
    const std::int64_t expect_len =
        (static_cast<std::int64_t>(k.n_in) * k.target_sr + k.orig_sr - 1) /
        k.orig_sr;
    CHECK(static_cast<std::int64_t>(got.size()) == expect_len);
    CHECK(static_cast<std::int64_t>(k.n_out) == expect_len);
    REQUIRE(static_cast<int>(got.size()) == k.n_out);

    // (2) A LOWER BOUND. A resampler that returned zeros, or one whose filter
    // normalization collapsed, passes any tolerance against a golden it is
    // being compared to only in absolute terms. The fixture signals reach 0.5
    // and 0.99, so 0.2 is far under the truth and far over zero.
    double peak = 0.0;
    for (float v : got) peak = std::max(peak, static_cast<double>(std::fabs(v)));
    CHECK(peak > 0.2);

    // (3) THE VALUES, against scipy's own output.
    double worst = 0.0;
    for (int i = 0; i < k.n_out; ++i)
      worst = std::max(worst, static_cast<double>(std::fabs(
                                  got[static_cast<std::size_t>(i)] - k.expected[i])));
    worst_all = std::max(worst_all, worst);
    MESSAGE(std::string(k.name) << ": " << k.orig_sr << " -> " << k.target_sr << " Hz, "
                   << k.n_in << " in, " << got.size()
                   << " out, peak " << peak << ", worst |got - scipy| " << worst);
    // The tolerance is ~2x the measured float floor and is derived in spec
    // §4.17.11 from this suite's own numbers. It is NOT a "close enough" band:
    // the port and scipy agree to ~1e-15 in double, and everything left is the
    // single narrowing store to `float`.
    CHECK(worst <= kResampleTol);

    // (4) THE DIFFERENCE ASSERTION, which is what makes (3) mean anything. A
    // nearest-sample decimation — mutation M3, a "resampler" with no
    // anti-alias filter — is computed HERE and must be FAR from the golden.
    // Without this, a tolerance of 6e-7 could be one that nothing can fail.
    std::vector<double> xin(static_cast<std::size_t>(k.n_in));
    for (int i = 0; i < k.n_in; ++i)
      xin[static_cast<std::size_t>(i)] = static_cast<double>(k.input[i]);
    const std::vector<double> naive =
        ref_resample::NearestSample(xin, k.orig_sr, k.target_sr);
    REQUIRE(naive.size() == got.size());
    double sep = 0.0;
    for (std::size_t i = 0; i < naive.size(); ++i)
      sep = std::max(sep, std::fabs(naive[i] - static_cast<double>(
                                                   k.expected[i])));
    MESSAGE(std::string(k.name) << ": nearest-sample decimation is " << sep
                   << " from the golden, against the value tolerance");
    CHECK(sep > 1e-2);

    // (5) AND THE INDEPENDENT REFERENCE, written from scipy's source a second
    // time and sharing no helper with `src/`.
    const std::vector<double> ref =
        ref_resample::ResamplePoly(xin, k.orig_sr, k.target_sr);
    REQUIRE(ref.size() == got.size());
    double worst_ref = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i)
      worst_ref = std::max(worst_ref,
                           std::fabs(ref[i] - static_cast<double>(
                                                  got[i])));
    worst_ref_all = std::max(worst_ref_all, worst_ref);
    CHECK(worst_ref <= kResampleTol);
  }
  MESSAGE("worst over all cases: |ours - scipy| " << worst_all
          << ", |ours - ref_resample| " << worst_ref_all);
}

TEST_CASE("dots3-note W7c-2: the resample seam is a NO-OP at the target rate") {
  // `if orig_sr_int == target_sr_int: return audio` (`audio.py:241-242`). This
  // is what keeps every 16 kHz waveform in this tree byte-identical across
  // W7c-2, and it is asserted rather than assumed.
  const std::vector<float> x = dots3_tiny::FixtureAudioF32(0);
  const std::vector<float> y =
      vllm::multimodal::ResampleAudioScipy(x.data(),
                                           static_cast<std::int64_t>(x.size()),
                                           16000, 16000);
  REQUIRE(y.size() == x.size());
  std::size_t moved = 0;
  for (std::size_t i = 0; i < x.size(); ++i)
    if (x[i] != y[i]) ++moved;
  MESSAGE("at 16000 -> 16000, " << moved << " of " << x.size()
                                << " samples moved");
  CHECK(moved == 0u);
}

TEST_CASE("dots3-note W7c-2: the seam REFUSES a rate it will not design a filter for") {
  const std::vector<float> x = dots3_tiny::FixtureAudioF32(0);
  const auto resample = [&x](int orig, int target) {
    std::string msg;
    try {
      vllm::multimodal::ResampleAudioScipy(
          x.data(), static_cast<std::int64_t>(x.size()), orig, target);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    return msg;
  };

  SUBCASE("a non-positive rate") {
    CHECK(resample(0, 16000).find("must be positive") != std::string::npos);
    CHECK(resample(44100, -1).find("must be positive") != std::string::npos);
  }

  SUBCASE("a reduced ratio past `kMaxPolyphaseRate`, which is a DIVERGENCE") {
    // The filter is `20 * max(up, down) + 1` taps and `max(up, down)` is chosen
    // by the REQUEST, because a WAV's `fmt ` chunk names its own rate. Upstream
    // has no such guard; upstream is also not reached from an HTTP body. The
    // message has to say both, so an operator is not told a port is missing.
    const std::string msg = resample(999983, 16000);
    INFO("message: ", msg);
    CHECK(msg.find("999983") != std::string::npos);
    CHECK(msg.find("100000") != std::string::npos);
    CHECK(msg.find("UPSTREAM HAS NO SUCH GUARD") != std::string::npos);
    CHECK(msg.find("DIVERGENCE") != std::string::npos);
    CHECK(msg.find("§4.17.10") != std::string::npos);
  }

  SUBCASE("...and every rate this row actually serves is FAR under the bound") {
    // The other half of the sentence above, so the bound is gated in BOTH
    // directions and cannot quietly become one that refuses real audio.
    for (const int sr : {44100, 48000, 22050, 8000, 32000, 11025, 44101}) {
      CAPTURE(sr);
      CHECK(resample(sr, 16000).empty());
    }
  }

  SUBCASE("an UPSAMPLE ratio past `kMaxUpsampleRatio`, which is a SECOND bound") {
    // `kMaxPolyphaseRate` BOUNDS THE FILTER AND NOT THE OUTPUT, and the two
    // come apart at a LOW `orig_sr`. `up` is `target_sr / gcd`, so it can never
    // exceed 16000 on this row; a 1 Hz `fmt ` chunk therefore reduces to
    // `max(up, down) = 16000`, sails under the 100000 filter bound, and asks
    // for `16000 *` the input in OUTPUT samples. Measured on the unguarded
    // tree: 20000 input samples became 320000000 output samples, 1220.7 MB, in
    // 2.301 s -- and TWICE per request, because the route resampled once for
    // the features and again for the hash. Under `ulimit -v 900000` the same
    // call threw `std::bad_alloc`, which is a bare `std::exception` and NOT
    // `InputValidationError`, so the server answered HTTP 500 for a property
    // of the REQUEST.
    for (const int sr : {1, 2, 1999}) {
      CAPTURE(sr);
      const std::string msg = resample(sr, 16000);
      INFO("message: ", msg);
      CHECK(msg.find("output samples") != std::string::npos);
      CHECK(msg.find("UPSTREAM HAS NO SUCH GUARD") != std::string::npos);
      CHECK(msg.find("DIVERGENCE") != std::string::npos);
      CHECK(msg.find("§4.17.10") != std::string::npos);
    }
    // The two bounds are INDEPENDENT and each catches what the other cannot.
    // 999983 -> 16000 trips the filter bound at a ratio of 16000/999983, which
    // is a DOWNsample; 1 -> 16000 trips this one at `max(up, down) = 16000`,
    // which is well under the filter bound. Naming both keeps a future repair
    // from folding them into one.
    CHECK(resample(999983, 16000).find("100000") != std::string::npos);
    CHECK(resample(1, 16000).find("100000") == std::string::npos);

    // BOTH DIRECTIONS, one rate apart. 2000 Hz reduces to 8/1, which is the
    // bound exactly and SERVES; 1999 Hz is coprime with 16000 and reduces to
    // 16000/1999 = 8.004, which is just past it and REFUSES.
    CHECK(resample(2000, 16000).empty());
    CHECK(!resample(1999, 16000).empty());
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. W8a (#2860): THE ONE-PASS APPLIER against `ref_apply`.
//
//    `ApplyPromptReplacements` is what replaces the two sequential expanders.
//    Spec §4.18.1 records why chaining them cannot work: each rebuilds the
//    whole id vector, so the second one's offsets are measured against the
//    first one's UN-expanded input and every span after the first is short by
//    the earlier expansions.
// ═══════════════════════════════════════════════════════════════════════════
namespace {

// The two dots3-note rules, at the fixture's own ids, built through the
// PRODUCTION helper. `ref_apply` is handed the same three ids per modality and
// builds its own target and content.
std::vector<vllm::multimodal::PromptReplacement> ProdRules(
    const std::vector<int>& image_counts,
    const std::vector<int>& audio_counts) {
  std::vector<vllm::multimodal::PromptReplacement> rules;
  if (!image_counts.empty()) {
    rules.push_back(vllm::multimodal::MakeTokenTripleReplacement(
        "image", dots3_tiny::kImgStartId, dots3_tiny::kImgPadId,
        dots3_tiny::kImgEndId, image_counts));
  }
  if (!audio_counts.empty()) {
    rules.push_back(vllm::multimodal::MakeTokenTripleReplacement(
        "audio", dots3_tiny::kAudStartId, dots3_tiny::kAudPadId,
        dots3_tiny::kAudEndId, audio_counts));
  }
  return rules;
}

std::vector<ref_apply::Rule> RefRules(const std::vector<int>& image_counts,
                                      const std::vector<int>& audio_counts) {
  std::vector<ref_apply::Rule> rules;
  if (!image_counts.empty()) {
    rules.push_back(ref_apply::Rule{
        "image",
        {dots3_tiny::kImgStartId, dots3_tiny::kImgPadId, dots3_tiny::kImgEndId},
        image_counts});
  }
  if (!audio_counts.empty()) {
    rules.push_back(ref_apply::Rule{
        "audio",
        {dots3_tiny::kAudStartId, dots3_tiny::kAudPadId, dots3_tiny::kAudEndId},
        audio_counts});
  }
  return rules;
}

std::vector<int> ToInt(const std::vector<std::int32_t>& v) {
  return std::vector<int>(v.begin(), v.end());
}

std::vector<std::int32_t> ToI32(const std::vector<int>& v) {
  return std::vector<std::int32_t>(v.begin(), v.end());
}

// Every case's assertion, so a new prompt shape costs one line.
void AgreesWithReference(const std::vector<int>& prompt,
                         const std::vector<int>& image_counts,
                         const std::vector<int>& audio_counts) {
  std::vector<vllm::multimodal::AppliedPromptUpdate> got;
  const std::vector<std::int32_t> ours =
      vllm::multimodal::ApplyPromptReplacements(
          ToI32(prompt), ProdRules(image_counts, audio_counts), &got);
  const ref_apply::Applied want =
      ref_apply::Apply(prompt, RefRules(image_counts, audio_counts));

  REQUIRE(ToInt(ours) == want.ids);
  REQUIRE(got.size() == want.spans.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    INFO("span ", i);
    CHECK(got[i].modality == want.spans[i].modality);
    CHECK(got[i].item_index == want.spans[i].item);
    CHECK(got[i].offset == want.spans[i].offset);
    CHECK(got[i].length == want.spans[i].length);
    // The span holds only pad ids, and it lies inside the output.
    REQUIRE(got[i].offset >= 0);
    REQUIRE(got[i].offset + got[i].length <= static_cast<int>(ours.size()));
    const std::int32_t pad = got[i].modality == "image"
                                 ? dots3_tiny::kImgPadId
                                 : dots3_tiny::kAudPadId;
    for (int t = got[i].offset; t < got[i].offset + got[i].length; ++t)
      CHECK(ours[static_cast<std::size_t>(t)] == pad);
  }
  // ASCENDING and DISJOINT, which `GetMmFeaturesInWindow`'s two binary searches
  // (`utils.cpp:9-50`) are a precondition of.
  for (std::size_t i = 1; i < got.size(); ++i) {
    CHECK(got[i - 1].offset + got[i - 1].length <= got[i].offset);
  }
}

}  // namespace

TEST_CASE("dots3-note W8a: the ONE-PASS applier agrees with an INDEPENDENT reference") {
  const int kImg = dots3_tiny::kImgStartId, kImgP = dots3_tiny::kImgPadId,
            kImgE = dots3_tiny::kImgEndId;
  const int kAud = dots3_tiny::kAudStartId, kAudP = dots3_tiny::kAudPadId,
            kAudE = dots3_tiny::kAudEndId;
  const int kText = 13;  // "hello" in the row's BPE fixture

  SUBCASE("one image alone") {
    AgreesWithReference({kImg, kImgP, kImgE, kText}, {4}, {});
  }
  SUBCASE("one audio alone") {
    AgreesWithReference({kAud, kAudP, kAudE, kText}, {}, {7});
  }
  SUBCASE("audio then text then image — the mixed request the seam builds") {
    AgreesWithReference({kAud, kAudP, kAudE, kText, kImg, kImgP, kImgE}, {4},
                        {7});
  }
  SUBCASE("image FIRST, so the rule order and the stream order disagree") {
    AgreesWithReference({kImg, kImgP, kImgE, kAud, kAudP, kAudE}, {4}, {7});
  }
  SUBCASE("two images and two audios, INTERLEAVED") {
    AgreesWithReference({kImg, kImgP, kImgE, kAud, kAudP, kAudE, kText, kImg,
                         kImgP, kImgE, kAud, kAudP, kAudE},
                        {4, 8}, {7, 3});
  }
  SUBCASE("ADJACENT targets with no text between them") {
    AgreesWithReference({kImg, kImgP, kImgE, kImg, kImgP, kImgE}, {4, 4}, {});
  }
  SUBCASE("a ZERO-length run, which a 1x1 grid under a 2x2 merge cannot make "
          "but the applier must still place") {
    AgreesWithReference({kAud, kAudP, kAudE, kText}, {}, {0});
  }
  SUBCASE("no rules at all leaves the stream byte-identical") {
    AgreesWithReference({kText, kText, kImgP, kAudP}, {}, {});
  }
}

TEST_CASE("dots3-note W8a: the applier keys on the TARGET TRIPLE, not on the pad id") {
  // Upstream's target is `[start, pad, end]` (`common/processor.py:749-756`,
  // `:777-783` @ `9035151d6`), so a BARE pad id in the user's own text is NOT a
  // placeholder. The pre-W8a `ExpandImagePlaceholders` keyed on the pad alone
  // and threw "more image placeholders than grids" at it, which reached the
  // client as a 400 naming an internal helper.
  const std::vector<std::int32_t> prompt{
      dots3_tiny::kImgPadId,   // a bare pad the user typed
      13,
      dots3_tiny::kImgStartId, dots3_tiny::kImgPadId, dots3_tiny::kImgEndId};
  std::vector<vllm::multimodal::AppliedPromptUpdate> got;
  const std::vector<std::int32_t> out =
      vllm::multimodal::ApplyPromptReplacements(prompt, ProdRules({4}, {}),
                                                &got);
  REQUIRE(got.size() == 1u);
  // The REAL placeholder, not the bare one at index 0.
  CHECK(got[0].offset == 3);
  CHECK(got[0].length == 4);
  // Two untouched ids (the bare pad and the text token) plus the replaced
  // triple, which is `[start] + 4 pads + [end]`.
  CHECK(out.size() == 2u + 6u);
  CHECK(out[0] == dots3_tiny::kImgPadId);  // left exactly where the user put it
}

TEST_CASE("dots3-note W8a: an item the prompt has no target for is REFUSED BY NAME") {
  // The one outcome that produces a fluent WRONG answer: the encoder never runs
  // for the dropped item, the placeholder run is never written, and the prompt
  // reads as if the user had not sent that media. Upstream reaches the same
  // conclusion through `_all_items_found` (`processor.py:896-903` @
  // `9035151d6`).
  const std::vector<std::int32_t> one_target{
      dots3_tiny::kImgStartId, dots3_tiny::kImgPadId, dots3_tiny::kImgEndId};
  CHECK_THROWS_WITH_AS(
      vllm::multimodal::ApplyPromptReplacements(one_target, ProdRules({4, 4}, {}),
                                                nullptr),
      doctest::Contains("1 'image' placeholder target(s) but the request "
                        "carries 2 item(s)"),
      std::runtime_error);

  // An EMPTY target is upstream's other planner arm and this port does not
  // represent it, so it is refused rather than treated as a no-op.
  std::vector<vllm::multimodal::PromptReplacement> empty_target;
  empty_target.push_back(vllm::multimodal::PromptReplacement{"image", {}, {}});
  CHECK_THROWS_WITH_AS(
      vllm::multimodal::ApplyPromptReplacements(one_target, empty_target,
                                                nullptr),
      doctest::Contains("EMPTY target"), std::runtime_error);
}

TEST_CASE("dots3-note W8a: the SEQUENTIAL two-pass expansion this replaces gets the second span WRONG") {
  // THE MEASUREMENT BEHIND SPEC §4.18.1, executed rather than argued. The two
  // shipped expanders each rebuild the whole id vector, so chaining them over
  // one prompt reports the image span at an offset that is short by exactly the
  // audio expansion — and it lands INSIDE the audio span, where the runner
  // would splice vision rows over audio rows.
  const std::vector<std::int32_t> prompt{
      dots3_tiny::kAudStartId, dots3_tiny::kAudPadId, dots3_tiny::kAudEndId,
      13,
      dots3_tiny::kImgStartId, dots3_tiny::kImgPadId, dots3_tiny::kImgEndId};

  std::vector<std::array<int, 2>> audio_spans;
  const std::vector<std::int32_t> after_audio =
      vllm::multimodal::ExpandAudioPlaceholders(prompt, dots3_tiny::kAudPadId,
                                                {7}, &audio_spans);
  REQUIRE(audio_spans.size() == 1u);
  CHECK(audio_spans[0][0] == 1);
  CHECK(audio_spans[0][1] == 7);

  // The chained pass, measured against the ORIGINAL ids, which is what makes it
  // wrong. `grid` (1, 4, 4) over merge 2 is FOUR placeholder rows.
  std::vector<std::array<int, 2>> chained_image;
  vllm::multimodal::ExpandImagePlaceholders(
      prompt, dots3_tiny::kImgPadId, /*merge_size=*/2,
      {std::array<std::int64_t, 3>{1, 4, 4}}, &chained_image);
  REQUIRE(chained_image.size() == 1u);
  MESSAGE("chained two-pass reports the image at offset "
          << chained_image[0][0] << "; the audio span is ["
          << audio_spans[0][0] << ", "
          << audio_spans[0][1] + audio_spans[0][0] << ")");
  CHECK(chained_image[0][0] == 5);
  // INSIDE the audio span, which is the defect in one line.
  CHECK(chained_image[0][0] < audio_spans[0][0] + audio_spans[0][1]);

  // The one-pass applier puts it where it belongs.
  std::vector<vllm::multimodal::AppliedPromptUpdate> got;
  const std::vector<std::int32_t> ours =
      vllm::multimodal::ApplyPromptReplacements(prompt, ProdRules({4}, {7}),
                                                &got);
  REQUIRE(got.size() == 2u);
  CHECK(got[0].modality == "audio");
  CHECK(got[0].offset == 1);
  CHECK(got[1].modality == "image");
  CHECK(got[1].offset == 11);
  CHECK(got[1].offset >= got[0].offset + got[0].length);
  CHECK(ours.size() == 16u);
  // And the chained arm's own OUTPUT stream is not this one either.
  CHECK(after_audio.size() != ours.size());
}
