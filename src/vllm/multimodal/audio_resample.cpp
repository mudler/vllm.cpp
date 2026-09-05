// The shared audio resample seam (dots3-note W7c-2, #2828).
//
// `scipy.signal.resample_poly` at its defaults, transcribed from scipy 1.17.1's
// own `scipy/signal/_signaltools.py::resample_poly` and
// `scipy/signal/_fir_filter_design.py::firwin`, which is the arm upstream's
// `resample_audio_scipy` reaches (`vllm/multimodal/audio.py:232-250` @
// `9035151d6`). The header says why THIS arm and not upstream's `pyav` default.
// `.agents/specs/dots3-note.md` §4.17.3 writes the algorithm out step by step
// and records that the transcription was checked against scipy rather than
// trusted.
#include "vllm/multimodal/audio_resample.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

#include "vllm/v1/engine/validation_error.h"

namespace vllm::multimodal {

namespace {

// `numpy.sinc`: `sin(pi x) / (pi x)`, and 1 at 0. `firwin` builds its low-pass
// band from `right * sinc(right * m)` (`_fir_filter_design.py`, the `bands`
// loop), so this is the ideal brick-wall impulse response before windowing.
double Sinc(double x) {
  if (x == 0.0) return 1.0;
  const double p = 3.14159265358979323846 * x;
  return std::sin(p) / p;
}

// `scipy.special.i0`, the modified Bessel function of the first kind, order 0,
// as its own power series `sum_k (x^2/4)^k / (k!)^2`.
//
// WRITTEN AS THE SERIES AND NOT AS THE USUAL POLYNOMIAL FIT. The Abramowitz
// and Stegun rational approximation that most C code reaches for is accurate to
// about 1e-7 relative, which is a `float` answer used to build a `double`
// filter; the series is accurate to `double` and, at the beta = 5 the kaiser
// default uses, its argument never exceeds 5, where it converges in about
// twenty terms.
double BesselI0(double x) {
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

// `firwin(numtaps, f_c, window=("kaiser", 5.0))` at `resample_poly`'s
// arguments, scaled by `up` — steps 3 and 4 of §4.17.3.
//
// `firwin` with `fs=None` takes `nyq = 1`, so `f_c` is already normalized. With
// one cutoff and `pass_zero=True`, `pass_nyquist` is false and the band list is
// the single `[0, f_c]`; the `left` term of the band loop is
// `0 * sinc(0 * m) == 0` and drops out. `scale=True` with a first band whose
// left edge is 0 takes `scale_frequency = 0`, so `c = cos(0) = 1` and the
// normalizer is the plain sum of the taps.
std::vector<double> DesignFilter(int64_t up, int64_t down) {
  const int64_t max_rate = std::max(up, down);
  const double f_c = 1.0 / static_cast<double>(max_rate);
  const int64_t half_len = 10 * max_rate;
  const int64_t numtaps = 2 * half_len + 1;
  const double alpha = 0.5 * static_cast<double>(numtaps - 1);
  const double i0_beta = BesselI0(5.0);

  std::vector<double> h(static_cast<std::size_t>(numtaps));
  for (int64_t i = 0; i < numtaps; ++i) {
    const double m = static_cast<double>(i) - alpha;
    // `kaiser(numtaps, 5.0, sym=True)`:
    // `i0(beta * sqrt(1 - ((n - alpha) / alpha)^2)) / i0(beta)`.
    const double r = m / alpha;
    const double arg = 1.0 - r * r;
    const double win =
        BesselI0(5.0 * std::sqrt(arg > 0.0 ? arg : 0.0)) / i0_beta;
    h[static_cast<std::size_t>(i)] = f_c * Sinc(f_c * m) * win;
  }
  double sum = 0.0;
  for (double v : h) sum += v;
  const double scale = static_cast<double>(up) / sum;
  for (double& v : h) v *= scale;
  return h;
}

// `_output_len(len_h, n_in, up, down)` from `scipy/signal/_upfirdn.py`: the
// length `upfirdn` produces, which is the decimation by `down` of a convolution
// of length `(n_in - 1) * up + len_h`.
int64_t UpfirdnOutputLen(int64_t len_h, int64_t n_in, int64_t up, int64_t down) {
  return ((n_in - 1) * up + len_h - 1) / down + 1;
}

int64_t Gcd(int64_t a, int64_t b) {
  while (b != 0) {
    const int64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

// `up` and `down` after upstream's own gcd reduction (`audio.py:244-249`).
// `resample_poly` reduces AGAIN by `math.gcd(up, down)`, which is a no-op on an
// already-reduced pair; both reductions are written here so a reader comparing
// against either source finds what they expect.
void ReducedRatio(int orig_sr, int target_sr, int64_t* up, int64_t* down) {
  const int64_t g = Gcd(orig_sr, target_sr);
  *up = static_cast<int64_t>(target_sr) / g;
  *down = static_cast<int64_t>(orig_sr) / g;
  const int64_t g2 = Gcd(*up, *down);
  *up /= g2;
  *down /= g2;
}

void ValidateRates(int orig_sr, int target_sr) {
  if (orig_sr <= 0 || target_sr <= 0) {
    throw v1::InputValidationError(
        "audio resample: a sampling rate must be positive, and this request "
        "carries orig_sr=" + std::to_string(orig_sr) +
        " target_sr=" + std::to_string(target_sr) + ".");
  }
  int64_t up = 0;
  int64_t down = 0;
  ReducedRatio(orig_sr, target_sr, &up, &down);
  const int64_t max_rate = std::max(up, down);
  if (max_rate > kMaxPolyphaseRate) {
    throw v1::InputValidationError(
        "audio resample: converting " + std::to_string(orig_sr) + " Hz to " +
        std::to_string(target_sr) + " Hz reduces to a polyphase ratio of " +
        std::to_string(up) + "/" + std::to_string(down) +
        ", whose anti-alias filter is 20 * " + std::to_string(max_rate) +
        " + 1 taps. This port REFUSES a reduced max(up, down) above " +
        std::to_string(kMaxPolyphaseRate) +
        " because the rate is named by the REQUEST, in a WAV `fmt ` chunk, and "
        "designing that filter costs the process time and memory before one "
        "audio sample is touched. UPSTREAM HAS NO SUCH GUARD and this is a "
        "recorded DIVERGENCE, not a missing port: see "
        ".agents/specs/dots3-note.md §4.17.10. Every ordinary rate reduces far "
        "below the bound (44100 -> 441, 48000 -> 3, 22050 -> 441, 8000 -> 2).");
  }
  // THE BOUND ABOVE IS ON THE FILTER AND NOT ON THE OUTPUT (PR #2842 F2). `up` is
  // `target_sr / gcd` and can never exceed `target_sr`, so a LOW `orig_sr`
  // gives a SMALL `max(up, down)` and a HUGE `up/down`: 1 -> 16000 reduces to
  // 16000/1, passes the check above, and then asks for 16000 output samples
  // per input sample. This one is on the RATIO, which is what actually sizes
  // the allocation, and it is checked here rather than beside `n_out` so that
  // nothing is allocated at all.
  if (up > static_cast<int64_t>(kMaxUpsampleRatio) * down) {
    throw v1::InputValidationError(
        "audio resample: converting " + std::to_string(orig_sr) + " Hz to " +
        std::to_string(target_sr) + " Hz reduces to a polyphase ratio of " +
        std::to_string(up) + "/" + std::to_string(down) +
        ", which turns every input sample into " + std::to_string(up) + "/" +
        std::to_string(down) + " output samples. This port REFUSES a reduced "
        "up/down above " + std::to_string(kMaxUpsampleRatio) +
        " because the rate is named by the REQUEST, in a WAV `fmt ` chunk, and "
        "an UPSAMPLE is an AMPLIFIER: a 40 KB `data` chunk declaring 1 Hz "
        "produced a 1220.7 MB buffer in 2.301 s, and the `std::bad_alloc` that "
        "follows under memory pressure is a bare `std::exception` rather than "
        "an InputValidationError, so this would be answered 500 for a property "
        "of the request. `kMaxPolyphaseRate` does NOT bound this, because it "
        "bounds max(up, down) — the FILTER — and `up` cannot separate the two "
        "cases: a coprime 44101 Hz, which this seam SERVES, also reduces to "
        "up = 16000. UPSTREAM HAS NO SUCH GUARD and this is a recorded "
        "DIVERGENCE, not a missing port: see .agents/specs/dots3-note.md "
        "§4.17.10. Every ordinary rate reduces far below the bound "
        "(44100 -> 160/441, 48000 -> 1/3, 22050 -> 320/441, 8000 -> 2/1, "
        "44101 -> 16000/44101), and the bound admits any source rate down to "
        "2000 Hz on a 16 kHz target.");
  }
}

}  // namespace

std::vector<float> ResampleAudioScipy(const float* samples, int64_t num_samples,
                                      int orig_sr, int target_sr) {
  ValidateRates(orig_sr, target_sr);
  if (num_samples <= 0) return {};

  // `if orig_sr_int == target_sr_int: return audio` (`audio.py:241-242`), and
  // `resample_poly`'s own `if up == down == 1: return x` beneath it. A waveform
  // already at the target rate cannot move by a bit through this call.
  if (orig_sr == target_sr) {
    return std::vector<float>(samples, samples + num_samples);
  }

  int64_t up = 0;
  int64_t down = 0;
  ReducedRatio(orig_sr, target_sr, &up, &down);
  if (up == 1 && down == 1) {
    return std::vector<float>(samples, samples + num_samples);
  }

  const int64_t half_len = 10 * std::max(up, down);
  std::vector<double> h = DesignFilter(up, down);

  const int64_t n_in = num_samples;
  const int64_t prod = n_in * up;
  const int64_t n_out = prod / down + ((prod % down) != 0 ? 1 : 0);

  // "Zero-pad our filter to put the output samples at the center" — step 5.
  // `n_pre_pad` shifts the filter so that the first kept sample is the one
  // whose window is centred on input sample 0, and `n_pre_remove` is how many
  // outputs that pad puts in front of it.
  const int64_t n_pre_pad = down - (half_len % down);
  const int64_t n_pre_remove = (half_len + n_pre_pad) / down;
  int64_t n_post_pad = 0;
  while (UpfirdnOutputLen(static_cast<int64_t>(h.size()) + n_pre_pad + n_post_pad,
                          n_in, up, down) < n_out + n_pre_remove) {
    ++n_post_pad;
  }

  std::vector<double> hh(static_cast<std::size_t>(n_pre_pad) + h.size() +
                         static_cast<std::size_t>(n_post_pad), 0.0);
  std::copy(h.begin(), h.end(), hh.begin() + n_pre_pad);
  const int64_t len_h = static_cast<int64_t>(hh.size());

  // `upfirdn(h, x, up, down)[n_pre_remove : n_pre_remove + n_out]` — step 6.
  //
  // `upfirdn` is the decimation by `down` of the convolution of `hh` with `x`
  // zero-stuffed by `up`, so the kept output is
  //
  //     y[i] = sum_j hh[(i + n_pre_remove) * down - j * up] * x[j]
  //
  // over every `j` whose filter index falls inside `hh`. The loop runs over
  // THOSE `j` — about `len_h / up` of them — rather than over the whole filter,
  // which is the polyphase structure and not an approximation of it: every `j`
  // the full convolution would visit lands on a zero of the zero-stuffed
  // signal.
  std::vector<float> out(static_cast<std::size_t>(n_out));
  for (int64_t oi = 0; oi < n_out; ++oi) {
    const int64_t t = (oi + n_pre_remove) * down;
    // `0 <= t - j * up < len_h`  <=>  `(t - len_h) / up < j <= t / up`.
    int64_t j_lo = (t - len_h + up) / up;
    if (j_lo < 0) j_lo = 0;
    int64_t j_hi = t / up;
    if (j_hi > n_in - 1) j_hi = n_in - 1;
    double acc = 0.0;
    for (int64_t j = j_lo; j <= j_hi; ++j) {
      acc += hh[static_cast<std::size_t>(t - j * up)] *
             static_cast<double>(samples[j]);
    }
    out[static_cast<std::size_t>(oi)] = static_cast<float>(acc);
  }
  return out;
}

}  // namespace vllm::multimodal
