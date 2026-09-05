#ifndef VLLM_MULTIMODAL_AUDIO_RESAMPLE_H_
#define VLLM_MULTIMODAL_AUDIO_RESAMPLE_H_

#include <cstdint>
#include <vector>

namespace vllm::multimodal {

// ── the SHARED audio resample seam (dots3-note W7c-2, #2828) ────────────────
//
// `scipy.signal.resample_poly` at its defaults, which is what upstream's own
// `resample_audio_scipy` calls (`vllm/multimodal/audio.py:232-250` @
// `9035151d6`).
//
// WHY THIS ARM AND NOT UPSTREAM'S DEFAULT. `AudioResampler`'s default method is
// `"pyav"` (`audio.py:283`), which is libswresample through PyAV. That arm is
// NOT bit-identical to itself: on ffmpeg 6.1.1, one binary and one input
// differing only in CPU dispatch produce 24691 differing samples of 32000 at a
// worst absolute difference of 9.686e-08. A bit-exact gate against it is
// impossible in principle, its option defaults come from an unpinned linked
// binary, and its auto-resolved `cutoff` is not readable from outside the
// source. `"scipy"` is ANOTHER ARM OF THE SAME SWITCH (`audio.py:305-316`), and
// vLLM already ships it in production for another model
// (`vllm/model_executor/models/phi4mm.py:580`). Measured distance from the real
// default at 44100 -> 16000, interior only, ON A 0 -> 7500 Hz SWEEP — content
// that reaches the OUTPUT Nyquist, which is what a speech encoder sees: scipy
// 51.78 dB, soxr 44.63, torchaudio 29.02. THE PROBE IS PART OF THE NUMBER. On
// content well below the new Nyquist the ordering inverts and soxr wins by
// 30 dB, because a resampler's transition band cannot matter where there is no
// energy in it. Spec §4.17.2 carries the four-probe table.
//
// See `.agents/specs/dots3-note.md` §4.17 for the decision, the algorithm
// written out step by step, and what its gate does and does not establish.
//
// OPTED INTO PER MODEL, NEVER BLANKET. Five rows in this tree refuse a rate
// mismatch and they are not all the same policy: Parakeet's refusal is
// upstream-faithful, because `feature_extraction_parakeet.py` raises rather
// than resampling, while dots3-note's upstream resamples. dots3-note opts in
// by calling this; nobody else is changed by its existence.

// The filter is `20 * max(up, down) + 1` taps and `max(up, down)` is chosen by
// the REQUEST, because a WAV's `fmt ` chunk names its own sample rate. A
// request declaring 999983 Hz reduces to `max(up, down) = 999983` and asks this
// process for a twenty-million-tap filter, and twenty million Bessel
// evaluations, before one audio sample is touched.
//
// THIS BOUND IS A DELIBERATE DIVERGENCE. Upstream has no such guard and would
// design that filter. Upstream is also not the surface this protects: this one
// is reached from an HTTP request body. 100000 caps the design at about two
// million taps and sixteen megabytes, and every real rate reduces far below it
// — 44100 -> 441, 48000 -> 3, 22050 -> 441, 8000 -> 2, and even a coprime
// 44101 Hz gives 44101.
inline constexpr int kMaxPolyphaseRate = 100000;

// THE FILTER BOUND IS NOT AN OUTPUT BOUND, and PR #2842's fresh review found the gap. `up` is
// `target_sr / gcd`, so on a 16 kHz target it can never exceed 16000: a `fmt `
// chunk declaring 1 Hz reduces to `up/down = 16000/1`, sails under
// `kMaxPolyphaseRate`, designs a cheap filter — and then asks for SIXTEEN
// THOUSAND output samples per input sample. Measured on the unguarded tree, a
// 40 KB `data` chunk produced a 1220.7 MB buffer in 2.301 s; under
// `ulimit -v 900000` the same call threw `std::bad_alloc`, which is a bare
// `std::exception` and NOT `vllm::v1::InputValidationError`, so the server
// answered HTTP 500 for a property of the REQUEST.
//
// SO THE SECOND BOUND IS ON THE RATIO, not on `max(up, down)` and not on `up`
// alone. `up` cannot separate the two cases: a coprime 44101 Hz — a DOWNsample
// this seam serves, and gates that it serves — also reduces to `up = 16000`.
// `up/down` separates them completely: 44101 gives 0.363 and 1 Hz gives 16000.
// Bounding the RATIO also never refuses a long clip, because it bounds the
// output as a multiple of an input the client already paid to upload.
//
// 8 IS FOUR TIMES THE LARGEST RATIO THIS ROW SERVES. The highest real
// upsample into 16 kHz is telephony's 8000 -> 16000 = 2; 11025 gives 1.451 and
// every rate at or above the target gives less than 1. The bound admits any
// source rate down to 2000 Hz on a 16 kHz target, and an 8 kHz source on a
// 48 kHz one, and it caps this seam's allocation at 32 bytes per input sample.
// It is gated in BOTH directions one hertz apart: 2000 Hz reduces to 8/1 and
// serves, 1999 Hz reduces to 16000/1999 = 8.004 and refuses.
//
// LIKE `kMaxPolyphaseRate`, THIS IS A DELIBERATE DIVERGENCE. Upstream has no
// such guard because upstream is not reached from an HTTP request body. See
// `.agents/specs/dots3-note.md` §4.17.10.
inline constexpr int kMaxUpsampleRatio = 8;

// `resample_audio_scipy(audio, orig_sr=..., target_sr=...)`.
//
// Returns the input unchanged when the two rates are equal, exactly as upstream
// does (`audio.py:241-242`), so a 16 kHz waveform on a 16 kHz checkpoint cannot
// move by a bit through this call.
//
// The filter design and the convolution run in `double` and narrow ONCE to
// `float` at the store. Upstream's arm narrows earlier — `resample_poly` does
// `h = xp.asarray(h, dtype=x.dtype)` before `h *= up`, so a `float32` input
// gets a `float32` filter and a `float32` accumulation. That is deliberately
// not mirrored, because it is not reproducible: a `float32` sum over the ~56
// taps each output touches at 44100 -> 16000 depends on summation order and on
// whether the compiler contracts a multiply-add. The `double` arm is
// order-stable to ~1e-16 relative, which is invisible after the narrowing
// store.
//
// Throws `vllm::v1::InputValidationError` — the rate is a property of the
// REQUEST, so the server answers HTTP 400 rather than 500 — when a rate is not
// positive, when the reduced `max(up, down)` exceeds `kMaxPolyphaseRate`, or
// when the reduced `up/down` exceeds `kMaxUpsampleRatio`.
std::vector<float> ResampleAudioScipy(const float* samples, int64_t num_samples,
                                      int orig_sr, int target_sr);

}  // namespace vllm::multimodal

#endif  // VLLM_MULTIMODAL_AUDIO_RESAMPLE_H_
