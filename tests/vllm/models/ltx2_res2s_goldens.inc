// GENERATED from Lightricks/LTX-2 @ fd4ded7f by
// scripts/gen-ltx2-res2s-goldens.py. Do not hand-edit.
#pragma once

#include <cstdint>

namespace vllm_test {

// res2s.py:4-22. `phi(j, z)` at j = 1 and j = 2, INCLUDING the small-z
// cliff: the guard is `abs(z) < 1e-10` and outside it the formula
// cancels catastrophically, so upstream's own phi2(-1e-10) is 0.0 and
// phi2(-1e-8) is 1.1102230246251563. These are upstream's values, not a
// series expansion's, and a 'better' port fails here.
inline constexpr double kLtx2PhiZ[] = {
    0.0, -1e-12, -1e-11,
    -1e-10, -1e-09, -1e-08,
    -1e-06, -0.001, -0.125,
    -0.25, -0.5, -1.0,
    -2.0, -5.0};
inline constexpr double kLtx2Phi1[] = {
    1.0, 1.0, 1.0,
    1.000000082740371, 0.9999999717180684, 0.999999993922529,
    0.9999994999843054, 0.9995001666249781, 0.9400247793232364,
    0.8847968677143805, 0.7869386805747332, 0.6321205588285577,
    0.43233235838169365, 0.1986524106001829};
inline constexpr double kLtx2Phi2[] = {
    0.5, 0.5, 0.5,
    0.0, 0.0, 1.1102230246251563,
    0.5000444502911705, 0.4998333750227957, 0.4798017654141091,
    0.46081252914247806, 0.4261226388505337, 0.36787944117144233,
    0.2838338208091532, 0.16026951787996344};
inline constexpr int64_t kLtx2PhiCount = 14;

// res2s.py:25-62, c2 = 0.5 (samplers.py:288).
inline constexpr double kLtx2Res2sCoeffH[] = {
    1e-12, 1e-10, 1e-08,
    1e-06, 0.01, 0.125,
    0.25, 0.5, 1.0,
    3.0, 7.0};
inline constexpr double kLtx2Res2sCoeffA21[] = {
    0.5, 0.5, 0.4999999969612645,
    0.4999998749477541, 0.498752080731768, 0.48469549749219354,
    0.4700123896616182, 0.44239843385719024, 0.3934693402873666,
    0.2589566132838567, 0.13854323093966878};
inline constexpr double kLtx2Res2sCoeffB1[] = {
    0.0, 1.000000082740371, -1.2204460553277836,
    -8.940059803563827e-05, -0.0016583582791218632, -0.019578751504981895,
    -0.036828190570575625, -0.06530659712633424, -0.103638323514327,
    -0.1387705935377022, -0.10220830485081611};
inline constexpr double kLtx2Res2sCoeffB2[] = {
    1.0, 0.0, 2.2204460492503126,
    1.000088900582341, 0.9966749833623112, 0.9596035308282183,
    0.9216250582849561, 0.8522452777010674, 0.7357588823428847,
    0.4555082374150809, 0.2449351788557369};
inline constexpr int64_t kLtx2Res2sCoeffCount = 11;

// BongOn: sigmas [0.9, 0.8, 0.7, 0.62], eta 0.5, h [0.117783, 0.133531, 0.121361], bong changed the result: True
inline constexpr double kLtx2Res2sBongOnEta = 0.5;
inline constexpr float kLtx2Res2sBongOnSigmas[] = {
    0.899999976f, 0.800000012f, 0.699999988f,
    0.620000005f};
inline constexpr int64_t kLtx2Res2sBongOnSigmaCount = 4;
inline constexpr int64_t kLtx2Res2sBongOnEvaluations = 6;
inline constexpr double kLtx2Res2sBongOnEvalSigmas[] = {
    0.8999999761581421, 0.8485281325067245, 0.800000011920929,
    0.7483314765582876, 0.699999988079071, 0.658786760603827};
inline constexpr int64_t kLtx2Res2sBongOnEvalStepIndices[] = {
    0, 0, 1,
    0, 2, 0};
inline constexpr float kLtx2Res2sBongOnVideo[] = {
    -0.322195023f, 0.323642969f, 0.699999988f,
    0.249027193f, 0.400000006f, 0.576920807f};
inline constexpr float kLtx2Res2sBongOnAudio[] = {
    0.244931653f, 0.400000006f, 0.0785670504f,
    0.699999988f, 0.159960002f, -0.233123973f};
inline constexpr bool kLtx2Res2sBongOnBongMoved = true;
inline constexpr float kLtx2Res2sBongOnNoBongVideo[] = {
    -0.21995157f, 0.451260477f, 0.699999988f,
    0.552530169f, 0.400000006f, 0.592931151f};

// BongOffByH: sigmas [0.9, 0.5, 0.25, 0.12], eta 0.5, h [0.587787, 0.693147, 0.733969], bong changed the result: False
inline constexpr double kLtx2Res2sBongOffByHEta = 0.5;
inline constexpr float kLtx2Res2sBongOffByHSigmas[] = {
    0.899999976f, 0.5f, 0.25f,
    0.119999997f};
inline constexpr int64_t kLtx2Res2sBongOffByHSigmaCount = 4;
inline constexpr int64_t kLtx2Res2sBongOffByHEvaluations = 6;
inline constexpr double kLtx2Res2sBongOffByHEvalSigmas[] = {
    0.8999999761581421, 0.670820384364601, 0.5,
    0.3535533905932738, 0.25, 0.1732050788211701};
inline constexpr int64_t kLtx2Res2sBongOffByHEvalStepIndices[] = {
    0, 0, 1,
    0, 2, 0};
inline constexpr float kLtx2Res2sBongOffByHVideo[] = {
    0.241032124f, 0.415953547f, 0.699999988f,
    0.426735073f, 0.400000006f, 0.526328683f};
inline constexpr float kLtx2Res2sBongOffByHAudio[] = {
    0.0932270736f, 0.400000006f, 0.0732859746f,
    0.699999988f, 0.0699004084f, 0.0386947282f};
inline constexpr bool kLtx2Res2sBongOffByHBongMoved = false;
inline constexpr float kLtx2Res2sBongOffByHNoBongVideo[] = {
    0.241032124f, 0.415953547f, 0.699999988f,
    0.426735073f, 0.400000006f, 0.526328683f};

// BongOffBySigma: sigmas [0.03, 0.028, 0.026, 0.025], eta 0.5, h [0.068993, 0.074108, 0.039221], bong changed the result: False
inline constexpr double kLtx2Res2sBongOffBySigmaEta = 0.5;
inline constexpr float kLtx2Res2sBongOffBySigmaSigmas[] = {
    0.0299999993f, 0.0280000009f, 0.0260000005f,
    0.0250000004f};
inline constexpr int64_t kLtx2Res2sBongOffBySigmaSigmaCount = 4;
inline constexpr int64_t kLtx2Res2sBongOffBySigmaEvaluations = 6;
inline constexpr double kLtx2Res2sBongOffBySigmaEvalSigmas[] = {
    0.029999999329447746, 0.028982753615772204, 0.02800000086426735,
    0.026981475821224496, 0.026000000536441803, 0.025495098020929436};
inline constexpr int64_t kLtx2Res2sBongOffBySigmaEvalStepIndices[] = {
    0, 0, 1,
    0, 2, 0};
inline constexpr float kLtx2Res2sBongOffBySigmaVideo[] = {
    0.119263649f, 0.260755271f, 0.699999988f,
    0.486651659f, 0.400000006f, 0.701571345f};
inline constexpr float kLtx2Res2sBongOffBySigmaAudio[] = {
    0.275088519f, 0.400000006f, 0.194240034f,
    0.699999988f, 0.11664068f, 0.0714727566f};
inline constexpr bool kLtx2Res2sBongOffBySigmaBongMoved = false;
inline constexpr float kLtx2Res2sBongOffBySigmaNoBongVideo[] = {
    0.119263649f, 0.260755271f, 0.699999988f,
    0.486651659f, 0.400000006f, 0.701571345f};

// TerminalZero: sigmas [1.0, 0.75, 0.5, 0.25, 0.0], eta 0.5, h [0.287682, 0.405465, 0.693147], bong changed the result: True
inline constexpr double kLtx2Res2sTerminalZeroEta = 0.5;
inline constexpr float kLtx2Res2sTerminalZeroSigmas[] = {
    1.0f, 0.75f, 0.5f,
    0.25f, 0.0f};
inline constexpr int64_t kLtx2Res2sTerminalZeroSigmaCount = 5;
inline constexpr int64_t kLtx2Res2sTerminalZeroEvaluations = 9;
inline constexpr double kLtx2Res2sTerminalZeroEvalSigmas[] = {
    1.0, 0.8660254037844386, 0.75,
    0.6123724356957945, 0.5, 0.3535533905932738,
    0.25, 0.016583123906848306, 0.0010999999940395355};
inline constexpr int64_t kLtx2Res2sTerminalZeroEvalStepIndices[] = {
    0, 0, 1,
    0, 2, 0,
    3, 0, 4};
inline constexpr float kLtx2Res2sTerminalZeroVideo[] = {
    0.445208162f, 0.445210904f, 0.699999988f,
    0.44826746f, 0.400000006f, 0.451412231f};
inline constexpr float kLtx2Res2sTerminalZeroAudio[] = {
    0.0724363402f, 0.400000006f, 0.082616441f,
    0.699999988f, 0.0833412334f, 0.0882256776f};
inline constexpr bool kLtx2Res2sTerminalZeroBongMoved = true;
inline constexpr float kLtx2Res2sTerminalZeroNoBongVideo[] = {
    0.443982691f, 0.447306544f, 0.699999988f,
    0.448465914f, 0.400000006f, 0.450835288f};

// Eta1: sigmas [0.9, 0.8, 0.7, 0.62], eta 1.0, h [0.117783, 0.133531, 0.121361], bong changed the result: True
inline constexpr double kLtx2Res2sEta1Eta = 1.0;
inline constexpr float kLtx2Res2sEta1Sigmas[] = {
    0.899999976f, 0.800000012f, 0.699999988f,
    0.620000005f};
inline constexpr int64_t kLtx2Res2sEta1SigmaCount = 4;
inline constexpr int64_t kLtx2Res2sEta1Evaluations = 6;
inline constexpr double kLtx2Res2sEta1EvalSigmas[] = {
    0.8999999761581421, 0.8485281325067245, 0.800000011920929,
    0.7483314765582876, 0.699999988079071, 0.658786760603827};
inline constexpr int64_t kLtx2Res2sEta1EvalStepIndices[] = {
    0, 0, 1,
    0, 2, 0};
inline constexpr float kLtx2Res2sEta1Video[] = {
    -0.513900995f, 0.366209567f, 0.699999988f,
    0.682274103f, 0.400000006f, -0.190410003f};
inline constexpr float kLtx2Res2sEta1Audio[] = {
    -0.362084478f, 0.400000006f, 0.0627472028f,
    0.699999988f, 0.390976906f, -0.051062014f};
inline constexpr bool kLtx2Res2sEta1BongMoved = true;
inline constexpr float kLtx2Res2sEta1NoBongVideo[] = {
    -0.510560513f, 0.363734573f, 0.699999988f,
    0.686390102f, 0.400000006f, -0.188381493f};

inline constexpr int64_t kLtx2Res2sLatentCount = 6;
inline constexpr float kLtx2Res2sVideo0[] = {
    0.0f, 0.166666672f, 0.333333343f,
    0.5f, 0.666666687f, 0.833333313f};
inline constexpr float kLtx2Res2sAudio0[] = {
    0.5f, 0.416666657f, 0.333333343f,
    0.25f, 0.166666672f, 0.0833333358f};
inline constexpr float kLtx2Res2sMask[] = {
    1.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 1.0f};
inline constexpr float kLtx2Res2sClean[] = {
    -0.300000012f, 0.200000003f, 0.699999988f,
    -0.100000001f, 0.400000006f, 0.0500000007f};

// samplers.py:160-170. `_get_new_noise` normalizes globally and then
// channelwise; the DRAW itself is torch.randn, whose stream this port
// does not have, so only the normalization is gated.
inline constexpr double kLtx2Res2sNoiseRaw[] = {
    -0.33333333333333326, -1.6666666666666667, 2.666666666666667,
    1.3333333333333335, 0.0, -1.3333333333333335};
inline constexpr double kLtx2Res2sNoiseNormalized[] = {
    -0.27066598098038347, -1.0826639239215337, 1.5563293906372047,
    0.7443314476960544, -0.06766649524509592, -0.8796644381862462};

}  // namespace vllm_test
