import importlib.util, sys, types, numpy as np, torch
from pathlib import Path
SRC = Path(__import__("os").environ.get("CAMPPLUS_LAYERS", "/tmp/campplus/layers.py"))
# load the UPSTREAM module by file path (no package __init__, no vllm deps)
spec = importlib.util.spec_from_file_location("cp_layers", SRC)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

MASK = (1 << 64) - 1
def fnv(n):
    h = 0xCBF29CE484222325
    for b in n.encode(): h ^= b; h = (h * 0x100000001B3) & MASK
    return h
def sm(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)
def rnd(name, n, scale=1.0):
    s = fnv(name); o = np.empty(n)
    for i in range(n): o[i] = (((sm((s + i) & MASK) >> 11) * 2.0**-53) * 2 - 1) * scale
    return o
def P(name, shape, scale=1.0):
    n = int(np.prod(shape))
    return torch.tensor(rnd(name, n, scale), dtype=torch.float64).reshape(shape).float()

torch.manual_seed(0)
C, T, BN, OUT, K, D = 6, 250, 4, 5, 3, 2   # T>100 so seg_pooling's ceil_mode bites
x = P("x", (1, C, T), 1.0)

# ---- StatsPool: mean + UNBIASED std ----
stats = m.StatsPool()(x)

# ---- BatchNorm1d in EVAL mode (running stats, not batch stats) ----
bn = torch.nn.BatchNorm1d(C).eval()
with torch.no_grad():
    bn.weight.copy_(P("bn.w", (C,), 0.5) + 1.0); bn.bias.copy_(P("bn.b", (C,), 0.3))
    bn.running_mean.copy_(P("bn.rm", (C,), 0.4)); bn.running_var.copy_(P("bn.rv", (C,), 0.2).abs() + 0.5)
bn_out = bn(x)

# ---- CAMLayer (incl. seg_pooling with ceil_mode) ----
cam = m.CAMLayer(BN, OUT, K, stride=1, padding=(K-1)//2*D, dilation=D, bias=False).eval()
xb = P("xb", (1, BN, T), 1.0)
with torch.no_grad():
    cam.linear_local.weight.copy_(P("cam.ll.w", tuple(cam.linear_local.weight.shape), 0.3))
    cam.linear1.weight.copy_(P("cam.l1.w", tuple(cam.linear1.weight.shape), 0.3))
    cam.linear1.bias.copy_(P("cam.l1.b", tuple(cam.linear1.bias.shape), 0.2))
    cam.linear2.weight.copy_(P("cam.l2.w", tuple(cam.linear2.weight.shape), 0.3))
    cam.linear2.bias.copy_(P("cam.l2.b", tuple(cam.linear2.bias.shape), 0.2))
cam_out = cam(xb)
seg = cam.seg_pooling(xb)


# ---- TransitLayer / DenseLayer / CAMDenseTDNNLayer / Block ----
IN, GROWTH, NL = 8, 3, 2
def fill(mod, prefix):
    with torch.no_grad():
        for n, q in mod.named_parameters():
            q.copy_(P(prefix + "." + n, tuple(q.shape), 0.3))
        for n, b in mod.named_buffers():
            if n.endswith("running_var"): b.copy_(P(prefix + "." + n, tuple(b.shape), 0.2).abs() + 0.5)
            elif n.endswith("running_mean"): b.copy_(P(prefix + "." + n, tuple(b.shape), 0.4))

xt = P("xt", (1, IN, T), 1.0)
transit = m.TransitLayer(IN, IN // 2, bias=True).eval(); fill(transit, "transit")
transit_out = transit(xt)

dense2d = m.DenseLayer(2 * C, 12, config_str="batchnorm_").eval(); fill(dense2d, "dense2d")
dense_out = dense2d(stats)                      # 2-D input path: unsqueeze -> conv -> squeeze

dl = m.CAMDenseTDNNLayer(IN, GROWTH, BN, K, dilation=D).eval(); fill(dl, "dl")
dl_out = dl(xt)

blk = m.CAMDenseTDNNBlock(NL, IN, GROWTH, BN, K, dilation=D).eval(); fill(blk, "blk")
blk_out = blk(xt)


# ---- BasicResBlock: Conv2d strides (stride, 1) -- FREQUENCY axis only ----
FIN, FPLANES, FH, FW = 2, 3, 16, 20
x4 = P("x4", (1, FIN, FH, FW), 1.0)
rb = m.BasicResBlock(FIN, FPLANES, stride=2).eval(); fill(rb, "rb")
rb_out = rb(x4)


# ---- the WHOLE CAMPPlus at reduced config ----
# Every parameter and buffer is rebuilt from the name stream by ONE rule the C++
# mirrors, so the manifest below is the contract: a tensor one side builds and
# the other does not is a failure, not a silent zero.
import importlib.util as _il
_dt = _il.spec_from_file_location("dt", SRC.parent / "DTDNN.py")
_dtm = _il.module_from_spec(_dt)
import sys as _sys
_sys.path.insert(0, str(SRC.parents[4]))
_dt.loader.exec_module(_dtm)

FEAT, EMB, GROW, BNSZ, INIT, FT = 32, 16, 2, 2, 8, 40
full = _dtm.CAMPPlus(feat_dim=FEAT, embedding_size=EMB, growth_rate=GROW,
                     bn_size=BNSZ, init_channels=INIT).eval()
MANIFEST = []
with torch.no_grad():
    for name, q in list(full.named_parameters()) + list(full.named_buffers()):
        if name.endswith("num_batches_tracked"):
            continue
        vals = P(name, tuple(q.shape) if q.dim() else (1,), 0.3)
        if name.endswith("running_var"):
            vals = vals.abs() + 0.5
        q.copy_(vals.reshape(q.shape))
        MANIFEST.append((name, list(q.shape)))
feats = P("feats", (1, FT, FEAT), 1.0)   # (B, T, F) -- forward permutes to (B,F,T)
_tap = {}
def _hook(_m, _i, o): _tap["tdnn"] = o.detach().clone()
full.xvector.tdnn.register_forward_hook(_hook)
full_out = full(feats)
tdnn_out = _tap["tdnn"]

def emit(f, name, t):
    a = np.asarray(t.detach().numpy(), dtype=np.float32).reshape(-1)
    f.write(f"inline constexpr float {name}[] = {{\n")
    for i in range(0, len(a), 6):
        f.write("    " + ", ".join(f"{v:.9e}F" for v in a[i:i+6]) + ",\n")
    f.write("};\n\n")

out = Path(sys.argv[1])
with out.open("w") as f:
    f.write("// GENERATED by scripts/gen-campplus-goldens.py -- do not edit.\n")
    f.write("// Upstream executed DIRECTLY (indextts/s2mel/modules/campplus/layers.py,\n")
    f.write("// index-tts @ main): it has no vllm dependency, so these are the real\n")
    f.write("// classes rather than a restatement. Weights rebuilt both sides from one\n")
    f.write("// FNV-1a -> splitmix64 stream; no weight byte is checked in.\n")
    f.write("#pragma once\n\n#include <cstdint>\n\nnamespace campplus_goldens {\n\n")
    for n, v in (("kChannels", C), ("kFrames", T), ("kBnChannels", BN), ("kOutChannels", OUT),
                 ("kKernel", K), ("kDilation", D), ("kSegLen", 100)):
        f.write(f"inline constexpr int64_t {n} = {v};\n")
    f.write("\n")
    emit(f, "kStats", stats); emit(f, "kBatchNormEval", bn_out)
    emit(f, "kSegPooling", seg); emit(f, "kCamOut", cam_out)
    for n, v in (("kIn", IN), ("kGrowth", GROWTH), ("kNumLayers", NL)):
        f.write(f"inline constexpr int64_t {n} = {v};\n")
    f.write("\n")
    emit(f, "kTransit", transit_out); emit(f, "kDense2d", dense_out)
    emit(f, "kDenseTdnnLayer", dl_out); emit(f, "kDenseTdnnBlock", blk_out)
    for n, v in (("kFcmIn", FIN), ("kFcmPlanes", FPLANES), ("kFcmH", FH), ("kFcmW", FW),
                 ("kFcmOutH", rb_out.shape[2]), ("kFcmOutW", rb_out.shape[3])):
        f.write(f"inline constexpr int64_t {n} = {v};\n")
    f.write("\n")
    emit(f, "kResBlock", rb_out)
    for n, v in (("kFeatDim", FEAT), ("kEmbedding", EMB), ("kGrowth2", GROW),
                 ("kBnSize", BNSZ), ("kInitChannels", INIT), ("kFullFrames", FT)):
        f.write(f"inline constexpr int64_t {n} = {v};\n")
    f.write(f"\ninline constexpr int64_t kManifestSize = {len(MANIFEST)};\n\n")
    f.write("struct ManifestEntry { const char* name; int64_t rank; int64_t d0, d1, d2, d3; };\n")
    f.write("inline constexpr ManifestEntry kManifest[] = {\n")
    for nm, sh in MANIFEST:
        d = list(sh) + [1, 1, 1, 1]
        f.write(f'    {{"{nm}", {len(sh)}, {d[0]}, {d[1]}, {d[2]}, {d[3]}}},\n')
    f.write("};\n\n")
    f.write(f"inline constexpr int64_t kTdnnChannels = {tdnn_out.shape[1]};\n")
    f.write(f"inline constexpr int64_t kTdnnFrames = {tdnn_out.shape[2]};\n\n")
    emit(f, "kTdnnOut", tdnn_out)
    emit(f, "kFullEmbedding", full_out)
    f.write("}  // namespace campplus_goldens\n")
print("wrote", out, "T=", T, "seg segments=", -(-T//100))
